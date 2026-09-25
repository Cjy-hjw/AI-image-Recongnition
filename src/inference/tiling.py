"""静态图滑窗 tiling 推理(Z1,新架构线第一波,2026-09-21)。

设计规格(设计文档已删)要点=tile 512/stride 384+概率域 max 融合,实现记录见总纲 §6.4 Z1 条。

## 为什么 tiling(零训练提升检测召回的推理侧手段)

现行静态链把任意 H×W 原图**直接 resize 到 512×512**
(`model_runner.preprocess_image`)。官方数据图幅跨度 6.5×(订正史 :685,典型
400×300 ~ 1024×768),长边 ≥700 的图经此压缩后小病灶像素数损失 30-50%——
1.42× 的像素保留率差足以把"IoU 差一点过 0.5"的临界带病灶压扁。训练分辨率
640 已判死(训练 config 误配),但**推理侧 tiling 从未实验过**(plan
swift-whistling-pearl Z1);本模块只改推理前向,权重零改动。

## 融合规则(🔴 预注册,改动须过 tests/test_tiling_inference.py)

- **掩膜 = 概率域逐像素 max**,不是平均。TTA 判死教训(第 9 次跑分 -0.94):
  平均操作压缩置信度分布、排序指标(AP)受损。max 保留尖峰,与
  `hysteresis_binarize` 的 seed 语义兼容(seed=高概率成核,重叠区取 max 不丢核)。
- **分类四头 + 良恶性 = 单 tile 输出,零融合**。取"融合图全局 argmax 像素
  (最强病灶响应点)所属 tile"的 outputs 直接 decode——位置头绝不能吃多
  tile 混合特征(左右叶语义),四头概率平均会把分布压平(与 TTA 覆辙同型)。
- **置信度解耦**:`detection_confidence` 在 `v21_output._write_static` 消费
  `soft_fullres`(= 融合后概率图 resize 回原图),实例区域内取均值。融合概率
  图取 max 后,实例若完整落在单一 tile 内,其 conf 与单 tile 路径逐位一致
  (max 只在 tile 重叠区改变数值);跨 tile 边界的实例,conf 反映"任一 tile 的
  峰值响应",不跨 tile 平均——这是 TTA 禁令的字面执行。

## 默认关(零回归红线)

`AICOMP_STATIC_TILING` 未设/"0" 时本模块零消费,`diagnose_static_case`
走原单前向路径逐位不变。判死回退同 TTA 先例:畸形值一律回关(安全侧)。
tiling 对 ≤512 的图是恒等路径(图小到放不进一个以上 tile 时仍走单前向,
不进融合代码,保证小图逐位零回归)。
"""

from __future__ import annotations

import os

import numpy as np
import torch

from src.inference.model_runner import preprocess_image, run_forward

# 静态图滑窗 tiling(Z1)。
# 🔴 2026-09-21 实现后即发现设计稿的"分类 tile 按融合胜出像素选"逻辑有
# 依赖循环(选 tile 依赖融合结果、融合又依赖逐 tile 前向)——本文件实际
# 实现改为**两遍式**:第一遍逐 tile 前向并融合概率;第二遍用融合图前景
# 中心的所属 tile 取分类 outputs。为省前向成本,第一遍时缓存每 tile 的
# outputs(引用不复制,显存代价仅在 GPU 上的 mask logits,≤4 tiles 可控)。
# 上面的模块 docstring 里"分类 tile=含最大分割实例"语义不变,实现从
# "胜出像素响应和"修正为"融合图全局 argmax 像素所属 tile"(见
# _select_classification_tile)。

# env 开关(默认关)。与 AICOMP_STATIC_TTA 同款三态解析:判死/未验完之前
# "忘了设 env"必须回退到关。
_ENV_TILING = "AICOMP_STATIC_TILING"
# tile 尺寸与重叠。tile=512 = 模型输入尺寸(不 resize,tile 原尺寸进模型);
# stride=384 = 重叠 128px,保证跨 tile 边界的病灶在至少一个 tile 内有完整
# 上下文(重叠小于病灶半径时会碎)。
_DEFAULT_TILE = 512
_DEFAULT_STRIDE = 384
_ENV_TILE = "AICOMP_TILING_TILE"
_ENV_STRIDE = "AICOMP_TILING_STRIDE"


def _tiling_enabled() -> bool:
    """解析 AICOMP_STATIC_TILING(畸形值回 False——未验证路径的安全侧=关)。"""
    raw = os.environ.get(_ENV_TILING, "0").strip().lower()
    return raw in ("1", "true", "on", "yes")


def _tiling_geometry() -> tuple[int, int]:
    """(tile, stride) 解析。非整数 env 回默认;数值越界 **fail-loud**(不静默钳)。

    Gate2 2026-09-21:tile∈[128,1024] 曾放行 640,preprocess 仍 resize 到 512,
    fusion `region`(tile²) vs `prob`(512²) RuntimeError;stride==tile 零重叠
    让 `_select_classification_tile` 缺上界时选中不含该像素的 tile。
    合法窗口:tile 必须 == `_DEFAULT_TILE`(模型 img_size);stride ∈
    [tile//2, tile)——重叠 >0,禁止滑窗退化成分块。
    """
    def _int_env(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            v = int(raw)
        except (TypeError, ValueError):
            return default
        return v

    tile = _int_env(_ENV_TILE, _DEFAULT_TILE)
    stride = _int_env(_ENV_STRIDE, _DEFAULT_STRIDE)
    if tile != _DEFAULT_TILE:
        raise ValueError(
            f"AICOMP_TILING_TILE must equal img_size={_DEFAULT_TILE}, got {tile}. "
            "tile≠img_size is resized back to 512 (benefit=0) and crashes fusion "
            "(region tile×tile vs prob 512×512)."
        )
    if not (_DEFAULT_TILE // 2 <= stride < _DEFAULT_TILE):
        raise ValueError(
            f"AICOMP_TILING_STRIDE must be in "
            f"[{_DEFAULT_TILE // 2}, {_DEFAULT_TILE}), got {stride}. "
            "stride>=tile is block-split (zero overlap)."
        )
    return tile, stride


def tile_origins(h: int, w: int, tile: int, stride: int) -> list[tuple[int, int]]:
    """滑窗起点 (y, x) 列表(含右/下边缘补齐,保证覆盖全图)。

    覆盖保证:起点序列按 stride 递增,最后一个起点钳到 `max(0, dim - tile)`
    ——否则尺寸非 stride 整数倍的图右侧/底部有永不被任何 tile 完整覆盖的
    条带(病灶在那里就直接消失,不是精度问题)。
    小图(两维都 ≤ tile)返回 [(0, 0)] 单起点,调用方走恒等路径。
    """
    def _axis(dim: int) -> list[int]:
        if dim <= tile:
            return [0]
        last = dim - tile
        pts = list(range(0, last + 1, stride))
        if pts[-1] != last:
            pts.append(last)
        return pts

    return [(y, x) for y in _axis(h) for x in _axis(w)]


def _crop_tile(img_np: np.ndarray, oy: int, ox: int, tile: int) -> np.ndarray:
    """零拷贝裁剪(超界钳位后 shape 可能小于 tile,pad 到 tile 见调用方)。"""
    h, w = img_np.shape[:2]
    y1, x1 = min(oy + tile, h), min(ox + tile, w)
    return img_np[oy:y1, ox:x1]


def _pad_square(tile_img: np.ndarray, tile: int) -> np.ndarray:
    """边缘 tile pad 到 tile×tile(右/下零填充,与 preprocess 的 resize 语义
    不同但合法:模型 assert 只看 512×512,零填充区概率图天然低响应)。"""
    if tile_img.shape[0] == tile and tile_img.shape[1] == tile:
        return tile_img
    out = np.zeros((tile, tile), dtype=tile_img.dtype)
    out[:tile_img.shape[0], :tile_img.shape[1]] = tile_img
    return out


def diagnose_static_case_tiled(model, img_np, config, calibrator, precision,
                              device, tile: int, stride: int) -> tuple:
    """滑窗多 tile 前向 → 概率域 max 融合 + 单 tile 分类头。

    Returns:
        (outputs_cls, mask_prob_fused):
        - outputs_cls: 分类 tile(融合图全局 argmax 像素所属 tile)的原始
          outputs dict,下游 decode 路径与单前向完全一致;
        - mask_prob_fused: [H, W] torch 张量(原图坐标系),供
          decode_predictions 的 mask_prob_override 消费。

    任何 tile 前向抛 RuntimeError OOM 时**整例冒泡**(不做 per-tile 降级:
    半 tiling 状态比整例 ERROR 更阴,与 D2 TTA 审查 F2 的哲学一致——但 tiling
    每张图 ≤4 前向、单 tile 显存与现行单前向相同,OOM 不可达)。
    """
    h, w = img_np.shape[:2]
    origins = tile_origins(h, w, tile, stride)

    img_size = config["data"]["img_size"]
    if tile != img_size:
        raise ValueError(
            f"tiling tile={tile} != config img_size={img_size}; "
            "preprocess resizes every tile to img_size, fusion then shape-mismatches."
        )
    prob_full = torch.full((h, w), -1.0, dtype=torch.float32)

    # 第一遍:逐 tile 前向 → max 融合;outputs 缓存供分类 tile 选择复用
    # (引用不复制;≤4 tiles 的 GPU mask logits 峰值与单前向同量级)。
    tile_outputs = []
    for oy, ox in origins:
        tile_img = _pad_square(_crop_tile(img_np, oy, ox, tile), tile)
        img_tensor = preprocess_image(tile_img, img_size, device)
        outputs = run_forward(model, img_tensor, precision, device)
        prob = torch.sigmoid(outputs["mask"].float())[0, 0].cpu()
        # 🔴 融合区裁剪到原图坐标系的实际区域:小图(dim≤tile)时 crop+pad 后
        # prob 是 tile×tile,但 prob_full 只有 dim×dim——直接广播会 shape
        # mismatch 崩,且 pad 区概率(模型对黑边的响应)必须丢弃不能进融合。
        region = prob_full[oy:oy + tile, ox:ox + tile]
        rh, rw = region.shape
        region[:] = torch.maximum(region, prob[:rh, :rw])
        tile_outputs.append((oy, ox, outputs))

    # 未覆盖像素(理论不可达:tile_origins 保证全覆盖;防御)置 0 概率。
    prob_full[prob_full < 0.0] = 0.0

    # 分类 tile:融合图全局 argmax 像素(= 最强病灶响应点)所属 tile。
    flat_i = int(torch.argmax(prob_full))
    cy, cx = divmod(flat_i, w)
    best_outputs = _select_classification_tile(tile_outputs, (cy, cx), tile)
    return best_outputs, prob_full


def _select_classification_tile(tile_outputs, argmax_px, tile: int) -> dict:
    """融合图全局 argmax 像素 (cy, cx) → 所属 tile 的 outputs。

    所属判据:像素落在 tile 半开区间 [oy, oy+tile) × [ox, ox+tile) 内
    (非 pad 区;Gate2:缺上界时 stride==tile 会选中不含该像素的更早 tile)。
    多个 tile 重叠时取**起点最靠近该像素的**(tile 内偏移越靠中心上下文越完整)。
    """
    cy, cx = argmax_px
    best = None
    best_d = None
    for oy, ox, outputs in tile_outputs:
        if not (oy <= cy < oy + tile and ox <= cx < ox + tile):
            continue
        # 该像素在该 tile 内的相对偏移(取 tile 中心的距离,近者优先)
        dy = abs((cy - oy) - tile // 2)
        dx = abs((cx - ox) - tile // 2)
        d = dy + dx
        if best_d is None or d < best_d:
            best_d = d
            best = outputs
    if best is None:
        # 不可达(全图被 tile 覆盖);防御回首个 tile。
        return tile_outputs[0][2]
    return best
