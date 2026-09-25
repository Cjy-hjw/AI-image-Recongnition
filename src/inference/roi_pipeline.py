"""推理侧 per-lesion ROI 分流(两阶段 ROI 范式推理面,2026-09-25 新架构线)。

## 定位

现行推理链把任意尺寸原图 resize 到 512² 后**整图一次前向**,四头输出是
**整图级**(`stage4_spatial` 全局池化,`heads/*.py` 无实例维度),于是
`v21_output._shared_nodule_context` 只能把**同一份四头值**写进每个结节的
`prediction.json`(`_lesion_level_head_preds` docstring 的"一个必须知道的前提"
那段已把这个局限写明)。本模块是二阶段范式的推理侧入口:分割实例 → 逐实例在
**原分辨率**裁 ROI → ROI 级前向 → **每结节一份四头值**。

## 契约(钉死,勿自创)

- 模型面 `model.forward_roi(roi_images, roi_meta)` 的逐病灶返回 dict 键与
  `src/training/collect_roi_records.py` 的输入契约**逐字一致**(那份是训练侧
  消费方,本模块是推理侧生产方,两侧必须对得上):
  `lesion_id` / `is_main` / `patient_id`(可选) / `malignancy`[2] logits /
  `location_probs`[7] / `ete_probs`[4] / `ete_logits`[4] / `t_conf` 标量 /
  `tirads_probs`[5]。
- ROI 元数据形态与 `src/dataset/roi_dataset.py` 的 `roi_meta` 对齐:
  `bbox_xyxy` 半开区间原图坐标 / `orig_w` / `orig_h` / `spacing_mm` /
  `lesion_id`。裁剪口径 = **训练侧同款紧切片**(`ROI_PAD_RATIO=0.0`,见该
  常量注释的 P0 对齐裁决)—— 复用 `nodule_split.crop_nodule` 的钳位/返回
  几何,边距参数显式传 0 对齐 `RoiDataset.crop_roi`;不另抄一份裁剪逻辑。

## 动态侧的内存红线(🔴 勿改)

动态病例逐实例**串行**消费、**帧在线裁**(读到一帧立刻裁该实例的 ROI 用完即弃),
**禁止**把 (3D 实例 × 全帧) 展开成列表预裁:单动态例 750 帧 × 每帧原分辨率
float32 ROI,数量级就是几十 GB —— 与 worker=1 也 OOM 的历史事故同源
(`memory/worker1-single-case-memory-oom-and-fixes.md`:单大动态例 30GB+,修法是
稀疏存储 + 用完即弃)。故本模块的接口设计成**生成器逐帧喂**,而不是"先给全部帧"。

## 零回归红线

`AICOMP_ROI_HEADS` 未设(=关)时本模块**零消费**:`case_pipeline` 不 import 它、
`v21_output` 走原整图共享路径,两者逐位不变。开了开关但模型没有 `forward_roi`
(如交付容器里的旧权重/旧代码)→ **fail-loud raise**,不静默回退整图路径 ——
静默回退会让"开了 ROI 却在跑整图"这种半开态伪装成成功,而 prediction.json
完全合法、只是四头全是整图共享值,本地永远查不出(与 TTA 判死回退、tiling
恒等路径同款风险面,处置取"显式报错"而非"静默劣化")。
"""

import os

import numpy as np
import torch

from src.inference.model_runner import preprocess_image, run_forward
from src.inference.nodule_split import crop_nodule

__all__ = [
    "ROI_PAD_RATIO",
    "RoiPipelineError",
    "roi_heads_enabled",
    "_roi_instance_bbox",
    "crop_roi_for_instance",
    "roi_meta_for_instance",
    "build_roi_batch",
    "run_roi_forward",
    "iter_instance_frames_static",
    "iter_instance_frames_dynamic",
]

# env 开关。三态解析:未设/"0"/垃圾串一律关(未验证路径的安全侧,与
# `_tiling_enabled`/`_shrink_enabled`/`_static_tta_enabled` 同款)。
_ENV_ROI_HEADS = "AICOMP_ROI_HEADS"

# 🔴 ROI 上下文边距 = 0.0(2026-09-26 R-A F1 P0 修):必须与训练侧
# `RoiDataset.crop_roi` 的 **bbox 紧切片**逐位同口径 —— 训练侧 GT bbox 是
# mask 精确外接框、无上下文外扩,35 epoch 里 ROI 头学的是紧框输入分布;
# 推理侧旧取 0.25 外扩(继承 `crop_nodule` 默认)→ 同一结节推理 ROI 多出成圈
# 外部组织,结节占比/band 分布系统性漂移,输出合法但静默劣化(实测:训练 ROI
# 结节占比 100% vs 推理侧 44.4%)。pad_to_square 差异经对拍为像素恒等
# (cv2 双线性按轴分离重采样),可忽略;真正差异只在边距,故只改此一处。
# 训练侧加 0.25 边距(交界判据增益)属下一轮设计议题,须两侧同步改。
ROI_PAD_RATIO = 0.0


class RoiPipelineError(RuntimeError):
    """ROI 分流不可用(模型无 forward_roi / 实例无有效 bbox / 裁剪为空)。

    独立于 `V21OutputError`:本异常是"ROI 路径自身不成立",由 `case_pipeline`
    或 `v21_output` 的调用方决定是整例 ERROR 还是回退 —— 但**不静默回退**(见
    模块 docstring 的零回归/半开态红线)。
    """


def roi_heads_enabled() -> bool:
    """解析 `AICOMP_ROI_HEADS`(畸形值回 False —— 未验证路径的安全侧=关)。

    与 `tiling._tiling_enabled` 逐字同款("1"/"true"/"on"/"yes" 为真,其余全假)。
    """
    raw = os.environ.get(_ENV_ROI_HEADS, "0").strip().lower()
    return raw in ("1", "true", "on", "yes")


def _roi_instance_bbox(inst: dict) -> tuple:
    """`split_nodules` 实例 → 原图坐标半开区间 `(x0, y0, x1, y1)`。

    🔴 与 `nodule_split` 的两套 bbox 口径分别对应,别混:
    - **2D 实例**:`bbox` 是 `(x, y, w, h)`(xywh),此处转成半开区间的
      `(x, y, x+w, y+h)`(与 `official_loader._roi_lesion_meta` 的
      `bbox_xyxy` 同口径:x1/y1 = 最大非零列/行 + 1)。
    - **3D 实例**:`bbox` 多带 `(z0, z1)` 尾部,前四项**仍是 (x, y, w, h)**
      (`_split_3d` 的 `nodules.append` 逐字如此),此处取前四项同样转 xyxy,
      **忽略 z** —— ROI 是逐帧 2D 裁剪,z 由逐帧循环决定。

    Raises:
        RoiPipelineError: 实例无 `bbox` 键或长度不足 4。
    """
    bb = inst.get("bbox")
    if bb is None or len(bb) < 4:
        raise RoiPipelineError(
            f"实例缺 bbox 或长度不足 4(收到 {bb!r})——ROI 裁剪无几何依据;"
            "实例应来自 nodule_split.split_nodules"
        )
    x, y, w, h = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
    return (x, y, x + w, y + h)


def crop_roi_for_instance(frame: np.ndarray, inst_bbox_xyxy, pad_ratio: float = ROI_PAD_RATIO):
    """原分辨率帧 + 实例 xyxy → `(roi_patch, (x0, y0, x1, y1))`。

    复用 `nodule_split.crop_nodule`(单一权威裁剪口径),把 `bbox_xyxy` 转成它要的
    `(x, y, w, h)`。返回的第二项是**实际生效**的裁剪框(已按图像边界钳位),
    消费方靠它把 ROI 坐标映射回原图(坐标可逆性是本模块的验收门之一)。

    🔴 精度口径:`crop_nodule` 内部对 pad 取 `int(round(w*pad_ratio))` 且按轴独立
    计算(细长结节不被短边撑方),本函数**不重复计算 pad**、不自己钳位 —— 任何
    二次换算都会与那条唯一权威分叉,而分叉的表现是"ROI 偏一点",不报错。
    """
    x0, y0, x1, y1 = inst_bbox_xyxy
    bbox_xywh = (int(x0), int(y0), int(x1) - int(x0), int(y1) - int(y0))
    return crop_nodule(frame, bbox_xywh, pad_ratio=pad_ratio)


def roi_meta_for_instance(inst: dict, lesion_id, orig_hw, bbox_xyxy=None,
                          spacing_mm=None, patient_id=None, is_main=1,
                          frame_index=None) -> dict:
    """构造喂给 `model.forward_roi` 的 per-lesion 元数据 dict。

    形态与 `src/dataset/roi_dataset.py` 的 `roi_meta` 对齐(同键同义),并补齐
    推理侧特有字段。键集:

    - `bbox_xyxy` — 半开区间原图坐标。🔴 E2 review P1-1 修复(2026-09-25):
      记**实例半开框**(与训练侧 `RoiDataset` 的 GT 半开同口径,loc 头
      CoordConv 坐标映射两侧一致);**不是** `crop_roi_for_instance` 的
      pad 后实际框(那是裁剪几何,patch 还原用,不冒充病灶位置元数据);
    - `orig_w` / `orig_h` — 原图尺寸(该帧尺寸);
    - `pad_hw` — `(0, 0)`:本模块走 `crop_nodule` 的按轴 pad(不 pad 到正方形),
      故无"pad 到方形"这一步的补边量;该键保留是为了**键集对齐**训练侧
      `RoiDataset.__getitem__` 的 meta(消费方按 `.get()` 读,缺键会静默取默认)。
      真正的裁剪几何以 `bbox_xyxy` 为准(那是唯一无损还原依据);
    - `spacing_mm` — 官方无 spacing 来源,恒 None 透传(`official_loader`
      docstring 的"不得编造"条);
    - `lesion_id` — 病灶唯一键(病灶级聚合前提,动态多帧同病灶同 id);
    - `patient_id` / `is_main` / `frame_index` — 推理侧补齐(训练侧 meta 里
      `patient_key` 承担患者键,此处两者都给,消费方读哪个都不缺);
    - `global_loc` — ROI 中心在原图的归一化坐标 `(cx/orig_w, cy/orig_h)`,
      与 `roi_dataset.expand_roi_samples` 同式(loc 头的全局位置输入)。
    """
    x0, y0, x1, y1 = bbox_xyxy
    ow, oh = int(orig_hw[1]), int(orig_hw[0])
    cx = ((int(x0) + int(x1)) / 2.0) / float(ow)
    cy = ((int(y0) + int(y1)) / 2.0) / float(oh)
    return {
        "bbox_xyxy": (int(x0), int(y0), int(x1), int(y1)),
        "orig_w": ow,
        "orig_h": oh,
        "pad_hw": (0, 0),
        "spacing_mm": spacing_mm,
        "lesion_id": lesion_id,
        "patient_key": patient_id,
        "patient_id": patient_id,
        "is_main": int(is_main),
        "frame_index": frame_index,
        "global_loc": (cx, cy),
        # is_dynamic 补齐(2026-09-25 推理侧接线):训练侧 `RoiDataset.__getitem__`
        # 的 meta 键集含 is_dynamic,消费方按键集契约校验时缺键会静默漏检
        # (键集对齐是 P1 验收门,见 tests/test_roi_inference.py 的键集断言)。
        # 静态(frame_index=None)→ False,动态 → True,与 roi_dataset 同义。
        "is_dynamic": frame_index is not None,
    }


def iter_instance_frames_static(frame: np.ndarray, inst: dict, lesion_id,
                                patient_id=None, is_main=1):
    """静态实例 → 单帧 ROI 生成器(yield `(roi_patch, roi_meta)`)。

    静态只有一帧,但**仍走生成器接口**——让动/静两条链路的消费端逐字相同
    (`case_pipeline` 侧一个循环吃两路),避免"静态走 A 接口、动态走 B 接口"
    导致两条链各自演化出错(本项目已有过多起镜像漂移事故)。
    """
    bbox = _roi_instance_bbox(inst)
    patch, actual_box = crop_roi_for_instance(frame, bbox)
    # 🔴 E2 review P1-1 修复(2026-09-25):meta 的 bbox_xyxy 此前记
    # **pad 后实际裁剪框**(actual_box),而训练侧 `RoiDataset` 的 meta 记
    # **GT 半开外接框**(r["bbox_xyxy"],无 pad;crop 只在右/下 pad-to-square
    # 不改框起点)。loc 头 CoordConv 的 x_orig 映射消费的就是这个 bbox——
    # 两侧口径漂移 pad 量(w×0.25 各方向)。现统一记**实例半开框**(bbox,
    # 与训练侧同口径);actual_box 仍是裁剪几何真相(patch 还原靠它,
    # pad_hw 键继续透传),但**不再冒充病灶位置元数据**。
    meta = roi_meta_for_instance(
        inst, lesion_id, orig_hw=frame.shape[:2], bbox_xyxy=bbox,
        patient_id=patient_id, is_main=is_main, frame_index=None)
    yield patch, meta


def iter_instance_frames_dynamic(frame_by_index, inst: dict, lesion_id,
                                 patient_id=None, is_main=1):
    """动态实例 → 逐帧 ROI 生成器(**帧在线消费,禁批量预裁**)。

    Args:
        frame_by_index: `{原始帧号: 原分辨率帧 np.ndarray}`。实例的
            `frame_indices` 是**原始帧号**,直接按它取 —— 不要用列表下标
            (抽样回填后下标与帧号不等,错取会静默拿到别的帧)。
        inst: `split_nodules(3D)` 的实例(含 `frame_indices` 与 xywh 形态 bbox)。
        lesion_id: 病灶唯一键(多帧同 id,病灶级聚合前提)。
        patient_id / is_main: 透传给 `roi_meta`。

    Yields:
        `(roi_patch, roi_meta)`,每帧一项。**逐帧调用方负责用完即弃**:
        ROI patch 不进任何跨帧列表。

    🔴 为何是生成器而不是返回 list:见模块 docstring 的内存红线。把逐帧 ROI
    攒成 list 就是"实例 × 全帧展开",单动态例几十 GB 级。
    """
    bbox = _roi_instance_bbox(inst)
    # 🔴 空间幅面按**实例 bbox 的并集**算一次(`_split_3d` 的 bbox 本就是该实例
    # 全部帧非零像素的并集,与 `official_loader._roi_lesion_meta` 的动态口径
    # 逐字一致),逐帧沿用同一 bbox —— 病灶级 ROI 元数据不随帧变(同 lesion_id
    # 只有一个 bbox,否则下游按 lesion_id 聚合会拿到互相矛盾的几何)。
    for fi in inst.get("frame_indices") or []:
        frame = frame_by_index.get(int(fi))
        if frame is None:
            # 实例帧必来自有预测的抽样帧,理论上不可达;仍显式跳过而非静默取
            # 邻近帧(取错帧 = 用别的帧的四头值当本帧的,完全不报错)。
            continue
        patch, actual_box = crop_roi_for_instance(frame, bbox)
        # 🔴 E2 review P1-1 修复(2026-09-25,与静态路同款):meta 的
        # bbox_xyxy 记**实例半开框**(训练侧同口径),非 pad 后裁剪框。
        meta = roi_meta_for_instance(
            inst, lesion_id, orig_hw=frame.shape[:2], bbox_xyxy=bbox,
            patient_id=patient_id, is_main=is_main, frame_index=int(fi))
        yield patch, meta


def build_roi_batch(patches, roi_metas, img_size: int, device):
    """ROI patch 列表 → 模型输入张量 `[N,1,S,S]`。

    `preprocess_image`(单一权威预处理:resize 到 img_size + 归一化到 [0,1])
    **逐张**调用后 stack。逐张而非一次 cv2.resize 多通道:与生产单帧路径
    (`case_pipeline`/`tiling`)走**同一个函数**,不引入第二条 resize 实现
    (镜像漂移风险面)。

    Raises:
        RoiPipelineError: patches 为空(调用方不该在零实例时调本函数)。
    """
    if not patches:
        raise RoiPipelineError("build_roi_batch 收到空 ROI 列表(零实例时不应调用)")
    tensors = [preprocess_image(p, img_size, device) for p in patches]
    return torch.cat(tensors, dim=0)


def run_roi_forward(model, img_tensor, roi_metas, precision, device):
    """ROI 批量前向 → `list[dict]`(每病灶一条,键同 `collect_roi_records` 输入契约)。

    🔴 **模型面契约缺失即 fail-loud**:`model` 上没有 `forward_roi` 说明加载的是
    不含 ROI 头的旧结构(交付容器里的旧权重/旧代码),此时**绝不回退整图路径**
    (静默回退 = 开了开关却跑整图,输出合法但四头全是整图共享值,本地查不出)。
    错误信息直指代码/权重成对耦合红线(`memory/deepsup-implementation-and-coupling`)。

    Raises:
        RoiPipelineError: 模型无 `forward_roi`,或返回条数与输入不齐。
    """
    forward_roi = getattr(model, "forward_roi", None)
    if forward_roi is None:
        raise RoiPipelineError(
            f"模型 {type(model).__name__} 没有 forward_roi —— 加载的是不含 ROI 头的"
            "旧结构。ROI 分流开了但模型侧未就位:按代码/权重成对耦合红线,更新模型"
            "代码后回退旧权重须连 src 一起回退;此处**不静默回退整图路径**"
            "(静默回退会让四头全是整图共享值而 prediction.json 完全合法)。"
        )
    with torch.inference_mode():
        # 🔴 E2 review P0-1 修复(2026-09-25):此前是三参调用
        # `forward_roi(model, img_tensor, roi_metas)` —— bound method 下模型
        # 实例落 self 后,img_tensor 挤进 roi_images 位、roi_metas 落
        # seg_features 位,真 MultiTaskThyroidModel 直接 AttributeError
        # (:461 `roi_images.shape`)。测试全绿纯因 FakeRoiModel 的 fake
        # 签名把生产错位写进了契约(fake 契约必须与真实签名逐参一致,
        # memory/fake-must-match-real-return-contract 同型)。
        out = forward_roi(img_tensor, roi_metas)
    # 真模型 forward_roi 返回**整包 dict**(含 roi_outputs 键+9 个观测键,
    # multi_task_model.py:583-594);fake 只回 list。两形态都吃:dict 取
    # "roi_outputs" 键,list 直用(FakeRoiModel 既有契约,测试锚)。🔴 此前
    # 直接把 dict 当 list 用,`len(dict)=11` 撞上"逐条对齐"守卫——那是
    # P0-1 修复后立刻暴露的第二层错位(fake 从未覆盖 dict 形态)。
    if isinstance(out, dict):
        if "roi_outputs" not in out:
            raise RoiPipelineError(
                "forward_roi 返回 dict 但缺 roi_outputs 键 —— 契约不符"
                f"(得到键: {sorted(out.keys())[:8]})"
            )
        roi_outputs = out["roi_outputs"]
    else:
        roi_outputs = out
    if len(roi_outputs) != len(roi_metas):
        raise RoiPipelineError(
            f"forward_roi 返回 {len(roi_outputs)} 条,输入 {len(roi_metas)} 条 —— "
            "逐病灶输出与输入必须逐条对齐(错位会让四头值挂到别的结节上,不报错)"
        )
    return roi_outputs
