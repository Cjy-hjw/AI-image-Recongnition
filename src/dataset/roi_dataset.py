"""per-lesion ROI 采样器 + 独立 collate —— 两阶段 ROI 范式的数据面(方案一 P1,2026-09-25)。

## 定位

第一阶段(现有分割器,不改)在全图出候选;第二阶段逐实例在**原分辨率**裁 ROI,
做 ROI 内掩膜精修 + 按病灶单独分类。本模块把 `build_official_samples(roi_meta=True)`
产出的 per-lesion 元数据(`roi_lesions` 键)展开成**病灶级训练样本**:

    每个样本 = (ROI 图像 tensor[1,S,S], roi_meta dict)

与 `OfficialThyroidDataset`(整图样本)严格分库 —— 共用一个 DataLoader 会让
batch 语义在"图级 [B]"与"病灶级 [B*K]"之间漂移(scoping §A.1 #13 风险行原文)。

## 每条 ROI 样本的产出契约

`__getitem__` 返回 `(img_t, roi_meta)`:

- `img_t`: `[1, roi_size, roi_size]` f32 ∈ [0,1] —— 原分辨率裁剪(bbox 半开区间
  xyxy 直接切片)→ **aspect 保不裁、短边 pad** → resize 到 roi_size。裁剪用
  `resize_img_to`/`resize_mask_to` 的口径(单一权威,勿另抄插值)。
- `roi_meta`: dict,键:
  - `bbox_xyxy` (tuple[int,int,int,int]) — 原图像素坐标半开区间,还原裁剪用;
  - `orig_w` / `orig_h` (int) — 原图尺寸(= 裁剪帧尺寸);
  - `pad_hw` (tuple[int,int]) — pad 到正方形前,右侧/下侧补的像素数(还原时裁掉);
  - `spacing_mm` (None) — 官方无 spacing 来源,恒 None 透传(见 official_loader
    `build_official_samples` docstring 的"不得编造"条);
  - `lesion_id` (str) — 病灶唯一键(与整图样本同式,病灶级聚合前提);
  - `patient_key` (str) — 患者分组切分键;
  - `is_dynamic` / `frame_index` — 动态样本携带该 ROI 对应的帧号(原 0-based);
  - `global_loc` (tuple[float,float]) — ROI 中心在原图的归一化坐标
    (cx/orig_w, cy/orig_h, ∈ [0,1])— loc 头的全局位置输入(方案一 #4);
  - `label 子集` — loc/ete/t/tirads/malignancy 的**训练侧 0-based 值 + valid_***,
    与 `_build_labels` 同键同口径。

## 🔴 独立 collate 的原因(勿"顺手"用 default_collate)

`lesion_id` 是 32 位 hex 字符串(或 `img_dir::roi` 复合串)。`default_collate` 对
list[str] 会按**字符维堆叠**成 [B, 32] 的字符矩阵——长度不齐时直接报错、
长度恰好对齐时静默变矩阵,两种都是错(下游 `aggregate_by_lesion` 等按 str 消费)。
本模块的 `roi_collate` 显式保 list[str];数值键堆成 batch tensor,其余 list 透传。

## 数据源三态

- 整图样本带 `roi_lesions` 键(build 时 roi_meta=True):正常展开;
- 整图样本**无**该键(旧 pkl / 默认 build):`__init__` 时跳过该样本并计数
  (`self.n_skipped_no_meta`)——元数据缺失的样本构不出 ROI 样本,跳过是唯一
  安全行为(不编造 bbox);数量在构造时 `_safe_print` 一行,零反馈才是最隐蔽的
  失败模式。
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# 复用(单一权威,勿另抄):resize 口径 / _read_dcm_gray / 动态帧读取 / 安全打印
from src.dataset.official_loader import (
    resize_img_to,
    resize_mask_to,
    _read_dcm_gray,
    _read_nifti_frame_gray,
    _read_nifti_binmasks,
    _safe_print,
)

__all__ = ["RoiDataset", "roi_collate", "expand_roi_samples",
           "roi_labels_from_meta"]

# roi_meta 里随 collate 堆成 batch tensor 的数值键(int/float 标量)。
# label 值与 valid_* 全部是 python int —— default 语义下堆成 [B] int64。
# 债6(2026-09-25):概念四键(composition/echogenicity/shape/margin +
# valid_*)并入——ROI 概念监督(concepts_roi)的标签经桥透传;foci 官方
# 无标注恒不携带(与 _ROI_LABEL_KEYS 同批改,键名单源)。
_ROI_SCALAR_KEYS = (
    "malignancy", "valid_malignancy",
    "lesion_loc", "valid_lesion_loc",
    "extrathy_ext", "valid_extrathy_ext",
    "clin_t_stage", "valid_clin_t_stage",
    "tirads", "valid_tirads",
    "is_main",
    "composition", "valid_composition",
    "echogenicity", "valid_echogenicity",
    "shape", "valid_shape",
    "margin", "valid_margin",
)

# 债1(2026-09-25 训练侧接线):数据面 meta 产**无前缀键**(lesion_loc/…),
# 损失/尺子面吃 **gt_ 前缀键**(`components/roi_heads.py` 的 targets 契约 +
# `collect_roi_records` 的 labels 契约)。改名映射单源在此;桥的**调用点**在
# 训练循环(run_roi_epoch/validate_roi_epoch,scripts/train_official.py),
# 别在数据面自动改键(推理侧 roi_pipeline 消费同一批无前缀 meta 键,改了
# 破坏已闭环的推理侧)。
# 债6(2026-09-25):概念四键同式加 gt_ 前缀(concepts_roi 消费 gt_composition
# 等——官方 xlsx 概念列实测名,勿用 DR 报告 comp/echo 简写)。
_ROI_LABEL_RENAME = {
    "malignancy": "gt_malignancy",
    "lesion_loc": "gt_lesion_loc",
    "extrathy_ext": "gt_extrathy_ext",
    "clin_t_stage": "gt_clin_t_stage",
    "tirads": "gt_tirads",
    "composition": "gt_composition",
    "echogenicity": "gt_echogenicity",
    "shape": "gt_shape",
    "margin": "gt_margin",
}


def roi_labels_from_meta(roi_meta_list: List[Dict]) -> List[Dict]:
    """per-lesion meta list → 损失/尺子面的 roi_labels(债1 的纯改名桥)。

    逐条 dict:5 个 label 键加 gt_ 前缀;valid_* / lesion_id / is_main /
    patient_key / patient_id 原样透传(消费方读的就是这些名字——两侧契约
    均为收编产物+测试钉死,勿动)。collate 后的 [K] int64 tensor 与原始
    python int 均可(透传原值,消费方 `_scalar`/`_label_val` 自解标量)。

    K 对齐:与输入等长逐条对位(roi_outputs 与 meta 同 list 长度,天然对齐)。
    collate 后的 metas 是 **dict-of-lists**(roi_collate 产物:数值键 [K]
    tensor、str/list 键 list)——本函数对两种形态统一:输入是 dict(批形态)
    时按键转置逐条重建,输入是 list[dict] 时逐条改名(幂等)。
    """
    if hasattr(roi_meta_list, "keys"):      # roi_collate 批形态 → 转置
        rows = []
        for k, v in roi_meta_list.items():
            if torch.is_tensor(v):
                v = v.tolist()
            if not isinstance(v, (list, tuple)):
                v = [v]                     # 标量键(理论不该有,防御)
            rows.append((k, list(v)))
        if not rows:
            return []
        K = min(len(v) for _, v in rows)
        roi_meta_list = [
            {k: v[i] for k, v in rows if i < len(v)} for i in range(K)]
    out: List[Dict] = []
    for m in roi_meta_list:
        d: Dict = {}
        for k, v in m.items():
            if k in _ROI_LABEL_RENAME:
                d[_ROI_LABEL_RENAME[k]] = v
            else:
                d[k] = v
        out.append(d)
    return out


def expand_roi_samples(samples: List[Dict]) -> List[Dict]:
    """整图样本列表 → ROI 样本列表(纯展开,不读像素)。

    每个**整图样本**的 `roi_lesions` 里每个病灶 → 按**交集帧**展开:
    - 动态整图样本自身带 `frame_index`(该样本 = "某病灶在帧 f 的一个样本"),
      ROI 展开时只取该帧落在此病灶 `frame_indices` 内的病灶 → 该样本产出
      "在帧 f 可见的全部病灶"的 ROI 样本各一条。同一(病灶,帧)组合在多
      个整图样本间天然重复(同图样本共享同一 `roi_lesions` 列表),按
      (lesion_id, frame_index) 去重——同病灶同帧只产一条 ROI 样本。
    - 静态整图样本(无帧维)→ 该图全部病灶各一条。
    无 `roi_lesions` 键的整图样本跳过(旧 pkl 半开态,允许——元数据是增强
    不是监督语义,跳过不产生"号称开了实际没开")。

    返回样本 dict 键:
      img_path, is_dynamic, frame_index(动态:该 ROI 样本的帧;静态 None),
      patient_key, lesion_id, 以及 roi_meta dict 的全部键(bbox_xyxy/orig_w/
      orig_h/spacing_mm/global_loc/label 子集)。
    """
    out: List[Dict] = []
    seen = set()          # (lesion_id, frame_index) 去重:同图样本共享 roi_lesions
    # 🔴 R3 P2-3 修(2026-09-26):丢弃点计数——bbox=None 的病灶(有标注但
    # mask 文件缺失,构不出 ROI 几何)在下方 continue 被静默丢,而
    # RoiDataset.__init__ 的 n_skipped_no_meta 只滤"手构入参无 bbox"(生产链
    # 经本函数展开后恒 0 = 死计数器)。真正的丢弃必须可观测:返回前打一行,
    # 零反馈静默丢监督是最隐蔽的失败模式(r7 同型)。
    n_dropped_no_bbox = 0
    for s in samples:
        lesions = s.get("roi_lesions")
        if not lesions:
            continue
        for les in lesions:
            if les.get("bbox_xyxy") is None:
                n_dropped_no_bbox += 1   # 无 mask / 读失败:构不出 ROI 几何,跳过
                continue
            x0, y0, x1, y1 = les["bbox_xyxy"]
            cx = ((x0 + x1) / 2.0) / float(les["orig_w"])
            cy = ((y0 + y1) / 2.0) / float(les["orig_h"])
            common = {
                "img_path": s["img_path"],
                "is_dynamic": s["is_dynamic"],
                "patient_key": s["patient_key"],
                # 债5(2026-09-25 训练侧接线):forward_roi 输出侧消费 patient_id
                # (multi_task_model.py:574 m.get("patient_id")),数据面此前只产
                # patient_key——loader 的 _entry 现已带 patient_id,这里透传;
                # 无键(手构/旧入口)→ None(forward_roi 侧 .get() 容忍)。
                "patient_id": les.get("patient_id"),
                # 债2:GT mask 路径随样本下发(RoiDataset.__getitem__ 据此产
                # roi_mask);无 mask 病灶(loader 无 mask 分支)→ None。
                "mask_path": les.get("mask_path"),
                "lesion_id": les["lesion_id"],
                "bbox_xyxy": les["bbox_xyxy"],
                "orig_w": les["orig_w"], "orig_h": les["orig_h"],
                "spacing_mm": les.get("spacing_mm"),
                "global_loc": (cx, cy),
                **{k: les[k] for k in _ROI_SCALAR_KEYS if k in les},
            }
            if s["is_dynamic"]:
                # 交集帧:整图样本帧 s.frame_index ∈ 此病灶 frame_indices 才产出
                # (该病灶在此帧有 GT 标注;其余病灶的帧不产此病灶的样本)。
                fidxs = [s["frame_index"]] if (
                    s["frame_index"] in (les.get("frame_indices") or [])) else []
                for fidx in fidxs:
                    key = (les["lesion_id"], fidx)
                    if key in seen:
                        continue
                    seen.add(key)
                    r = dict(common)
                    r["frame_index"] = fidx
                    out.append(r)
            else:
                key = (les["lesion_id"], None)
                if key in seen:
                    continue
                seen.add(key)
                r = dict(common)
                r["frame_index"] = None
                out.append(r)
    if n_dropped_no_bbox:
        _safe_print(
            f"--> [roi_dataset] ⚠️ expand_roi_samples 丢弃 {n_dropped_no_bbox} 个"
            f"无 bbox 病灶(mask 缺失/读失败,该病灶零 ROI 监督,无独立整图样本)")
    return out


class RoiDataset(Dataset):
    """病灶级 ROI 数据集。`__getitem__` → `(img[1,S,S] f32∈[0,1], roi_meta dict)`。

    裁剪管线(坐标可逆性是 P1 验收门:裁→还原逐像素相等):
        原分辨率帧 → bbox 切片(半开区间,零换算)→ 右/下 pad 到正方形
        (pad_hw 记录补了多少,还原时裁掉)→ resize 到 roi_size
        (`resize_img_to` 单一权威口径)。
    读取路径与 `OfficialThyroidDataset` 同源:`_read_dcm_gray`(静态)/
    `_read_nifti_frame_gray`(动态指定帧)。
    """

    def __init__(self, roi_samples: List[Dict], roi_size: int = 224):
        self.roi_size = int(roi_size)
        # 无 bbox 的样本在 expand 阶段已滤;这里再防一层(手构样本容错),
        # 计数并打一行 —— 静默跳过是零反馈失败模式。
        keep = [r for r in roi_samples if r.get("bbox_xyxy") is not None]
        self.n_skipped_no_meta = len(roi_samples) - len(keep)
        if self.n_skipped_no_meta:
            _safe_print(
                f"--> [roi_dataset] ⚠️ {self.n_skipped_no_meta}/{len(roi_samples)} "
                f"条 ROI 样本无 bbox(旧 pkl 无 roi_lesions 键或 mask 读失败),已跳过")
        self.samples = keep

    def __len__(self) -> int:
        return len(self.samples)

    def _read_frame(self, r: Dict) -> np.ndarray:
        """整图原分辨率帧(与整图 loader 同源读取)。"""
        if r["is_dynamic"]:
            return _read_nifti_frame_gray(r["img_path"], r["frame_index"])
        return _read_dcm_gray(r["img_path"])

    def crop_roi(self, frame: np.ndarray, bbox_xyxy) -> Tuple[np.ndarray, Tuple[int, int]]:
        """原分辨率帧 + bbox → (ROI 正方形原图, pad_hw)。纯函数,供推理端复用。

        pad 只补右/下(左/上零偏移,还原 = ROI[:h-pad_h, :w-pad_w] 无坐标平移)。
        """
        x0, y0, x1, y1 = bbox_xyxy
        roi = frame[y0:y1, x0:x1]
        h, w = roi.shape[:2]
        side = max(h, w)
        pad_h, pad_w = side - h, side - w
        if pad_h or pad_w:
            roi = np.pad(roi, ((0, pad_h), (0, pad_w)),
                         mode="constant", constant_values=0)
        return roi, (pad_h, pad_w)

    def __getitem__(self, idx):
        r = self.samples[idx]
        frame = self._read_frame(r)
        roi, (pad_h, pad_w) = self.crop_roi(frame, r["bbox_xyxy"])
        roi = resize_img_to(roi, self.roi_size)
        img_t = torch.tensor(roi, dtype=torch.float32).unsqueeze(0) / 255.0

        meta: Dict = {
            "bbox_xyxy": tuple(r["bbox_xyxy"]),
            "orig_w": r["orig_w"], "orig_h": r["orig_h"],
            "pad_hw": (pad_h, pad_w),
            "spacing_mm": r.get("spacing_mm"),
            "lesion_id": r["lesion_id"],
            "patient_key": r["patient_key"],
            # 债5:forward_roi 输出侧读 patient_id(multi_task_model.py:574);
            # 推理侧 roi_pipeline 的 meta 两个键都有,训练侧此处对齐。
            "patient_id": r.get("patient_id"),
            "is_dynamic": r["is_dynamic"],
            "frame_index": r.get("frame_index"),
            "global_loc": tuple(r["global_loc"]),
        }
        for k in _ROI_SCALAR_KEYS:
            if k in r:
                meta[k] = r[k]

        # ── 债2:roi_mask 生产(2026-09-25 训练侧接线)──────────────────────
        # GT mask 的**该样本帧**用**与图像完全相同的几何管线**裁剪:
        # 同 bbox 切片 → 同 pad → resize(INTER_NEAREST,resize_mask_to 单一权威,
        # 勿用 resize_img_to——双线性会在边界造 0/1 之外的中间值)。
        # 消费方:forward_roi 的 ETE band / T 头 d_mm(multi_task_model.py:497-510,
        # 缺键 → 全零 = 阴性语义,模型侧兜底)——此前"测试手工装 mask 掩盖生产
        # 面悬空"(Gate2 B-P1-2),本处即生产者。
        # 帧选择口径:
        #   静态:mask 全帧里唯一非零帧(_read_nifti_binmasks 返回的 binf 取 [0]);
        #   动态:该样本 frame_index 在 mask 的 idxs 里 → 对应帧;不在 = 本帧无
        #         该病灶 GT → 全零图(阴性帧语义,非缺键——动态交集帧展开下
        #         frame_index 必 ∈ frame_indices,此分支是防御性兜底)。
        # mask_path 缺失(loader 无 mask 病灶/mock 手构)→ **不写键**,由
        # forward_roi 侧全零兜底(别自己造零图:键缺失=模型侧语义,两者可区分)。
        _mp = r.get("mask_path")
        if _mp is not None:
            m_frame = self._gt_mask_for_sample(_mp, r)
            if m_frame is not None:
                m = self._crop_mask_like_image(m_frame, r["bbox_xyxy"], (pad_h, pad_w))
                m = resize_mask_to(m, self.roi_size)
                m = np.ascontiguousarray(m)
                # 🔴 值域断言:binmasks 已 {0,1}、INTER_NEAREST 不产中间值,
                # 任何 >1 都意味着上游 mask 被污染(前景面积会系统性错)。
                assert m.max() <= 1, (
                    f"roi_mask 值域 {m.max()} > 1(应 {0,1} 二值)——GT mask "
                    f"被污染,检查 {_mp}")
                meta["roi_mask"] = torch.tensor(
                    m, dtype=torch.float32).unsqueeze(0)
        return img_t, meta

    def _gt_mask_for_sample(self, mask_path: str, r: Dict) -> Optional[np.ndarray]:
        """GT mask 文件 → **该样本帧的原分辨率** mask (H,W) 二值图。

        帧选择口径(与病灶元数据的 frame_indices 同源,`_read_nifti_binmasks`):
          - 静态:取唯一帧(binf[0];静态 mask 实为 3D 单帧,与 loader
            `_static_mask` 取首帧语义一致);
          - 动态:frame_index ∈ idxs → 对应帧;不在 → 全零图(**本帧无该病灶
            GT = 阴性帧语义,返回零图而非 None**——动态交集帧展开下
            frame_index 必 ∈ frame_indices,此分支是防御性兜底,零图语义与
            forward_roi 的"缺 mask→全零"一致)。
        读失败(空 binf)→ None(**不写 roi_mask 键**,forward_roi 全零兜底,
        与 loader 收集端对读失败病灶跳过的容错语义一致——键缺失可区分于
        "零图"两种状态)。

        返回整帧(裁剪在 `_crop_mask_like_image`,与图像管线逐步同几何)。
        """
        binf, idxs = _read_nifti_binmasks(mask_path)
        if not binf:
            return None
        if r["is_dynamic"]:
            fidx = r.get("frame_index")
            if fidx not in idxs:
                return np.zeros_like(binf[0])
            return binf[idxs.index(fidx)]
        else:
            # 静态:唯一帧(静态 mask 实为 3D 单帧,binf[0] 即该帧)。
            return binf[0]

    def _crop_mask_like_image(self, mask_frame: np.ndarray, bbox_xyxy,
                              pad_hw: Tuple[int, int]) -> np.ndarray:
        """GT mask 帧 → ROI 域,与 crop_roi 逐步同几何(纯函数,供单测对拍)。

        bbox 半开切片 → 右/下 pad 到同 side → 调用方再 resize_mask_to。
        pad 值 0(与图像 pad 同为背景填充;mask 语义 0=无 GT)。
        """
        x0, y0, x1, y1 = bbox_xyxy
        m = mask_frame[y0:y1, x0:x1]
        pad_h, pad_w = pad_hw
        if pad_h or pad_w:
            m = np.pad(m, ((0, pad_h), (0, pad_w)),
                       mode="constant", constant_values=0)
        return m


def roi_collate(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, Dict]:
    """ROI batch 的显式 collate —— 🔴 勿换 default_collate。

    - 图像 → [B,1,S,S] 堆叠;
    - `lesion_id`/`global_loc` 等保持 **list**(lesion_id 是 32 位 hex 串,
      default_collate 会按字符维堆成 [B,32] 矩阵 —— 长度对齐时静默变矩阵、
      不齐时报错,两种都是错);
    - 数值标量键(label 值 + valid_*)→ [B] int64;
    - frame_index 恒 int 或 None:全 int 时 [B] int64,含 None(静态)时保 list
      (default_collate 对 None 的报错没有语义,保 list 是唯一无损选择);
    - 其余键(bbox_xyxy/pad_hw 等 tuple)逐元素转 list[tuple];
    - `roi_mask`(债2,2026-09-25)→ **list 透传,不 stack**:forward_roi 侧
      per-lesion 消费(H' 可不同;模型面 multi_task_model:497-510 的组装段自带
      "逐条 .detach().float() + 按首条栅格 pad"逻辑,stack 会破坏该契约)。
      批内混有缺 mask 样本时 keys 交集逻辑让整批丢该键(模型侧全零兜底;
      训练批来自同 loader,全缺/全有的批占绝对多数,混批是边缘态)。
    """
    imgs = torch.stack([b[0] for b in batch], dim=0)
    keys = [k for k in batch[0][1] if all(k in b[1] for b in batch)]
    out: Dict = {}
    out["lesion_id"] = [b[1]["lesion_id"] for b in batch]      # 显式保 list[str]
    for k in keys:
        if k == "lesion_id":
            continue
        vals = [b[1][k] for b in batch]
        if k in _ROI_SCALAR_KEYS:
            out[k] = torch.tensor(vals, dtype=torch.int64)     # label/valid:堆 [B]
        elif k == "roi_mask":
            out[k] = list(vals)                                # [B][[1,S,S] tensor]
        elif k == "frame_index":
            if all(v is not None for v in vals):
                out[k] = torch.tensor(vals, dtype=torch.int64)
            else:
                out[k] = vals                                   # 静/动混批:保 list
        elif k == "global_loc":
            out[k] = [tuple(v) for v in vals]
        else:
            # bbox_xyxy / pad_hw / orig_w / orig_h / spacing_mm / is_dynamic 等
            # tuple 或 None:保 list[tuple](bbox 数值消费在训练循环里自行
            # torch.tensor 化,这里不猜 dtype)。
            out[k] = [tuple(v) if isinstance(v, tuple) else v for v in vals]
    return imgs, out
