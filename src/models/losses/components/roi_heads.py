# -*- coding: utf-8 -*-
"""ROI(per-lesion)四头损失编排 —— 两阶段 ROI 范式损失组(方案一 P3,2026-09-25)。

与 `components/official_heads.py`(整图版编排)的**同型结构**:固定 λ 外挂,
**不进 FAMO**(理由同那边 docstring:①`famo.py:92-95` 的 `load_state_dict`
校验通道数,扩通道破坏 checkpoint 兼容;②masked loss 整批无有效样本时精确
返回 0.0,FAMO 会把这个"0"当"该任务已收敛"而持续压低其权重,方向恰好反了)。
插入点在整图三头外挂之后、`compute_regularizers` 之前。

## 输入契约(两端口已钉死,本模块只做搬运与门控)

**outputs** 含 `roi_outputs` 键(list[dict],长度 K = 病灶数)——模型面
forward_roi 的输出,每个 dict 含一病灶:
  `malignancy` [2] logits / `location_probs` [7] / `ete_probs` [4] /
  `ete_logits` [4] / `t_conf` 标量 / `tirads_probs` [5]。
  (验证侧 `collect_roi_records` 消费同一键集——单源契约,此处不重定义。)

**targets** 含 `roi_labels` 键(list[dict],长度 K,与 roi_outputs 逐条对齐)
——数据面 `roi_dataset` 的 per-lesion 标签,每条 dict 含
  `gt_malignancy`/`gt_lesion_loc`(0-based)/`gt_extrathy_ext`/
  `gt_clin_t_stage`/`gt_tirads` + `valid_*` 标志 + `lesion_id`/`is_main`。
  (collate 后的 int64 [K] tensor 元素由本模块逐条取出;键名以
  collect_roi_records 的记录契约为准——gt_ 前缀。)

## 🔴 良恶性(malignancy)不在本编排内

契约的 `malignancy` [K,2] logits 是**整图 malignancy 损失**
(`compute_malignancy_losses`,AUCM+SupCon)的输入形态——ROI 范式下良恶性
仍由整图链路监督(AUCM 是 pairwise 损失,与 per-lesion CE 不同式,不在此
复制一份;整图 outputs["malignancy"] 的 batch 级统计由模型侧契约保证仍产出)。
本编排只管四条 per-lesion 损失:location/ete/clin_t/tirads。

## fail-loud 语义(deepsup 同型)

- `roi_head_weight > 0` 而 targets 无 `roi_labels` 键(或 outputs 无
  `roi_outputs` 键)→ **raise**:新损失配旧模型/旧数据输出,拒绝静默空算
  (r7 事故同型雷:静默丢变量 = 空微调被误读为损失无效)。
- `roi_head_weight == 0`(默认):键存在与否都不算、不 raise——**零回归**,
  loss_dict 的 roi_* 键恒 0.0(键恒在,接线被删 JSONL 立刻可见,不自证)。
- 长度不齐(roi_outputs vs roi_labels)→ raise(与 collect_roi_records
  同款校验,错误前置到损失层)。

## 归一口径

四条损失各自内部 masked 归一(`(loss*v).sum()/(v.sum()+eps)`,分母只数
该头自己的有效病灶数)——**四头分母互不相同**(各头 valid/is_main 人口
不一样),这是与整图版一致的行为(整图版各头也是独立归一)。λ 在编排层
统一乘:`total = λ·(L_loc + L_ete + L_t + L_tirads)`(λ 是单开关,四头
内部相对权重沿用整图版的 1:1:1:1,不另立四个 λ——调参面最小化,真需要
分头调时再扩,勿预先发明)。
"""

from typing import Dict, List, Tuple

import torch

from src.models.losses.components.location_roi import compute_location_roi_loss
from src.models.losses.components.extension_roi import compute_extension_roi_loss
from src.models.losses.components.clinical_t_roi import compute_clinical_t_roi_loss
from src.models.losses.components.tirads_roi import compute_tirads_roi_loss
from src.models.losses.components.concepts_roi import compute_roi_concept_losses

__all__ = ["compute_roi_head_losses", "compute_roi_concept_losses",
           "ROI_CONCEPT_KEYS"]


def _scalar(entry, key, default=0):
    """per-lesion dict 取标量:int/float/0 维张量均可;缺失给 default。

    只接受标量(与 collect_roi_records._label_val 同判):batch 张量在这里
    是"labels 仍是整图 tensor dict"的误接线指纹,取 v[0] 会静默把整批标签
    当单病灶标签,必须 fail-loud。
    """
    if not isinstance(entry, dict):
        return default
    v = entry.get(key, default)
    if isinstance(v, torch.Tensor):
        if v.numel() != 1:
            raise ValueError(
                f"per-lesion 字段 {key} 必须是标量,得到 shape {tuple(v.shape)}"
                " —— roi_labels 疑似整图 batch 张量误接 per-lesion 链路"
            )
        v = v.reshape(()).item()
    elif isinstance(v, (list, tuple)):
        if len(v) != 1:
            raise ValueError(
                f"per-lesion 字段 {key} 必须是标量,得到长度 {len(v)} 的序列"
            )
        v = v[0]
    return int(v)


def _to_k_tensor(vals: List, device) -> torch.Tensor:
    """标量 list → [K] float tensor(device 上)。"""
    return torch.as_tensor(
        [float(v) for v in vals], dtype=torch.float32, device=device)


# 四头 detail 键(零回归/空批路径恒 0 的键集合,单源——键改名只改这里)。
_ROI_DETAIL_KEYS = (
    "roi_location", "roi_location_side", "roi_location_level",
    "roi_ete", "roi_ete_ce", "roi_clin_t", "roi_clin_t_ce",
    "roi_tirads", "roi_tirads_ce", "roi_tirads_ordinal",
)


def _zero_detail() -> Dict[str, torch.Tensor]:
    """四头恒 0 detail(dict 每次新建,防调用方原地改写互染)。"""
    return {k: torch.zeros(()) for k in _ROI_DETAIL_KEYS}


def compute_roi_head_losses(
    outputs: Dict,
    targets: Dict,
    lambda_roi: float = 0.0,
    ete_class_weights: torch.Tensor = None,
    ete_focal_gamma: float = 2.0,
    tirads_class_weights: torch.Tensor = None,
    tirads_label_smoothing: float = 0.0,
    tirads_ordinal_weight: float = 0.0,
    concept_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """ROI 四头损失加权编排。

    Args:
        outputs: 模型 forward 输出;ROI 范式下须含 `roi_outputs`(list[dict])。
        targets: 须含 `roi_labels`(list[dict],与 roi_outputs 逐条对齐)。
            整图/legacy 调用方无此键。
        lambda_roi: ROI 四头总权重(**默认 0.0 = 关闭**)。>0 时四条损失
            以 1:1:1:1 内部相对权重求和后乘 λ。
        ete_class_weights / ete_focal_gamma: 透传 extension_roi(语义同整图版
            `compute_extension_loss` 的对应参数;单一配置源,不另立 roi_ 前缀)。
        tirads_class_weights / tirads_label_smoothing / tirads_ordinal_weight:
            透传 tirads_roi(语义同整图版 compute_tirads_loss)。
        concept_weight: ROI 概念头监督权重(债6,方案 A 简化版,**默认 0.0 =
            关闭零回归**)。>0 时对 composition/echogenicity/shape/margin 四键
            做 per-lesion CE;foci 无官方标注恒零损失。与 lambda_roi **独立**
            (概念监督在 roi 四头关闭时也可单独开——如纯概念预标定)。
    Returns:
        (weighted_total, detail)。detail 键 `roi_location`/`roi_location_side`/
        `roi_location_level`/`roi_ete`/`roi_ete_ce`/`roi_clin_t`/
        `roi_clin_t_ce`/`roi_tirads`/`roi_tirads_ce`/`roi_tirads_ordinal`,
        均为**未乘 λ 的原始损失值**(监控要看任务本身的量级,乘了 λ 就分不清
        "损失降了"和"权重小"——与整图版 detail 同约定)。
        权重 0(默认)时 total 是 python 0.0、各 detail 键恒为 0.0 张量
        (键恒在,消费端无条件 update 不查权重,防"接线被删 JSONL 静默丢分项";
        deepsup/seg_extra 同款)。

    `weighted_total` 在权重 0 时是 python `0.0`(不是 tensor)—— 调用方一律
    `total_loss = total_loss + roi_total`,标量 0 与 tensor 相加安全且不建图。
    """
    detail: Dict[str, torch.Tensor] = {}

    if lambda_roi <= 0.0:
        # 零回归路径:不算、不 raise、detail 键恒 0(张量形态与权重>0 时
        # 一致,JSONL 逐键 round 不需特判)。概念监督独立于 lambda_roi——
        # 概念权重>0 时仍要走自己的监督(纯概念标定场景)。
        if concept_weight <= 0.0:
            return 0.0, _zero_detail()
        c_total, c_detail = compute_roi_concept_losses(
            outputs, targets.get("roi_labels", []), lambda_concept=concept_weight)
        return c_total, {**_zero_detail(), **c_detail}

    # ── fail-loud(权重>0 而输入无 ROI 键 = 版本错配,拒绝静默空算)──
    missing = [k for k in ("roi_outputs", "roi_labels")
               if (k not in outputs if k == "roi_outputs" else k not in targets)]
    if missing:
        raise ValueError(
            f"roi_head_weight={lambda_roi} > 0 但输入缺 ROI 键 {missing}"
            " —— ROI 损失需要 outputs['roi_outputs'](模型 forward_roi 输出)"
            " 与 targets['roi_labels'](per-lesion 标签 list);"
            "新损失配旧模型/旧数据输出,拒绝静默空算"
            "(检查模型前向与训练循环的 ROI 组装)"
        )
    roi_outputs: List[Dict] = outputs["roi_outputs"]
    roi_labels: List[Dict] = targets["roi_labels"]
    if len(roi_outputs) != len(roi_labels):
        raise ValueError(
            f"roi_outputs({len(roi_outputs)}) 与 roi_labels({len(roi_labels)})"
            " 长度不齐 —— per-lesion 输出与标签必须逐条对齐"
        )
    K = len(roi_outputs)
    if K == 0:
        # 空 batch(理论上 DataLoader 不产,防御):全 0,不 raise。
        if concept_weight <= 0.0:
            z = torch.zeros(())
            return 0.0, {k: z.clone() for k in _ROI_DETAIL_KEYS}
        c_total, c_detail = compute_roi_concept_losses(
            outputs, roi_labels, lambda_concept=concept_weight)
        return c_total, {k: torch.zeros(()) for k in _ROI_DETAIL_KEYS} | c_detail

    # ── 逐条抽取张量化(device 取第一个头的输出;各头契约同源)──────
    # location_probs / ete_logits / t_conf / tirads_probs 直接用模型输出的
    # 张量(保梯度);is_main 从 roi_labels 取(数据面权威,模型输出里的
    # is_main 是给验证侧的,训练监督门以标签侧为准)。
    first = roi_outputs[0]
    ref = (first["location_probs"] if isinstance(first.get("location_probs"), torch.Tensor)
           else first["ete_logits"] if isinstance(first.get("ete_logits"), torch.Tensor)
           else first["t_conf"] if isinstance(first.get("t_conf"), torch.Tensor)
           else first["tirads_probs"])
    device = ref.device if isinstance(ref, torch.Tensor) else torch.device("cpu")

    def _stack(key, width=None):
        """roi_outputs 的 [width]/标量键 → [K,width]/[K] 张量(保梯度)。

        0 维标量张量(如 t_conf 逐病灶 torch.tensor(0.5))堆叠时天然 [K];
        [width] 向量 reshape 到统一宽;list 标量走 as_tensor。
        """
        rows = [r[key] for r in roi_outputs]
        if isinstance(rows[0], torch.Tensor):
            if width is not None:
                return torch.stack([r.reshape(width) for r in rows])
            # 标量堆:0 维 reshape(-1) 会变 [1],要 reshape(()) 保 0 维后
            # torch.stack 自然升 [K]。
            return torch.stack([r.reshape(()) for r in rows])
        return torch.as_tensor(rows, dtype=torch.float32, device=device)

    location_probs = _stack("location_probs", 7)
    ete_logits = _stack("ete_logits", 4)
    t_conf = _stack("t_conf")            # [K](标量堆)
    tirads_probs = _stack("tirads_probs", 5)

    loc_idx = torch.as_tensor(
        [_scalar(l, "gt_lesion_loc") for l in roi_labels],
        dtype=torch.long, device=device)
    loc_valid = _to_k_tensor(
        [_scalar(l, "valid_lesion_loc") for l in roi_labels], device)
    ete_idx = torch.as_tensor(
        [_scalar(l, "gt_extrathy_ext") for l in roi_labels],
        dtype=torch.long, device=device)
    ete_valid = _to_k_tensor(
        [_scalar(l, "valid_extrathy_ext") for l in roi_labels], device)
    t_idx = torch.as_tensor(
        [_scalar(l, "gt_clin_t_stage") for l in roi_labels],
        dtype=torch.long, device=device)
    t_valid = _to_k_tensor(
        [_scalar(l, "valid_clin_t_stage") for l in roi_labels], device)
    tirads_idx = torch.as_tensor(
        [_scalar(l, "gt_tirads") for l in roi_labels],
        dtype=torch.long, device=device)
    tirads_valid = _to_k_tensor(
        [_scalar(l, "valid_tirads") for l in roi_labels], device)
    is_main = _to_k_tensor(
        [_scalar(l, "is_main", 1) for l in roi_labels], device)

    # ── 四头各自算(内部 masked 归一 + is_main 门,detail 未乘 λ)────
    loc_total, loc_detail = compute_location_roi_loss(
        location_probs, loc_idx, loc_valid, is_main=is_main)
    ete_total, ete_detail = compute_extension_roi_loss(
        ete_logits, ete_idx, ete_valid, is_main=is_main,
        class_weights=ete_class_weights, focal_gamma=ete_focal_gamma)
    t_total, t_detail = compute_clinical_t_roi_loss(
        t_conf, t_idx, t_valid, is_main=is_main)
    tirads_total, tirads_detail = compute_tirads_roi_loss(
        tirads_probs, tirads_idx, tirads_valid, is_main=is_main,
        class_weights=tirads_class_weights,
        label_smoothing=tirads_label_smoothing,
        ordinal_weight=tirads_ordinal_weight)

    total = lambda_roi * (loc_total + ete_total + t_total + tirads_total)
    detail.update(loc_detail)
    detail.update(ete_detail)
    detail.update(t_detail)
    detail.update(tirads_detail)

    # ── 债6(2026-09-25):ROI 概念头监督(方案 A 简化版)──────────────────
    # RoiConceptHead 此前全仓无损失消费方 = 18 维随机噪声旁路拼给 TI-RADS 头
    # (detach 后 TI-RADS 梯度也不回流它)。本项对 composition/echogenicity/
    # shape/margin 四概念做 per-lesion CE;foci 无官方标注(official_loader
    # `_build_labels`:占位零 + valid_foci 恒 0)→ 零损失零梯度。
    concept_total, concept_detail = compute_roi_concept_losses(
        outputs, roi_labels, lambda_concept=concept_weight)
    if concept_total is not None:
        total = total + concept_total
    detail.update(concept_detail)
    return total, detail
