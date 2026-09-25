# -*- coding: utf-8 -*-
"""ROI(per-lesion)概念监督损失 —— 债6(方案 A 简化版,DR 终审采纳,2026-09-25)。

## 为什么存在(债6,J.5 P2-3 实锤)

`RoiConceptHead`(`src/models/roi_heads.py`)产五概念 logits(18 维)拼给
`RoiTiradsHead` 的概念旁路,而 `RoiTiradsHead.forward` 对概念 logits 全键
`.detach()`(🔴 硬红线,DR+主会话一致:防 TI-RADS 梯度直接拽概念表征、绕过
概念头自己的监督)。**detach 是对的,但它使 RoiConceptHead 在"没有独立监督"
时永久随机**——18 维噪声旁路污染 TI-RADS 头输入。本模块给它自己的监督:
对官方有标注的四个概念做 per-lesion CE。

🔴 **2026-09-26 R2 review P0-1 定案:官方表无独立概念列,本监督在官方数据
上结构性恒空(roi_concept_weight 须保持 0)**。依据与前置(拆复合串+对齐
TN3K 口径)见 `official_loader._parse_concept` docstring。本模块数学/梯度
路径已被独立对拍过(R2 PASS-1/2),数据侧解锁后即可直接启用。

## 契约(键名实测,勿照抄外部报告简写)

- **概念 logits 来源**:`outputs["roi_concepts"]`(模型面 `forward_roi` 的返回
  dict 键,`multi_task_model.py:587`),值 = `RoiConceptHead.forward(pooled)` 的
  dict,五键 `logits_comp`[K,4] / `logits_echo`[K,4] / `logits_shape`[K,2] /
  `logits_margin`[K,4] / `logits_foci`[K,4]。
  🔴 **不是** `roi_outputs` 逐病灶 dict 里的键——forward_roi 只把五键 dict 放
  在顶层 `roi_concepts`,不随 per-lesion dict 下发(若 call 方只传
  `{"roi_outputs": ...}` 窄 dict,概念监督读不到 → fail-loud,见下)。
- **标签键**(数据面 `_ROI_LABEL_RENAME` 单源,`roi_dataset.py:88`):
  `gt_composition`/`gt_echogenicity`/`gt_shape`/`gt_margin` + `valid_*` 同名
  (数据面 `_build_labels` 产 `composition`/`valid_composition` 等,桥加 gt_
  前缀后正好落在这些名上——**不新增改名映射**,桥已有,零数据面改动)。
  🔴 foci 官方无标注(`official_loader._build_labels`:占位零 + valid_foci
  恒 0)→ 本损失**零 foci 项、零梯度**(不做蒸馏——DR 的 A+C 混合已被主会话
  裁决降级为纯 A)。

## 缺失语义(fail-loud 分层)

- `lambda_concept <= 0`(默认):不算、不 raise、恒 0 detail(键恒在,
  与四头零回归同款)。
- 权重>0 而 outputs 无 `roi_concepts` 键(窄 dict 误传)→ **raise**:
  概念监督需要完整 forward_roi 返回,窄化 `{"roi_outputs": ...}` 是常见误传
  形态,静默跳过 = "号称开了实际没开"(r7 事故同型)。
- 权重>0 而 targets 无 `roi_labels` → 概念标签无从取 → raise(同四头
  fail-loud 语义)。
- per-lesion 标签缺失(`valid_*=0`,数据面缺列/空单元格):**不进分母**
  (valid 屏蔽,与四头 masked 归一同范式)。占位值 0 + valid=0 → 零贡献。
- 🔴 R2 review P1-3(2026-09-26):**有限性门**——logits 含 NaN/Inf 时 raise
  (0×NaN=NaN 不被 v 屏蔽,会经共享 total 把全部损失项毒成 NaN,GradScaler
  静默跳过 step = 训练卡死无报错)。fail-loud 而非 nan_to_num(静默降级
  伪装成健康收敛,违反本仓纪律)。
- **is_main 门:有意不加**(R2 P2-1 订正,2026-09-26)——概念是病灶级
  合法监督(次病灶的概念标注同样教概念头),且**不因 lambda_roi=0 而变**:
  纯概念标定(lambda_roi=0、概念权重>0)时本损失是唯一项,仍是全 ROI
  样本的面(次病灶含在内)。这不是"与四头的门不同"——四头 lambda_roi=0
  时根本不跑,不存在对比;写了反会误导读者去找"门差异"。

## 归一范式

四概念各 masked 归一后求和(1:1:1:1,与四头内部相对权重同约定——不另立
四 λ,调参面最小化),再乘 `lambda_concept`。k_ref 分母平滑同四头
(`clamp(v.sum()+eps, min=4)`):n_valid≥4 逐位不变,n_valid∈(0,4) 压 4×,
全 0 → 精确 0。

## JSONL 观测

detail 键(未乘 λ 的原始量级,与四头同约定):
`roi_concept`(四概念和)/`roi_concept_comp`/`roi_concept_echo`/
`roi_concept_shape`/`roi_concept_margin`,另有**有效计数键**
`concept_n_valid_comp` 等(标量 float,计数惯例照 `n_cc_*`:权重>0 时恒
出现,回读可判"监督人口是不是零")——计数键恒在恒数值(不是 0 张量,
是 float),权重 0 时恒 0.0。

⚠️ 计数键消费端:train_official 的 `_roi_detail_extra` 对 detail 逐键
`float(v)` round 6 → 计数自动进 JSONL(`loss_concept_n_valid_comp` 前缀;
数值是"每 batch 的有效数",epoch 级聚合由调用方/回读侧均值——与
`n_cc_*` 的 per-batch 语义一致,勿当全量口径)。
"""

from typing import Dict, List, Tuple

import torch

__all__ = ["compute_roi_concept_losses", "ROI_CONCEPT_KEYS"]

_EPS = 1e-8
# k_ref 分母平滑(债6 同批,2026-09-25,DR 机制 1):语义见 location_roi._K_REF。
_K_REF = 4.0

# (logits 键, 标签键, valid 键, 类数)——键名实测单源:
# logits 侧 = RoiConceptHead.forward 返回键;标签侧 = roi_labels_from_meta
# 桥产物(_build_labels 无前缀键 + gt_ 改名)。foci 有意不在表内(零监督)。
ROI_CONCEPT_KEYS = (
    ("logits_comp", "gt_composition", "valid_composition", 4),
    ("logits_echo", "gt_echogenicity", "valid_echogenicity", 4),
    ("logits_shape", "gt_shape", "valid_shape", 2),
    ("logits_margin", "gt_margin", "valid_margin", 4),
)

# valid_* 键 → detail 短名(与 logits 键后缀 comp/echo/shape/margin 一致,
# 单源映射,勿在多处手写切片)。
_VALID_TO_SHORT = {
    "valid_composition": "comp",
    "valid_echogenicity": "echo",
    "valid_shape": "shape",
    "valid_margin": "margin",
}

_CONCEPT_DETAIL_KEYS = ("roi_concept", "roi_concept_comp", "roi_concept_echo",
                        "roi_concept_shape", "roi_concept_margin")


def _zero_concept_detail() -> Dict[str, torch.Tensor]:
    """概念监督恒 0 detail + 计数键恒 0.0(键恒在,权重 0 时 JSONL 仍可见)。"""
    out = {k: torch.zeros(()) for k in _CONCEPT_DETAIL_KEYS}
    for s in _VALID_TO_SHORT.values():
        out[f"concept_n_valid_{s}"] = 0.0
    return out


def compute_roi_concept_losses(
    outputs: Dict,
    roi_labels: List[Dict],
    lambda_concept: float = 0.0,
) -> Tuple[object, Dict[str, torch.Tensor]]:
    """ROI 概念头 per-lesion CE(四概念,masked 归一,foci 零项)。

    Args:
        outputs: 模型 forward_roi 的**完整**返回 dict(须含 `roi_concepts`:
            五键 logits dict)。🔴 只传 `{"roi_outputs": ...}` 窄 dict 会
            fail-loud(概念 logits 在顶层键,不在 per-lesion dict 里)。
        roi_labels: 数据面桥产物(list[dict],gt_ 前缀标签 + valid_*)。
        lambda_concept: 权重(默认 0.0 = 关闭)。与 roi 四头的 lambda_roi
            独立。
    Returns:
        (weighted_total, detail)。权重 0(早退路径)时 total 是 python 0.0、
        detail 键恒在(损失键恒 0 张量、计数键恒 0.0);**权重>0 但整批无有效
        样本时 total 是 tensor(0.)**(数值恒 0;R2 P2-2 订正——torch 域
        clamp/加减的返回天然是张量,调用方一律 `+` 相加安全,无功能影响)。
        detail 的损失值未乘 λ(监控约定同四头)。logits 含非有限值 → raise
        (P1-3 有限性门,见模块 docstring)。
    """
    if lambda_concept <= 0.0:
        return 0.0, _zero_concept_detail()

    if "roi_concepts" not in outputs:
        raise ValueError(
            f"roi_concept_weight={lambda_concept} > 0 但 outputs 缺 "
            "'roi_concepts' 键 —— 概念 logits 在 forward_roi 返回的顶层 "
            "'roi_concepts' dict(五键 logits_*),不在 per-lesion "
            "roi_outputs dict 里;若训练循环只传 {\"roi_outputs\": ...} "
            "窄 dict,概念监督被静默丢 = 号称开了实际没开,拒绝空算"
            "(把 forward_roi 的完整返回传进来)"
        )
    concepts: Dict = outputs["roi_concepts"]
    if not isinstance(concepts, dict) or not ROI_CONCEPT_KEYS:
        raise TypeError(
            f"outputs['roi_concepts'] 应为五键 dict(logits_*),got {type(concepts)}"
        )

    total = 0.0
    detail: Dict[str, torch.Tensor] = {}
    n_valid_out: Dict[str, float] = {}
    device = torch.device("cpu")

    for logit_key, gt_key, valid_key, n_cls in ROI_CONCEPT_KEYS:
        if logit_key not in concepts:
            raise ValueError(
                f"outputs['roi_concepts'] 缺概念键 '{logit_key}'"
                f"(RoiConceptHead.forward 的五键契约,got {list(concepts)})"
            )
        logits = concepts[logit_key]
        if not isinstance(logits, torch.Tensor) or logits.dim() != 2 \
                or logits.shape[-1] != n_cls:
            raise ValueError(
                f"{logit_key} 应为 [K,{n_cls}] 张量,got "
                f"{tuple(logits.shape) if isinstance(logits, torch.Tensor) else type(logits)}"
            )
        K = logits.shape[0]
        device = logits.device
        if len(roi_labels) != K:
            raise ValueError(
                f"roi_labels({len(roi_labels)}) 与 {logit_key} 的 K({K})"
                " 长度不齐 —— per-lesion 输出与标签必须逐条对齐"
            )

        # 标签/有效标志:缺失键 = 该概念无标注(valid=0,零贡献零分母)。
        # _scalar 的 default 语义:键缺席 → 0 占位 + valid=0(数据面
        # `_build_labels` 对缺列概念正是这个产出形态,此处同款兜底)。
        # (函数内 import:顶层 import 会与 roi_heads 循环——那边顶层 import
        # 本模块的 compute_roi_concept_losses,本模块顶层 import 它的
        # _scalar 会炸 partially initialized;函数内延迟到调用时两模块均完成)
        from src.models.losses.components.roi_heads import _scalar
        idx = torch.as_tensor(
            [_scalar(l, gt_key, 0) for l in roi_labels],
            dtype=torch.long, device=device)
        v = torch.as_tensor(
            [float(_scalar(l, valid_key, 0)) for l in roi_labels],
            dtype=torch.float32, device=device)
        n_valid_out[_VALID_TO_SHORT[valid_key]] = float(v.sum())

        # 概率域 CE(与 tirads_roi 同式:logits 直接 log_softmax,fp32 归约)。
        # 🔴 R2 review P1-3 修(2026-09-26):有限性门(fail-loud)——
        # `0×NaN=NaN` 不被 v 屏蔽,(per_sample*v).sum() 会把单个非有限
        # logits 扩散成整批 NaN → 共享 total 全项 NaN → GradScaler 静默跳过
        # step(训练原地卡死不报错)。与全项目"宁可 ERROR 也不静默"纪律一致,
        # 此处 raise 而非 nan_to_num(静默降级会把数值病态伪装成健康收敛)。
        # 纵然 NaN 大概率来自上游(autocast/发散),fail-loud 才能触发排查。
        if not bool(torch.isfinite(logits).all()):
            raise ValueError(
                f"{logit_key} 含非有限值(NaN/Inf)—— 概念监督拒绝把 NaN "
                f"扩散进整批总损失(0×NaN=NaN 不被 valid 屏蔽,GradScaler "
                f"会静默跳过该 step,训练卡死无报错)。排查上游前向/AMP。"
            )
        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        log_pt = log_probs.gather(1, idx.unsqueeze(1)).squeeze(1)   # [K]
        per_sample = -log_pt
        denom = torch.clamp(v.sum() + _EPS, min=_K_REF)
        loss_c = (per_sample * v).sum() / denom
        total = total + lambda_concept * loss_c
        detail[f"roi_concept_{_VALID_TO_SHORT[valid_key]}"] = loss_c.detach()

    detail["roi_concept"] = (
        sum(detail[f"roi_concept_{s}"] for s in
            ("comp", "echo", "shape", "margin")))
    for s, n in n_valid_out.items():
        detail[f"concept_n_valid_{s}"] = n
    # 🔴 foci 恒零梯度:不在 ROI_CONCEPT_KEYS 里 = logits_foci 从不进
    # 任何损失项(测试钉死:concept 权重>0 时 logits_foci.grad is None)。
    return total, detail
