"""ROI(per-lesion)ETE 损失 —— 两阶段 ROI 范式损失组(方案一 P3,2026-09-25)。

与 `components/extension.py`(整图版)的**同式数学**:masked CB-Focal
(Class-Balanced Focal,γ 调制)——整图版吃 `ete_logits` [B,4];ROI 版吃
ROI 模型 forward_roi 契约输出的 `ete_logits` [K,4](**logits 域**,与整图版
同入口;`ete_probs` [K,4] 是给验证/指标侧的,损失侧不消费它,避免
probs→logits 反解的数值损失)。

## 与整图版的口径对齐(勿漂移成两把尺子)

- **标签语义不变**:0-based 4 类(官方码 1..4 减一),`_parse_ete_code` 同源。
- **valid 过滤**:per-lesion `valid_extrathy_ext`。
- **is_main 门**:与整图版 train_step 同判据(v × main_w)。
  ⚠️ **不在此处加恶性限定**——官方 ETE 只在恶性主病灶上*评分*,但训练侧
  `train_step.py` 的整图实现只有 `is_main × valid` 门(没有 gt_malignancy
  过滤;良性样本的 ETE=1无侵犯 是合法监督,不是噪声)。ROI 版照抄同口径,
  不自创"更严格"的门(会静默丢 2410+ 良性样本的 ETE=无 监督,与整图口径
  漂移成两把尺子)。**加了恶性门 ≠ 更对**,这是评分口径(仅恶性)与监督
  口径(全部主病灶)的有意区分,见 `train_step.py` docstring ③。
- **qwk 分支不带入**:整图版有 LA-SoftQWK-OD 分支(#128 判死后默认
  qwk_weight=0 走 CB-Focal 老路径)。ROI 版**只实现 CB-Focal 分支**——
  #128 已判死关闭,qwk 路径整体不进新架构(避免判死路线在新代码里复活);
  若日后要重启,须先重过整图版同款的 DR 依据,勿在 ROI 版里静默带上。
- **归一范式**:`(loss * v).sum() / (v.sum() + eps)`(与全项目同范式)。

## 从整图版原样搬的数学(勿改)

- Focal 调制:`(1 - pt)^γ`,γ=0 退化加权 CE;🔴 γ∈(0,1) 在 p_t=1.0 处
  NaN 梯度(整图版实测坐实的坑),本版同款 assert 拦死。
- CB 权重按**真实标签**索引(非预测——按预测索引会让模型通过"少预测
  稀有类"自行降权,与校正意图相反)。
- 越界正负不对称拦下界(idx=-1 走 python 负索引静默取末元素=最稀有的
  血管类,表现为"稀有类多了一批假样本"无报错)。
- fp32:autocast 下 logits 可能 bf16,log_softmax 在低精度丢有效位。
"""

import torch
import torch.nn.functional as F
from typing import Optional

__all__ = ["compute_extension_roi_loss"]

_EPS = 1e-8
# k_ref 分母平滑(债6 同批,2026-09-25,DR 机制 1):语义见 location_roi._K_REF。
_K_REF = 4.0


def compute_extension_roi_loss(
    ete_logits: torch.Tensor,
    ete_idx: torch.Tensor,
    ete_validity: torch.Tensor,
    is_main: torch.Tensor = None,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
):
    """ROI ETE 的 masked CB-Focal。

    Args:
        ete_logits: [K,4] ROI 头契约输出的原始 logits。
        ete_idx: [K] long,0-based 类别标签(官方输出码 −1)。无效样本处值任意。
        ete_validity: [K] float,1 = 该病灶有 ETE 标注。
        is_main: [K] float/int,1 = 主病灶。None → 全当主病灶。
        class_weights: [4] CB 权重,None → 不加权(等价全 1)。
        focal_gamma: Focal 的 γ。0 → 退化加权 CE。🔴 必须是 0 或 ≥1
            (γ∈(0,1) 在 p_t=1.0 处 NaN 梯度,整图版实测坐实)。
    Returns:
        (total, detail)。detail 含 `roi_ete`(未乘权重的原始损失)与
        `roi_ete_ce`(纯 CE 对照,不加权不调制,监控用)。
        整批无有效样本 → 精确 0、梯度 0、无 NaN。
    """
    assert ete_validity.dim() == 1, (
        f"ete_validity 必须是 [K] 一维,got {tuple(ete_validity.shape)}"
        " —— [K,1] 会隐式广播成 [K,K] 静默放大 loss(QA_VP §2.2)"
    )
    assert ete_idx.shape == ete_validity.shape, (
        f"标签与 validity 形状不一致: {tuple(ete_idx.shape)} vs "
        f"{tuple(ete_validity.shape)}"
    )
    num_classes = ete_logits.shape[-1]
    assert ete_logits.dim() == 2 and num_classes == 4, (
        f"ete_logits 应为 [K,4],got {tuple(ete_logits.shape)}"
        " —— ETE 4 类实测定稿(1无 2包膜 3软组织 4血管)"
    )
    assert focal_gamma == 0.0 or focal_gamma >= 1.0, (
        f"focal_gamma 必须是 0(关闭)或 ≥1,got {focal_gamma}"
        " —— γ∈(0,1) 在 p_t=1.0 处梯度为 NaN,且 loss 值看不出异常"
    )
    assert int(ete_idx.min()) >= 0 and int(ete_idx.max()) < num_classes, (
        f"ETE 标签越界: min={int(ete_idx.min())}, max={int(ete_idx.max())},"
        f" 合法区间 [0,{num_classes - 1}](无标注样本应填占位 0 + validity=0)"
    )

    K = ete_logits.shape[0]
    device = ete_logits.device

    if is_main is None:
        main_w = torch.ones(K, dtype=torch.float32, device=device)
    else:
        assert is_main.shape == ete_validity.shape, (
            f"is_main 与 validity 形状不一致: {tuple(is_main.shape)} vs "
            f"{tuple(ete_validity.shape)}"
        )
        main_w = is_main.to(device=device, dtype=torch.float32)

    v = (ete_validity.to(device) * main_w).float()

    # fp32 归约(autocast 语义同整图版:log_softmax 在 bf16 丢有效位)。
    logits = ete_logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(1, ete_idx.unsqueeze(1)).squeeze(1)  # [K]
    pt = log_pt.exp()
    focal = (1.0 - pt).pow(focal_gamma) if focal_gamma != 0.0 else torch.ones_like(pt)

    if class_weights is None:
        w_c = torch.ones_like(log_pt)
    else:
        assert class_weights.shape == (num_classes,), (
            f"class_weights 形状应为 [{num_classes}],got {tuple(class_weights.shape)}"
        )
        # 🔴 按**真实标签**索引,不是按预测(见模块 docstring)。
        w_c = class_weights.to(log_pt.dtype).to(log_pt.device)[ete_idx]

    per_sample = -w_c * focal * log_pt
    denom = torch.clamp(v.sum() + _EPS, min=_K_REF)
    total = (per_sample * v).sum() / denom
    ce_plain = (-log_pt * v).sum() / denom
    return total, {"roi_ete": total.detach(), "roi_ete_ce": ce_plain.detach()}
