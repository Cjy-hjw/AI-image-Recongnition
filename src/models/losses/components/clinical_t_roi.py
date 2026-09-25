"""ROI(per-lesion)临床 T 损失 —— 两阶段 ROI 范式损失组(方案一 P3,2026-09-25)。

## 🔴 与整图版的关键差异:吃 `t_conf`,不吃 logits

整图版(`components/clinical_t.py`)吃 `t_logits` [B,2](`ClinicalTHead` 的
原始 logits,损失内部自己做 log_softmax)。ROI 版**没有这个输入**——ROI 模型
forward_roi 契约钉死的输出键是 `t_conf` [K](P(T=1),softmax 后的**正类
概率**,与官方 `T_ClinSig.T_conf` 字段直接对应的那个量;collect_roi_records
的记录契约同键)。

故本实现的数学是**从概率出发的 BCE**:

    L = −[y·log(p) + (1−y)·log(1−p)],p = t_conf.clamp(eps, 1−eps)

与整图版从 logits 走 `F.cross_entropy` 的**期望值完全一致**(cross_entropy
就是 softmax+log+NLL;二分类下 NLL(logits) 与 BCE(sigmoid 正类概率)恒等),
差别仅在数值路径:softmax→log 的 torch 稳定实现 vs 手动 clamp+log。clamp
下界 _EPS=1e-8 与全项目 masked 范式一致;sigmoid 饱和时 t_conf 可精确为
0.0/1.0,不 clamp 会 -inf。

**为什么不在 ROI 版里另要 logits**:契约已钉死(模型面并行开发,不得自创
键名);且 t_conf 路径 = 官方评分直接消费的量(AUC 用它当 score),损失
优化它 = 优化交付物本身,口径更近。梯度经 t_conf 回流 ROI 头。

## 与整图版的口径对齐(勿漂移成两把尺子)

- **标签语义不变**:0/1(官方 `T_ClinSig.T` 与 0-based 下标同值,不 ±1)。
- **is_main 门 + 恶性限定**:与整图版 train_step 同判据(v × main_w),
  **不加** gt_malignancy 过滤——官方 T 指标只在恶性主病灶上*评分*,但训练侧
  监督吃全部主病灶(良性 T=0 无 是合法监督)。见 `train_step.py` docstring ③
  与 extension_roi.py 同款说明。
- **无 Focal/label_smoothing**:整图版刻意不加(阳性率 35.7% 均衡任务;
  LS 会钳死 T_conf 上界,官方读这个字段)。ROI 版照抄——**别把 CB 权重
  抄过来**(千级计数下 CB 静默退化全 1,整图版实测坐实)。
- **归一范式**:`(loss * v).sum() / (v.sum() + eps)`。
"""

import torch

__all__ = ["compute_clinical_t_roi_loss"]

_EPS = 1e-8
# k_ref 分母平滑(债6 同批,2026-09-25,DR 机制 1):语义见 location_roi._K_REF。
_K_REF = 4.0


def compute_clinical_t_roi_loss(
    t_conf: torch.Tensor,
    t_idx: torch.Tensor,
    t_validity: torch.Tensor,
    is_main: torch.Tensor = None,
):
    """ROI 临床 T 的 masked BCE(概率域)。

    Args:
        t_conf: [K] ROI 头契约输出的 P(T=1)(float,∈[0,1];sigmoid 饱和端点
            0.0/1.0 合法,内部 clamp 防 log(0))。
        t_idx: [K] long,标签 ∈ {0,1}。无效样本处的值任意。
        t_validity: [K] float,1 = 该病灶有 T 标注。
        is_main: [K] float/int,1 = 主病灶。None → 全当主病灶。
    Returns:
        (total, detail)。detail 含 `roi_clin_t`(原始损失)与
        `roi_clin_t_ce`(同值——概率域 BCE 本身就是"纯 CE",无加权无调制,
        保留双键与 ETE/TI-RADS 的 detail 形态对称,消费端 JSONL 逐键 round
        不需特判)。整批无有效样本 → 精确 0、梯度 0、无 NaN。
    """
    assert t_validity.dim() == 1, (
        f"t_validity 必须是 [K] 一维,got {tuple(t_validity.shape)}"
        " —— [K,1] 会隐式广播成 [K,K] 静默放大 loss(QA_VP §2.2)"
    )
    assert t_idx.shape == t_validity.shape, (
        f"标签与 validity 形状不一致: {tuple(t_idx.shape)} vs "
        f"{tuple(t_validity.shape)}"
    )
    assert t_conf.dim() == 1, (
        f"t_conf 必须是 [K] 一维(契约钉死:0 维标量 list 堆成 [K]),"
        f"got {tuple(t_conf.shape)} —— [K,1] 会隐式广播成 [K,K]"
    )
    assert int(t_idx.min()) >= 0 and int(t_idx.max()) < 2, (
        f"临床 T 标签越界: min={int(t_idx.min())}, max={int(t_idx.max())},"
        " 合法区间 [0,1](无标注样本应填占位 0 + validity=0)"
    )

    K = t_conf.shape[0]
    device = t_conf.device

    if is_main is None:
        main_w = torch.ones(K, dtype=torch.float32, device=device)
    else:
        assert is_main.shape == t_validity.shape, (
            f"is_main 与 validity 形状不一致: {tuple(is_main.shape)} vs "
            f"{tuple(t_validity.shape)}"
        )
        main_w = is_main.to(device=device, dtype=torch.float32)

    v = (t_validity.to(device) * main_w).float()

    # 概率域 BCE:fp32 归约(autocast 下 t_conf 可能 bf16)。
    # 🔴 clamp 上界用 1−_EPS 的**字面量**,不是表达式 `1.0 - _EPS`——fp32 下
    # `1.0 - 1e-8` 舍入回 1.0(实测),p=1.0 时 log(1−p)=−inf 毒化整批。
    # 字面量 0.99999999 同样舍入;真正稳定的做法是**双端字面量下界抬升**:
    # 对 y=1 只用 log(p)(p≥eps 有限),对 y=0 只用 log(1−p)(1−p≥eps),
    # 两支按标签分流,任何一支都不碰 0 的对数。
    p = t_conf.float().clamp(min=_EPS)
    one_minus_p = (1.0 - t_conf.float()).clamp(min=_EPS)
    y = t_idx.to(device).to(p.dtype)
    per_sample = -(y * p.log() + (1.0 - y) * one_minus_p.log())
    total = (per_sample * v).sum() / torch.clamp(v.sum() + _EPS, min=_K_REF)
    return total, {"roi_clin_t": total.detach(), "roi_clin_t_ce": total.detach()}
