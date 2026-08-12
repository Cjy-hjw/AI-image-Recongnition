"""Post-hoc 置信度校准(温度缩放)。

神经网络对分类置信度普遍过度自信(predicted prob 虚高于真实通过率)。
温度缩放(Temperature Scaling, Guo et al. 2017)是 post-hoc 方法:模型冻结,
只在**独立验证集**上为每个任务学一个标量温度 T,推理时 `softmax(logits / T)`。
T>1 压低过度自信,T<1 反向;**不改 argmax 预测类别、不改准确率**,只调概率刻度。

设计约束(本仓核心):
- 真实数据 2026-08-01 初赛才到 → 现在**只搭骨架,不拟合任何 T**。
- 参数文件缺失 / 字段缺失 → 该任务 **T=1.0(降级=不校准)**,输出与无校准逐位相同。
- 有序任务(TI-RADS composition/echogenicity/margin 等级有序)预留 `fit_ordinal_temperature`,
  跨级错误代价更高,真数据到位后再启用。

插入点唯一: `postprocess.decode_predictions()` 在 softmax/sigmoid **之前**调用 `apply`。
mask 为分割,不走温度缩放(分割校准是另一套,本期不碰)。
"""

import json
import os

import numpy as np
import torch

# 走温度缩放的分类任务(mask 分割不在内)。foci 为多标签 sigmoid,其余为 softmax。
CALIBRATED_TASKS = (
    "composition",
    "echogenicity",
    "shape",
    "margin",
    "foci",
    "malignancy",
)


class TemperatureScaler:
    """每任务一个标量温度。缺参数即 T=1.0 降级(不改变任何输出)。"""

    def __init__(self, temperatures: dict | None = None):
        # 缺省全 1.0:apply 等于恒等,decode_predictions 行为不变。
        self.temperatures = {t: 1.0 for t in CALIBRATED_TASKS}
        if temperatures:
            for task, temp in temperatures.items():
                if task not in CALIBRATED_TASKS:
                    print(f"[WARNING] calibration: 忽略未知任务 '{task}'")
                    continue
                if not (temp and temp > 0):
                    print(f"[WARNING] calibration: 任务 '{task}' 温度非法({temp}),降级 T=1.0")
                    continue
                self.temperatures[task] = float(temp)

    @classmethod
    def from_file(cls, path: str | None) -> "TemperatureScaler":
        """从 JSON 参数文件构造。文件缺失/损坏 → 全 T=1.0 降级,打印 WARNING 不抛异常。"""
        if not path or not os.path.exists(path):
            print(
                f"[WARNING] calibration: 参数文件不存在({path}),全任务 T=1.0(不校准)。"
                "等初赛真实数据到位后用 fit_temperature 拟合并写入。"
            )
            return cls(None)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARNING] calibration: 参数文件解析失败({e}),全任务 T=1.0 降级。")
            return cls(None)
        temps = data.get("temperatures", data)  # 兼容 {"temperatures": {...}} 或扁平 {...}
        print(f"[INFO] calibration: 已加载温度参数 {temps}")
        return cls(temps)

    def apply(self, task: str, logits: torch.Tensor) -> torch.Tensor:
        """对指定任务 logits 做温度缩放。T=1.0 时返回原张量(恒等)。

        在 softmax/sigmoid **之前**调用。不就地修改输入。
        """
        temp = self.temperatures.get(task, 1.0)
        if temp == 1.0:
            return logits
        return logits / temp

    def is_identity(self) -> bool:
        """全任务 T=1.0(纯降级)时为 True —— 用于断言无行为变化。"""
        return all(t == 1.0 for t in self.temperatures.values())


# ---------------------------------------------------------------------------
# 离线拟合(真实数据到位后跑;现在不调用)。
# ---------------------------------------------------------------------------

def fit_temperature(
    logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50, lr: float = 0.01
) -> float:
    """二分类/多分类温度缩放:LBFGS 最小化温度后 NLL,学单个标量 T。

    Args:
        logits: [N, C] 未经 softmax 的原始 logits(独立验证集上前向所得)。
        labels: [N] int64 类别索引。
    Returns:
        最优温度 T(float)。

    原理: 冻结模型,仅优化 T 使 `CrossEntropy(softmax(logits/T), labels)` 最小。
    softmax 单调,T 不改 argmax,故准确率不变,只校准置信度刻度。
    """
    logits = logits.detach().float()
    labels = labels.detach().long()
    log_t = torch.zeros(1, requires_grad=True)  # 优化 log T 保证 T>0
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def fit_ordinal_temperature(
    logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50, lr: float = 0.01
) -> float:
    """有序任务温度缩放(TI-RADS 等级有序:TR5→TR1 比 TR5→TR4 错得更重)。

    当前实现: 在标准温度缩放 NLL 上加**距离加权**项,惩罚预测期望等级与真实等级的偏离,
    使校准后概率质量更贴近真实等级邻域。真实数据到位后可替换为累积链接(cumulative-link)模型。

    Args:
        logits: [N, C] 有序类别 logits(类索引 0..C-1 即等级序)。
        labels: [N] int64 等级索引。
    Returns:
        最优温度 T(float)。
    """
    logits = logits.detach().float()
    labels = labels.detach().long()
    num_classes = logits.shape[1]
    grades = torch.arange(num_classes, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled = logits / log_t.exp()
        loss = nll(scaled, labels)
        # 期望等级与真实等级的平方距离(有序惩罚)
        probs = torch.softmax(scaled, dim=1)
        expected_grade = (probs * grades).sum(dim=1)
        ordinal_penalty = ((expected_grade - labels.float()) ** 2).mean()
        total = loss + 0.1 * ordinal_penalty
        total.backward()
        return total

    optimizer.step(closure)
    return float(log_t.exp().item())


# ---------------------------------------------------------------------------
# 诊断指标: ECE + 可靠性图。
# ---------------------------------------------------------------------------

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error(分箱置信度 vs 真实准确率的加权差)。

    Args:
        probs: [N, C] softmax 概率,或 [N] 二分类正类概率。
        labels: [N] 真实类别索引。
    Returns:
        ECE ∈ [0,1],越小越校准。
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if probs.ndim == 1:  # 二分类正类概率 → 补成两列
        probs = np.stack([1.0 - probs, probs], axis=1)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = in_bin.mean()
        if prop > 0:
            acc_bin = accuracies[in_bin].mean()
            conf_bin = confidences[in_bin].mean()
            ece += np.abs(conf_bin - acc_bin) * prop
    return float(ece)


def reliability_diagram(
    probs: np.ndarray, labels: np.ndarray, save_path: str, n_bins: int = 15, title: str = ""
) -> None:
    """绘制可靠性图(每箱准确率 vs 平均置信度,对角线=完美校准)。失败不抛(无 matplotlib 时跳过)。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] calibration: 未安装 matplotlib,跳过可靠性图。")
        return

    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=1)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() > 0:
            centers.append((lo + hi) / 2)
            accs.append(accuracies[in_bin].mean())
            confs.append(confidences[in_bin].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="完美校准")
    ax.bar(centers, accs, width=1.0 / n_bins, alpha=0.7, edgecolor="black", label="准确率")
    ax.set_xlabel("置信度")
    ax.set_ylabel("准确率")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(title or f"Reliability Diagram (ECE={compute_ece(probs, labels, n_bins):.4f})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
