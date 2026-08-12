"""由分割 soft mask 派生的标量/局部特征工具（纯张量函数，无 nn 状态）。

从 `multi_task_model.py` 外移至此的原因（rules.md §6）：这些是纯数值工具，
不依赖模型状态，挂在 `MultiTaskThyroidModel` 上属层级错配；且 2026-08-12 阶段 2.0
新增后该文件已达 311 行、越过 300 行分解红线。

本模块服务 Phase 2 三头：
  - `nodule_scale_from_mask` —— 临床 T 头的结节尺度输入（阶段 2.0 ✅）
  - 交界带（膨胀−腐蚀）提取 —— 侵犯头的 ROI 局部特征（阶段 2.2 待补）

⚠️ 本模块所有函数**一律返回 detach 后的张量**：消费方是三头，而三头梯度**不得**
回流分割主干（会破坏 `gate8_adaptor_isolation` 的精确零断言前提，并复刻 2026-08-11
Stage2 崩塌的四因之一——物理先验通路梯度由 0 到满强度）。
"""

import torch

__all__ = ["nodule_scale_from_mask"]


def nodule_scale_from_mask(soft_mask: torch.Tensor) -> torch.Tensor:
    """由 soft mask 导出归一化结节尺度标量，供临床 T 头消费。

    Args:
        soft_mask: [B, 1, H, W] sigmoid 后的概率图。
    Returns:
        [B, 1] —— **归一化等效直径** = sqrt(前景面积 / 图像面积)，值域 [0, 1]，
        且**已 detach**（即便入参带梯度）。

    为什么是等效直径而非「等效体积」（`docs/训练架构方案.md` 的措辞）:
      ① T 分期临床判据是**最大径**（T1≤2cm / T2 2-4cm / T3>4cm），线性尺度直接对应;
      ② 面积^1.5 会把典型结节（面积占比 ~1%）压到 1e-3 量级，MLP 输入数值塌陷。
    本标量是架构方案所称「结节等效体积标量」的尺度代理。

    🔴 fp16 陷阱:求和累加器**必须**是 fp32,不可删。512² 有 262144 个像素,远超 fp16
    最大值 65504(QA_VP §1.3),朴素求和直接溢出成 inf。
    **触发路径是推理而非训练**:训练侧 autocast 用 bf16(`src/training/loop/train_step.py:91`,
    指数范围同 fp32,不溢出);真正会 fp16 的是推理 `model_runner.run_forward(precision="fp16")`。
    别因为「训练跑得好好的」就以为这条防护多余。
    用 `sum(dtype=torch.float32)` 而非 `.float().sum()`:前者直接以 fp32 累加,后者
    先物化一份整张 [B,1,H,W] 的 fp32 拷贝(实测 fp16/batch=4 峰值 6.29MB vs 2.10MB)。
    二者数值严格等价,在 14G 显存约束下取更省的写法。
    `tests/test_head_interfaces.py::test_nodule_scale_survives_fp16_sum_overflow`
    用反证锁死这条(该测试自带「反证失效」保护,防它哪天变成永远通过的假门)。

    `clamp(min=0)` 不是冗余:本函数是**公开函数**,契约不限定入参来自 sigmoid。
    若未来有人传 logits 进来,负面积会让 `sqrt` 产出 NaN 并静默污染 T 头输入。

    ⚠️ **下游消费方注意维度**:返回 `[B,1]` 而非 `[B]`。QA_VP §1.5/§2.2 点名的坑——
    `[B,1] * [B]` 会隐式广播成 `[B,B]`(实测)且**不报错**,`cat([B],[B,1],dim=1)` 则
    直接 IndexError。临床 T 头 concat 全局特征时须确认两侧都是 2D。
    """
    # 入参形状硬校验:C>1 时 `sum(dim=(2,3))` 返回 [B,C] 而分母仍只是 H*W,
    # 结果语义错**且不崩**(静默错误比崩溃阴)。3D 输入则会在 dim=3 越界处崩,
    # 但报错信息指向 torch 内部、不指向契约,所以这里一并拦下给出可读原因。
    assert soft_mask.dim() == 4 and soft_mask.shape[1] == 1, (
        f"nodule_scale_from_mask 期望 [B,1,H,W],got {tuple(soft_mask.shape)}"
    )
    area = soft_mask.detach().sum(dim=(2, 3), dtype=torch.float32)  # [B, 1]
    n_pixels = float(soft_mask.shape[-2] * soft_mask.shape[-1])
    return torch.sqrt(torch.clamp(area / n_pixels, min=0.0))
