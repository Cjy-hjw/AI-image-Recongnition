"""由分割 soft mask 派生的标量/局部特征工具（纯张量函数，无 nn 状态）。

从 `multi_task_model.py` 外移至此的原因（rules.md §6）：这些是纯数值工具，
不依赖模型状态，挂在 `MultiTaskThyroidModel` 上属层级错配；且 2026-08-12 阶段 2.0
新增后该文件已达 311 行、越过 300 行分解红线。

本模块服务 Phase 2 三头：
  - `nodule_scale_from_mask` —— 临床 T 头的结节尺度输入（阶段 2.0 ✅）
  - `boundary_band_from_mask` —— 侵犯头的 ROI 交界带局部特征（阶段 2.2 ✅）

⚠️ 本模块所有函数**一律返回 detach 后的张量**：消费方是三头，而三头梯度**不得**
回流分割主干（会破坏 `gate8_adaptor_isolation` 的精确零断言前提，并复刻 2026-08-11
Stage2 崩塌的四因之一——物理先验通路梯度由 0 到满强度）。
"""

import torch
import torch.nn.functional as F

__all__ = ["nodule_scale_from_mask", "boundary_band_from_mask"]


def nodule_scale_from_mask(soft_mask: torch.Tensor) -> torch.Tensor:
    """由 soft mask 导出归一化结节尺度标量，供临床 T 头消费。

    Args:
        soft_mask: [B, 1, H, W] sigmoid 后的概率图。
    Returns:
        [B, 1] —— **归一化等效直径** = sqrt(前景面积 / 图像面积)，值域 [0, 1]，
        且**已 detach**（即便入参带梯度）。

    为什么是等效直径而非「等效体积」（`docs/训练与架构/训练架构方案.md` 的措辞）:
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


def boundary_band_from_mask(
    soft_mask: torch.Tensor, kernel_size: int = 5
) -> torch.Tensor:
    """由 soft mask 用「膨胀 − 腐蚀」取出结节交界带，供侵犯头（ETE）做局部加权池化。

    Args:
        soft_mask: [B, 1, H, W] sigmoid 后的概率图。
        kernel_size: 形态学核边长，**必须是奇数且 ≥3**。
    Returns:
        [B, 1, H, W] 非负权重图，轮廓处最大、结节内部与远处为 0，且**已 detach**。

    ## 为什么是交界带而不是整个结节

    甲状腺外侵犯（ETE）的判据是**肿瘤与包膜/毗邻组织的界面**是否被突破 —— 证据全在
    轮廓一圈，结节实心内部的纹理与 ETE 无关。用交界带加权池化等于把分类器的注意力
    钉在解剖学上真正相关的窄环上，而不是让它自己从整图里找。

    ## 带宽实测（不要按核宽推算）

    `dilate(k) − erode(k)` 的带宽是 **`kernel_size − 1` 像素**，不是 `kernel_size`：
    40×40 画布上 20×20 实心方块实测，k=3 → 每边 2px、k=5 → 4px、k=7 → 6px。
    故 `docs/训练与架构/训练架构方案.md` 所称「5px 交界带」对应 **k=5（实测 4px）**或 k=7（6px），
    这里取 k=5。

    ## 三个必须知道的行为（均已实测，`tests/test_extension_head.py` 锁死）

    1. **空 mask / 全 1 mask → 带恒为 0**。前者是阴性病例，后者是分割崩溃（整图判前景）；
       两种情形都没有"界面"可言，返回 0 是正确语义。下游加权池化必须能吞下
       `分母=0`（见 `ExtensionHead` 的 eps），否则阴性病例直接 NaN。
    2. 🔴 **贴图像边界的结节，其贴边那一侧没有带**。`F.max_pool2d` 的 padding 语义是
       隐式 −inf，于是 `-max_pool2d(-m)`（腐蚀）在越界方向取不到背景值、不会内缩。
       实测：20×20 画布左上角 10×10 前景块，图像角点 (0,0) 带值 0、块的内侧边
       (9,5) 带值 1.0。这是**有意保留的行为** —— 图像边缘外的解剖结构本就未知，
       在那里伪造一条"界面"只会喂给 ETE 头假证据。
       ⚠️ 该行为**只区分得出 zero-padding 的错误实现**（改用零填充会让角点冒出带）；
       reflect 填充实测与 −inf 在所有检查点逐位相同，测试抓不到。别高估这条护栏。
    3. **降采样到特征分辨率后带很窄**。512² 的带在 16×16 特征图上：20px 结节 → 仅
       **4 个** cell 非零，50px → 4 个，100px → 12 个，200px → 28 个（实测）。
       即小结节的局部特征实际只读 4/256 个 cell，方差偏高 —— 但每个 cell 的感受野
       覆盖大片图像，并非只看 32×32 的原图块。若真实数据上 ETE 明显欠拟合，
       **第一个该试的就是加全局池化特征**，而不是先去调采样倍率。

    🔴 **band 一旦被转回低精度再求和就会溢出**（`ExtensionHead` 内 `w` 会 `.to(feat.dtype)`
    变回 fp16，故其求和写的是 `sum(dtype=torch.float32)`）：512² 有 262144 个像素，
    fp16 朴素求和实测直接 `inf`（同 `nodule_scale_from_mask` 的陷阱，QA_VP §1.3）。
    本函数返回的**是 fp32**（内部 `.float()`），所以直接对返回值求和不会溢出 ——
    风险点在下游的精度回转，不在这里。`ExtensionHead` 的做法是先
    `adaptive_avg_pool2d` 到 16×16 再求和：pool 保均值精确（实测 0.5 → 0.5），
    256 个 ≤1 的数相加不可能溢出。

    ℹ️ 同款可微腐蚀 `-max_pool2d(-m)` 在 `physical_prior.py` 里也用了一次（算周长时
    取单侧内边 `m − erode(m)`），**语义不同故未合并**：那里要的是单侧窄边、且**必须
    保留梯度**（物理先验通路有意不 detach，见 `training/stages.py`），本函数则是双侧
    带且一律 detach。合并会让 detach 语义打架，得不偿失；此处交叉引用仅供检索。
    """
    assert soft_mask.dim() == 4 and soft_mask.shape[1] == 1, (
        f"boundary_band_from_mask 期望 [B,1,H,W],got {tuple(soft_mask.shape)}"
    )
    assert kernel_size >= 3 and kernel_size % 2 == 1, (
        f"kernel_size 必须是 ≥3 的奇数,got {kernel_size}"
        " —— 偶数核的 padding 不对称,带会往一侧偏移"
    )
    pad = kernel_size // 2
    m = soft_mask.detach().float()
    dilated = F.max_pool2d(m, kernel_size=kernel_size, stride=1, padding=pad)
    # 腐蚀 = 对偶的膨胀。`clamp(min=0)` 兜的是 soft mask 下 dilate<erode 不可能发生、
    # 但浮点误差可能产出 -1e-8 的情形 —— 负权重会让加权池化的分母与分子反号。
    eroded = -F.max_pool2d(-m, kernel_size=kernel_size, stride=1, padding=pad)
    # `clamp(min=0)` 在数学上**恒真**（同一窗口的 max ≥ min，浮点也不会翻转）——
    # 实测 200 组随机 soft mask + 二值/软阈值/软边共 0 个负例。保留它纯属零成本的
    # 结构性防御（万一将来换非 max_pool 的实现），**不要因为"它从没触发过"就当成
    # 需要清理的死代码**；同样也别指望它兜住什么真实的数值问题。
    return (dilated - eroded).clamp(min=0.0)
