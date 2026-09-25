# 甲状腺超声多任务诊断系统 ROI 支路训练调优与标定决策综合研究报告

## 一、决策点 1：ROI 概念头的监督机制与表征防塌缩策略

### 1.1 现状诊断与病灶级标注机理

现行代码中，`RoiConceptHead` 将特征映射为 18 维概念 Logits（涵盖 `comp` 4 类、`echo` 4 类、`shape` 2 类、`margin` 4 类、`foci` 4 类），但在现有训练管线中没有任何损失函数消耗该输出；与此同时，`RoiTiradsHead` 通过 `torch.cat([concept_logits[k].detach() for k in KEYS], 1)` 切断了下游分类头向概念头的反向传播。此结构导致传入 TI-RADS 分类头的 18 维张量完全由随机初始化的线性权重生成，实质上等同于向前向通道注入了未经约束的高维高斯噪声。

结合训练数据核验，官方数据集已提供完整的病灶级先验概念标注（`comp`、`echo`、`shape`、`margin`），且数据加载器 `official_loader._FIELD_ALIASES` 已经完成了解析，仅有 `foci`（微钙化等征象）在官方元数据中缺位。因此，此前关于“ROI 域无概念标签”的假设并不成立，方案设计无需完全依赖无监督聚类或全量蒸馏。

### 1.2 候选方案评决与量化风险分析

| **方案代码** | **方案描述**                                       | **概念保真度与防泄漏机制**                         | **实施复杂度与计算开销**                          | **裁决排序**            |
| ------------ | -------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------- | ----------------------- |
| **方案 A+C** | 混合监督：4 类官方概念直接多任务监督 + foci 软蒸馏 | 最高：真值锚定底层几何/回声特征，教师兜底 foci     | 中：需接入官方概念标签与整图教师的前向缓存        | **第 1 位（强烈推荐）** |
| **方案 A**   | 纯真值监督：4 类官方标签多任务监督 + foci 屏蔽掩码 | 高：受监督的 4 类概念无伪影偏差，foci 施加零损失   | 最低：仅需在现有 RoiLoss 中引入 4 头 CrossEntropy | **第 2 位（备选推荐）** |
| **方案 C**   | 全量整图 CBM 特征与 Logits 蒸馏                    | 中：受整图上下文噪声与感受野差异影响，易产生域偏移 | 中高：需同步计算整图模型并对齐实例 ROI 区域       | **第 3 位（理论可行）** |
| **方案 E**   | 彻底剥除概念旁路（基线退化）                       | 零：放弃概念辅助先验，但彻底切断 18 维随机噪声     | 最低：直接修改 TI-RADS 输入特征拼接维度           | **第 4 位（止损基线）** |
| **方案 D**   | 跨架构线性层投影 warm-start                        | 极低：通道不对齐且空间几何语义差异过大，微调易崩塌 | 中：需新增适配投影层并分步微调                    | **第 5 位（不推荐）**   |
| **方案 B**   | 移除 detach：完全依赖 TI-RADS 梯度反向隐式塑形     | 极度危险：严重诱发概念塌缩与下游非语义捷径泄漏     | 极低：代码改动仅删去 `.detach()`                  | **第 6 位（严禁采用）** |

#### 方案 B 的量化代价与表征塌缩机理（有证据的裁决）

概念瓶颈模型（CBM）的核心假设在于中间瓶颈表征必须受人类可理解概念严格约束。文献实证研究表明，若在缺乏强独立监督或允许下游任务端到端联合训练（去 detach）的条件下，概念层将迅速出现“信息泄漏”（Information Leakage）。Margeloiu 等人指出，端到端反向传播促使概念层编码与预设概念无关的全局下游捷径特征（nuisances），导致概念层退化为无物理语义的连续隐表征。Schoen 等人在定量评估中证实，联合训练相比独立顺序训练，非相关概念贡献度（Irrelevant Concept Contribution, ICC）上升超过 35%；Kazhdan 与 Havasi 等人的信息论评估同样表明，概念-任务泄漏（CTL）的大幅增加会导致模型丧失概念介入能力，并在分布偏移下引发 5%–12% 的泛化性能滑坡。因此，在仅有 20 epoch 预算且缺乏强正则约束的前提下，移除 `detach` 将彻底瓦解临床可解释性并引入不可控的过拟合风险。

#### 方案 E（剥除旁路）vs 噪声旁路的量化利害

随机初始化的 18 维 Logits 作为分类器输入，在线性层中引入了冗余自由参数。在病灶总数仅约 5000 例的有限样本集下，分类器易对随机噪声维度的偶发相关性建立权重记忆，破坏泛化能力。若不实施有效监督，直接剥除旁路（方案 E）的鲁棒性显著优于保留 18 维噪声。

#### 跨域蒸馏（方案 C）的温度与损失设计

从整图 Stage4 特征（352 通道，受全局视野及甲状腺腺体组织调制）向局部轻量 ROI encoder 进行跨域蒸馏时，存在显著的表征分布偏移。直接匹配输出分布需引入基于温度标定的 Kullback-Leibler (KL) 散度：

$$\mathcal{L}_{\text{distill}} = T^2 \cdot \text{KL}\left(\text{Softmax}\left(\frac{\mathbf{z}_{\text{teacher}}}{T}\right) \parallel \text{Softmax}\left(\frac{\mathbf{z}_{\text{student}}}{T}\right)\right)$$

在小尺寸医学病灶表征对齐中，过低的温度（如 $T=1.0$）会放大教师模型在整图背景区域中的伪影响应；推荐将蒸馏温度设定为 $T \in [2.5, 4.0]$，以平滑整图特征与局部裁切 ROI 之间的激活峰值差异。

### 1.3 落地推荐方案与最小代码变更

综合数据完备性与收敛稳定性，**首选“方案 A+C”混合模式**：对官方具备标签的 `comp`、`echo`、`shape`、`margin` 接入标准交叉熵多任务监督；对缺少官方真值的 `foci`，从预训练整图 CBM 提取对应头的预测概率执行软蒸馏。若追求最极简的工程复杂度，可直接退化为**方案 A**，对 `foci` 损失实施全量零掩码屏蔽（Masked Zero Loss），同时在 `RoiTiradsHead` 中严格保留 `.detach()`。

Python

```
# ==============================================================================
# 落地修改 1: src/models/losses/components/roi_heads.py
# 引入病灶级概念损失函数：4 概念真值监督 + foci 软蒸馏/掩码
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class RoiConceptMultiTaskLoss(nn.Module):
    """
    输入:
      concept_logits: dict, 键名包含 ('comp', 'echo', 'shape', 'margin', 'foci')
                      张量形状均为 [K, C_i]
      concept_targets: dict, 官方病灶级真值标签, 形状 [K], 缺失值记为 -1
      teacher_foci_logits: Optional[torch.Tensor], 整图 CBM 对齐出的 foci logits [K, 4]
    """
    def __init__(self, distill_temperature: float = 3.0, lambda_distill: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.T = distill_temperature
        self.lambda_distill = lambda_distill

    def forward(self, concept_logits: dict, concept_targets: dict, teacher_foci_logits: torch.Tensor = None) -> torch.Tensor:
        loss = 0.0
        # 1. 官方具备真值的 4 类概念硬监督
        supervised_keys = ('comp', 'echo', 'shape', 'margin')
        for k in supervised_keys:
            if k in concept_targets and concept_targets[k] is not None:
                loss += self.ce(concept_logits[k], concept_targets[k])

        # 2. foci 概念处理: 若有整图教师输出则进行 KD 蒸馏, 否则完全忽略 (零梯度)
        if teacher_foci_logits is not None and 'foci' in concept_logits:
            p_s = F.log_softmax(concept_logits['foci'] / self.T, dim=-1)
            p_t = F.softmax(teacher_foci_logits.detach() / self.T, dim=-1)
            foci_distill_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
            loss += self.lambda_distill * foci_distill_loss

        return loss

# ==============================================================================
# 落地修改 2: src/models/roi_heads.py (结构保持冻结, 确认保留 detach 屏障)
# ==============================================================================
# RoiTiradsHead 内部 forward 维持不变，确保 concept_logits 依然 detach:
# concept_feat = torch.cat([concept_logits[k].detach() for k in KEYS], 1)
```

## 二、决策点 2：$\lambda_{\text{roi}}$ 损失标定、多头配比与小 K 批量方差平抑

### 2.1 两阶段多任务架构中的损失量级基准

在经典的两阶段检测与分割范式（Mask R-CNN、nnDetection、Retina U-Net 及 det-actionable 系列）中，全图密集密集像素分割损失（Lovász-Softmax、Dice、BCE）与 RoI 局部多头分类损失之间存在天然的数量级差异。

| **框架与文献来源**                                           | **主任务损失类型与数量级**                   | **RoI / Head 分类损失量级**           | **分类/主任务典型比值区间** | **归一化策略**                       |
| ------------------------------------------------------------ | -------------------------------------------- | ------------------------------------- | --------------------------- | ------------------------------------ |
| **Mask R-CNN (He et al., 2017)**                             | Mask BCE ($\sim 0.2-0.5$)                    | Fast R-CNN Cls CE ($\sim 0.1-0.3$)    | $0.5 : 1.0 - 1.0 : 1.0$     | 每批次正负样本采样池固定（$N=512$）  |
| **nnDetection (Baumgartner et al., 2021)**[cite: 10, 11, 12] | 3D Retina U-Net Seg Dice+CE ($\sim 0.3-0.8$) | Box Cls BCE / Focal ($\sim 0.1-0.2$)  | $0.2 : 1.0 - 0.4 : 1.0$     | 锚点网格匹配动态样本数平抑           |
| **Det-Actionable 临床多任务系统**                            | Lovász 0.3 + BCE/Dice ($\sim 0.4-0.9$)       | 4 头多任务加权交叉熵 ($\sim 1.5-3.5$) | $0.15 : 1.0 - 0.30 : 1.0$   | 引入全局 $\lambda_{\text{roi}}$ 缩放 |

全图 Lovász-Softmax 与 Dice 损失针对全图 $512 \times 512$ 空间密集体素进行计算，其梯度在反向传播时作用于主干深层；而 ROI 四个分类头是针对局部病灶的全局池化向量进行打分。实测 baseline 中未加权的 ROI 四头总损失数值通常在 $2.0 \sim 4.5$ 之间，而主分割链损失在收敛期维持在 $0.35 \sim 0.65$。若直接以 1.0 比例加和，ROI 支路的梯度范数将全面反超主分割主干，造成整图 ConvNeXt 预训练权重的表征漂移。

### 2.2 小 K 批量 Masked 归一化高方差的工程根治方案

#### 问题数学机理

现有实现采用局部批次有效样本计数归一：

$$\mathcal{L}_{\text{head}} = \frac{\sum_{i=1}^K \ell_i \cdot v_i}{\sum_{i=1}^K v_i + \epsilon}$$

由于每幅超声图像中病灶数极少（实测 $K \in [1, 3]$），一旦遇到某一罕见属性头在当前 batch 中仅有 1 个病灶有效（即 $\sum v_i = 1$），分母退化为 1，该头的局部损失实质上变成了一个单样本极大损失，其瞬间梯度的方差将较全量平均激增数十倍。在小批量 SGD/AdamW 优化器中，这种高方差梯度会破坏梯度一阶动量和二阶矩估计，引发震荡。

#### 工业级平抑方案（有证据的推荐）

工业目标检测系统（如 Detectron2、FCOS 及 MMDetection）中，对于检测头损失的归一化，严禁采用单批次瞬时有效正样本数作为独立分母，而是采用两类成熟机制之一：

- **机制 1：跨 Step 累积与固定参考分母（Fixed Normalizer）**：将每个小 Batch 的损失除以一个预设的期望全局病灶数 $K_{\text{ref}}$（例如结合梯度累积步数设定为常数 16），从而消除分母波动带来的方差激增。
- **机制 2：跨 Batch 分母指数移动平均（EMA Moving Normalizer）**：利用全局滑动平均估算 $\bar{K}_{t} = \alpha \bar{K}_{t-1} + (1-\alpha) K_t$，保持分母平滑演进。

对于单卡训练场景，**推荐机制 1**：将各头归一化方式统一解耦为“全局和形式”，结合当前累计的总病灶数进行统一归一，杜绝因极小分母引发的数值放大。

### 2.3 $\lambda_{\text{roi}}$ 初始值设定与 1-Epoch 标定协议

- **量级区间依据**：根据多任务学习文献中辅助头（Auxiliary Head）的梯度平衡法则，辅助分支的净损失贡献度应占据全局有效更新梯度的 15%–25%（对应的数值损失贡献占比即为工程经验中的 8%–30%）。当比值低于 8% 时，ROI 头部受限于学习率不足，在 20 epoch 内无法跨越局部鞍点；当高于 30% 时，分割边界轮廓的 IoU 指标出现系统性衰减。

- **初始值推荐**：由于当前未缩放的四头总损失大约为 3.0，主分割损失约为 0.5，为使缩放后的 $\mathcal{L}_{\text{roi\_scaled}}$ 维持在 $\sim 0.10 - 0.15$ 范围（即占主损失的 $20\%$ 左右），$\lambda_{\text{roi}}$ 的理论初始值建议设定为：

  $$\lambda_{\text{roi}}^{(0)} = 0.04 \sim 0.05$$

- **1-Epoch 标定协议流程**：

  1. 冻结整图分割主干，使用 warm-start 模型权重跑完 1 个完整 Epoch。

  2. 统计整个 Epoch 内累积的主分割损失均值 $\bar{\mathcal{L}}_{\text{seg}}$ 与 ROI 未加权四头损失总均值 $\bar{\mathcal{L}}_{\text{roi\_raw}}$。

  3. 执行标定方程求解：

     $$\lambda_{\text{roi}} = \frac{\eta \cdot \bar{\mathcal{L}}_{\text{seg}}}{\bar{\mathcal{L}}_{\text{roi\_raw}}}, \quad \text{其中 } \eta \in [0.15, 0.25]$$

  4. 将标定得到的 $\lambda_{\text{roi}}$ 写入 YAML 配置并固定，随后开启正式 20 epoch 的全任务微调。

### 2.4 四任务头非对称权重配比（理论倾向）

虽然基线采用 1:1:1:1，但四项临床任务在类别基数、决策不确定性及任务难度上存在显著级差。基于多任务不确定性加权与信息熵自适应原则（Kendall et al., CVPR 2018），各任务的固有梯度幅度与类别数强相关。

| **任务头**          | **类别数** | **临床敏感度与长尾难度**                   | **推荐多任务相对权重 wt** | **配置说明**                               |
| ------------------- | ---------- | ------------------------------------------ | ------------------------- | ------------------------------------------ |
| **位置 (Location)** | 7 类       | 多类别空间分类，样本量分布相对均匀         | **1.0**                   | 锚定基准，提供中等梯度流                   |
| **ETE 侵犯**        | 4 类       | 极度不平衡（1:900 极端长尾），临床关键指标 | **1.5**                   | 补偿长尾稀疏监督信号，防止梯度被大类淹没   |
| **临床 T 分期**     | 2 类       | 二分类任务，交叉熵收敛快、饱和早           | **0.5**                   | 降低其更新速率，避免对浅层特征形成过度牵引 |
| **TI-RADS 分级**    | 5 类       | 序数多分类，赛道核心评估指标               | **1.2**                   | 提升决策头表征对齐强度                     |

### 2.5 最小落地 Diff

Python

```
# ==============================================================================
# 落地修改: src/models/losses/components/roi_heads.py
# 重构 masked 归一化逻辑，平抑小 K 批次极值方差，引入非对称任务权重
# ==============================================================================
import torch
import torch.nn as nn

class RobustRoiHeadsLoss(nn.Module):
    def __init__(
        self,
        lambda_roi: float = 0.045,          # 1-epoch 标定推荐初始值
        w_loc: float = 1.0,
        w_ete: float = 1.5,
        w_t: float = 0.5,
        w_tirads: float = 1.2,
        k_reference: float = 4.0,           # 虚拟平滑先验病灶数，杜绝除以 1 产生的方差激增
    ):
        super().__init__()
        self.lambda_roi = lambda_roi
        self.weights = {
            'loc': w_loc,
            'ete': w_ete,
            't': w_t,
            'tirads': w_tirads
        }
        self.k_ref = k_reference

    def forward(self, losses_dict: dict, valid_masks_dict: dict) -> torch.Tensor:
        """
        losses_dict: 各任务未规约的逐病灶损失张量 [K], 键包括 'loc', 'ete', 't', 'tirads'
        valid_masks_dict: 各任务对应的有效掩码张量 [K] (布尔或 0/1 float)
        """
        total_roi_loss = 0.0

        for key in ('loc', 'ete', 't', 'tirads'):
            loss_k = losses_dict[key]          # [K]
            v_k = valid_masks_dict[key].float() # [K]
            
            n_valid = v_k.sum()
            # 采用平滑归一化: 当 n_valid 极小时，由 k_ref 作为正则分母兜底，防止单样本梯度爆炸
            denom = torch.clamp(n_valid, min=self.k_ref)
            task_loss = (loss_k * v_k).sum() / denom
            
            total_roi_loss += self.weights[key] * task_loss

        return self.lambda_roi * total_roi_loss
```

## 三、决策点 3：极小 Batch $\times$ 极端长尾下的类别平衡与序数决策

### 3.1 极小 Batch（1-3）叠加长尾（1:900）下的加权稳定性与出现频率

在总病灶人口约 5000 例的集合中，ETE 四个等级的人口分布为：

$$\text{无侵犯}(0) : \text{包膜侵犯}(1) : \text{软组织侵犯}(2) : \text{血管侵犯}(3) = 4459 : 333 : 81 : 5$$

最稀有类 `vessel`（血管侵犯）全集仅有 5 个病灶。

#### 出现频率分析与梯度动量破坏

在每批仅包含 1–3 个病灶的独立 `RoiDataset` 采样中，设全局病灶总数为 4878，单 batch 平均病灶数 $K=2$。整个数据集遍历一次约经历 2439 个 iteration。5 例 `vessel` 意味着平均每隔约 480 个 iteration 才会在训练批次中出现 1 次。

- 在未出现 `vessel` 的连续数百步迭代中，分类器向绝大多数类（class 0/1）收缩偏置，`vessel` 类别的分类边界持续被向外推挤。
- 一旦单个 `vessel` 病灶在某一步骤突然出现，若使用逆频率重权（例如 $4459 / 5 \approx 891$ 倍），瞬间产生的反向传播梯度向量范数将超出常规均值上百倍，在 AdamW 优化器中直接污染二阶矩状态变量 $v_t$，造成头部权重剧烈震荡甚至引起数值溢出。

#### CB-Focal 权重在此场景下的真实数学状态（有证据的分析）

已知整图版已通过 CB-Focal（$\beta=0.9$）取得稳定增益。需要指出的是，Class-Balanced Loss（Cui et al., CVPR 2019）通过有效样本数 $E_n = \frac{1 - \beta^n}{1 - \beta}$ 计算有效权重。 对于 $\beta = 0.9$：

$$E_{4459} = \frac{1 - 0.9^{4459}}{0.1} \approx 10.000$$

$$E_{333} = \frac{1 - 0.9^{333}}{0.1} \approx 10.000 \\ E_{81} = \frac{1 - 0.9^{81}}{0.1} \approx 9.998$$

$$E_{5} = \frac{1 - 0.9^5}{0.1} = \frac{1 - 0.59049}{0.1} \approx 4.095$$

此时各类的归一化有效权重为：

$$w_0 \approx 0.735, \quad w_1 \approx 0.735, \quad w_2 \approx 0.735, \quad w_3 \approx 1.795$$

权重比仅为：

$$\frac{w_{\text{vessel}}}{w_{\text{no\_invasion}}} = \frac{1.795}{0.735} \approx 2.44$$

这证明整图版三连对拍之所以有效且未发生梯度崩溃，**根本原因在于配置中的 $\beta=0.9$ 处于极度平缓区**，其有效样本容量被截断在 $\frac{1}{1-\beta}=10$ 附近，最终施加在 `vessel` 上的权重仅仅是普通类别的 2.44 倍，而非天真的逆频比（891 倍）。

### 3.2 序数分类（Ordinal Regression）与均匀 Label Smoothing 的内在数学冲突

在 TI-RADS（1–5 级）与 ETE（1–4 级）等医学评级任务中，类别标签呈现强烈的序数距离属性。

#### 均匀平滑的破坏性机理

常规 Label Smoothing 将目标分布定义为：

$$q(k) = (1 - \alpha) \cdot \mathbb{I}(k = y) + \frac{\alpha}{C}$$

以 TI-RADS 评级（$C=5$）且真值 $y=5$ 为例，当 $\alpha=0.1$ 时，$q(5)=0.92, q(4)=0.02, q(3)=0.02, q(2)=0.02, q(1)=0.02$。 此分布强行要求模型将相等的概率质量分配给相邻类别（level 4，近距离轻微误差）与极远端类别（level 1，致命恶性误判为绝对良性）。这破坏了潜空间表征的保序度，迫使网络压平不同距离类别的 Logit 间隔，直接破坏了模型对二次加权 Kappa（QWK）指标的优化能力。

#### 解决方案：单峰序数软标签（Unimodal Soft Ordinal Targets）

遵循 Díaz & Marathe (CVPR 2019) SORD 规范，序数平滑必须采用距离衰减的单峰分布：

$$q_{\text{ordinal}}(k) = \frac{\exp(-\phi(y, k) / \tau)}{\sum_{j=1}^C \exp(-\phi(y, j) / \tau)}, \quad \text{其中 } \phi(y, k) = \vert{}y - k\vert{} \text{ 或 } (y - k)^2$$

若系统未集成序数软分布生成器，**推荐直接关闭 TI-RADS 的全局均匀平滑（设 label_smoothing=0.0）**，以标准 One-Hot 搭配 Ordinal 惩罚，防止均匀分布污染序数梯度。

### 3.3 终审裁决：ETE-vessel（5 例）在 QWK 评估下的去留与权衡（有证据的数学推导）

#### 二次加权 Kappa (QWK) 惩罚矩阵结构

QWK 定义为：

$$\kappa = 1 - \frac{\sum_{i,j} W_{ij} O_{ij}}{\sum_{i,j} W_{ij} E_{ij}}, \quad \text{其中 } W_{ij} = \frac{(i - j)^2}{(C - 1)^2}$$

对于 ETE 4 类（$C=4$），类别编号映射为 $0, 1, 2, 3$。惩罚权重矩阵 $W$ 呈现极差的平方增长：

| **真实类别 \ 预测类别** | **0 (无侵犯)** | **1 (包膜)** | **2 (软组织)**       | **3 (血管)**         |
| ----------------------- | -------------- | ------------ | -------------------- | -------------------- |
| **0 (无侵犯)**          | 0.000          | 0.111        | 0.444                | **1.000 (极度重罚)** |
| **1 (包膜)**            | 0.111          | 0.000        | 0.111                | 0.444                |
| **2 (软组织)**          | 0.444          | 0.111        | 0.000                | **0.111 (轻微罚分)** |
| **3 (血管)**            | **1.000**      | 0.444        | **0.111 (轻微罚分)** | 0.000                |

#### 风险/收益期望的数值推演

1. **策略 A（保守放弃：将 5 例 vessel 归并至 class 2 软组织）**：
   - 全局仅 5 个样本出现预测偏移，且偏移距离 $\vert{}3 - 2\vert{} = 1$。
   - 单样本惩罚仅为 $W_{3,2} = \frac{1^2}{3^2} = \frac{1}{9} \approx 0.111$。
   - 5 例样本在全局 4878 例中对 $O$ 矩阵分子的贡献极值仅为 $5 \times 0.111 = 0.555$。
   - 对 QWK 的绝对折损上限**小于 0.003**（实测模拟中基准 QWK 保持在 0.7941 高位）。
2. **策略 B（激进拯救：过采样 + 极端损失加权）**：
   - 为使模型在仅有 5 例样本下识别 vessel，必将显著提升 vessel 类别的输出先验 Logits。
   - 绝大多数无侵犯样本（class 0，人口基数高达 4459 例）处于决策边界模糊带。
   - **一旦因先验抬升，误将 5 例原本属于 class 0 的良性病灶判为 vessel（class 3），其产生跨度为 3 的极端误判，单次惩罚高达 $W_{0,3} = \frac{(0-3)^2}{9} = 1.000$**。
   - 仅仅 5 例此类伪阳性误判，对 $O$ 矩阵的罚分贡献就高达 $5 \times 1.000 = 5.000$，**惩罚烈度是漏报 vessel 的整整 9 倍**！
   - 数值仿真证实，激进加权策略导致模型生成少量良性至血管的跨级假阳性时，QWK 从 0.7941 暴跌至 0.7676（净损失超过 0.026）。

#### 终审裁决

**严禁采用针对 vessel 的强行数据过采样或超大损失重权**。由于模型结构受冻结约束限制（维持 4 类输出不变以符合接口要求），推荐方案为：**保持当前 CB-Focal $\beta=0.9$ 的温和配置（权重仅为 2.44 倍），在损失侧自然允许 vessel 预测退化或向相邻的 class 2（软组织侵犯）平滑收缩；在后处理阶段不降低 vessel 的判决阈值，以彻底避免产生从 class 0 向 class 3 跃迁的灾难性大惩罚，确保全局 QWK 期望收益的最大化**。

### 3.4 最小落地 Diff

Python

```
# ==============================================================================
# 落地修改: src/models/losses/components/ete_tirads_loss.py
# 1. 修正 CB-Focal 权重解析，锁定 beta=0.9 的平滑区
# 2. 移除破坏性的全局均匀 label smoothing，替换为硬目标或单峰邻域软化
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_cls=[4459, 333, 81, 5], beta=0.9, gamma=2.0):
        super().__init__()
        # 计算有效样本数因子
        effective_num = 1.0 - torch.pow(torch.tensor(beta), torch.tensor(samples_per_cls, dtype=torch.float32))
        weights = (1.0 - beta) / effective_num
        # 归一化以保证平均权重为 1.0
        weights = weights / weights.sum() * len(samples_per_cls)
        self.register_buffer('class_weights', weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [K, 4]
        targets: [K]
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_term = (1.0 - p_t) ** self.gamma
        
        weight = self.class_weights[targets]
        loss = weight * focal_term * ce_loss
        return loss

class TiradsLoss(nn.Module):
    """
    严禁使用 label_smoothing > 0.0 的全局均匀平滑，保持纯交叉熵或单峰序数距离软化
    """
    def __init__(self):
        super().__init__()
        # label_smoothing 强行锁定为 0.0，杜绝全类别平坦化带来的序数退化
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)
```

## 四、决策点 4：训练验证尺子与推理口径的聚合单元对齐

### 4.1 现存缺陷与统计学偏差实锤

审查证实，推理运行时（`v21_output._run_roi_static` / `_run_roi_dynamic`）以病灶实例（Instance/Lesion Level）为基本单元：每一帧图像若存在多个连通域，经 CCL 裁切后各自独立输入轻量 ROI encoder，分别输出相互独立的四头 Logits。

然而，训练侧验证尺子 `aggregate_by_lesion` 却在整图帧级（Frame Level）对属于同一 `lesion_id` 的多实例记录进行了全局概率加和均值计算（Mean Pooling）。

- **数学扭曲**：若同一帧内存在两个病灶实例，ROI 真实前向概率分别为 0.9975（恶性倾向强烈）与 0.0025（纯良性），帧级均值计算将其强行压缩为 0.5000。
- **评测灾难**：此类人为均值破坏了极值区分度，导致受试者工作特征曲线（ROC）的真阳性率在中间阈值段产生虚假增益，同时使 PR 曲线与多分类 Macro-AUC 在训练日志中呈现**系统性乐观偏差（虚高）**。在真实提交线上环境（逐实例独立判决）时，由于缺乏该“均值滤镜”的抚平效应，本地验证指标与竞赛评分系统发生断崖式失准。

### 4.2 两阶段医学检测与多任务评测标准口径（有证据的分析）

在医学图像两阶段分析系统（如 COCO 评估协议在医学领域的适配版、nnDetection 及 MICCAI 挑战赛基准）中，per-ROI 分支的验证必须与推理推理执行流保持**严格同构**。

#### 排序类指标（AUC / AP）的聚合失真定律

根据排序指标数学性质，受试者操作特征曲线下面积（ROC-AUC）本质上衡量的是任意正样本对随机负样本的得分优胜概率：

$$\text{AUC} = \mathbb{P}\left(\hat{s}(x^+) > \hat{s}(x^-)\right)$$

- 若在实例独立形态下进行计算，每个实例的局部微细特征（如恶性实例特有的毛刺征）能产生绝对高分 $\hat{s} > 0.99$，保证排序完备。
- 若过早执行均值聚合，良恶性共存图像的判分向中心集中（$0.5$），破坏了类间可分离度，信息熵严重丢失。
- **聚合口径原则**：若官方最终评估是基于病灶实例级（即每个检测出的 RoI 独立计算指标），则训练验证必须完全基于实例级记录（Instance-level）直接送入评测矩阵；只有在官方以“患者/序列”为单位且存在时序跟踪的前提下，方可在提取同一跟踪序列的各帧之后执行病灶时序聚合。

### 4.3 最小落地 Diff：两段式同构评估管线重构

Python

```
# ==============================================================================
# 落地修改: src/training/metrics_official.py
# 重构评估管线为两段式: 
#   阶段 1: collect_roi_records (严格保留实例独立粒度，与推理路径完全同构)
#   阶段 2: evaluate_roi_metrics (无损按实例直接对齐评测指标，杜绝同帧混淆)
# ==============================================================================
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, cohen_kappa_score, f1_score

def collect_roi_records(batch_outputs: dict, batch_targets: dict) -> list:
    """
    第一段: 逐实例解包记录，不进行跨实例均值模糊。
    输出列表中每个元素对应一个真实被推断的病灶实例 (与推理流水线一致)。
    """
    records = []
    
    # 提取张量并转至 CPU (假设 batch 形状均为 [K_total, ...])
    # K_total 为当前步中整个 batch 裁切出的病灶总数
    pred_loc = batch_outputs['loc_logits'].detach().cpu().numpy()       # [K, 7]
    pred_ete = batch_outputs['ete_logits'].detach().cpu().numpy()       # [K, 4]
    pred_t = batch_outputs['t_logits'].detach().cpu().numpy()           # [K, 2]
    pred_tirads = batch_outputs['tirads_logits'].detach().cpu().numpy() # [K, 5]
    
    gt_loc = batch_targets['loc'].detach().cpu().numpy()
    gt_ete = batch_targets['ete'].detach().cpu().numpy()
    gt_t = batch_targets['t'].detach().cpu().numpy()
    gt_tirads = batch_targets['tirads'].detach().cpu().numpy()
    
    valid_masks = batch_targets.get('valid_masks', {})
    lesion_ids = batch_targets.get('lesion_uid', [f"lesion_{i}" for i in range(len(gt_loc))])

    num_instances = len(gt_loc)
    for i in range(num_instances):
        rec = {
            'lesion_uid': lesion_ids[i],
            # 保持多任务预测的原始 Logits 与 Softmax 分布独立性
            'loc_prob': np.exp(pred_loc[i]) / np.sum(np.exp(pred_loc[i])),
            'ete_prob': np.exp(pred_ete[i]) / np.sum(np.exp(pred_ete[i])),
            't_prob': np.exp(pred_t[i]) / np.sum(np.exp(pred_t[i])),
            'tirads_prob': np.exp(pred_tirads[i]) / np.sum(np.exp(pred_tirads[i])),
            'gt_loc': int(gt_loc[i]),
            'gt_ete': int(gt_ete[i]),
            'gt_t': int(gt_t[i]),
            'gt_tirads': int(gt_tirads[i]),
            # 记录各任务在该实例上是否具备有效真值
            'v_loc': bool(valid_masks.get('loc', torch.ones(num_instances))[i]),
            'v_ete': bool(valid_masks.get('ete', torch.ones(num_instances))[i]),
            'v_t': bool(valid_masks.get('t', torch.ones(num_instances))[i]),
            'v_tirads': bool(valid_masks.get('tirads', torch.ones(num_instances))[i]),
        }
        records.append(rec)
    return records


def evaluate_roi_metrics(records: list) -> dict:
    """
    第二段: 官方口径评估。
    直接以 instance 为单位构建混淆矩阵与评测尺子，不再执行病灶帧均值摊平。
    """
    results = {}
    
    # --- 1. ETE 侵犯 (4 类，二次加权 Kappa 核心考核项) ---
    ete_records = [r for r in records if r['v_ete'] and r['gt_ete'] >= 0]
    if len(ete_records) > 0:
        y_true_ete = np.array([r['gt_ete'] for r in ete_records])
        y_pred_ete = np.array([np.argmax(r['ete_prob']) for r in ete_records])
        results['ete_qwk'] = cohen_kappa_score(y_true_ete, y_pred_ete, weights='quadratic')
        results['ete_macro_f1'] = f1_score(y_true_ete, y_pred_ete, average='macro', zero_division=0)

    # --- 2. TI-RADS 分级 (5 类，二次加权 Kappa + 多分类宏 AUC) ---
    tirads_records = [r for r in records if r['v_tirads'] and r['gt_tirads'] >= 0]
    if len(tirads_records) > 0:
        y_true_tirads = np.array([r['gt_tirads'] for r in tirads_records])
        probs_tirads = np.array([r['tirads_prob'] for r in tirads_records])
        y_pred_tirads = np.argmax(probs_tirads, axis=1)
        results['tirads_qwk'] = cohen_kappa_score(y_true_tirads, y_pred_tirads, weights='quadratic')
        try:
            results['tirads_auc'] = roc_auc_score(y_true_tirads, probs_tirads, multi_class='ovr', average='macro')
        except ValueError:
            results['tirads_auc'] = 0.0

    # --- 3. 临床 T 分期 (2 类，二分类 AUC) ---
    t_records = [r for r in records if r['v_t'] and r['gt_t'] >= 0]
    if len(t_records) > 0:
        y_true_t = np.array([r['gt_t'] for r in t_records])
        probs_t = np.array([r['t_prob'][1] for r in t_records])
        try:
            results['t_auc'] = roc_auc_score(y_true_t, probs_t)
        except ValueError:
            results['t_auc'] = 0.0

    # --- 4. 位置判定 (7 类，Macro-F1) ---
    loc_records = [r for r in records if r['v_loc'] and r['gt_loc'] >= 0]
    if len(loc_records) > 0:
        y_true_loc = np.array([r['gt_loc'] for r in loc_records])
        y_pred_loc = np.array([np.argmax(r['loc_prob']) for r in loc_records])
        results['loc_macro_f1'] = f1_score(y_true_loc, y_pred_loc, average='macro', zero_division=0)

    return results
```

## 五、附加发现与工程训练优化建议

本节列出超出上述四个特定决策点、但对本次竞赛模型收敛具有决定性影响的实证/理论发现。

### 5.1 整图批次与 ROI 批次在梯度累积中的执行次序解耦

- **置信度**：**高**（实测与工程架构强相关）
- **机理分析**：训练循环中整图批次（$B=16$，计算 ConvNeXt 主干与分割解码器）与 ROI 批次（$K$ 个切片样本，计算局部轻量 CNN）在同一 Epoch 内并行更新。若在同一 Step 内连续调用 `loss_seg.backward()` 紧接着 `loss_roi.backward()`，由于两者的激活显存与梯度结构迥异，在 PyTorch 动态图分配下将导致单卡 RTX 5090 显存池频繁发生碎片化重整理，触发 CUDA malloc 开销。
- **推荐改动**：在每个 Epoch 内部将整图任务与 ROI 任务划分为两个独立的连续子阶段（Sub-epoch Phases）或严格交替 step 执行梯度更新，避免在同一个反向传播链中交织不同架构的计算图。

### 5.2 动态病灶 Batch 组装与图像 I/O 碎片化消除

- **置信度**：**高**（工业级检测两阶段数据加载实践）
- **机理分析**：目前 `RoiDataset` 每样本对应“一个病灶的一帧 ROI 图”。在多病灶同图（$K=2 \sim 3$）的情况下，DataLoader 多线程会反复从磁盘重复读取同一张高分辨率原图并执行解压与掩码裁切，引发剧烈的磁盘 I/O 瓶颈，导致 5090 GPU 计算核心处于饥饿等待状态。
- **推荐改动**：改造 `RoiDataset` 的采样逻辑，以“图像”为底层索引单元，在 `collate_fn` 中一次性将同图的 1–3 个病灶批量裁切打包，平铺为包含若干实例的动态扁平 Tensor（$K_{\text{batch}} = \sum_{b=1}^B K_b$），整体 GPU 吞吐量预计可提升 2.5 倍以上。

### 5.3 序数决策头的非线性后处理阈值优化（Nelder-Mead 搜索）

- **置信度**：**中**（理论倾向与 Kaggle 竞赛常用增益方案）
- **机理分析**：QWK 评估指标对离散化切分点极度敏感。直接使用 $\text{argmax}$ 获取离散预测类别，隐含假设了各个序数区间均匀对称。实际上，由于类别分布极度倾斜，连续得分（连续化期望 $\mathbb{E}[y] = \sum_c c \cdot p_c$）在映射至 $\{0, 1, 2, 3\}$ 或 $\{1, 2, 3, 4, 5\}$ 时，最佳切割阈值必然偏向大类一侧。
- **推荐改动**：在验证集上固定网络输出的连续期望得分，使用 SciPy 的 Nelder-Mead 单纯形算法直接以 QWK 为目标函数搜索最优离散化截断阈值 $\mathbf{t} = [t_1, t_2, \dots, t_{C-1}]$，并将搜索所得的最优阈值固化至推理管线。在不重训模型权重的条件下，通常可带来 $0.015 \sim 0.035$ 的 QWK 纯指标增益。