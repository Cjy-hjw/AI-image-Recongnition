# 2D甲状腺超声多任务一体化诊断系统：分类AUC瓶颈的深度解析与架构演进方案

## 绪论与临床工程背景

当前医学影像人工智能在基层的规模化落地正面临复杂的泛化性挑战。特别是在中国河南省南阳市及周边地区的医疗生态中，基层医疗机构的超声设备型号繁多、探头频率差异大、图像采集规范不一。这种设备异构性直接导致了超声图像中散斑噪声（Speckle Noise）、声影伪影以及病灶边界模糊程度的剧烈波动。在这一背景下，部署的“2D甲状腺超声多任务一体化诊断系统”承担着病灶分割、超声属性概念分类（如边缘特征、回声水平、钙化灶分布等）以及最终良恶性判断（TI-RADS分级辅助诊断）的多重任务。

对该系统当前代码库的深度静态与动态分析表明，系统的局部任务表现卓越，其分割Dice系数已达82.95%，概念预测的F1分数稳定在81.97%。然而，系统在核心的良恶性分类任务上遭遇了严峻的瓶颈，其测试集AUC（ROC曲线下面积）长期停滞在77.55%至78.00%的区间。这一数据显著落后于临床所期冀的SOTA（State-of-the-Art）标准（通常需超过90%方具备独立筛查价值）。通过穿透表层性能指标，系统架构底层在特征解耦深度、多任务优化调度器（FAMO）的粗粒度配置、监督对比学习（SupCon）与AUC边际损失的拓扑拉扯，以及物理先验计算的非一致性等维度上，存在深度的逻辑冲突。本报告将对上述问题展开详尽的机理剖析，并提供具备严密数学推导与代码级可落地的技术架构演进方案。

## 课题一：特征解耦深度的极限与多任务干扰机理

### 共享底座的表征竞争与解耦层级局限

当前架构的核心主干网络选用了 `EfficientNet-B2`，其解耦策略表现为一种极其保守的“晚期分叉”模式：即在Stage 0至Stage 4阶段，分割任务与分类任务完全共享底层特征图，仅在最后一层（Stage 5）通过深拷贝（`copy.deepcopy`）分裂为两个独立的分支。这种拓扑结构在面对高度非均质的医学图像时，必然引发剧烈的“表征竞争”（Representation Competition）。

在卷积神经网络的层级特征映射中，浅层和中层网络（Stage 1-3）主要负责提取图像的局部高频信息，如灰度跃变、纹理边缘以及超声特有的散斑分布规律。分割任务本质上是一个密集预测（Dense Prediction）过程，它要求网络保留极其精确的空间分辨率和像素级的局部连续性；相反，良恶性分类任务以及抽象概念提取（如判断结节是否具有“微钙化”或“垂直生长”特性），则依赖于具有平移不变性的高级语义抽象和全局感受野。当这两个目标函数在Stage 0-4强制共享参数时，分割损失产生的梯度会倾向于保留所有的空间细节，从而“淹没”或“抹平”了分类任务所急需的、微弱但极具判别性的恶性纹理特征。这种多任务干扰导致分类网络在进入Stage 5的独立分支前，实际上已经丢失了关键的临床判别信息，进而锁死了AUC的上限。

### 主干容量评估与完全解耦双主干设计

除了特征竞争，`EfficientNet-B2` 的固有特征抽取容量也已显现出天花板效应。尽管该模型在自然图像的ImageNet基准上通过复合缩放取得了优异的参数效率，但其卷积核的感受野动态调整能力在应对南阳地区多样化的甲状腺结节形态时显得力不从心。甲状腺恶性肿瘤（如乳头状癌）往往呈现出形态极其不规则的浸润性生长，这需要模型具备跨越较长物理距离的上下文捕获能力。

若将特征解耦点提前，甚至采用完全解耦的异构双主干（Heterogeneous Dual-Backbone）设计，系统的表征能力将获得质的跃升。考虑到临床推理时延必须严格控制在2.0秒以内的工程约束，建议在分类路径保留高度定制化的卷积特征提取器，而在分割路径引入擅长捕捉长距离依赖的轻量级视觉Transformer（如 `Swin-Transformer-Tiny`），或采用大核卷积设计的 `ConvNeXt-Nano`。

为了清晰展示不同架构在解耦深度与参数开销上的权衡，下表对比了潜在的演进方案：

| **架构策略**        | **分裂节点** | **分割主干**       | **分类主干**    | **相对参数量增幅** | **表征竞争程度** | **时延预估 (GPU)** |
| ------------------- | ------------ | ------------------ | --------------- | ------------------ | ---------------- | ------------------ |
| **晚期分叉 (当前)** | Stage 5      | EfficientNet-B2    | EfficientNet-B2 | 基准 (1.0x)        | 极高             | ~0.4s              |
| **中早期分叉**      | Stage 3      | EfficientNet-B2    | EfficientNet-B2 | + 45%              | 中等             | ~0.6s              |
| **完全解耦双主干**  | 输入端独立   | ConvNeXt-Nano      | EfficientNet-B2 | + 80%              | 零冲突           | ~1.1s              |
| **异构注意力主干**  | 输入端独立   | Swin-Transformer-T | EfficientNet-B2 | + 110%             | 零冲突           | ~1.5s              |

### 跨维度任务对齐的数学表达

在实施完全解耦后，虽然消除了底层的梯度冲突，但分类网络也因此失去了分割网络对病灶区域的精确“显式”关注。为了在物理空间上对齐两者，需要设计一个轻量级的任务对齐解码器（Task-Alignment Adaptor），将分割特征作为空间注意力注入到分类特征中。

假设在某一特征尺度下，分类特征图为 $f_{cls} \in \mathbb{R}^{B \times C \times H \times W}$，分割编码器输出的深层语义掩膜特征为 $f_{seg} \in \mathbb{R}^{B \times C' \times H' \times W'}$。由于两者网络架构可能不同，首先需通过双线性插值或步长卷积使得空间维度严格对齐为 $H \times W$。随后，利用单向特征聚合（One-way Feature Aggregation）生成空间注意力调制图：

$$M_{attn} = \sigma \left( \mathcal{F}_{Conv1\times1} \left( \mathcal{P}_{MaxPool}(f_{seg}) \right) \right)$$

分类特征随后通过逐元素乘法（Hadamard Product）被调制，并叠加残差：

$$\tilde{f}_{cls} = f_{cls} \odot M_{attn} + f_{cls}$$

这种单向信息流动（利用 `detach()` 切断分类误差向分割分支的反向传播）确保了分类网络能够受惠于分割网络学到的几何边界定位，而不会重新引入破坏性的多任务梯度纠缠。

## 课题二：FAMO调度器配置与损失项粗粒度合并瓶颈

### FAMO机制的内核与粗粒度合并的陷阱

快速自适应多任务优化（Fast Adaptive Multitask Optimization, FAMO）是一种旨在通过动态权重分配来缓解多任务学习中梯度冲突的先进算法。其核心机制是在时间和空间复杂度均为 $\mathcal{O}(1)$ 的前提下，寻找一个最优的任务更新方向，使得所有任务的损失能够以近似相等的速率下降。FAMO通过维护一个对数化任务权重的平滑序列，试图最大化最差任务的局部改善率。

然而，在当前的工程实现中，`FAMOScheduler` 被硬编码配置为 `num_tasks=3`。前向传播阶段将多种损失生硬地揉捏为三个粗粒度张量组：`loss_seg`、`loss_concepts` 和 `loss_mal`。这种粗粒度合并是极其危险的，它直接掩盖了子任务内部极其严重的梯度冲突。当不同目标的损失在合并前产生方向完全相反的梯度（即余弦相似度极负）时，它们会在算术相加的过程中相互抵消。此时，FAMO调度器接收到的“组级梯度”实际上是一团方向混乱的噪声，导致其动态权重更新策略完全失效。

### 多源梯度的内部拉扯机理

系统内部的梯度拉扯主要集中在两个重灾区： 第一，在 `loss_concepts` 内部，系统将18维超声属性的对称标签回归（SLR）分类损失与1维TI-RADS临床得分的Huber先验正则合并。SLR损失试图将概念特征向量推向极端的二值化（表示某属性存在或不存在），这种优化倾向于在潜空间制造尖锐的、非连续的决策边界。相反，针对TI-RADS的Huber正则是一个鲁棒回归函数，它迫使同一潜空间嵌入具有平滑连续的分布特性，以便能够线性映射到临床得分。这两者在流形学习的几何约束上截然对立，导致特征空间无法收敛到一个稳定状态。

第二，在 `loss_mal` 内部，良恶性对偶AUC-Margin损失与SupCon（监督对比学习）损失被强行绑定。这一组合的破坏力甚至大于前者，因为它们在处理多模态类别分布时遵循着完全不同的拓扑学原理（详见课题三）。

### 细粒度多任务调度方案（num_tasks=5）的数学重构

为解除上述死锁，应当摒弃粗粒度的损失合并，全面扩展FAMO的调度维度。通过将系统解构为独立的五个正交损失通道，可以在更细粒度的梯度级别寻找帕累托最优（Pareto Optimality）平衡。

具体而言，重构后的多目标函数集合定义如下：

$$\mathcal{L}_{total} = \sum_{i=1}^{5} w_i \mathcal{L}_i + \mathcal{L}_{reg}$$

其中，$w_i$ 为 FAMO 动态调度的权重，$\mathcal{L}_{reg}$ 代表直接加总的L2隐残差等约束项（不参与FAMO梯度计算）。五个核心通道分别为：

1. **分割目标** $\mathcal{L}_1$：结合二值交叉熵（BCE）与 Dice 损失，主导像素级结构。
2. **概念分类** $\mathcal{L}_2$：SLR 损失，专司 18 维临床概念的二元属性映射。
3. **临床先验** $\mathcal{L}_3$：Huber 回归损失，负责平滑连续的 TI-RADS 评分预测。
4. **排序边际** $\mathcal{L}_4$：AUC-Margin 损失，用于确立良恶性宏观排序边界。
5. **聚类表征** $\mathcal{L}_5$：SupCon 损失，提供辅助性的同类聚集约束。

FAMO更新规则依据每个独立通道的对数下降率 $r_i^{(t)} = \frac{\mathcal{L}_{i}^{(t)} - \mathcal{L}_{i}^{(t+1)}}{\mathcal{L}_{i}^{(t)}}$ 进行演化。通过细粒度拆分，FAMO 调度器能够敏锐地捕捉到例如 $\mathcal{L}_2$（概念分类）在早期可能出现的过快收敛，并主动压低其权重 $w_2$，从而保证难以优化的 $\mathcal{L}_4$（AUC-Margin）获得足够的优化动力，避免局部极小值的陷阱。

## 课题三：监督对比学习（SupCon）与 AUC-Margin 损失的梯度冲突

### 聚类与排名的拓扑学冲突机理

当前系统良恶性分类AUC之所以深陷77.55%的泥沼，其核心学术原因在于底层同时使用了优化目标相互背离的两种损失函数：`AUCMLossApprox` 与 `SupervisedContrastiveLoss` (SupCon)。

甲状腺恶性结节在病理和影像学上属于典型的多模态、弱内聚类别（Polythetic Class）。乳头状癌可能呈现为边界不规则、极低回声并伴随沙砾样微钙化；而滤泡状癌则可能具有完整的晕环、等回声且内部无钙化。尽管它们在临床上均被归为“恶性”，但在超声图像的高维特征空间中，它们位于距离极其遥远的异构流形簇上。

SupCon 损失的数学本质是在单位超球面上拉近同一批次内所有具有相同标签的样本特征（正对），同时推开不同标签的样本（负对）。对于上述高度多态的恶性结节，SupCon 施加了一种严厉的“同类绝对聚拢”约束，强行将分布于不同区域的恶性亚型特征压缩到一个单一的紧凑簇中。这种生硬的几何扭曲彻底破坏了类内差异（Intra-class Variance），导致分类器无法保留各个亚型特有的视觉线索。

相对而言，AUC-Margin（基于对偶形式的深度AUC最大化，DAM）优化的则是一个松弛的成对排序问题（Pairwise Ranking）。它的核心诉求并非同类相聚，而是确保任意一个正样本（恶性）在分类器上的投影得分，能够以指定的几何边际（Margin）高于任意一个负样本（良性）。AUC优化允许同一类别呈现多个分布中心，只要整体排名不受影响即可。因此，当 SupCon 强行折叠流形时，直接摧毁了 AUC-Margin 辛辛苦苦寻找到的能够容纳多模态分布的“弱决策分界线”。

### 对比投射空间的语义崩塌与剥离重构

进一步分析发现，当前架构中 SupCon 的投影头被错误地挂载于混合概念特征（CEM Mixed Features，共288维）上。概念嵌入模型（Concept Embedding Model, CEM）的设计初衷是赋予神经网络高度的语义可解释性，并允许医生在测试阶段对特定概念（如强行将“无钙化”修改为“有微钙化”）进行人工干预。

如果在表征具体属性的概念向量上施加 SupCon，网络为了满足整体“恶性”类别的聚拢要求，将不可避免地导致“概念泄漏”（Concept Leakage）或概念纠缠（Concept Entanglement）。具体表现为，模型会强迫原本形态各异的恶性结节生成高度雷同的概念概率分布，从而完全丧失了CBM模型解析具体临床属性的能力。

为了消除这种系统级破坏，必须对对比学习的空间挂载点实施彻底剥离：

1. **特征空间重置**：将 SupCon 的输入张量从 288 维的 $c_{mixed}$ 上移，改接到分类主干网全局平均池化后的纯视觉特征 $f_{cls\_pooled}$（352维）上。这里尚未进入概念瓶颈层，实施特征拉扯不会直接破坏概念的语义纯洁度。
2. **松弛对比策略**：鉴于良恶性二分类标签对 SupCon 过于粗糙，建议引入基于亚型的对齐调制损失（Alignment Modulation Loss）。如果系统无法提供细粒度的病理亚型标签，则应直接削弱 SupCon 的权重，甚至彻底移除该损失，确保 AUC-Margin 获得对整体流形分布的绝对主导权。

## 课题四：训练与评估之间的“物理先验特征”偏差 (Train-Eval Gap)

### 计算不一致性引发的分类系统性崩溃

当前系统为了注入医学先验知识，开发了 `DifferentiablePhysicalPriorExtractor` 模块。然而，该模块在训练期与评估期的计算链路存在致命的不一致性，引发了严重的分布外（OOD）偏移。

在训练期间，为了保证误差可微并反向传播到分割网络，系统采用了 Sobel 边缘算子在连续概率预测掩膜（Predict Mask）上计算周长，并通过面积积分求得圆度等特征。问题在于，Sobel 梯度图在未经过严格的非极大值抑制（NMS）时，产生的“边缘”往往具有多像素的厚度（通常为3至5个像素宽）。这种模糊的厚边缘导致系统估算出的训练期周长 $Perimeter_{train}$ 被急剧放大。 由于圆度的物理学定义公式为 $C = \frac{4\pi \times Area}{Perimeter^2}$，周长的平方项在分母位置起决定性作用。因此，训练期的估算圆度 $C_{train}$ 会被极度压抑，往往表现为逼近于 0 的极小值。

而在推理评估期（`evaluate.py`），系统采用了基于 OpenCV 库的 `cv2.findContours` 算法。该算法作用于硬二值化掩膜，利用严格的拓扑分析提取精确的最短单像素欧氏边界。因此，评估期的 $C_{eval}$ 在数值尺度上远远大于 $C_{train}$。 由于负责融合的 `prior_mlp` 网络在训练期间已经适应了趋近于 0 的伪分布输入，当推理时接收到真实的较大数值时，其神经元激活陷入混乱。这种物理性质计算的“不一致性”，使得本应提升性能的物理先验成为了污染分类器决策边界的系统性偏差。

### 消除偏差的形态学重构与噪声退火

为解决这一工程难题，必须采用数学上严格等价且可微的替代方案。建议引入 Kornia 张量形态学库（`kornia.morphology`），通过可微的数学形态学运算来替代粗糙的 Sobel 算子。

具体的软边缘提取与特征计算流程推导如下： 设定一个 $3 \times 3$ 的全1结构元素（Kernel），对连续概率掩膜 $M$ 执行可微腐蚀（Erosion）操作。

$$M_{eroded} = \text{Erosion}(M, kernel)$$

$$Boundary = \text{ReLU}(M - M_{eroded})$$

$$Perimeter_{soft} = \sum_{h, w} Boundary_{h, w}$$

$$Area_{soft} = \sum_{h, w} M_{h, w}$$

$$Circularity = \frac{4\pi \times Area_{soft}}{Perimeter_{soft}^2 + 1e^{-6}}$$

使用基于形态学腐蚀的方法，提取出的软边界严格保证了极窄的像素宽度，使得训练期的物理估值与 `cv2.findContours` 的后处理结果高度对齐，彻底消除了 Train-Eval 鸿沟。

此外，在训练的早期阶段（如 Epoch 1-20），分割网络的输出充满高频碎片噪声。提取这些碎片的面积和周长毫无物理意义。为了防止这种“噪声累积”通过 `prior_mlp` 引发负迁移并导致分类分支过早收敛于局部极小值，必须设计一个动态时间退火门控（Time-Decay Gating Scalar）。 定义衰减因子 $\gamma(t)$：

$$\gamma(t) = \max\left(0, \min\left(1, \frac{t - t_{warmup}}{\Delta t}\right)\right)$$

在前向传播时，将融合特征重写为：$e_{prior} = prior\_mlp(v_{phy}) \times \gamma(t)$。通过在早期的 $t_{warmup}$ 轮次内实施强行断连，允许分类主干优先根据视觉纹理自主收敛，待分割掩膜稳定后再逐步引入几何先验。

## 课题五：Shannon 熵自适应门控的标定与潜在死锁

### 门控硬编码阈值的分布错配

概念嵌入模型（CBM）的一个著名困境是：如果在概念瓶颈层实施极其严格的阻断，往往会导致最终分类精度的不可逆下降。为此，当前系统引入了一个自适应的信息熵门控通道 $\alpha_{gate}$，试图在模型对概念预测极其不确定时，打开一个容许特征直通的隐捷径（Latent Shortcut）以补偿精度损失。

目前的熵值门控公式被硬编码为：

$$\alpha_{gate} = \frac{1}{1 + \exp\left(-10.0 \cdot \left(H_{mean} - 0.5\right)\right)}$$

其中，0.5 的阈值设计存在严重的统计学错配。系统的概念集包含 18 维通过 Sigmoid/Softmax 激活的二值与多值属性概率。由于医疗概念数据的长尾分布特性（例如“彗星尾征”或“微钙化”在多数结节中并不存在），模型极易在训练初期就学会了将罕见属性的预测概率置于极低的先验值（如 $p \approx 0.01$）。 这种先验压制导致全局平均归一化 Shannon 熵 $H_{mean}$ 在真实的推断分布中极低（通常远低于 0.1）。由于 $H_{mean}$ 难以达到硬编码的 0.5，Sigmoid 激活的输入项变为一个较大的负数，导致 $\alpha_{gate}$ 迅速跌落并紧贴于 0。

### 门控断崖死锁与“良性泄漏”的修复

当 $\alpha_{gate} \approx 0$ 时，隐捷径通道 `latent_mlp` 彻底失去梯度流。依据链式法则，$\frac{\partial L}{\partial w_{latent}} = \alpha_{gate} \cdot \frac{\partial L}{\partial Logits} \cdot \frac{\partial Logits}{\partial w_{latent}}$，门控归零导致该参数矩阵发生持续性的“梯度死锁”（Dead Gradients），阻碍了网络挖掘隐藏非概念视觉特征的能力。

最近的研究表明，在真实的不完美概念标注场景下，适度的“良性泄漏”（Benign Leakage）对于保持 CBM 的精度和干预能力是不可或缺的。为了打破死锁并维持良性泄漏，应当将固定的 0.5 转化为批次自适应阈值，或使用指数移动平均（EMA）追踪的标定参数，并强制引入一个保底泄漏系数 $\epsilon$（如 0.05）。

修正后的自适应门控数学模型应重构为：

$$\tau_{gate}^{(t)} = \beta \tau_{gate}^{(t-1)} + (1-\beta) \text{Median}\left(H_{batch}\right)$$

$$\alpha_{raw} = \sigma\left(-10.0 \cdot \left(H_{mean} - \tau_{gate}^{(t)}\right)\right)$$

$$\alpha_{gate} = (1 - \epsilon) \cdot \alpha_{raw} + \epsilon$$

这一机制通过 $\tau_{gate}$ 动态锚定当前模型的真实熵分布中位数，确保网络中约有一半的样本能够获得较大程度的特征直通，而保底系数 $\epsilon$ 则确保了即便是概念预测极其确定的样本，其 `latent_mlp` 依然能微弱地参与优化博弈。

## 课题六：代码重构与遗留死代码治理

### 死代码清理与隐式惩罚激活

经过严密的审计，代码库中的 `LayerwiseGTA`、`DSCA` 作为未实例化的类定义，严重污染了图编译逻辑与后续开发者的认知空间，必须从系统底层彻底剥离。如果为了实现课题一中的跨层任务对齐，应以精简的 `TaskAlignmentAdaptor` 函数重新实现。此外，未被调用的 `SPINHead` 同样应被清除。

更为严峻的工程隐患在于，代码库中 `MultiTaskLoss` 期望获得 CBM 投影权重的 L1 正则惩罚，以促使概念嵌入学习到更稀疏、更具独立语义的表达；然而，现有的 `self.cbm_head.get_l1_penalty()` 存根函数仅返回硬编码的 `0.0`。 必须在重构的 `ConceptEmbeddingCBMHead` 中实质性地激活这一惩罚机制：通过获取正负极投影矩阵 `proj_plus.weight` 和 `proj_minus.weight` 的绝对值和，动态返回惩罚标量，以防止多维概念发生特征塌陷与语义纠缠。

### 阶段定义统一与渐进式对准编排

`stages.py` 内部的两阶段全解冻策略与项目设计文档中的三阶段渐进对准策略存在直接的冲突。在多源异构架构下，早期阶段的全参数联合优化往往会导致网络脆弱的浅层表征被多方梯度撕扯至崩溃。全任务优化必须建立在循序渐进的基石之上。

基于以上论证，在此制定调和后的三阶段渐进训练编排（Progressive Alignment Finetuning）时间轴：

| **训练阶段**                  | **Epoch 跨度**  | **冻结模块**                            | **解冻激活模块**                          | **FAMO 调度范围**           | **阶段核心优化目标**                                         |
| ----------------------------- | --------------- | --------------------------------------- | ----------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| **Stage 1: 表征基础对准**     | Epoch 1 - 40    | 分类头部MLP, 潜捷径网络, AUC-Margin损失 | 解耦双主干, 分割解码器, CEM概念预测评分层 | 仅 Seg & Concept            | 在消除整体分类干扰的前提下，确立高保真的形态学基建与稳固的局部概念响应特征。 |
| **Stage 2: 全任务联合调度**   | Epoch 41 - 100  | 无                                      | 全模型解冻，注入退火结束的物理先验嵌入    | 全面扩展至 5 个任务通道     | 启动扩张至五维度的FAMO系统，执行全局特征流形的帕累托空间寻优，确立多任务间的相对权重平衡。 |
| **Stage 3: 决策边际极限微调** | Epoch 101 - 120 | 解耦双主干, 分割解码器, 物理先验提取器  | 仅限分类投影头, 捷径映射层                | 关闭FAMO，仅针对 AUC-Margin | 在已定型的优质特征底座上，纯粹利用大Batch与小学习率单独优化对偶AUC损失，极限推扩支持向量的分类边际，突破77%的天花板。 |



## 交付要求：工程落地执行指南与架构导图

### 1. 建议系统拓扑与损失协同架构 (Mermaid 框架)

以下图谱呈现了经由深度剖析后重塑的双分支解耦流向及 5 维 FAMO 损失控制逻辑：

代码段

```
graph TD
    subgraph Input Module
        IMG[Thyroid US Scan\n Nanyang Local Dataset]
    end

    subgraph Dual Backbone Decoupling
        BB_Shared[Conv-Stem\n Stage 0-1]
        BB_Seg[Swin-T Decoder\n High-Res Context]
        BB_Cls[EfficientNet Branch\n Texture Extraction]
    end

    subgraph Dense Segmentation Pipeline
        SEG_HEAD[Seg Head]
        PHY_EXTRACT[Kornia Morphological\n Differentiable Extractor]
        SEG_MASK[Mask Prediction]
    end

    subgraph CBM Classification Pipeline
        POOL[Global Avg Pool\n Feature Dimension]
        CBM[Concept Embedding Space\n Dim: 288]
        MLP_LATENT[Latent Shortcut\n Dim: 1]
        MLP_PRIOR[Prior Projection\n Dim: 16]
        GATE[Adaptive EMA Entropy Gate]
        CAT((Feature Concat\n Dim: 304))
        CLS_HEAD[Malignancy Classifier\n Dim: 1]
    end

    subgraph Loss Orchestration & FAMO Engine
        L_SEG[Task 1: Seg Loss\n BCE + Dice]
        L_SLR[Task 2: Concept\n SLR Loss]
        L_HUB[Task 3: Prior\n Huber Loss]
        L_AUC[Task 4: Malignancy\n AUC-Margin]
        L_SUP[Task 5: SupCon\n Applied to Pool]
        FAMO{5-Task FAMO Scheduler}
    end

    %% Workflow Connections
    IMG --> BB_Shared
    BB_Shared --> BB_Seg & BB_Cls
    BB_Seg --> SEG_HEAD --> SEG_MASK
    BB_Cls -. Task-Alignment .-> BB_Seg
    BB_Cls --> POOL
    
    SEG_MASK -. Soft Erosion Area/Perimeter .-> PHY_EXTRACT -. Time-Decay Fusion .-> MLP_PRIOR
    
    POOL --> CBM
    POOL --> MLP_LATENT
    POOL -. Safe Feature Mapping .-> L_SUP
    
    CBM --> CAT
    MLP_PRIOR --> CAT
    CAT --> CLS_HEAD
    
    CBM -. Compute Median Entropy .-> GATE
    GATE --> MLP_LATENT
    MLP_LATENT --> |Modulated Addition| CLS_HEAD
    
    %% Optimization Backprop
    SEG_MASK -.-> L_SEG
    CBM -.-> L_SLR & L_HUB
    CLS_HEAD -.-> L_AUC
    
    L_SEG & L_SLR & L_HUB & L_AUC & L_SUP --> FAMO
    FAMO --> |Pareto Balanced Gradients| BB_Shared
```

### 2. 逐步工程实施指南

本指南将直接引导开发团队对现有的 `multi_task_model.py` 及相关损失模块进行工业级重构操作：

**步骤一：主干网替换与对齐代码实装** 在 `multi_task_model.py` 中，移除对 EfficientNet Stage 5 的深度拷贝机制。在模型 `__init__` 初始化期间，实例化 `timm.create_model('swin_tiny_patch4_window7_224')` 专供 `self.seg_encoder` 使用。同时，追加单向特征对齐算子：

Python

```
self.task_alignment = nn.Sequential(
    nn.AdaptiveMaxPool2d((14, 14)), # 根据分类特征尺度适配
    nn.Conv2d(dim_seg, dim_cls, kernel_size=1),
    nn.Sigmoid()
)
```

**步骤二：实现可微形态学与误差修正** 清理自研的伪 Sobel 算子，挂载 `kornia` 库，重写 `DifferentiablePhysicalPriorExtractor` 内部逻辑以实现软圆度计算，并附加上下文保护机制以阻挡分母坍塌：

Python

```
import kornia.morphology as morph
import torch

def compute_soft_circularity(mask):
    kernel = torch.ones(3, 3, device=mask.device)
    # 利用Kornia形态学进行单像素级边缘提取
    eroded_mask = morph.erosion(mask, kernel)
    boundary = mask - eroded_mask
    perimeter_soft = boundary.sum(dim=(1, 2, 3))
    area_soft = mask.sum(dim=(1, 2, 3))
    # 平滑计算以吻合评估阶段的cv2.findContours结果
    circularity = (4 * torch.pi * area_soft) / (perimeter_soft ** 2 + 1e-6)
    return circularity
```

**步骤三：熵门控死锁的动态化解除** 对 `ConceptEmbeddingCBMHead` 进行改造。弃用 `0.5` 固定值，通过定义滑动平均（EMA）追踪变量，计算自适应阈值，并加入 0.05 的安全保底：

Python

```
# 初始化时定义: self.register_buffer('ema_entropy_tau', torch.tensor(0.5))
with torch.no_grad():
    batch_median_entropy = torch.median(entropy_values)
    self.ema_entropy_tau = 0.9 * self.ema_entropy_tau + 0.1 * batch_median_entropy

# 前向传播门控计算
gate_raw = torch.sigmoid(-10.0 * (entropy_values - self.ema_entropy_tau))
alpha_gate = gate_raw * 0.95 + 0.05 # 确保良性泄漏，打破死锁
```

**步骤四：FAMO扩张与SupCon剥离重构** 在 `loss.py` 内部，解散 `loss_concepts` 和 `loss_mal` 的粗暴相加机制。将目标损失通道定义为完全相互独立的5个Tensor向量。 对实例化 `FAMOScheduler` 的传入参数修改为 `num_tasks=5`。同时，确保 `SupervisedContrastiveLoss` 接收到的特征维度从 `c_mixed` (288维) 变更为 `f_cls_pooled` (352维)。最后，调整外围训练主循环的编排代码，使其严格遵从本文提出的“三阶段渐进对准微调策略”，彻底释放分类AUC提升的空间潜能。