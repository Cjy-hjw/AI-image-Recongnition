# 极端不平衡四类有序标签下的加权 Kappa 优化与 ETE 预测策略研究报告

## 1. 架构特征变动与损失机制解耦分析 (Q0)

在上一轮 35 epoch 的训练实验中，甲状腺外侵犯（Extrathyroidal Extension, ETE）预测头的输入特征存在严重的局部通路缺失。由于分割主干被冻结，分割预测图呈现恒定高斯平滑状态（`val_dice` 恒为 0.0007，预测概率处处约为 0.4425），导致通过膨胀与腐蚀算法提取的包膜-结节交界带（`local` 特征向量）与交界带面积占比（`band_ratio`）持续退化为绝对常量零。ETE 头实际上仅依赖 EfficientNet-B2 的全局池化特征（`global`）与 CBM 边缘概念（`margin`）进行决策。

针对下一轮训练中分割主干改为全程自训练、`local` 通路首次引入真实解剖学结构特征的变动，系统性评估表明：**绝不能采用“仅解除分割冻结、保持现有 CB-Focal 损失重跑”的单变量递进策略，损失函数与输入特征解耦机制必须在下一轮训练中同时重构。**

理由在于，上一轮 ETE 加权 Kappa 恒为 0 且后续转负的根本原因并非单一维度的特征缺失，而是损失函数的设计存在结构性缺陷。在极度不平衡（最大类与最小类比例达 781:1）且以加权 Kappa 为硬预测评测指标的优化目标下，Class-Balanced Focal Loss 产生了严重的决策边界偏移。即使 `local` 特征在下一轮提供了高鉴别力的交界带信息，若损失函数继续对极稀有类（如样本数仅为 5 的血管侵犯类）施加高达 20.4 倍的类权重放大与 Focal 动态压制，模型依然会过度预测稀有类，导致在硬预测（$\text{argmax}$）下触发加权 Kappa 距离惩罚机制，使指标不可逆地陷入负值区间。

为了在单轮训练中同时改善输入质量与损失驱动方向，并精准区分两者的贡献，必须在模型结构与监控维度上建立解耦防护机制：

1. **输入特征维度的门控残差融合机制**：在 ETE 头的特征融合层引入自适应门控网络（Gated Feature Fusion）或带 DropPath 的残差连接。在训练初期（分割主干尚未收敛、`local` 特征质量较低时），门控机制能够自动降低 `local` 特征的权重，防止噪声干扰 `global` 与 `margin` 特征；随着分割 Dice 提升，门控自动放行交界带特征，实现解剖学先验的平滑注入。
2. **梯度与特征贡献度的在线归因监控**：通过在训练日志中记录 `global`、`local` 和 `margin` 三路特征在 MLP 第一层的梯度范数（Gradient Norm）以及自适应门控权重的演化曲线，可以直接观测到 `local` 特征从“零信息”向“高鉴别力特征”转变的过程，从而在无需多轮对比实验的前提下，从物理机制上完成特征改善与损失优化贡献的归因分离。

## 2. 极度不平衡有序分类的损失函数机制分析与选型 (Q1)

### 2.1 Class-Balanced Focal Loss 导致 Kappa 转负的数学机理

加权 Kappa 指标（Quadratic Weighted Kappa, QWK）衡量的是真实标签与预测标签之间的距离一致性。设类别总数为 $K=4$，观测混淆矩阵为 $O \in \mathbb{R}^{K \times K}$（其中 $O_{ij}$ 表示真实类别为 $i$ 且预测类别为 $j$ 的样本比例），期望混淆矩阵为 $E = r c^T$（其中 $r, c \in \mathbb{R}^K$ 分别为真实标签与预测标签的边缘概率分布）。二次加权矩阵定义为 $W_{ij} = \frac{(i - j)^2}{(K - 1)^2}$。二次加权 Kappa 的计算公式为：

$$\kappa = 1 - \frac{\sum_{i=0}^{K-1} \sum_{j=0}^{K-1} W_{ij} O_{ij}}{\sum_{i=0}^{K-1} \sum_{j=0}^{K-1} W_{ij} E_{ij}}$$

上一轮训练采用的 Class-Balanced Focal Loss（$\beta=0.99, \gamma=2.0$）导致 $\kappa$ 转负（$-0.054 \sim -0.0039$）的数学机理如下：

- **Focal Loss 的梯度压制效应**：$\gamma=2.0$ 使高置信度主导类（类别 0，占比 92.02%）的梯度贡献被压制了约 39 倍。模型主干放弃了对 92% 主导类的边界维护，专注于降低少数类的 Focal 损失。
- **类权重的过度放大与概率转移**：CB 权重为血管侵犯类（类别 3，样本数 5）分配了 20.4 倍的梯度权重。在 Softmax 输出端，模型倾向于提高类别 2 与类别 3 的 Logits 值。
- **硬预测（Argmax）下的巨额二次惩罚**：当模型对真实为类别 0（无侵犯）的样本产生误判时，若将其预测为类别 3（血管侵犯），单样本产生的二次距离惩罚项 $W_{0,3} = \frac{(0 - 3)^2}{(4 - 1)^2} = 1.0$。由于类别 0 的基数巨大（3906 样本），只要模型因过度激励少数类而对类别 0 产生仅 2% 的伪阳性误判（约 78 个样本预测为类别 3），累积的非对角惩罚项即达到 $78 \times 1.0 = 78.0$。相比之下，若模型将所有样本均预测为主导类 0（常数预测），对 5 个类别 3 样本产生的最大惩罚仅为 $5 \times 1.0 = 5.0$。

当模型过度预测稀有类导致的实际错误距离惩罚超过了随机或常数预测的期望惩罚时，$\sum W_{ij} O_{ij} > \sum W_{ij} E_{ij}$，直接导致 $\kappa$ 跌破 0 变成负值。这一实测现象证实：在硬预测加权 Kappa 评测体系下，对 0.1% 量级的极稀有类进行高倍率加权或 Focal 压制，会导致模型发生灾难性的类别过度预测。

### 2.2 候选损失函数评估与推荐排序

针对四类极度不平衡有序分类任务，结合单轮训练且评测指标为加权 Kappa 的严格约束，对六种候选损失函数方案进行多维度评估与对比：

| **候选方案**                                                 | **优化目标与 Kappa 对齐度**          | **极稀有类 (n=5) 稳健性**                      | **超参敏感度 / 零调参可用性**           | **推荐排序** |
| ------------------------------------------------------------ | ------------------------------------ | ---------------------------------------------- | --------------------------------------- | ------------ |
| **(d) 可微加权 Kappa 损失 + Logit 调整 (LA-SoftQWK)**        | 极高（直接优化连续化 Kappa 指标）    | 高（通过边缘分布项自适应约束）                 | 中（$\tau=0.5$ 论文推荐值，零调参可用） | **1**        |
| **(b) 有序距离加权交叉熵 (CDW-CE)**                          | 高（显式注入二次距离惩罚 $W_{ij}$）  | 高（共享边界，防局部过拟合）                   | 高（超参仅 $p=2$，零调参可用）          | **2**        |
| **(e) 连续回归 + 极值分位数切分 (Regression + Quantile)**    | 中高（MSE 与 QWK 距离项同构）        | 中（受极小值拉动影响较敏感）                   | 低（需要在训练内学到阈值或动态分位数）  | **3**        |
| **(f) 动态数据重采样 + 经典交叉熵 (Balanced Sampling + CE)** | 低（交叉熵未利用有序距离信息）       | 低（极稀有类 $n=5$ 过采样极易过拟合）          | 高（采样比例需仔细调整）                | **4**        |
| **(c) CORAL / CORN 累积二分类**                              | 中（考虑有序性，但各二分类器解耦）   | 极低（第 3 个二分类器仅 5 个正例，梯度易崩溃） | 中（需对各二分类子网络设独立权重）      | **5**        |
| **(a) 经典加权交叉熵 / CB-Focal (当前失效方案)**             | 负相关（无视距离且极易引发过度预测） | 极低（实测导致 Kappa 恒 0 后转负）             | 低（对 $\beta, \gamma$ 极度敏感）       | **6**        |



#### 方案推荐理由详解：

- **首选方案（Rank 1）：Logit 调整下的可微加权 Kappa 损失 (LA-SoftQWK)** 可微加权 Kappa 损失 通过 Softmax 输出的概率分布构建连续可微的混淆矩阵 $\hat{O}$ 和期望矩阵 $\hat{E}$。该损失函数直接对齐评测指标 $\kappa$ 的分子与分母。为了防止在 $n=5$ 的极稀有类上产生梯度发散，必须结合 Logit Adjustment (LA) 技术。LA 通过在输入端注入类别先验 $\tau \log \pi_k$（其中 $\pi_k = n_k / N$），将决策边界平移，强制要求模型对类别 3 的置信度显著高于先验时才输出类别 3 的硬预测。这种组合在数学上天然阻止了误判主导类导致的巨额距离惩罚，且 $\tau=0.5 \sim 1.0$ 为通用理论值，无需额外的超参搜索。

- **次选方案（Rank 2）：类距离加权交叉熵损失 (Class Distance Weighted CE, CDW-CE)** CDW-CE 损失 在标准交叉熵的基础上，根据目标类别 $y_i$ 与预测分布 $p_{i,k}$ 之间的距离对交叉熵进行动态惩罚：

  $$\mathcal{L}_{\text{CDW-CE}} = -\sum_{k=0}^{K-1} y_{i,k} \log(p_{i,k}) + \lambda \sum_{k=0}^{K-1} (k - y_i)^2 p_{i,k}$$

  该损失函数保留了交叉熵的平滑凸优化特性，同时引入了二次距离惩罚，促使错误预测的概率分布向真实标签的邻域收敛，而不是跨越式地预测到极稀有类。该方案参数极少（$\lambda=1.0, p=2$），在极端不平衡下的训练稳定性极高。

### 2.3 评估指标（Linear vs Quadratic Kappa）的稳健性分析

官方规则指明指标为“加权 Kappa”，未说明是线性加权（Linear）还是二次加权（Quadratic）。

- **权重矩阵差异**：线性加权权重为 $W^{\text{lin}}_{ij} = \frac{\vert{}i - j\vert{}}{K - 1}$；二次加权权重为 $W^{\text{quad}}_{ij} = \frac{(i - j)^2}{(K - 1)^2}$。对于 4 类别任务，邻级误判（如 0 判 1）两者惩罚比例相似（$1/3$ vs $1/9$），但跨级误判（如 0 判 3）二次加权施加了 9 倍的惩罚，而线性加权仅施加 3 倍惩罚。
- **稳健性推导**：基于二次距离 $W^{\text{quad}}$ 设计的损失函数（如 LA-SoftQWK 或 CDW-CE）对跨级大误差施加了更严格的梯度约束。在凸包优化理论中，二次连续凸上界严格控制了线性误差的上界。因此，针对 Quadratic 优化出的模型在 Linear 口径下依然能保持高度对齐与高分；反之，若针对 Linear 优化，模型可能会容忍适度的跨级误判，导致在 Quadratic 口径下被严重扣分。 鉴于单轮提交的赌注性质，方案必须坚决基于 **Quadratic 假设** 进行损失函数构建。

### 2.4 $n=5$ 极稀有类别的防过拟合与类不平衡校正

在血管侵犯类仅有 5 个样本的极端条件下，传统的类不平衡校正（如 CB 权重或高倍率动态加权）与有序损失之间存在严重的对抗作用：

- **对抗机制**：类不平衡校正倾向于拉高少数类的概率，而有序损失（加权 Kappa）在多数类基数巨大时强烈惩罚少数类的伪阳性预测。当两者叠加时，CB 权重的梯度拉力强行破坏了 Kappa 距离控制，导致训练崩溃。
- **统一解法：有序软标签 Smoothing 与 Logit 偏移** 应当放弃传统的硬类重加权（CB 权重），转而采用**有序与类平衡统一校正策略**：
  1. **Logit 偏移校正 (Logit Adjustment)**：利用边缘分布先验在推理 Logit 上施加负偏移量 $\Delta_k = \tau \log(n_k / N)$，仅调整决策边界，不改变梯度更新的相对距离标度。
  2. **高斯有序软标签 (Gaussian Ordinal Label Smoothing)**：将类别 3（血管侵犯）的 one-hot 编码 $[0, 0, 0, 1]$ 替换为有序高斯分布软标签 $[0.0, 0.01, 0.14, 0.85]$。对于仅有的 5 个样本，软标签不仅为相邻类别（类别 2）提供了合理的概率梯度过渡，而且防止了 Softmax 极值优化导致的过拟合与特征空间过拟合坍塌。

## 3. 标签结构重构与训练期并类代价推导 (Q2)

虽然官方最终推理输出必须为 4 类之一（0-based 索引 0, 1, 2, 3），但在训练期是否可以将样本极少的类别 2（软组织侵犯，67 例）与类别 3（血管侵犯，5 例）合并为“高阶侵犯类”（合并后共 72 例），推理时再通过次级二分类器或规则拆分？针对这一设想，在加权 Kappa 口径下进行精确的距离惩罚代价推导：

### 3.1 预估损失与收益计算

假设评测集共有 $N=4245$ 个样本，类别分布与全数据集一致。 混淆矩阵中，若将 5 个真实为类别 3 的样本误判为类别 2，在 Quadratic 权重下导致的单样本惩罚为 $W_{3,2} = \frac{(3 - 2)^2}{9} = \frac{1}{9}$。5 个样本全错的总惩罚为：

$$\text{Error}_{\text{mis\_3\_to\_2}} = 5 \times \frac{1}{9} = 0.555$$

若为了强行区分类别 2 与类别 3，模型在训练时产生了严重的伪阳性，导致主导类 0（3906 例）中仅有 0.5%（约 20 个样本）被过度预测为类别 3，则产生的单样本惩罚为 $W_{0,3} = \frac{(0 - 3)^2}{9} = 1.0$。20 个样本的总惩罚为：

$$\text{Error}_{\text{false\_positive\_3}} = 20 \times 1.0 = 20.0$$

上述推导表明：**类别 3 对整体 Kappa 值的最大正向贡献上限极低（全对仅消除 0.555 个惩罚单位），但其引发过度预测所带来的负向风险上限极大（少许误判即产生数十个惩罚单位）。**

### 3.2 并类方案的风险与最终结论

尽管从数学上暂存类别 3 似乎能避免过拟合，但**在训练期显式合并类别 2 与 3 是不可行的**，原因在于：

1. **解耦拆分难题**：由于只有一次提交机会，若训练期仅输出 3 个类别的 Logits，推理期缺乏经过端到端联合优化的第 4 类 Logit 表达，任何后处理规则拆分（如设置固定 Sigmoid 阈值）都极易在硬预测 $\text{argmax}$ 时产生系统性偏置。
2. **正确解法：隐式连续表征而非硬合并** 最佳策略是不在结构上硬合并类别，而是采用**连续分布正则化（Soft Target Scaling）**。在损失函数中，将类别 3 的真实目标设定为包含类别 2 概率的软分布，使模型在分类器最后一层的权重矩阵中，类别 2 与类别 3 的方向向量保持高度关联。这既保留了 4 类的原生 Logits 输出架构（满足官方接口契约），又在梯度层面达到了与“合并训练”等价的稳健收敛效果。

## 4. 评测子集约束下的样本选择策略与域偏移防护 (Q3)

官方评测规则明确限定：加权 Kappa **仅在匹配成功、恶性且为主病灶（`is_main == 1` 且 `gt_malignancy == 1` 且 `valid_extrathy_ext == 1`）的子集上计算**。基于全数据集 4245 行 ETE 非空样本、主病灶 4877 行及恶性率 53% 的统计数据，由于许多次病灶也带有 ETE 标签，且主病灶中包含良性结节，评测子集真实的样本规模 $N_{\text{eval}}$ 预估在 1000 至 1500 之间。

针对不同评估子集规模的可能性，训练样本选择策略与防过拟合机制的分析如下表所示：

| **评估子集可能规模**                                      | **数据利用策略**                                 | **领域偏移 (Domain Shift) 风险**                        | **关键防过拟合与稳健性手段**                                 |
| --------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------- | ------------------------------------------------------------ |
| **规模 A ($N \approx 200 \sim 500$)**                     | 必须使用**全量 4245 行样本**进行训练             | 高（良性/次病灶的图像特征可能与恶性主病灶存在微小差异） | 1. 在 ETE 头的 MLP 拼接输入中加入 `is_main` 与 `gt_malignancy` 辅助特征； 2. 应用强退化 Dropout ($p=0.4$) 与特征层 L2 正则化。 |
| **规模 B ($N \approx 1000 \sim 1500$)（高概率真实区间）** | 建议使用**全量 4245 行样本**进行训练             | 中（恶性主病灶占全量数据的 30%~40%，具备分布主导性）    | 1. 引入评价子集掩码加权：训练时全量样本算损失，但评价子集样本施加 1.5 倍 Loss 权重； 2. 结合 Logit Adjustment 进行全局概率平移。 |
| **规模 C ($N \approx 2500$)**                             | 使用**全量 4245 行样本**训练，或仅用恶性样本训练 | 极低（评测集已覆盖大部分数据集）                        | 1. 依靠连续加权 Kappa 损失进行全局收敛； 2. 使用标准的 Weight Decay ($10^{-4}$) 即可。 |



### 结论与数据利用方案

**必须使用全量 4245 行有标注样本训练 ETE 头，决不能仅挑选恶性主病灶样本训练。**

理由在于：类别 3（血管侵犯）全表仅有 5 例，若进一步限定在“恶性主病灶”子集内，该类样本数极有可能降至 1~2 例甚至为 0，导致深度神经网络根本无法学习到血管侵犯的解剖学表征。通过使用全量数据，良性与次病灶中的包膜侵犯及软组织侵犯样本为 ETE 头提供了宝贵的形态学边界特征。为消除分布偏移，训练时可以通过将主病灶与恶性状态作为条件特征标量（Conditioning Scalars）与 `global`、`local` 特征一同拼接注入 ETE 头的 MLP 最后一层，使模型显式感知评测环境。

## 5. 训练过程的早期诊断与可观测性监控体系 (Q4)

为防止在 24 小时云端训练中出现中期坍塌（如在第 15 epoch 损失看似下降，但预测在第 25 epoch 突然聚向极稀有类导致 Kappa 爆破转负），必须在 JSONL 日志中增加动态监控指标。基于每 epoch 的验证集输出，必须实时记录并监控以下四项关键信号：

### 1. 验证集硬预测分布直方图 (`val_pred_hist`)

- **记录内容**：验证集经过 $\text{argmax}$ 后，预测为类别 0, 1, 2, 3 的样本绝对数量占比 $[P_0, P_1, P_2, P_3]$。
- **判定阈值与预警信号**：正常收敛的分布中 $P_0$ 应保持在 85% ~ 95% 区间。若出现 $P_3 > 2.0\%$（即预测类别 3 超过 20 例）或 $P_0 < 80\%$，表明模型已发生**少数类过度激励**，Kappa 指标即将剧烈下跌并转负。

### 2. 非对角线大跨度误判率 (`val_extreme_error_rate`)

- **记录内容**：混淆矩阵中 $\vert{}i - j\vert{} \ge 2$ 的样本比例，即把无侵犯（类 0）错判为软组织/血管侵犯（类 2/3），或反之的样本总数。
- **判定阈值与预警信号**：若该比例上升超过 1.5%，无论当前 Loss 是否下降，必须立即判定 ETE 优化方向失败，因为二次距离惩罚 $W_{ij}$ 正在迅速积聚。

### 3. 可微 Soft-Kappa 与硬 Argmax-Kappa 的离散偏差 (`val_kappa_gap`)

- **记录内容**：在验证集上同时计算连续 Soft-Kappa（由概率分布矩阵算得）和硬 Argmax-Kappa。
- **判定阈值与预警信号**：若 Soft-Kappa 持续上升（如达到 0.4），但 Argmax-Kappa 恒为 0，说明 Softmax 输出的概率峰值不够陡峭，无法跨越 $\text{argmax}$ 门槛；若 Soft-Kappa 为正但 Argmax-Kappa 为负，表明概率分布存在严重的长尾不确定性，需要提高 Logit Adjustment 的 $\tau$ 值。

### 4. ETE 预测 Logits 各类均值与范数 (`val_logit_mean_per_class`)

- **记录内容**：记录 4 个类别 Logits 的平均值 $[\bar{z}_0, \bar{z}_1, \bar{z}_2, \bar{z}_3]$。
- **判定阈值与预警信号**：若稀有类的 Logit 均值 $\bar{z}_3$ 超过主导类 $\bar{z}_0$（即 $\bar{z}_3 - \bar{z}_0 > 0$），属于严重预警，表明网络已经对极稀有类产生了过度的无偏偏移。

## 6. 单轮训练风险控制方案与 PyTorch 代码骨架 (Q5)

### 6.1 主方案设计：LA-SoftQWK-OD 复合损失

由于没有中途修改超参的机会，主方案采用**对齐指标的可微加权 Kappa 损失（SoftQWK）为主体，融合 Logit Adjustment 边界平移与有序距离交叉熵正则化（LA-SoftQWK-OD）**。

该损失函数在启动时一次性组装完毕，完全自适应全过程训练，无需在训练中途通过外部 Python 脚本强行修改超参：

1. **先验 Logit 调整**：将原始 Logits 加上静态先验 $\Delta_k = \tau \cdot \log(n_k / N)$（设定 $\tau=0.5$），平滑修正边界。
2. **Soft-QWK Loss 驱动**：连续化计算混淆矩阵，直接最大化 QWK 指标。
3. **样本级距离正则化 (Ordinal Distance Penalty)**：引入连续二次距离正则项，约束不确定的概率分布向真值邻域集中。

### 6.2 代码骨架实现

以下代码可直接替换 `src/models/losses/components/extension.py` 中的实现，完全兼容masked计算范式（`validity` 掩码），并符合 `(total_loss, loss_detail_dict)` 的接口规范。

Python

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitAdjustedSoftQWKLoss(nn.Module):
    """
    针对极端不平衡有序分类的 Logit 调整可微二次加权 Kappa 复合损失函数 (LA-SoftQWK-OD)
    结合了:
    1. Logit Adjustment (Menon et al., 2020) 动态校正决策边界
    2. Soft Quadratic Weighted Kappa Loss (de La Torre et al., 2018) 直接优化评测指标
    3. Distance-Weighted Penalty (Polat et al., 2024) 约束连续概率分布
    """
    def __init__(self, class_counts, tau=0.5, dist_weight=0.2, eps=1e-7):
        super().__init__()
        self.num_classes = len(class_counts)
        self.tau = tau
        self.dist_weight = dist_weight
        self.eps = eps

        # 1. 计算类先验 Logit Adjustment 偏移量
        counts_tensor = torch.tensor(class_counts, dtype=torch.float32)
        priors = counts_tensor / counts_tensor.sum()
        # log_prior 静态存在 buffer 中，无需梯度更新
        self.register_buffer("log_priors", torch.log(priors + self.eps))

        # 2. 预计算 Quadratic 加权矩阵 W (K x K), 归一化到 [0, 1]
        w_mat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                w_mat[i, j] = ((i - j) ** 2) / ((self.num_classes - 1) ** 2)
        self.register_buffer("W", w_mat)

    def forward(self, ete_logits, ete_idx, ete_validity):
        """
        ete_logits: [B, 4] 原始模型输出
        ete_idx: [B] 真实标签 (0, 1, 2, 3)
        ete_validity: [B] 掩码 (1.0 表示有效样本, 0.0 表示忽略)
        """
        # 仅保留有效样本
        valid_mask = (ete_validity > 0.5)
        if valid_mask.sum() == 0:
            zero_loss = (ete_logits * 0.0).sum()
            return zero_loss, {"loss_ete_qwk": 0.0, "loss_ete_dist": 0.0, "loss_ete_total": 0.0}

        logits = ete_logits[valid_mask]       # [N_valid, 4]
        targets = ete_idx[valid_mask].long()   # [N_valid]

        # A. Logit Adjustment 边界平移
        adjusted_logits = logits + self.tau * self.log_priors.unsqueeze(0)
        probs = F.softmax(adjusted_logits, dim=1) # [N_valid, 4]

        # 构建 One-hot 真实分布
        one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).float() # [N_valid, 4]

        # B. 连续混淆矩阵 O 与 期望矩阵 E 的计算
        # O_ij: 真实为 i 预测为 j 的加权概率分布
        N_valid = probs.size(0)
        O = torch.matmul(one_hot_targets.t(), probs) / (N_valid + self.eps) # [4, 4]

        # 边缘概率分布
        hist_true = one_hot_targets.sum(dim=0) / (N_valid + self.eps) # [4]
        hist_pred = probs.sum(dim=0) / (N_valid + self.eps)          # [4]
        E = torch.outer(hist_true, hist_pred)                         # [4, 4]

        # 计算 Soft QWK Score
        num = (self.W * O).sum()
        den = (self.W * E).sum() + self.eps
        soft_qwk_loss = num / den  # 最小化 num/den 等价于最大化 QWK

        # C. 样本级有序二次距离正则化 (Distance Penalty)
        # 针对每个样本计算预测分布到真值索引的二次距离
        class_indices = torch.arange(self.num_classes, device=logits.device, dtype=torch.float32)
        # distance_sq: [N_valid, 4], 包含每个样本预测到所有类别的距离 (k - y_i)^2 / (K-1)^2
        distance_sq = ((class_indices.unsqueeze(0) - targets.unsqueeze(1).float()) ** 2) / ((self.num_classes - 1) ** 2)
        distance_loss = (distance_sq * probs).sum(dim=1).mean()

        # D. 复合总损失
        total_loss = soft_qwk_loss + self.dist_weight * distance_loss

        loss_detail = {
            "loss_ete_qwk": soft_qwk_loss.item(),
            "loss_ete_dist": distance_loss.item(),
            "loss_ete_total": total_loss.item()
        }

        return total_loss, loss_detail


def compute_extension_loss(ete_logits, ete_idx, ete_validity,
                           class_counts=[3906, 267, 67, 5],
                           la_tau=0.5, dist_weight=0.2):
    """
    符合 src/models/losses/components/extension.py 签名的包装函数
    """
    criterion = LogitAdjustedSoftQWKLoss(
        class_counts=class_counts,
        tau=la_tau,
        dist_weight=dist_weight
    ).to(ete_logits.device)

    return criterion(ete_logits, ete_idx, ete_validity)
```

### 6.3 超参推荐值来源与按数据集重算规范

| **参数名**     | **推荐值**           | **来源与作用**                              | **必须按本数据集重算/配置的部分**                            |
| -------------- | -------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| `class_counts` | `[3906, 267, 67, 5]` | 本地全量表实测统计                          | **必须严格按最新全量数据硬编码**，用于计算 `log_priors`。    |
| `la_tau`       | `0.5`                | Menon et al. (2020) 理论最佳区间（0.5~1.0） | 勿设为论文默认的 1.0；在 $n=5$ 的极端情况下，$\tau=0.5$ 可避免过度推挤 Logit。 |
| `dist_weight`  | `0.2`                | Polat et al. (2024) 距离正则化权重          | 无需修改，作为软约束辅助 SoftQWK 在早期平滑收敛。            |
| `focal_gamma`  | `0.0` (已废弃)       | 实测验证                                    | **必须保持 0.0**，彻底禁用 Focal 压制。                      |
| `cb_beta`      | `0.0` (已废弃)       | 实测验证                                    | **必须保持 0.0**，彻底禁用 CB 线性重加权。                   |



### 6.4 本地验证与实测检查清单

在正式提交 24 小时云端训练前，必须在本地使用小批量数据对以下环节进行实测断言验证：

1. **梯度可微性与数值稳定性断言**： 输入全零 Logits 或随机 Logits，断言 `total_loss` 产生的梯度无 `NaN` 或 `Inf`；断言在常数预测下 `soft_qwk_loss` 计算结果为 1.0（对应 QWK = 0）。
2. **Logit Adjustment 边界方向测试**： 人工构造一个类别 0 与类别 3 接近相等的 Logits 输出向量 `[2.0, 0.0, 0.0, 2.1]`。验证在经过 `la_tau=0.5` 的 `log_priors` 调整后，`probs` 中类别 0 的概率重新占据绝对主导。这证明模型只有在对类别 3 具备极高信心时才会在硬预测中输出类别 3。
3. **特征门控融合接口测试**： 输入全零的 `local` 特征向量（模拟前几个 epoch 分割未收敛状态），断言 ETE 头正向传播不报错且梯度能够正常回传至 `global` 通路。