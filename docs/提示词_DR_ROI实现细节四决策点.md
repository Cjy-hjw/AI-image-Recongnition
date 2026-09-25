# DR 提示词:ROI 两阶段范式实现细节四决策点(2026-09-25 v3,E1 review F1 并入)

## 0. 背景与约束(先读,防跑偏)

**任务**:甲状腺超声多任务诊断系统,2026 全国赛赛道七。架构 = 单模型多任务:整图分割(ConvNeXt 主干,512²)+ 分割 mask 实例化(CCL)→ 逐实例裁 ROI → **独立 ROI encoder(轻量 CNN)+ 四个 per-lesion 分类头**(位置 7 类/ETE 侵犯 4 类/临床 T 2 类/TI-RADS 5 类)+ 概念头旁路。推理时 ROI pipeline 从分割结果在线裁 ROI(逐实例串行+chunk=16)。

**硬约束(评审此调研的框架)**:
- 训练数据 = 官方标注:per-lesion 有四头标签(位置/ETE/T/TI-RADS/良恶性)+ 二值 GT mask + **病灶级概念标注(comp/echo/shape/margin,官方 xlsx 列已核实存在,`official_loader._FIELD_ALIASES` 已解析——决策点 1 方案 A 的"ROI 域概念标签"其实可直接取官方列,非只能蒸馏;foci 无官方标注)**。
- 训练侧 ROI batch 形态:独立 RoiDataset,每样本 = 一个病灶的一帧 ROI 图(512²)+ roi_mask + 标签;实测 K(每图病灶数)1-3;**总病灶人口约 5 千级,稀有类 ETE-vessel 仅 5 例**(4459:333:81:5)。
- 头结构已冻结落地(1518 测试绿,冻结日 09-27):本调研**只指导训练配方/损失/标定决策**,除非发现正确性级 bug,不出重构建议。
- 平台:单卡 RTX 5090 32G,20 epoch 预算,基座 warm-start 自已训好的整图模型(L2 ep16)。
- 已有判死结论(勿推翻除非有硬证据):ETE 稀有类加权(CB-Focal β=0.9)是起飞必要条件,LA-SoftQWK 替代已三连对拍证伪。

## 决策点 1(最紧):ROI 概念头的监督方式

**现状**(真实代码,PyTorch):

```python
# src/models/roi_heads.py —— ROI 域轻量概念头(与整图 CBM 同键语义)
class RoiConceptHead(nn.Module):
    _CONCEPT_DIMS = (4, 4, 2, 4, 4)   # comp/echo/shape/margin/foci → 18 维 logits
    def __init__(self, in_channels, hidden_dim=128, dropout=0.3):
        self.proj = nn.Sequential(nn.Linear(in_channels, 128),
                                  nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.head_comp = nn.Linear(128, 4)   # echo/shape/margin/foci 同构五个头

class RoiTiradsHead(nn.Module):
    def forward(self, roi_spatial, concept_logits: dict):
        # 🔴 概念 logits 由调用方 detach —— TI-RADS 梯度不得反向重塑概念
        concept_feat = torch.cat([concept_logits[k].detach() for k in KEYS], 1)  # [K,18]
        tirads_logits = self.classifier(torch.cat([g, concept_feat], 1))        # 5 类
```

**问题**:`RoiConceptHead` 全仓**无损失消费方**——概念头收不到任何监督,等于把随机初始化的 18 维特征拼给 TI-RADS 头。这 18 维现在纯是噪声旁路。

**候选方案**(请逐一裁决可行性+证据):
- **A 辅助监督**:概念头加独立损失。但 ROI 域概念标签不存在——伪标签从哪来?聚类质量可控吗?
- **B 去 detach**:TI-RADS 梯度隐式回流概念头。CBM 文献的 leakage 问题:detach 正是防"概念塌缩成下游捷径",去掉的量化代价?
- **C 蒸馏**:从整图 CBM(ConceptEmbeddingCBMHead,五概念有真监督、已训好)蒸馏 18 维 logits 到 ROI 概念头。teacher(整图 352 通道 stage4 特征)与 student(ROI encoder pooled 特征)**输入分布不同**,跨域蒸馏的温度/损失式?
- **D 投影 warm-start**:整图 CBM 头部 Linear 权重直接搬运(输入维度不匹配,需插投影层)——值不值?
- **E(基线)**:概念头**整个摘掉**,TI-RADS 头只吃 roi_spatial 特征——18 维噪声旁路 vs 无旁路,哪个更害?

**请调研**:CBM 在"概念无标签、只有下游可导梯度"场景的文献/工业实践(概念瓶颈 leakage、concept-free CBM 变体、cross-domain concept distillation);每方案给 leakage/塌缩风险的量化证据;最后**给一个明确排序 + 推荐方案的最小落地 diff**(以我们的真实类名/张量形状写)。

## 决策点 2:λ_roi(ROI 四头总权重)标定与量级

**现状**:

```python
# src/models/losses/components/roi_heads.py
total = lambda_roi * (loc_total + ete_total + t_total + tirads_total)
# 各头内部 masked 归一:(loss*v).sum()/(v.sum()+eps),分母=该头自己的有效病灶数
# 主分割损失(整图链):Lovász 0.3 + BCE/Dice,已是线上 69.64 基线的有效配方
```

单 λ 四头 1:1:1:1,config 占位待 1ep 标定。**背景**:训练循环里 ROI 支路与整图支路**同 epoch 并行**(先整图批,后 ROI 批,梯度累积);整图批 B=16、ROI 批 K 合计同量级。

**请调研**:
- 两阶段范式(Mask R-CNN 及其后继、nnDetection、det-actionable 系列)里 per-ROI 分类损失对主检测/分割损失的**典型量级占比**(RoI head 的 cls loss vs box/mask loss 的文献值);
- **小 K 批 masked 归一的高方差**:单 batch 一条有效病灶时,该头损失退化为单样本 CE——方差极大。工业标准处理?(跨 batch 分母累积/EMA/把归一挪到 optimizer 层/放大 ROI batch 伪批量)
- λ 初始值建议 + 标定协议(我们传统是 1ep 标定使新项占净主项 8-30%,这个区间有无依据?);
- 四头 1:1:1:1 有无文献依据按任务难度/类别数重配(位置 7 类 vs T 2 类同权是否失衡)。

## 决策点 3:小 K × 极端长尾的类平衡

**现状**:ETE 头 focal 调制+class_weights(透传整图版同配置);TI-RADS CE+可选 label_smoothing 0.1+ordinal 权重;**per-lesion 人口 4459:333:81:5(无侵犯:包膜:软组织:血管),训练时每 batch 只见 1-3 个病灶**。

```python
# ROI ETE:CB-Focal β=0.9(整图版三连对拍证实的必要条件,判死勿翻案)
# focal 与 class_weights 同键透传,单一配置源
```

**请调研**:
- 极小 batch(1-3)×长尾(1:900)下 focal×类权重组合的稳定性证据;梯度方差与稀有类 epoch 级出现频率(5 例 ÷ ~5k 病灶 ÷ 20ep ≈ 每 20-100 batch 才见 1 次 vessel)——这种频率下任何加权都救不救得动?文献的 per-ROI 类平衡标准做法(effective number/logit adjustment/loss-side re-weighting vs data-side 过采样/分桶采样);
- ordinal label(ETE 1-4、TI-RADS 1-5)与 label_smoothing 并用的已知相互作用(smoothing 会把 ordinal 相邻类的软目标抹平,方向与 ordinal 惩罚一致还是冲突?);
- **最后裁决**:vessel 类(5 例)在加权 Kappa 评分下,是"救"(过采样+重权)还是"放弃归入软组织"(4 类并 3 类)——按 Kappa 的数学性质给依据,不要只说"try both"。

## 决策点 4(新增,2026-09-25 E1 review 实锤):训练尺子与推理口径的聚合单元不一致

**现状**:**推理 ROI 路是病灶级**(同一帧两实例各自独立 ROI 前向,得 0.9975/0.0025 两个独立置信度),**训练侧验证尺子 `aggregate_by_lesion` 是帧级**(逐帧整图四头记录 → 按 lesion_id 帧均值,同两实例被摊平成同一个 0.5)。已实跑复现。

```python
# src/training/metrics_official.py
def aggregate_by_lesion(records):  # 帧级记录 → 按 lesion_id 均值
# 推理侧:v21_output._run_roi_static/_run_roi_dynamic 逐实例独立前向
# 后果:ROI 训练的本地 val 数字系统性乐观,判线对拍对不上
```

**请调研**:两阶段范式里 per-ROI 分支的**验证指标聚合单元**标准做法(instance-level vs frame/image-level 聚合);多实例同帧场景下按实例独立算指标 vs 先按 lesion 聚合再算指标的取舍依据(排序类指标 AUC/AP 在哪种聚合下信息损失更小);官方口径(帧级 IoU 算术平均 → 病灶级匹配)与训练内尺子对齐的最小改动路径。**要求**:给一个与推理路径同构的最小落地 diff(`collect_roi_records → lesion 聚合 → metrics_official` 两段式)。

## 输出要求(硬格式)

1. 每决策点:**文献/工业来源(可核实的具体出处)+ 方案裁决排序 + 推荐方案 + 风险量化 + 最小落地 diff**(以本提示词贴的真实类名/形状写 PyTorch 片段);
2. 明确二分:"有证据的推荐"(引来源) vs "理论倾向"(标注无实测依据);
3. 超出四决策点的强相关发现(如 per-lesion batch 构造、梯度累积次序)单列"附加发现"节,每条标置信度(高/中/低);
4. **不做**:整体架构重构建议、换范式建议(det-actionable 之外的 YOLO/Transformer 方向)、与我们判死结论冲突且无新证据的翻案。
