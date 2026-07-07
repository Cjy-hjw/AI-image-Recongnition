# Deep Research 提示词：MIL Phase2 解冻策略调研

## 背景与现状

项目：甲状腺超声 figshare 27021604（8500帧，842患者，histo_label患者级良恶性标签）。

已知结论：
- 单帧 CE 预训练：无泄漏 patient_val_auc ≈ 0.50（患者级弱标签天花板，已对拍实验证实）
- MIL Phase1（冻结 EfficientNet-B2 backbone，只训 GatedAttentionPooling + 分类头）：best_patient_val_auc = 0.5653
- MIL Phase2（LLRD 解冻 backbone，backbone_lr_scale=0.1 / 0.01，η=0.75）：不稳定，在 0.46-0.56 剧烈震荡，无法超越 Phase1

## 核心问题

1. **患者级弱标签 MIL 的 fine-tuning 上界是多少？** 在 ImageNet pretrained backbone + 患者级弱标签（非帧级精确标注）场景下，ABMIL/Gated-ABMIL 解冻 backbone 是否通常能超越冻结 backbone？有无文献给出典型 AUC 范围？

2. **LLRD 解冻为何失效？** 在超声/病理弱监督 MIL 任务中，LP-FT（先冻结再解冻）策略是否已知有灾难性遗忘风险？有哪些替代方案（如渐进解冻、LoRA、adapter fine-tuning、对比预训练）在小数据集患者级弱标签上有实证效果？

3. **当前 0.5653 是否已是该数据集的合理天花板？** figshare 27021604（ThyUS2Path）的图像分辨率（下采样到512×512）是否会抹掉甲状腺微钙化等关键判别特征？有无该数据集的公开 benchmark AUC？

4. **在等待官方标注数据（2026-08-01）前，MIL backbone 还能做什么预热？** 是否有无监督/自监督预训练方案（如 SimCLR、MAE、DINO on thyroid ultrasound）能提升 backbone 的超声领域适应性，从而在有真实标签后 fine-tuning 效果更好？

## 期望输出

- 对每个问题：引用关键论文（含年份、数据集规模、AUC 数值）
- 明确回答：在当前条件下（842 patients，患者级弱标签，ImageNet init），Phase2 LLRD 解冻是否值得继续优化，还是应该放弃等官方数据
- 如果建议继续：给出具体超参建议（lr、解冻策略、epoch 数）和成功条件
- 如果建议放弃：给出等官方数据后的最优 fine-tuning 策略建议

## Grounding 上下文（代码片段）

```python
# 当前 MIL 架构
class GatedAttentionPooling(nn.Module):
    # feature_dim=1408 (EfficientNet-B2 pooled), attn_dim=128
    # Ilse et al. 2018 Gated-ABMIL

class ThyroidMILNet(nn.Module):
    # backbone: timm efficientnet_b2, in_chans=1, num_classes=0 → 1408-dim
    # ImageNet safetensors loaded locally
    # Phase1 freeze_backbone=True, Phase2 unfreeze via LLRD

# 数据
# 842 patients, 8500 frames, patient_grouped_split(val_frac=0.2)
# train: 673 patients (benign:254, malignant:419)
# val:   169 patients (benign:64,  malignant:105)
# max_frames_per_bag=32, img_size=512×512, grayscale

# Phase2 结果（backbone_lr_scale=0.01, η=0.75）
# epoch1: 0.4842, epoch2: 0.5132, epoch6: 0.5629, epoch7: 0.5158 (不稳定)
```
