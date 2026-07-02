# Phase 4 实施进度报告

> **当前日期**: 2026-07-01 · **距官方数据发布**: 31 天（2026-08-01 初赛启动后才有真数据）
> **性质说明**: 本阶段所有数据管道/损失组件均为**官方数据到位前的框架预备**,尚未用真实数据训练验证。
> 详细规格见各模块 `AGENTS.md`,权重清单见 `docs/预训练权重打包清单.md`。

---

## 一、已落地(可用/已测)

### 预训练权重离线打包 ✅
- 4 个模型共 546.60 MB,离线缓存 `.cache_weights/`(**不入库**,由 `scripts/download_pretrain_weights.py` 按需下载):
  ConvNeXt-Nano (59.54M) · EfficientNet-B2 (35.18M) · SSCD-ResNet50 (94.22M) · MedSAM ViT-B (357.67M)
- Manifest(含 MD5/SHA256)`.cache_weights/weights_manifest.json`;清单文档 `docs/预训练权重打包清单.md`。

### 数据管道框架 ✅(骨架,单测通过)
- **JPEG 压缩增强** `jpeg_augment.py`:抵抗 PNG→JPG 域偏移(实证 25-26% 崩塌),Q~U(30,95)。
- **动态采样器** `stage_sampler.py`:防小样本遗忘,70/30→50/50 分阶段。
- **多源适配器** `multiset_adapter.py`:统一异构标注接口。**⚠️ 当前为空壳**——具体子类标签映射(TI-RADS→概念、metadata→malignancy)未实现,未实现任务 `task_mask` 一律 valid=0,从源头杜绝假标签污染(见 `src/dataset/AGENTS.md`)。

### 损失/优化组件 ✅(已清理)
- **掩膜多任务损失**:`multitask.py` forward 注入 `valid_mask`,由 components 在 per-sample 层**真正零梯度**屏蔽无标注样本。
- **PCGrad** `pcgrad.py`:梯度冲突消解。**⚠️ 实验性,未接入主循环**——与 FAMO 互斥、当前不兼容 AMP/GradScaler,激活前须过 Gate2(见 `src/training/AGENTS.md §0`)。

---

## 二、本轮代码质量清理(commit 54f0302 后续)

前一版 Phase4 代码经 Gate2 复审发现是**半成品含 P0 缺陷**(测试通过但集成层错误),已清理:
- **删** `masked_multitask.py`:其后处理缩放 mask 是错误近似(只缩小损失量级、不屏蔽梯度),且绕过 FAMO。已被 `multitask.py` 的 valid_mask 真屏蔽取代;仅保留任务名→loss key 映射表 `TASK_TO_LOSS_KEYS` 搬入 `multitask.py`。
- **修** adapter 假标签污染:未实现映射的任务 valid 改 0。
- **修** PCGrad:增 `shared_parameters` 只对共享编码器投影(原对全模型含私有 head 投影是错的);补实验性边界标注。
- **修** sampler 归一化负数越界;合并 JPEG 两个重复类;移除各文件内联 `__main__` 测试。
- 净变更 −560/+92 行;全套 **pytest 161 passed 零回归**;3 个 `AGENTS.md` 同步。

---

## 三、待办

| 项 | 状态 | 说明 |
|---|---|---|
| ThyroidXL 获取 | ⏸️ 阻塞 | 需人工申请 HF 访问(1-3 天);`docs/ThyroidXL数据集获取指南.md` |
| ThyUS2Path 获取 | ⏸️ 待定 | 原生 JPG,用于域适应;公开集是否真有用**待官方数据到位后再判** |
| 概念损失 per-sample mask | P1 | 当前全 batch,暂靠 adapter valid=0 兜底 |
| adapter 真实 schema | 官方数据后 | 空壳子类需按真标注重写并拆分(`rules.md §6`) |
| PCGrad 激活 | 官方数据后 | 需解决 AMP 兼容 + Gate2 复审,或维持 FAMO 单方案 |

**核心卡点**:官方数据 2026-08-01 才可用、禁本地、JPG、含动静态图——真实性/规范性/良恶性三项在此前无法真训。公开数据集是否纳入训练,待官方数据到位后再决策,不预先假设有用。
