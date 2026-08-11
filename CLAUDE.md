# CLAUDE.md — 项目级指令(每次会话自动加载)

> 甲状腺超声多任务诊断系统(2026 全国医保影像AI识图大赛 · 赛道七)。
> 下方 `@` 导入的运行规则与质量门**强制生效**,等同正文。
另外别忘了用context-mode;
## 强制规则(导入)

@rules.md

## 按需读(不常驻)

- `QA_VP.md` — 代码审查 / 验证门细则,由 `rules.md §2` 在跑 Gate 2 时触发去读。
- 各目录 `AGENTS.md` — 模块级规则(架构 / 训练 / 推理 / 数据 / 测试 / gatekeeper),**改某模块前先读其 `AGENTS.md`**;总索引见根目录 `AGENTS.md`。

## 开发命令（AI 无法从代码推断）

```bash
# 完整测试套件（本地 Windows 须用 A800/CI，GBK bug 见 memory/gate1-windows-gbk-bug.md）
pytest tests/ -v

# 单文件测试
pytest tests/test_pathology_dataset.py -v

# Gate1 模型验证（仅服务器/CI，本地 GBK 会报错）
python scripts/verify_model.py

# 病理预训练（A800，ImageNet迁移学习，全量8500张）
# 依赖 checkpoints/{efficientnet_b2,convnext_nano}_pretrained.safetensors；缺权重时才加 --no-pretrained 走随机初始化
# ⚠️预训练当前未成功：无泄漏val AUC≈0.50(patient级弱标签难题)，方案调研中，勿据此checkpoint微调；详见 src/dataset/AGENTS.md + memory/patient-leakage-and-probe-trap.md
python scripts/pretrain_malignancy.py --epochs 30 --batch-size 32

# 冒烟验证管道（快速3epoch）
python scripts/pretrain_malignancy.py --smoke --batch-size 8

# MIL 阶段1（冻结backbone，目标patient级AUC>0.65）Phase1 best=0.5653
python scripts/pretrain_malignancy_mil.py --epochs 30 --batch-size 4 --data-root data/27021604_extract

# MIL 阶段2（LLRD解冻backbone，从Phase1 checkpoint热启动，目标AUC>0.65）
python scripts/pretrain_malignancy_mil.py --epochs 30 --batch-size 4 --data-root data/27021604_extract \
    --unfreeze-backbone --resume-from checkpoints/malignancy_mil_best.pth \
    --backbone-lr-scale 0.1 --llrd-eta 0.75 --out checkpoints/malignancy_mil_phase2_best.pth

# 一键同步本地代码到A800服务器（跳过data/.git/__pycache__/.md/账号密码，按文件大小增量上传）
# 凭证：环境变量优先，未设则自动读账号密码/deploy.env（无需每次手动export）
python scripts/deploy.py

# 重训 AuthenticityClassifier（修复 gatekeeper 维度失配 / 特征维度变更后必跑）
# A800 上运行；产出 checkpoints/authenticity_model.pth (13维 + 标准化参数)
python scripts/retrain_authenticity.py --data-root data/27021604_extract

# ThyroidXL LP-FT-LoRA 训练（A800，官方数据前的管道验证实验）
# --out 显式指定版本化文件名，勿用默认 checkpoints/thyroidxl_best.pth
# （首版同名产物因训练脚本曾漏调 BN freeze 已判定报废，见 src/dataset/AGENTS.md）
# ⚠️训练循环内部报告的 patient-AUC 不可直接采信，须先合并再用 predict.py 真实链路重测
python scripts/train_thyroidxl.py --data-root data/public_datasets/ThyroidXL --lp-epochs 10 --ft-epochs 40 --batch-size 16 --out checkpoints/thyroidxl_best_v2.pth

# 合并 LoRA checkpoint 为部署用纯 Conv2d 权重（predict.py/evaluate.py 直接可加载）
# v2 已通过验收（CPU/GPU一致性+LoRA-vs-合并一致性+真实predict.py链路AUC=0.8514），可信
python scripts/merge_lora_checkpoint.py --in checkpoints/thyroidxl_best_v2.pth --out checkpoints/thyroidxl_merged_v2.pth --lora-r 8 --lora-alpha 16

# 27021604(ThyUS2Path) LP-FT-LoRA 训练——MIL vs LoRA 对拍实验（已完成）
# 结果 best patient-AUC=0.5531，与MIL基线0.5653、随机猜0.50同噪声区间：瓶颈是数据本身非MIL框架，详见 src/dataset/AGENTS.md
python scripts/train_thyus2path_lora.py --data-root data/27021604_extract --lp-epochs 10 --ft-epochs 40 --batch-size 16
```

## Git 规范（AI 无法从 git log 推断格式约定）

**分支命名**：`feat/<功能>` / `fix/<问题>` / `chore/<维护>` / `refactor/<模块>` / `docs/<文档>`

**Commit message 格式**（Conventional Commits）：
```
feat(模块): 简短描述（≤70字）

详细说明（可选）

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

**PR 规范**：
- 标题 ≤ 70 字，用 Conventional Commits 前缀
- 描述包含：改动摘要 / 测试说明 / 是否影响推理容器

**CI 验证**（rules.md §2 强制）：push 后用 `gh run list --branch <branch> --limit 1`（走代理）核对 headSha 匹配且 `conclusion=success`。



- **比赛约束**:赛道七**仅「自有模型组」**,交付为**常驻 HTTP 推理服务**(容器 `ENTRYPOINT = python -m src.service.app`,uvicorn 监听 8000;`/call` 收批量目录任务丢线程池、`/health` 供平台探针、跑完回调;逐病例输出 `{PatientId}/prediction.json` + 逐结节 `.nii.gz`)。`predict.py`/`video_predict.py`/`entrypoint.sh`(按扩展名分发)已从生产入口**降级为本地/CI 单文件调试旁路**。打包 gatekeeper + TaskAlignmentAdaptor + CBM 报告生成器。
- **当前卡点**:官方数据 **2026-08-01 初赛启动**后才可用、**禁本地、JPG、含动静态图**。**2026-07-28 官方赛前动员会已豁免赛道七全部医保安全考核**(真实性/规范性无 Sins 得分,总分 = 临床诊断得分 × 100),故卡点收敛为**良恶性/分级/分型这一条线**——真实性/规范性降级为不计分的防御性工程,勿再按"核心三项"分配精力。另有两项官方新约束:① 所有分类/分级/分型**必须输出概率分布,否则计 0 分**(已落地);② 新增 ETE 4级分级 / 7类病灶位置 / T临床显著性三项维度,**官方只给数量未给类别定义**,是当前最大未知。进度见 `docs/之前表现.md`、`docs/比赛对齐与风险.md`、`docs/赛前动员会-五大核心评审目标与属性分类.md`(官方原文)。
- **环境**:训练在服务器 A800;本地 Python 见全局 `~/.claude/CLAUDE.md`。
- **代理配置(本地访问外网必需)**:访问 HuggingFace/GitHub/外网资源时,**PowerShell 必须先设置代理环境变量**:
  ```powershell
  $env:http_proxy="http://127.0.0.1:1203"
  $env:https_proxy="http://127.0.0.1:1203"
  $env:NODE_OPTIONS="--dns-result-order=ipv4first"
  ```
  适用场景:`pip install`、`huggingface_hub` 下载、`git push/fetch`(见 `rules.md` §5 已有临时环境变量规则)、`curl`/`wget` 等。

