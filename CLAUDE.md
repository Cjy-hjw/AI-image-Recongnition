# CLAUDE.md — 项目级指令(每次会话自动加载)

> 甲状腺超声多任务诊断系统(2026 全国医保影像AI识图大赛 · 赛道七)。
> 下方 `@` 导入的运行规则与质量门**强制生效**,等同正文。
> 别忘了用context-mode,但不该用的时候别用;
> 别盲目自信，有充足的信息再动手;

## ⚠️ 动手前必读(防 compact 失忆)

@docs/项目现状总纲.md

上文已 `@`-import,每次会话自动加载,不必再主动 Read——项目定位 / 赛道评分口径 / 上云流程 / 平台约束 / 数据格式 / 三头架构 / 卡点 / 交付债的**单一权威扁平快照**。血泪教训:compact 会压丢现状,导致按过时文档空谈已解决问题(把 120G 配额当磁盘故障、把"数据已在云上"当"等官方数据")。与 `memory/project-status-arch-plan-ready.md`(分层演进指针)**互为进度权威**,冲突时以二者为准、勿信旧散落文档。**改动代码/得新情报后须按总纲 §9 同步维护该文**(过时等同未完成改动)。

🔴 **评分口径改动前必读官方原文 `docs/8.1赛事公开资料/初赛评审规则简要说明.md`**(2026-08-12 官方发布,总纲 §1 是对它的转述+解读,凡涉及评测指标/权重/匹配规则/输出字段的代码改动,以此原文逐句核对为准,不得只读总纲的转述——本会话已两次因"拿自己的推理代替官方原文的空白"而误判,见 `docs/待官方答复清单.md` §三)。

🔴 **数据格式改动前必读 `docs/训练与架构/训练数据定稿.md`**(目录层级 / xlsx 字段解析 / 静动判据 / 编码表的单一权威;与上条评审规则互补不重叠——评审规则管"怎么算分",本文管"数据长什么样",凡涉及 `official_loader`/标签构造/mask 对齐的代码改动,以此文核对为准,不得凭印象猜字段名或编码)。

🔴 **输出契约/训练日志改动前必读 `docs/8.1赛事公开资料/赛事开发规范-赛道七V2.1提取.md`**(官方 V2.1 规范原文,管"交付物长什么样"——`prediction.json` 必填字段/概率分布强制要求的字面措辞、JSONL 训练日志的路径与逐行必填字段;与上两条互补不重叠,评审规则管算分、训练数据定稿管输入数据、本文管输出格式。凡涉及 `prediction_json.py`/`v21_output.py`/`jsonl_logger.py` 的代码改动,以此原文核对为准)。
## 强制规则(导入)

@rules.md

## 按需读(不常驻)

- `QA_VP.md` — 代码审查 / 验证门细则,由 `rules.md §2` 在跑 Gate 2 时触发去读。
- 各目录 `AGENTS.md` — 模块级规则(架构 / 训练 / 推理 / 数据 / 测试 / gatekeeper),**改某模块前先读其 `AGENTS.md`**;总索引见根目录 `AGENTS.md`。

## 开发命令（AI 无法从代码推断）

```bash
# 完整测试套件（本地可跑，实测 960 collected / 959 passed+1 skipped，2026-08-20
# backlog #128 落地后 `Demo_D` 主环境实测。演进链 891→923→929→940→959，明细见 `tests/AGENTS.md`。
# 精确计数以 pytest --collect-only 实跑为准；
# ⚠️ 测试总数跨 Python 环境不可比，
# torch2.8 预演环境待下次云端对拍重新确认，判回归须同环境前后对拍）
pytest tests/ -v

# 单文件测试
pytest tests/test_pathology_dataset.py -v

# Gate1 模型验证（2026-08-12 起本地可跑：runner.py 已补 encoding=utf-8，此前 GBK 崩）
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

# 官方三头（位置/ETE/临床T）A/B/C 调度训练（Phase E，2026-08-13）
# 本地冒烟：合成标签验管道数值健康（无官方数据也能跑，含全 invalid 批 / 峡部 / ETE 血管边界）
python scripts/train_official.py --mock-labels --smoke --log-dir logs --batch-size 4

# 云端真数据正式训练（官方数据禁本地；A=1-10 冻主干温升三头 / B=11-30 LLRD / C=31-35 校准）
# ⚠️ 先跑预抽取 PNG 缓存（下面一条命令，只跑一次，约 35-45 分钟），再用 --samples-pkl 启动训练——
# 否则每个 epoch 对每个动态样本全量解压两个 258MB 体积，35 epoch 纯解压 ~24 小时就是白烧
# （动态 mask 文件 0.2-0.3MB 是压缩后大小，解压后与影像同体素数，勿据文件大小判"小头"）。
# JSONL 日志默认落 V2.1 强制路径 /2026aicompetition/workspace/logs（不传 --log-dir）
# 默认保留 legacy 多任务损失守护旧任务不退化；纯头训练传 --freeze-legacy
# 🔴 backlog #129（2026-08-17 实测定案）：--epochs 24 省 31% —— Stage C（ep31-35）
# 5 个 epoch 的指标总跨度 composite 仅 0.00047，纯浪费；Stage A（ep1-10）经日志重解析
# 证实三头真的在学（val_dice+0.334/tirads_kappa+0.179），不砍。24 epoch 下四道护栏
# （epoch>=绝对值 形式，见 train_official.py 顶部 SEG_*/COMPOSITE_WARN 常量）逐位适用，
# 无需改代码；--epochs 缺省仍是 STAGE_C_END=35，需显式传 24。
python scripts/data/preextract_dynamic.py --data-root /2026aicompetition/datasets/training/annotation \
    --xlsx <标注.xlsx> --cache-dir /2026aicompetition/workspace/cache [--limit 3]  # 先 --limit 3 验管道，约 1 分钟，再全量约 35-45 分钟
python scripts/train_official.py --data-root /2026aicompetition/datasets/training/annotation \
    --xlsx <标注.xlsx> --resume-from checkpoints/best_model_mixtrain_xl_e40.pth \
    --samples-pkl /2026aicompetition/workspace/cache/samples.pkl \
    --epochs 24 \
    --out checkpoints/official_heads_seg_r6_ep24.pth
# 🔴 --out 必须显式传版本化文件名（不要用默认 official_heads_best.pth）：那是
# --mock-labels --smoke 冒烟产物的默认路径，与真实产物同名同量级、文件系统分不出来
# （memory: weights-inventory-where-each-checkpoint-lives 记过同类误判）。
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
- **当前卡点**:官方数据 **已于 2026-08-01 上线云端**(`/2026aicompetition`,**禁本地下载**、JPG、含动静态图)。**2026-07-28 官方赛前动员会已豁免赛道七全部医保安全考核**(真实性/规范性无 Sins 得分)。🔴 **总分公式已被 2026-08-12《初赛评审规则简要说明》官方确证:赛道七总分 = Sclin(无 ×100;Sclin = 目标三60%+目标四40%,目标五初赛不计分),此前"总分=临床诊断得分×100"是 07-28 动员会旧口径,已过时**,详见 `docs/项目现状总纲.md` §1。故卡点收敛为**良恶性/分级/分型这一条线**——真实性/规范性降级为不计分的防御性工程,勿再按"核心三项"分配精力。另有两项官方新约束:① 所有分类/分级/分型**必须输出概率分布,否则计 0 分**——良恶性/TI-RADS/Location 已落地,但 🔴 **TI-RADS 与 ETE 在 V2.1 schema 里没有概率字段位,与该要求构成结构性矛盾,未解决**(见 `docs/待官方答复清单.md` Q4,含零成本对冲方案);② 新增 ETE 4级分级 / 7类病灶位置 / T临床显著性三项维度,**官方只给数量与字段类型、未给判定语义**(类别编码已从官方数据实测得出:位置7类/侵犯4类,见 `docs/项目现状总纲.md` §4)。
  🔴 **但这些"未知"都不阻塞开训**(2026-08-12 用户拍板):它们只影响并类/损失加权/阈值,**三头已在其全部未公布的情况下实现完毕**。当前真前置详见 `docs/项目现状总纲.md` §7(backlog)与 `docs/训练与架构/三头实现记录.md` §5.1。
  进度权威 = `docs/项目现状总纲.md`;官方原文见 `docs/官方原文/赛前动员会-五大核心评审目标与属性分类.md`;历史来龙去脉见 `docs/历史存档/比赛对齐与风险.md`(已标历史存档)。
- **环境**:**训练在云端官方容器**(A800 自 2026-08-12 起不再承担训练,仅作 Linux+真 CUDA 契约验证);**模拟云环境用本地 venv `C:\Users\cjy\envs\torch28`**(py3.12+torch2.8,A800 因 glibc 2.23 装不了 torch2.7+);本地常规 Python 见全局 `~/.claude/CLAUDE.md`。
- **代理配置(本地访问外网必需)**:访问 HuggingFace/GitHub/外网资源时,**PowerShell 必须先设置代理环境变量**:
  ```powershell
  $env:http_proxy="http://127.0.0.1:1203"
  $env:https_proxy="http://127.0.0.1:1203"
  $env:NODE_OPTIONS="--dns-result-order=ipv4first"
  ```
  适用场景:`pip install`、`huggingface_hub` 下载、`git push/fetch`(见 `rules.md` §5 已有临时环境变量规则)、`curl`/`wget` 等。

