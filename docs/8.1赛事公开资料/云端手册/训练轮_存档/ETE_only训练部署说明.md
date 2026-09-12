# ETE-only 训练增量包部署说明(cloud_train_ete_only.zip)

> D10-15 ETE 头部小试的**训练侧**改动(不碰推理、不碰权重、不碰 cloud_infer)。
> 本地 commit `b90e480`,全量 1089+1 绿,mock 冒烟+负测试通过。
>
> 包内 2 文件:
> - `src/training/schedule_official.py`(ete_only 冻结分支)
> - `scripts/train_official.py`(--ete-only 开关+启动硬门+基线留档)

## 部署

训练代码树 = **`/root/cloud_train_pkg`**(有 scripts/、src/、configs/ 的那层;
640 轮起的惯例,r6/r7 任务书的"项目根"同指此处):

```bash
cd /root/cloud_train_pkg
unzip -o /path/to/cloud_train_ete_only.zip
```

覆盖 2 文件:`src/training/schedule_official.py` + `scripts/train_official.py`。
覆盖后核验:
```bash
cd /root/cloud_train_pkg
find src scripts -name "*.py" | wc -l    # 期望 ≈69(防解压抹掉兄弟文件)
python -c "from src.training.schedule_official import configure_official_stage; print('import OK')"
grep -n "ete_only" src/training/schedule_official.py | head -3   # 应有命中
```

## 🔴 开跑前必查两条(防白烧)

1. **config 的 ETE 损失配方**(云 config 是可变状态,r7 事故同型):
   ```bash
   grep -A1 "ete_cb_beta\|ete_class_counts" /root/cloud_train_pkg/configs/thyroid_config.yaml
   ```
   **必须是**:`ete_cb_beta: 0.9` / `ete_focal_gamma: 0.0` / `ete_la_tau: 0.0` /
   `ete_qwk_weight: 0.0` / `ete_class_counts: [4459, 333, 81, 5]`(Q7 新表,
   旧值 [3906,267,67,5] 是 r7 前的旧表)。不对先改 config 再开训。
2. **确认 r5 权重路径**(热启动源,训练树里):`checkpoints/official_heads_seg_best.pth`
   (827 键那份;不在 cloud_train_pkg 就从 workspace cp 过去)。

## 训练命令

```bash
cd /root/cloud_train_pkg
mv logs/train.jsonl logs/train_backup_$(date +%m%d_%H%M).jsonl 2>/dev/null  # JSONL 是追加,先挪走旧的
nohup python -u scripts/train_official.py \
    --data-root /2026aicompetition/datasets/training/annotation \
    --xlsx <标注.xlsx> \
    --samples-pkl /2026aicompetition/workspace/cache/samples.pkl \
    --resume-from checkpoints/official_heads_seg_best.pth \
    --ete-only --freeze-legacy \
    --epochs 8 --batch-size 16 \
    --out checkpoints/ete_only_finetune_v1.pth \
    > logs/train_ete_only.log 2>&1 &
```

⚠️ `--ete-only` 的三个硬门(漏传直接 parser.error,是故意的):
- 必须配 `--resume-from`(r5 全量权重);必须版本化 `--out`;
- 启动横幅会打 `[ETE-only 配方] cb_beta=... class_counts=...`,开跑后 1 分钟内核对该行
  与上面第 1 条一致。

## 判读门(跑完看 JSONL)

- warm-start 后第一行 `epoch=0, phase=val` = r5 基线全指标(同权重同 split,这是
  对拍基准,别拿文档里的峰值比)。
- 8 epoch 的 `ete_weighted_kappa`:**≥0.24 且其余四项(mal/loc/t/tirads)各不降超
  0.01**(对照 epoch=0 基线,不是峰值)→ 过门,线上跑 1 次看总分;
- 不过门 → 权重作废,r5 仍是交付权重,**不重试**。

## 训练时长预期

仅 ETE 头 7 个参数键可训,前向/损失全量照算——单 epoch 耗时与 r5 相当(~95s),
8 epoch ≈ 13 分钟。
