#!/usr/bin/env bash
# seg_only v2 快照批量诊断(2026-09-18,云端 /root/cloud_train_pkg 下跑)
# 逐个加载 5 快照 + r5 基线,各跑一轮 diagnose_detection(官方口径),
# 结束把 6 份报告的静态/动态关键指标抽成一张对比表(快照筛选就靠它)。
# nohup 挂后台一次跑完,不占跑分名额。
set -u
cd /root/cloud_train_pkg

# 六个权重:r5 基线 + 5 快照。名字与任务书 §2 的产物对齐;
# 快照文件名若不同(任务书 --out=seg_only_finetune_v2.pth),
# 改这里即可,不用动其它行。
CKPT_DIR=/root/cloud_train_pkg/checkpoints
BASELINE=$CKPT_DIR/official_heads_seg_best.pth   # r5(0.8227/0.7955 锚点)
SNAPS="$CKPT_DIR/seg_only_finetune_v2_ep4.pth \
$CKPT_DIR/seg_only_finetune_v2_ep8.pth \
$CKPT_DIR/seg_only_finetune_v2_ep12.pth \
$CKPT_DIR/seg_only_finetune_v2_ep16.pth \
$CKPT_DIR/seg_only_finetune_v2_ep20.pth"

OUTROOT=/2026aicompetition/workspace/segonly_v2_diag
mkdir -p "$OUTROOT"

run_one() {
    w="$1"
    tag=$(basename "$w" .pth)
    if [ -f "$OUTROOT/$tag/report.json" ]; then
        echo "[skip] $tag 已有报告,跳过(删 $OUTROOT/$tag 重跑)"
        return 0
    fi
    mkdir -p "$OUTROOT/$tag"
    echo "===== [start] $tag $(date +%H:%M:%S) ====="
    PYTHONPATH=/2026aicompetition/workspace/pylibs python -u -m scripts.diagnose_detection \
        --data-root /2026aicompetition/datasets/training/annotation \
        --xlsx /2026aicompetition/datasets/training/annotation/甲状腺_标注结果_训练集.xlsx \
        --weights "$w" \
        --config configs/thyroid_config.yaml \
        --samples-pkl /2026aicompetition/workspace/cache/samples.pkl \
        --log-dir "$OUTROOT/$tag" \
        > "$OUTROOT/$tag/run.log" 2>&1
    rc=$?
    echo "===== [done rc=$rc] $tag $(date +%H:%M:%S) ====="
}

# r5 基线排第 1 个:跑挂了后面不用看(一切 Δ 的锚点)。
for w in "$BASELINE" $SNAPS; do
    run_one "$w"
done

# ── 汇总对比表 ──────────────────────────────────────────────────────────
echo "汇总 → $OUTROOT/summary.md"
python - "$OUTROOT" <<'PYEOF'
import json, sys, glob, os
out_root = sys.argv[1]
rows = []
for rp in sorted(glob.glob(os.path.join(out_root, "*", "report.json"))):
    tag = os.path.basename(os.path.dirname(rp))
    try:
        r = json.load(open(rp, encoding="utf-8"))
    except Exception as e:
        rows.append((tag, f"读报告失败: {e}")); continue
    st, dy = r.get("static", {}), r.get("dynamic", {})
    def g(d, k): return d.get(k) if isinstance(d.get(k), (int, float)) else float("nan")
    rows.append((tag, (
        g(st, "match_rate"), g(st, "ap_current"), g(st, "mean_dice"),
        g(dy, "match_rate"), g(dy, "ap_current"))))
# r5 行放最上(锚点),其余按快照 epoch 排
def _key(t):
    if "seg_best" in t: return (0, 0)
    try: return (1, int(t.rsplit("_ep", 1)[1].split(".")[0]))
    except Exception: return (2, 0)
rows.sort(key=lambda x: _key(x[0]))
lines = ["# seg_only v2 快照筛选对比", "",
         "| 权重 | 静态匹配率 | 静态AP | 静态Dice | 动态匹配率 | 动态AP |",
         "|---|---|---|---|---|---|"]
for tag, v in rows:
    if isinstance(v, str):
        lines.append(f"| {tag} | {v} | | | | |"); continue
    m, a, d, dm, da = v
    lines.append(f"| {tag} | {m:.4f} | {a:.4f} | {d:.4f} | {dm:.4f} | {da:.4f} |")
base = next((v for t, v in rows if "seg_best" in t and not isinstance(v, str)), None)
if base:
    lines += ["", "Δ vs r5(静态):", "",
              "| 权重 | Δ匹配率 | ΔAP |",
              "|---|---|---|"]
    for tag, v in rows:
        if isinstance(v, str) or "seg_best" in tag: continue
        lines.append(f"| {tag} | {v[0]-base[0]:+.4f} | {v[1]-base[1]:+.4f} |")
    lines += ["", "晋级判据(plan 止损线):Δ匹配≥+0.012 且 ΔAP≥+0.008。"]
open(os.path.join(out_root, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
PYEOF
echo "全部完成 $(date +%H:%M:%S)。对比表: $OUTROOT/summary.md"
