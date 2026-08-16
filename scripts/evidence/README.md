# scripts/evidence/ — 关键判据的可复跑实证

这里放**支撑某个已落地决策的实测脚本**，不是测试、不进 CI、不参与训练或推理。

**为什么单独建这个目录**：本项目多次发生「结论被写进文档，但支撑它的实验脚本随
临时文件一起清掉，结论退化成无出处的自述」。而 `.gitignore` 里 `/tmp_*` 会把
验证脚本一并排除。凡是**文档里点名引用的实证**，移到这里入库。

判定标准：文档（总纲 / 订正史 / rules）里写了「实测 XXX」并据此下了决策 → 脚本进这里。
一次性排查、中间产物、审查 agent 的临时脚本 → 留在 `tmp_*` 里随手清掉即可。

## 现有脚本

| 脚本 | 支撑的结论 | 出处 |
|---|---|---|
| `seg_guard_v2_falsified.py` | 分割护栏 v2 判据（`spread = p_max − p_mean`）**8 个场景判错 3 个** —— 有结构但整体偏低（max=0.45，spread=0.396）会被放行而 mask 仍全零、Sclin 仍 0（假的安全感）；常量图 + 4 个异常像素同样骗过 | [订正史 §分割护栏](../../docs/历史存档/总纲订正史.md) |
| `seg_guard_v3_judge.py` | v3 判据（`seg_over_half_frac >= 1e-4`）同样 8 个场景**全判对** | 同上 |
| `seg_spread_threshold_probe.py` | 冻结态 vs 可训态的输出分布对拍：冻结态 spread≈0.0005（复现云端 0.0009 指纹），可训态 5 步即 0.0515、40 步 0.819 | 同上 |

## 跑法

```bash
PYTHONIOENCODING=utf-8 python scripts/evidence/seg_guard_v2_falsified.py
PYTHONIOENCODING=utf-8 python scripts/evidence/seg_guard_v3_judge.py
PYTHONIOENCODING=utf-8 python scripts/evidence/seg_spread_threshold_probe.py   # 需 GPU，约 1 分钟
```

前两个是纯 numpy、秒级；第三个要建真实模型跑 40 个优化步。

⚠️ **护栏本身的回归防线是 `tests/test_seg_guards.py`（31 项 + 9 路变异全抓），
不是这里**。这些脚本回答的是「阈值为什么定在这个数」，测试回答的是「行为有没有被改坏」。
