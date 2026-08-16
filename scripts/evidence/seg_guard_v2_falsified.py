"""主会话独立验证:spread 判据的反向漏洞。

命题:`spread < 0.01` → 分割没学起来 → 应 abort。
反向质疑:**有空间结构但整体偏低**的输出(max=0.45, mean=0.05)spread=0.40 很大,
护栏放过 —— 但 `pred_mask=(p>0.5)` 仍然**全零**,Sclin 照样 0。

若成立,护栏需补「max 是否越过 0.5」的条件,否则是假的安全感。
"""
import numpy as np

THRESH = 0.01

cases = [
    # (名称, 构造的 prob 图, 期望的真实结论)
    ("上一轮真实失败态(处处 0.445)",
     np.full((512, 512), 0.4454, dtype=np.float32), "该杀"),
    ("常量图但整体 >0.5(mask 全 1,同样是废输出)",
     np.full((512, 512), 0.7, dtype=np.float32), "该杀"),
    ("有空间结构但整体偏低(max=0.45)",
     None, "该杀:mask 仍全零"),
    ("有空间结构且越过 0.5(正常在学)",
     None, "该放行"),
    ("常量图 + 极少数异常像素(攻击 max 的稳健性)",
     None, "该杀:整体无结构"),
]

# case 3: 有结构但 max<0.5
g = np.zeros((512, 512), dtype=np.float32) + 0.05
g[230:281, 230:281] = 0.45
cases[2] = (cases[2][0], g, cases[2][2])

# case 4: 有结构且 max>0.5
g2 = np.zeros((512, 512), dtype=np.float32) + 0.05
g2[230:281, 230:281] = 0.95
cases[3] = (cases[3][0], g2, cases[3][2])

# case 5: 常量图 + 4 个异常像素
g3 = np.full((512, 512), 0.4454, dtype=np.float32)
g3[0, 0] = g3[0, 1] = g3[1, 0] = g3[1, 1] = 0.99
cases[4] = (cases[4][0], g3, cases[4][2])

print(f"{'场景':<40} {'max':>7} {'mean':>7} {'spread':>8} {'>0.5像素':>9} "
      f"{'护栏判定':>10}  真实结论")
print("-" * 110)
for name, p, truth in cases:
    pmax, pmean = float(p.max()), float(p.mean())
    spread = pmax - pmean
    over = int((p > 0.5).sum())
    verdict = "abort" if spread < THRESH else "放行"
    mark = ""
    if verdict == "放行" and truth.startswith("该杀"):
        mark = "  <<< 漏放!"
    if verdict == "abort" and truth == "该放行":
        mark = "  <<< 误杀!"
    print(f"{name:<40} {pmax:>7.4f} {pmean:>7.4f} {spread:>8.4f} {over:>9d} "
          f"{verdict:>10}  {truth}{mark}")

print()
print("补上 max>0.5 条件后(abort 若 spread<0.01 **或** max<=0.5):")
print("-" * 110)
for name, p, truth in cases:
    pmax, pmean = float(p.max()), float(p.mean())
    spread = pmax - pmean
    over = int((p > 0.5).sum())
    verdict = "abort" if (spread < THRESH or pmax <= 0.5) else "放行"
    mark = ""
    if verdict == "放行" and truth.startswith("该杀"):
        mark = "  <<< 仍漏放!"
    if verdict == "abort" and truth == "该放行":
        mark = "  <<< 误杀!"
    print(f"{name:<40} {pmax:>7.4f} {pmean:>7.4f} {spread:>8.4f} {over:>9d} "
          f"{verdict:>10}  {truth}{mark}")
