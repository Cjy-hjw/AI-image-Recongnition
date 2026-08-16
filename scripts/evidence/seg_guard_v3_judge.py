"""主会话:用 seg_over_half_frac 替代 spread 作为主判据的验证。

上一版 spread 判据被自己证伪(两个漏放场景,见 tmp_main_spread_reverse_hole.py)。
根因:max 与 mean 都是单点/全局标量,不稳健 —— 4 个异常像素就能骗过 max。

改用 `seg_over_half_frac`(越过 0.5 的像素占比):它**直接就是**预测 mask 的前景占比,
也正是官方 IoU 匹配实际消费的量。判据变成"预测 mask 里到底有没有前景",与评分口径
同构,不再是间接的分布形状代理。

阈值下界怎么定:真实病灶占 512² 的 0.1-3%。取 1e-4(≈26 个像素)
作为"几乎什么都没预测"的下界 —— 比最小真实病灶(0.1% ≈ 262 像素)低两个数量级,
不会误杀(最小真实病灶 262 像素,10 倍余量);比"恒全零"高,给数值噪声留余量。
🔴 **不设上界**(2026-08-16 实测否掉):曾加过"占比>0.5 → abort"防"全图判成前景",
但随机初始化时输出普遍 >0.5(frac≈1.0),收敛到真值必然**穿过 0.5**:
    step0 1.000 → step10 0.926 → step20 0.555 → step25 0.219 → step60 0.0099(=真值)
上界门会在 step10-20 那段打死正常训练。真正的"全图判前景"崩溃态会一直卡在高位,
由 ep20 的 dice 兜底门抓。
"""
import numpy as np

LO = 1e-4   # 只有下界。上界门已被实测否掉,见文件末尾的收敛轨迹说明

cases = []
g = lambda v: np.full((512, 512), v, dtype=np.float32)

cases.append(("上一轮真实失败态(处处 0.445)", g(0.4454), "该杀"))
cases.append(("常量图整体 >0.5(mask 全 1)", g(0.7), "该放行"))  # 本判据不抓,交 ep20 dice 兜底门

a = g(0.05); a[230:281, 230:281] = 0.45
cases.append(("有结构但 max=0.45(mask 仍全零)", a, "该杀"))

b = g(0.05); b[230:281, 230:281] = 0.95
cases.append(("有结构且越过 0.5(正常在学,前景1%)", b, "该放行"))

c = g(0.4454); c[0, 0] = c[0, 1] = c[1, 0] = c[1, 1] = 0.99
cases.append(("常量图 + 4 个异常像素", c, "该杀"))

d = g(0.05); d[250:262, 250:262] = 0.95     # 144 像素 ≈ 0.055%
cases.append(("小病灶(0.055%,接近真实下限)", d, "该放行"))

e = g(0.05); e[128:384, 128:384] = 0.95     # 25% 前景
cases.append(("大病灶(25%)", e, "该放行"))

f = g(0.05); f[0, 0] = 0.99                 # 1 个像素
cases.append(("只有 1 个像素越过 0.5", f, "该杀"))

print(f"{'场景':<38} {'over_half':>10} {'spread':>8}  {'新判据':>8} {'旧判据':>8}  真实结论")
print("-" * 108)
bad_new = bad_old = 0
for name, p, truth in cases:
    over = float((p > 0.5).mean())
    spread = float(p.max() - p.mean())
    new_v = "放行" if over >= LO else "abort"
    old_v = "放行" if spread >= 0.01 else "abort"
    m1 = "" if (new_v == "放行") == (truth == "该放行") else "  <<<错"
    m2 = "" if (old_v == "放行") == (truth == "该放行") else "  <<<错"
    if m1:
        bad_new += 1
    if m2:
        bad_old += 1
    print(f"{name:<38} {over:>10.6f} {spread:>8.4f}  {new_v:>8}{m1:<6} {old_v:>8}{m2:<6} {truth}")

print()
print(f"新判据(over_half_frac >= {LO},无上界)判错 {bad_new}/{len(cases)}")
print(f"旧判据(spread >= 0.01)      判错 {bad_old}/{len(cases)}")
