"""重算推理服务的静态 import 闭包 —— `scripts/pack_upload.py` 的 `_CLOSURE` 依据。

**什么时候要跑**:`src/` 下新增/删除 import 之后。漏更新 = 上云包少文件 =
云端 `ImportError` 起不来服务,而这在本地仓库里查不出来(仓库有全量文件)。

用法:
    python scripts/verify_closure.py            # 打印闭包 + 与 pack_upload 对账
    python scripts/verify_closure.py --list     # 只打印清单(可直接贴进 pack_upload)

## 为什么用静态闭包而不是运行时探针

运行时探针只覆盖**本次实际走到**的分支。推理服务有大量条件路径(gatekeeper
降级 / 静动分发 / 坏帧兜底 / 报告生成),某次跑没触发 ≠ 部署时不会触发 ——
按运行时结果裁剪就是埋雷。静态闭包按 import 语句做传递闭包,覆盖所有可能路径,
只会多留不会少留。

实测差异(2026-08-17):运行时 30 个 vs 静态 41 个,少的 11 个正是当次没走到的
降级分支(如 `gatekeeper/*` 在无权重时提前返回,整个目录看起来"没用到")。
"""
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = "src.service.app"   # 容器 ENTRYPOINT: python -m src.service.app


def mod_to_path(mod):
    """模块名 → 文件路径(优先 x.py,其次 x/__init__.py);不是包内模块返回 None。"""
    base = os.path.join(REPO, *mod.split("."))
    if os.path.isfile(base + ".py"):
        return base + ".py"
    init = os.path.join(base, "__init__.py")
    return init if os.path.isfile(init) else None


def imports_of(path, mod):
    """解析文件里所有指向 src.* 的 import 目标。

    `from src.a import b` 的 b 可能是子模块也可能是符号,两种都加进候选 ——
    `mod_to_path` 会把非模块的那个过滤掉,宁可多试不可漏。
    """
    out = set()
    try:
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    except SyntaxError as e:
        print(f"[WARN] 解析失败 {path}: {e}", file=sys.stderr)
        return out

    parts = mod.split(".")
    # __init__.py 的包名就是模块名自身;普通模块要去掉最后一段
    pkg = parts if path.endswith("__init__.py") else parts[:-1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names if a.name.startswith("src."))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # 相对 import
                basep = pkg[: len(pkg) - node.level + 1] if node.level > 1 else pkg
                base = ".".join(basep + ([node.module] if node.module else []))
            elif node.module and node.module.startswith("src."):
                base = node.module
            else:
                continue
            if not base.startswith("src."):
                continue
            out.add(base)
            out.update(f"{base}.{a.name}" for a in node.names if a.name != "*")
    return out


def compute_closure():
    reachable, queue = set(), [ENTRY]
    while queue:
        mod = queue.pop()
        if mod in reachable:
            continue
        p = mod_to_path(mod)
        if p is None:
            continue
        reachable.add(mod)
        # 导入子模块隐含导入其所有父包的 __init__
        parts = mod.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent.startswith("src") and mod_to_path(parent) and parent not in reachable:
                queue.append(parent)
        queue.extend(imports_of(p, mod))
    return sorted(
        os.path.relpath(mod_to_path(m), REPO).replace("\\", "/") for m in reachable
    )


def main():
    files = compute_closure()
    if "--list" in sys.argv:
        print("\n".join(files))
        return

    print(f"入口 {ENTRY} 的静态 import 闭包: {len(files)} 个 .py\n")

    # 各目录占比(看清哪些目录整个用不上)
    onshelf = set()
    for root, _, fs in os.walk(os.path.join(REPO, "src")):
        for fn in fs:
            if fn.endswith(".py"):
                onshelf.add(os.path.relpath(os.path.join(root, fn), REPO).replace("\\", "/"))
    used = set(files)
    tops = sorted({f.split("/")[1] for f in onshelf if f.count("/") >= 1})
    print("--- 各目录 闭包内/总数 ---")
    for t in tops:
        tot = sum(1 for f in onshelf if f.split("/")[1] == t)
        u = sum(1 for f in used if f.split("/")[1] == t)
        print(f"  src/{t:<12} {u:>3}/{tot:<3}{'   <= 整目录不可达' if u == 0 else ''}")

    # 与 pack_upload.py 的 _CLOSURE 对账
    try:
        sys.path.insert(0, os.path.join(REPO, "scripts"))
        import pack_upload
        declared = set(pack_upload._CLOSURE)
    except Exception as e:  # noqa: BLE001
        print(f"\n[WARN] 无法读取 pack_upload._CLOSURE 对账: {e}")
        return

    stale = declared - used
    fresh = used - declared
    print("\n--- 与 scripts/pack_upload.py 的 _CLOSURE 对账 ---")
    if not stale and not fresh:
        print("  ✅ 一致")
    else:
        for f in sorted(fresh):
            print(f"  🔴 闭包新增但 _CLOSURE 未收录(上云会漏文件!): {f}")
        for f in sorted(stale):
            print(f"  ⚠️  _CLOSURE 有但已不可达(可删): {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
