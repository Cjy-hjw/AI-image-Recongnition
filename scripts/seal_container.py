#!/usr/bin/env python3
"""容器交付密封脚本 — 计算交付物 SHA-256 并注入 entrypoint.sh。

防篡改签名链:打包流水线在交付前跑本脚本,把以下指纹写死进 entrypoint.sh 的
EXPECTED_* 占位位;容器运行时(PRODUCTION=1)重算并比对,任一不符即阻断启动。

校验对象:
  - predict.py / video_predict.py  单图与动态图两个交付入口
  - src/ 全树(所有 *.py)            核心推理逻辑(原 entrypoint 只校验入口脚本,
                                     src/ 是防篡改盲区 — 本脚本闭合之)
  - checkpoints/best_model.pth      模型权重(缺失则跳过,注入占位零值)

src 树聚合哈希算法(entrypoint.sh 必须字节级复现):
  1. 找 src/ 下所有 *.py,路径取相对项目根的 POSIX 形式(正斜杠)
  2. 按字节序排序(Python str.sort() 按 Unicode codepoint;UTF-8 编码下
     codepoint 序 == 字节序,故与 bash LC_ALL=C sort 一致 —— 含中文路径亦然)
  3. 每文件算内容 SHA-256(hex lowercase)
  4. 拼行 "<sha>  <relpath>\n"(两空格,模仿 sha256sum 输出)
  5. 对拼接串的 UTF-8 字节算 SHA-256 → 聚合指纹

一致性边界(两端会分叉,须避免):
  - 非 UTF-8 文件名 或 构建机非 UTF-8 locale → Python 解码与 bash 原始字节分叉
  - src 内 .py 符号链接 → bash `find -type f` 排除、Python os.walk 纳入(已在
    src_tree_sha256 用 islink 跳过对齐;但仍建议 src 内禁 .py symlink)
  - .py 行尾 CRLF/LF → 须 Win 构建/Linux 运行同 checkout,已由 .gitattributes 锁 LF

用法:
  python scripts/seal_container.py            # 注入 entrypoint.sh
  python scripts/seal_container.py --dry-run  # 只打印,不写文件
"""
import argparse
import hashlib
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENTRYPOINT = os.path.join(_ROOT, "scripts", "entrypoint.sh")
_ZERO = "0" * 64


def _sha256_file(path):
    """流式算单文件 SHA-256(hex),大权重文件不撑爆内存。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def src_tree_sha256(root=_ROOT):
    """src/ 全树聚合指纹(算法见模块 docstring,与 entrypoint.sh 配对)。"""
    src_dir = os.path.join(root, "src")
    rels = []
    for dirpath, _dirs, files in os.walk(src_dir):
        for fn in files:
            if fn.endswith(".py"):
                ap = os.path.join(dirpath, fn)
                if os.path.islink(ap):
                    continue  # 与 bash `find -type f` 对齐:排除符号链接,防两端分叉
                rel = os.path.relpath(ap, root).replace(os.sep, "/")
                rels.append(rel)
    rels.sort()  # ASCII 路径:Unicode codepoint 序 == LC_ALL=C 字节序
    lines = "".join(f"{_sha256_file(os.path.join(root, r))}  {r}\n" for r in rels)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest(), rels


def _inject(text, var, value):
    """把 entrypoint.sh 里 `VAR="..."` 的值替换为 value(保留行尾注释)。"""
    # 匹配:  VAR="<64hex or anything>"   后可跟空格+# 注释
    pat = re.compile(rf'(^{re.escape(var)}=")[^"]*(")', re.MULTILINE)
    new, n = pat.subn(rf'\g<1>{value}\g<2>', text)
    if n != 1:
        sys.exit(f"[seal] 注入失败:entrypoint.sh 中 {var}= 匹配到 {n} 处(应为 1)")
    return new


def main():
    ap = argparse.ArgumentParser(description="计算交付物 SHA-256 并注入 entrypoint.sh")
    ap.add_argument("--dry-run", action="store_true", help="只打印指纹,不改 entrypoint.sh")
    args = ap.parse_args()

    predict_sha = _sha256_file(os.path.join(_ROOT, "predict.py"))
    video_sha = _sha256_file(os.path.join(_ROOT, "video_predict.py"))
    src_sha, src_files = src_tree_sha256(_ROOT)

    weights_path = os.path.join(_ROOT, "checkpoints", "best_model.pth")
    if os.path.isfile(weights_path):
        weights_sha = _sha256_file(weights_path)
    else:
        weights_sha = _ZERO
        print("[seal] 警告:checkpoints/best_model.pth 不存在,权重指纹注入零值占位")

    print(f"[seal] predict.py        : {predict_sha}")
    print(f"[seal] video_predict.py  : {video_sha}")
    print(f"[seal] src/ 树({len(src_files)} 文件): {src_sha}")
    print(f"[seal] best_model.pth     : {weights_sha}")

    if args.dry_run:
        print("[seal] --dry-run:未写入 entrypoint.sh")
        return

    with open(_ENTRYPOINT, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    text = _inject(text, "EXPECTED_PREDICT_SHA256", predict_sha)
    text = _inject(text, "EXPECTED_VIDEO_PREDICT_SHA256", video_sha)
    text = _inject(text, "EXPECTED_SRC_TREE_SHA256", src_sha)
    text = _inject(text, "EXPECTED_WEIGHTS_SHA256", weights_sha)
    with open(_ENTRYPOINT, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"[seal] 已注入 {_ENTRYPOINT}")


if __name__ == "__main__":
    main()

