#!/bin/bash
# _src_tree_hash.sh — src/ 全树聚合指纹(单一事实来源)。
#
# 被 entrypoint.sh 与 tests/test_entrypoint_dispatch.py 共同 source,确保
# 「容器运行时校验」与「测试」算的是同一份函数,杜绝复制粘贴 drift(reviewer E3)。
#
# 算法与 scripts/seal_container.py 的 Python 实现字节级配对:
#   找 src 下所有 *.py → 相对路径 POSIX 形式 → LC_ALL=C 字节序排序
#   → 每文件 "sha256  relpath\n" 拼接(两空格)→ 对拼接串算 SHA-256
#
# 一致性边界(reviewer A 类):构建与运行须同在 UTF-8 locale 的 Linux;
#   src 内禁止 .py 符号链接(find -type f 排除 symlink,Python os.walk 会纳入,
#   两端会分叉);.py 行尾须一致(已由 .gitattributes 锁 LF)。
compute_src_tree_sha256() {
    find src -type f -name '*.py' | LC_ALL=C sort | while read -r f; do
        sha=$(sha256sum "$f" | awk '{print $1}')
        printf '%s  %s\n' "$sha" "$f"
    done | sha256sum | awk '{print $1}'
}
