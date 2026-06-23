import sys
import os

# Windows 控制台默认 GBK,无法编码门里的 ✔/✘ 等字符 —— 强制 UTF-8 输出。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.verify.runner import verify  # noqa: E402  (sys.path 和 UTF-8 重配必须在此前执行)


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
