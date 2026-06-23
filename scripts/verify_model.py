import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.verify.runner import verify


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
