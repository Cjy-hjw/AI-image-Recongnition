"""JSON Lines 训练日志 —— 官方 V2.1《赛事开发规范（赛道七）》硬约束。

原文(`docs/8.1赛事公开资料/赛事开发规范-赛道七V2.1提取.md` §日志输出规范):

> 参赛队伍**必须**采用 JSON Lines (JSONL) 格式输出训练日志,每行一个有效 JSON
> 对象,如未能读取到指定日志文件,则**可能被标记为可疑对象**,一切后果自负。
> 日志路径**必须为** `/2026aicompetition/workspace/logs`。

字段(√必填 / ○可选):
  √ timestamp    string  ISO 8601(带毫秒 + Z,UTC),如 2025-04-28T10:30:00.123Z
  √ epoch        int     当前训练轮次(从 1 开始)
  √ step         int     当前迭代步数
  √ phase        string  运行阶段:train / val / test
  √ mode         string  模型状态:training / inference
  √ data_source  string  数据来源标识,如 official/train_v1
  ○ loss         float   损失值(训练/验证阶段)
  ○ lr           float   当前学习率
  ○ checkpoint   string  保存的模型文件路径(官方样例里恒出现,缺省写 null)
  ○ pretrained_from string 加载的预训练权重路径/来源标识

🔴 **路径**:`OFFICIAL_LOG_DIR` 是云端容器内的**绝对路径**,本地(Windows)不可写。
   本地冒烟须显式传 `log_dir="logs"`;上云跑正式训练时用默认值(勿改)。
🔴 **逐行 flush**:审查方随时读取日志文件,不 flush 会让缓冲区里的行读不到,
   在"未能读取到指定日志文件"的判罚口径下等于自曝。故每行 write 后立即 flush。
"""
import json
import os
from datetime import datetime, timezone
from typing import Optional

# V2.1 规定的云端绝对路径。本地不可写,冒烟测试须用 log_dir="logs" 覆盖。
OFFICIAL_LOG_DIR = "/2026aicompetition/workspace/logs"

_PHASES = ("train", "val", "test")
_MODES = ("training", "inference")


def _iso_now() -> str:
    """ISO 8601 UTC 带毫秒 + Z,与 V2.1 样例逐字符同形态(2025-04-28T10:30:00.123Z)。

    一次取 now 派生秒与毫秒,避免两次 now() 跨毫秒边界拼出错位时间戳。
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class JsonlLogger:
    """追加式 JSONL 日志写入器。每次 `log()` 落一行并 flush。

    用法(context manager 保证 close):
        with JsonlLogger(log_dir="logs", data_source="official/train_v1") as lg:
            lg.log(epoch=1, step=0, phase="train", loss=2.3, lr=1e-3)
    """

    def __init__(
        self,
        log_dir: str = OFFICIAL_LOG_DIR,
        filename: str = "train.jsonl",
        data_source: str = "official/train_v1",
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self.data_source = data_source
        # 追加而非覆盖:续训/多次运行的日志接续,审查看得到完整历史。
        self._f = open(self.path, "a", encoding="utf-8")

    def log(
        self,
        epoch: int,
        step: int,
        phase: str,
        *,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
        mode: str = "training",
        data_source: Optional[str] = None,
        checkpoint: Optional[str] = None,
        pretrained_from: Optional[str] = None,
    ) -> dict:
        """写一条日志。返回落盘的 dict(供测试断言,不改变副作用)。"""
        assert phase in _PHASES, f"phase 必须是 {_PHASES},got {phase!r}"
        assert mode in _MODES, f"mode 必须是 {_MODES},got {mode!r}"
        rec = {
            "timestamp": _iso_now(),
            "epoch": int(epoch),
            "step": int(step),
            "phase": phase,
            "mode": mode,
            "data_source": data_source or self.data_source,
        }
        if loss is not None:
            rec["loss"] = float(loss)
        if lr is not None:
            rec["lr"] = float(lr)
        # checkpoint 是可选字段,但官方样例每行恒带(train 步为 null)。原样保留 None
        # → json.dumps 写成 null,与样例逐字段对齐。
        rec["checkpoint"] = checkpoint
        if pretrained_from is not None:
            rec["pretrained_from"] = pretrained_from
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()  # 审查随时读取,必须逐行落盘(见模块 docstring)
        return rec

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
