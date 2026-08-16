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

**额外诊断字段(`extra=`,2026-08-14 新增)**:官方原文措辞是「一个规范的日志条目
**应包含**以下字段」——给的是**最小必填集,未禁止额外字段**(与 `TI_RADS_Probs`
零成本对冲同一判断逻辑:不违反明文禁令,官方不读也无损失)。故 `extra` 里的键
原样并入行尾,**不得覆盖上述规范字段**(同名会被 assert 拦下)。

动机(勿删,这是它存在的唯一理由):2026-08-14 首次真数据训练的日志只有总 loss,
出现 ~5 / ~0.8 双峰却**无法定位来自哪个头**——`train_batch` 返回的 `loss_dict`
在 `run_official_epoch` 里被直接丢弃(`scripts/train_official.py`)。没有子损失
分解,任何根因分析都只能靠猜;而本项目已两次栽在"推断被当成事实"上
(`image_property`/`header=1`,见 `memory/inferred-facts-must-not-enter-hard-fail-paths.md`)。
同理由记录**四头各自的有效标签数**(`n_valid_*`):判断"某头是否真的在受监督"
不能靠 loss 是否非零——masked CE 在 valid 全 0 时也返回 0,与"学得好"无法区分。

🔴 **路径**:`OFFICIAL_LOG_DIR` 是云端容器内的**绝对路径**,本地(Windows)不可写。
   本地冒烟须显式传 `log_dir="logs"`;上云跑正式训练时用默认值(勿改)。
🔴 **逐行 flush**:审查方随时读取日志文件,不 flush 会让缓冲区里的行读不到,
   在"未能读取到指定日志文件"的判罚口径下等于自曝。故每行 write 后立即 flush。
"""
import json
import math
import os
from datetime import datetime, timezone
from typing import Optional

# V2.1 规定的云端绝对路径。本地不可写,冒烟测试须用 log_dir="logs" 覆盖。
OFFICIAL_LOG_DIR = "/2026aicompetition/workspace/logs"

_PHASES = ("train", "val", "test")
_MODES = ("training", "inference")
# V2.1 规范字段名 —— `extra` 不得覆盖(见 log() 的 assert)。
_RESERVED = frozenset({
    "timestamp", "epoch", "step", "phase", "mode", "data_source",
    "loss", "lr", "checkpoint", "pretrained_from",
})


def _drop_nonfinite(d: dict) -> dict:
    """丢掉值为 NaN/Inf 的键 —— **JSON 合规硬约束,勿删**。

    🔴 `json.dumps` 默认 `allow_nan=True`,会写出**裸 `NaN` / `Infinity`**,
    这不是标准 JSON(RFC 8259 只允许有限数值)。实测:
        json.dumps({"loss_seg": float("nan")}) → '{"loss_seg": NaN}'
    Python 自己的 `json.loads` 宽容能读回,但**严格解析器(Go/Java/多数评审脚本)
    会直接拒绝整行**。V2.1 原文:「如未能读取到指定日志文件,则**可能被标记为
    可疑对象**,一切后果自负」—— 一个 NaN 就可能让整份日志不可解析。

    而 NaN 分项在本项目是**预期会出现的**,不是异常:损失模块自带
    `torch.isnan` 守卫(`losses/malignancy.py`、`losses/segmentation.py`),
    ETE 的 focal 在 p_t=1.0 时实测产生过 NaN 梯度 —— 35 epoch × 24 个分项,
    撞上是迟早的事。

    **丢弃而非填 0**:填 0 会把"这项炸了"伪装成"这项很好",正是本项目最忌讳的
    静默失真;缺字段则一眼看得出异常。丢弃只影响诊断字段,不影响规范字段。
    """
    return {k: v for k, v in d.items()
            if not (isinstance(v, float) and not math.isfinite(v))}


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
        extra: Optional[dict] = None,
    ) -> dict:
        """写一条日志。返回落盘的 dict(供测试断言,不改变副作用)。

        `extra`:额外诊断字段(子损失分解 / 各头有效标签数等),原样并入行尾。
        见模块 docstring —— 官方给的是最小必填集,额外键不违规。**不得覆盖规范
        字段**(同名 assert 拦下:静默覆盖 `loss` 会让官方读到错误的值,比崩溃阴)。
        """
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
        if extra:
            clash = _RESERVED & set(extra)
            assert not clash, f"extra 不得覆盖 V2.1 规范字段: {sorted(clash)}"
            rec.update(_drop_nonfinite(extra))
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
