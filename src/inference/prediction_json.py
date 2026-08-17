"""按官方规范构造 `prediction.json`(2026-08-01 发布,现以《赛事开发规范 V2.1》为准;V1.0/V2.0 已由提交方删除,仅留 V2.1)。

本模块只做**数据结构组装与合规校验**,不含任何模型推理逻辑 —— 目的是让输出契约
有一个单一权威实现,两条输入链路(静态图/动态图)共用,新增字段只改这一处。

## 官方 schema(必填项一个都不能少,缺字段该结节该项判 0)

```
StudyUID              str    V2.1 起填 PatientId(目录以 PatientId 为一级,非真正 StudyUid)
ProcessingTime_ms     int    推理总耗时(毫秒)
Prediction.MaskFiles  str[]  所有预测结节 Mask 相对病例一级目录的路径列表
Prediction.Nodules[]  array  结节详情(阴性时为空数组 [])
  ├ NoduleID                str    如 "N1"
  ├ MaskFile                str    对应 mask 相对病例一级目录路径(如 SERIES_UID/nodule_001.nii.gz)
  ├ Detection.confidence    float  [0,1] 病灶级检测置信度
  ├ Malignancy.predicted    bool   true=恶性
  ├ Malignancy.confidence   float  [0,1] 恶性置信度
  ├ TI_RADS_Class           int    1..5
  ├ ExtrathyroidalExtension int    1无 2包膜 3软组织 4血管
  ├ Location                int    1..7(仅无结节时可为 0)
  ├ LocationProbs           float[7]  顺序固定且**和为 1**
  ├ T_ClinSig.T             int    0/1
  └ T_ClinSig.T_conf        float  [0,1]
Interpretation        object 推荐(Conclusion + KeyFindings)
```

## 三个新维度的类别映射(本模块最需要小心的地方)

《数据格式说明》给的是 **xlsx 训练标签的类别**,与**输出编码**并不是一一对应,
两处类别数都不同 —— 直接把训练标签的 argmax 当输出值填会系统性错位:

| 维度 | xlsx 训练标签 | 输出编码 | 差异 |
|:--|:--|:--|:--|
| `extrathy_ext` | 甲状腺包膜/毗邻软组织/血管/**无**/**无病灶(阴性)**/**NA-UNK** (6类) | `1`无 `2`包膜 `3`软组织 `4`血管 (4值) | 训练侧 6 类,输出只有 4 个合法值;后两类须归并到 `1`(无) |
| `lesion_loc` | 左上/左下/左中/右上/右下/右中/峡部/**无(阴性)**/**NA-UNK** (9类) | `1..7` + 概率分布 `float[7]` | 训练侧 9 类,输出只有前 7 类;后两类不进分布 |
| `clin_t_stage` | 有/无 (2类) | `T` = 1/0 | 直接对应 |

`_ETE_LABEL_TO_CODE` / `LOCATION_LABELS` 把这层转换固化下来,训练侧读 xlsx 与
推理侧写 json 共用同一份定义,避免两边各写一套导致顺序悄悄分叉。

## 占位值的定位（🔴 2026-08-17 起生产路径已无占位）

`build_nodule_entry` 对**未提供**的维度仍会填保守占位:ETE=1(无侵犯)、Location
均匀分布、T=0 —— 这条兜底路径**保留不删**,供 CLI 调试等不产四头概率的调用方使用。

但**生产路径已不再走它**:backlog #116b 于 2026-08-17 把四个官方头
(位置/ETE/临床T/TI-RADS)的真实 softmax 接进了 `v21_output._shared_nodule_context`,
每个字段都由调用方显式传入。**若在生产输出里看到 `ETE=1 / LocationProbs 均匀 /
T=0 / T_conf=0.0` 这组值,那不是"占位正常",而是四头没接上的故障信号**
(最可靠的单条判据:`LocationProbs` 是均匀分布 —— 真实预测几乎不可能均匀)。
"""

import math
import re
from typing import Optional

# 类别定义的**单一权威已下沉到 `src/models/label_schema.py`**（2026-08-12）。
# 原因：`.importlinter` 契约 2 规定 `src.models` 是最底层、不得依赖 `src.inference`，
# 而位置头 `models/heads/location.py` 必须按同一份顺序构造标签 —— 常量留在本文件
# 会让 CI 的 import-linter job 直接红。此处 re-export 保持对外 import 路径不变。
# **改类别定义请改 `label_schema.py`，不要在这里另抄一份。**
from src.models.label_schema import (  # noqa: F401  (re-export，保向后兼容)
    ETE_CAPSULE,
    ETE_CODES,
    ETE_NONE,
    ETE_SOFT_TISSUE,
    ETE_VESSEL,
    LOCATION_LABELS,
    LOCATION_NULL_LABELS,
    NUM_LOCATIONS,
    NUM_TIRADS_CLASSES,
    TIRADS_CODES,
    TIRADS_LABELS,
    _ETE_LABEL_TO_CODE,
    _TIRADS_ROMAN_TO_CODE,
)

# 无结节时 Location 的合法特例(规范:"1~7,仅无结节时可为 0")。
LOCATION_NONE = 0


class PredictionSchemaError(ValueError):
    """产出的 JSON 不符合官方 schema。

    独立子类而非裸 ValueError:调用方(HTTP 服务层)需要区分「我方组装逻辑有 bug」
    与「模型输出数值异常」——前者是必须修的代码缺陷,后者走既有的坏帧/ERROR 兜底。
    """


def ete_label_to_code(label) -> int:
    """xlsx 的 `extrathy_ext` 中文标签 → 输出编码。未知标签按最保守的 `1`(无侵犯)处理。"""
    if label is None:
        return ETE_NONE
    return _ETE_LABEL_TO_CODE.get(str(label).strip(), ETE_NONE)


def location_label_to_index(label):
    """xlsx 的 `主病灶位置` 值 → 0-based 索引(用于构造训练标签)。

    **必须容忍"数字+中文"混写**:2026-08-01 实测官方标注表的单元格不是纯中文,
    而是形如 `2左中叶` / `5右中叶`(编码前缀与标签写在同一格)。纯字符串相等匹配
    会对**所有**行返回 None,导致位置任务一个训练样本都拿不到,而代码不报任何错。
    同类混写遍布该表:`1阳性`/`4恶性肿瘤`/`5 V类`/`1有`/`0无`。

    解析优先级:① 先取数字前缀(最权威,直接就是官方编码);② 无数字前缀时回退到
    中文标签匹配(兼容纯文本导出的场景)。

    Returns:
        int 0..6,或 `None` 表示该样本无位置标签(阴性/未知),**不应参与位置损失**。
        返回 None 而非默认 0:把未知样本当"左上叶"训练会污染整个分布。
    """
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None

    # ① 数字前缀:`2左中叶` → 2 → 0-based 1
    m = re.match(r"^\s*(\d+)", s)
    if m:
        code = int(m.group(1))
        if 1 <= code <= NUM_LOCATIONS:
            return code - 1
        return None  # 0 或 >7:阴性/未知编码

    # ② 回退中文匹配(去掉可能的空白)
    for null_label in LOCATION_NULL_LABELS:
        if null_label in s:
            return None
    for i, name in enumerate(LOCATION_LABELS):
        if name in s:
            return i
    return None


def uniform_location_probs():
    """均匀位置分布(占位用)。

    7 个 1/7 的浮点和存在舍入误差,故把残差并入**首位**保证严格和为 1 —— 规范明写
    "顺序固定且和为 1",严格校验的评测脚本会拒绝 0.9999999。

    残差并入首位而非末位:`build_nodule_entry` 在未显式传 location 时取 argmax,
    末位补偿会让均匀分布的 argmax 恒落在第 7 类("峡部")—— 峡部是甲状腺解剖上
    体积最小、结节发生率最低的分区,占位输出系统性报峡部是明确有偏的错误信号。
    并入首位则退化为"左上叶",属常见分区,占位偏差最小。

    ⚠️ 该偏差是**刻意保留**的:占位输出必须明显有偏,避免被误认为真实预测。
    🔴 **2026-08-17 起生产不再走本函数**(#116b:`v21_output._shared_nodule_context`
    显式传真实 `location_probs`)。本函数只服务 CLI 调试等不产四头概率的调用方。
    → 因此**在生产 prediction.json 里看到均匀 LocationProbs = 四头没接上的故障
    信号**,不是"占位正常"。
    """
    p = 1.0 / NUM_LOCATIONS
    probs = [p] * NUM_LOCATIONS
    probs[0] += 1.0 - sum(probs)
    return probs


def normalize_location_probs(probs):
    """把任意非负向量归一化成严格和为 1 的 7 维分布。

    模型 softmax 输出在 fp16→fp32→python float 的转换后常有 1e-7 量级的和偏差;
    规范要求"和为 1",严格解析的评测脚本可能按 `abs(sum-1) < eps` 校验。此处显式
    重归一化并把残差并入最大项(而非末项)——并入最大项的相对扰动最小。

    Raises:
        PredictionSchemaError: 长度不为 7、含负值/NaN/Inf、或总和为 0(无法归一化)。
    """
    vals = [float(x) for x in probs]
    if len(vals) != NUM_LOCATIONS:
        raise PredictionSchemaError(
            f"LocationProbs 长度必须为 {NUM_LOCATIONS},收到 {len(vals)}")
    for v in vals:
        if not math.isfinite(v):
            raise PredictionSchemaError(f"LocationProbs 含非有限值: {vals}")
        if v < 0:
            raise PredictionSchemaError(f"LocationProbs 含负值: {vals}")

    total = sum(vals)
    if total <= 0:
        raise PredictionSchemaError("LocationProbs 全为 0,无法归一化")

    vals = [v / total for v in vals]
    imax = vals.index(max(vals))
    vals[imax] += 1.0 - sum(vals)
    return vals


def _clamp01(x, field):
    v = float(x)
    if not math.isfinite(v):
        raise PredictionSchemaError(f"{field} 非有限值: {x}")
    return min(1.0, max(0.0, v))


def _validate_relative_mask_path(mask_file: str) -> None:
    r"""校验 MaskFile 是「相对病例一级目录的路径」,符合官方 V2.1 契约。

    官方硬约束(违反会被评测判违规):
      1. **禁绝对路径** —— 不以 `/`、`\` 或盘符(如 `C:`)开头。绝对路径让评测
         无法在病例目录下定位文件,且暴露提交方本地文件系统结构。注意不能用
         `os.path.isabs` 判定:它在 Windows 上对 `/abs/x.nii.gz` 返回 False(Windows
         仅认盘符开头为绝对路径),会让评测平台(Linux)判违规的 POSIX 绝对路径
         在本地静默通过,故显式查首字符与盘符,跨平台一致。
      2. **禁 `..` 越界** —— 路径任意段不得为 `..`。`../` 可跳出病例目录读到
         提交根以外文件,是评测侧明令禁止的越界访问。

    合法形态示例:`SERIES_UID/nodule_001.nii.gz`(静态图)、
    `ACCESSION_NUM/nodule_001.nii.gz`(动态图)。纯文件名也合法(V2.0 旧格式
    仍兼容,但 V2.1 起官方要求相对路径)。

    Raises:
        PredictionSchemaError: 绝对路径或含 `..` 段。
    """
    s = str(mask_file)
    if not s:
        raise PredictionSchemaError("MaskFile 不得为空")
    # 绝对路径:不以 `/` 或 `\` 开头(POSIX 绝对路径),也不是 Windows 盘符
    # `C:` 形态。**不能只用 `os.path.isabs`** —— 它在 Windows 上对 `/abs/x.nii.gz`
    # 返回 False(Windows 仅认盘符开头为绝对路径),会让评测平台(Linux)会判违规的
    # POSIX 绝对路径在本地静默通过,两边行为分叉。此处显式查首字符与盘符,跨平台一致。
    if s.startswith(("/", "\\")):
        raise PredictionSchemaError(
            f"MaskFile 禁绝对路径(V2.1 须相对病例一级目录): {s}")
    if re.match(r"^[A-Za-z]:[\\/]", s):
        raise PredictionSchemaError(
            f"MaskFile 禁绝对路径(V2.1 须相对病例一级目录): {s}")
    # 按路径分隔符切分,逐段查 `..`(覆盖正斜杠/反斜杠、混合写法)。
    parts = re.split(r"[\\/]+", s)
    if any(part == ".." for part in parts):
        raise PredictionSchemaError(
            f"MaskFile 禁含 `..`(V2.1 禁越界访问病例目录外文件): {s}")


def build_nodule_entry(
    nodule_id: str,
    mask_file: str,
    detection_confidence: float,
    malignancy_confidence: float,
    tirads_class: int,
    ete_code: int = ETE_NONE,
    location_probs=None,
    location: Optional[int] = None,
    t_clin_sig: int = 0,
    t_confidence: float = 0.0,
    tirads_probs=None,
    ete_probs=None,
    malignancy_threshold: float = 0.5,
) -> dict:
    """组装单个结节条目,含全部必填字段的合规校验。

    Args:
        nodule_id: 如 `"N1"`。
        mask_file: mask 文件路径(**相对病例一级目录的路径**,V2.1 契约 ——
            如 `SERIES_UID/nodule_001.nii.gz` 或 `ACCESSION_NUM/nodule_001.nii.gz`;
            纯文件名也合法但已非官方推荐形态)。禁绝对路径、禁 `..` 越界,
            违反抛 `PredictionSchemaError`(见 `_validate_relative_mask_path`)。
        detection_confidence: [0,1],病灶级(动态图代表整个序列)。
        malignancy_confidence: [0,1] 恶性概率。`Malignancy.predicted` 由它与
            `malignancy_threshold` 比较得出,**不单独传入** —— 两者分开传会出现
            `predicted=true` 而 `confidence=0.2` 这类自相矛盾的输出。
        tirads_class: 1..5。
        ete_code: 1..4,见 `ETE_CODES`。默认 1(无侵犯)。
        location_probs: 7 维概率分布。None 时用均匀分布占位。
        location: 1..7。None 时取 `argmax(location_probs)+1`,保证与分布一致
            (显式传入且与 argmax 不符时以传入值为准,允许调用方用先验覆盖)。
            **注意**:使用默认均匀分布占位时 argmax 恒为 0 → Location=1(左上叶),
            这是**刻意选择**——占位输出必须明确有偏,避免被误认为真实预测。
            若分类头未训练,应显式传 `location=LOCATION_NONE`(0) 或文档化此局限。
        t_clin_sig: 0/1。
        t_confidence: [0,1]。
        tirads_probs: 可选 5 维 TI-RADS 概率分布;`None` 时**不写出该键**。
        ete_probs: 可选 4 维 ETE 概率分布;`None` 时**不写出该键**。
            🔴 这两项是 **#116a 零成本对冲键**(`TI_RADS_Probs`/`ETE_Probs`),
            **不在官方 schema 内**。背景:评审规则 `:185` 是无条件全称句
            「所有分类/分级/分型任务必须输出概率分布,否则计 0 分」,而官方字段表
            对这两项明写「该字段类型为整数(int)」、只给 int 位,两条明文互相打架。
            ⚠️ 同文 `:251` 又正面强制「概率向量的键名必须与下表中的键名完全一致」,
            而表里根本没有这两个概率键 → **我方自拟的键名必然对不上,拿分概率≈0,
            这是纯防御性下注**,不是"官方读则拿分"。净收益仍 ≥0(int 字段照常合规、
            零损失),故仍做。赛道七全文**无**"不得包含额外字段"条款(该条款在
            《初赛评审规则简要说明》`:540`「其他Finding类型不得包含上述属性字段」,
            位于**赛道八**章节 `:390` 起,不适用赛道七),故不会被判违规。
            决策见 `docs/待官方答复清单.md` Q4。
            **默认 None 而非空列表**:现有直调本函数的调用方(CLI 调试、旧测试)
            不传这两项,写出 `null` 会污染 JSON。

    Raises:
        PredictionSchemaError: 任一字段越界或类型不合规;`mask_file` 不是合法
            相对路径(绝对路径或含 `..`)。
    """
    _validate_relative_mask_path(mask_file)

    if location_probs is None:
        probs = uniform_location_probs()
    else:
        probs = normalize_location_probs(location_probs)

    if location is None:
        location = probs.index(max(probs)) + 1  # 1-based
    location = int(location)
    if location != LOCATION_NONE and not (1 <= location <= NUM_LOCATIONS):
        raise PredictionSchemaError(
            f"Location 必须为 1..{NUM_LOCATIONS}(无结节时可为 0),收到 {location}")

    ete_code = int(ete_code)
    if ete_code not in ETE_CODES:
        raise PredictionSchemaError(
            f"ExtrathyroidalExtension 必须为 {ETE_CODES} 之一,收到 {ete_code}")

    tirads_class = int(tirads_class)
    if tirads_class not in TIRADS_CODES:
        raise PredictionSchemaError(f"TI_RADS_Class 必须为 {TIRADS_CODES} 之一,收到 {tirads_class}")

    t_clin_sig = int(t_clin_sig)
    if t_clin_sig not in (0, 1):
        raise PredictionSchemaError(f"T_ClinSig.T 必须为 0 或 1,收到 {t_clin_sig}")

    mal_conf = _clamp01(malignancy_confidence, "Malignancy.confidence")

    entry = {
        "NoduleID": str(nodule_id),
        "MaskFile": str(mask_file),
        "Detection": {"confidence": _clamp01(detection_confidence, "Detection.confidence")},
        # bool(...) 显式转换:numpy.bool_ 不是 Python bool,json.dump 会抛
        # "Object of type bool_ is not JSON serializable" —— 整份 prediction.json
        # 写不出去,比字段错更严重。
        "Malignancy": {
            "predicted": bool(mal_conf >= malignancy_threshold),
            "confidence": mal_conf,
        },
        "TI_RADS_Class": tirads_class,
        "ExtrathyroidalExtension": ete_code,
        "Location": location,
        "LocationProbs": probs,
        "T_ClinSig": {"T": t_clin_sig, "T_conf": _clamp01(t_confidence, "T_ClinSig.T_conf")},
    }

    # --- #116a 对冲键:仅在调用方显式提供时写出(见 Args 说明) ---
    # 长度按各自类别数硬校验:长度错说明上游取错了头(如把 5 维 TI-RADS 填进
    # ETE 位),此时静默写出等于交一份自相矛盾的 JSON,不如当场炸。
    # 🔴 **同时校验有限性**(2026-08-17 终审 P2.4):`json.dumps` 把 NaN 写成裸
    # 字面量 `NaN`,这是**非标准 JSON**,严格解析器(Go/Rust/Jackson 默认)会拒绝
    # **整份** prediction.json —— 该病例所有指标不予计分,比单字段错严重得多。
    # 当前上游 `decode_predictions._assert_finite` 已硬拦 NaN,此处是**第二道
    # 防线**:未来若头实现变更或有人绕过 decode 直调本函数,这道校验仍在。
    def _finite_probs(vals, field, n_expect):
        out = [float(x) for x in vals]
        if len(out) != n_expect:
            raise PredictionSchemaError(
                f"{field} 长度必须为 {n_expect},收到 {len(out)}")
        for v in out:
            if not math.isfinite(v):
                raise PredictionSchemaError(f"{field} 含非有限值(NaN/Inf): {out}")
        return out

    if tirads_probs is not None:
        entry["TI_RADS_Probs"] = _finite_probs(
            tirads_probs, "TI_RADS_Probs", len(TIRADS_CODES))
    if ete_probs is not None:
        entry["ETE_Probs"] = _finite_probs(ete_probs, "ETE_Probs", len(ETE_CODES))

    return entry


def build_prediction_json(
    study_uid: str,
    nodules,
    processing_time_ms: int,
    interpretation: Optional[dict] = None,
) -> dict:
    """组装完整 `prediction.json`。

    Args:
        study_uid: **V2.1 起此值填 PatientId**(目录结构以 PatientId 为一级目录,
            `StudyUID` 字段在官方 V2.1 契约里值即 PatientId,而非真正的 StudyUid)。
            本函数不对其做语义校验,只原样写入,调用方负责传入正确的 PatientId。
        nodules: `build_nodule_entry` 产出的条目列表。**空列表即阴性病例**
            (规范:阴性时 `Nodules` 为空数组 `[]`),是合法输出而非错误。
        processing_time_ms: 推理总耗时(毫秒)。
        interpretation: 可选,`{"Conclusion": str, "KeyFindings": [str, ...]}`。

    Raises:
        PredictionSchemaError: NoduleID 或 MaskFile 重复 —— 二者都是评测脚本用来
            索引结节的键,重复会让后出现的条目覆盖先出现的,静默丢结节。
    """
    entries = list(nodules)

    ids = [n["NoduleID"] for n in entries]
    if len(set(ids)) != len(ids):
        raise PredictionSchemaError(f"NoduleID 重复: {ids}")

    mask_files = [n["MaskFile"] for n in entries]
    if len(set(mask_files)) != len(mask_files):
        raise PredictionSchemaError(f"MaskFile 重复: {mask_files}")

    payload = {
        "StudyUID": str(study_uid),
        "ProcessingTime_ms": int(processing_time_ms),
        "Prediction": {
            "MaskFiles": mask_files,
            "Nodules": entries,
        },
    }
    if interpretation is not None:
        payload["Interpretation"] = interpretation
    return payload


def validate_prediction_json(payload: dict) -> None:
    """交付前自检:逐字段核对官方 schema。

    定位是**发布闸门**而非日常路径 —— `build_*` 已在构造时校验,本函数用于验证
    "手工拼装/历史产物/跨版本反序列化"来的 payload,以及在 HTTP 服务写盘前做最后
    一道拦截。宁可此处报错也不要写出一份结构不合规的 json。

    Raises:
        PredictionSchemaError: 首个不合规项(报错信息含字段路径)。
    """
    for key in ("StudyUID", "ProcessingTime_ms", "Prediction"):
        if key not in payload:
            raise PredictionSchemaError(f"缺必填顶层字段: {key}")

    if not isinstance(payload["ProcessingTime_ms"], int):
        raise PredictionSchemaError("ProcessingTime_ms 必须是 int")

    pred = payload["Prediction"]
    for key in ("MaskFiles", "Nodules"):
        if key not in pred:
            raise PredictionSchemaError(f"缺必填字段: Prediction.{key}")

    nodules = pred["Nodules"]
    mask_files = pred["MaskFiles"]
    if not isinstance(nodules, list) or not isinstance(mask_files, list):
        raise PredictionSchemaError("Prediction.Nodules / MaskFiles 必须是数组")

    # MaskFiles 与各结节的 MaskFile 必须互相覆盖:评测按 MaskFiles 找文件、按
    # Nodules[].MaskFile 关联属性,两边对不上等于该结节的 mask 或属性丢失。
    if sorted(mask_files) != sorted(n.get("MaskFile") for n in nodules):
        raise PredictionSchemaError(
            f"Prediction.MaskFiles 与 Nodules[].MaskFile 不一致: "
            f"{mask_files} vs {[n.get('MaskFile') for n in nodules]}")

    # V2.1:MaskFile 必须是相对病例一级目录的路径,禁绝对路径/`..`越界。
    # build 侧已校验,本闸门对手工拼装/跨版本反序列化的畸形 payload 同样拦截。
    for mf in mask_files:
        _validate_relative_mask_path(mf)

    for i, n in enumerate(nodules):
        path = f"Prediction.Nodules[{i}]"
        for key in ("NoduleID", "MaskFile", "Detection", "Malignancy",
                    "TI_RADS_Class", "ExtrathyroidalExtension", "Location",
                    "LocationProbs", "T_ClinSig"):
            if key not in n:
                raise PredictionSchemaError(f"{path} 缺必填字段: {key}")

        if n["TI_RADS_Class"] not in TIRADS_CODES:
            raise PredictionSchemaError(f"{path}.TI_RADS_Class 越界: {n['TI_RADS_Class']}")
        if n["ExtrathyroidalExtension"] not in ETE_CODES:
            raise PredictionSchemaError(
                f"{path}.ExtrathyroidalExtension 非法: {n['ExtrathyroidalExtension']}")
        if n["Location"] != LOCATION_NONE and not (1 <= n["Location"] <= NUM_LOCATIONS):
            raise PredictionSchemaError(f"{path}.Location 越界: {n['Location']}")

        probs = n["LocationProbs"]
        if len(probs) != NUM_LOCATIONS:
            raise PredictionSchemaError(
                f"{path}.LocationProbs 长度必须为 {NUM_LOCATIONS},实为 {len(probs)}")
        # 容差 1e-6:浮点序列化/反序列化的往返误差量级远小于此,而真正的构造错误
        # (漏归一化、少一类)偏差在 1e-2 以上,该阈值能区分二者。
        if abs(sum(probs) - 1.0) > 1e-6:
            raise PredictionSchemaError(
                f"{path}.LocationProbs 和不为 1: {sum(probs)}")

        for fpath, val in (
            (f"{path}.Detection.confidence", n["Detection"]["confidence"]),
            (f"{path}.Malignancy.confidence", n["Malignancy"]["confidence"]),
            (f"{path}.T_ClinSig.T_conf", n["T_ClinSig"]["T_conf"]),
        ):
            if not math.isfinite(val) or not (0.0 <= val <= 1.0):
                raise PredictionSchemaError(f"{fpath} 越界 [0,1]: {val}")

        if n["T_ClinSig"]["T"] not in (0, 1):
            raise PredictionSchemaError(f"{path}.T_ClinSig.T 非法: {n['T_ClinSig']['T']}")
        if not isinstance(n["Malignancy"]["predicted"], bool):
            raise PredictionSchemaError(
                f"{path}.Malignancy.predicted 必须是 bool(numpy.bool_ 不可 JSON 序列化)")
