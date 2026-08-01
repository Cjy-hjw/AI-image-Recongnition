"""按官方规范构造 `prediction.json`(2026-08-01《赛事开发规范 V1.0》§五.4)。

本模块只做**数据结构组装与合规校验**,不含任何模型推理逻辑 —— 目的是让输出契约
有一个单一权威实现,两条输入链路(静态图/动态图)共用,新增字段只改这一处。

## 官方 schema(必填项一个都不能少,缺字段该结节该项判 0)

```
StudyUID              str    检查唯一标识
ProcessingTime_ms     int    推理总耗时(毫秒)
Prediction.MaskFiles  str[]  所有预测结节 Mask 文件名列表
Prediction.Nodules[]  array  结节详情(阴性时为空数组 [])
  ├ NoduleID                str    如 "N1"
  ├ MaskFile                str    对应 mask 文件名
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

## 占位值的定位

三个新维度的分类头尚未训练(官方数据 2026-08-01 才可用)。`build_nodule_entry` 对
未提供的维度填**保守占位**:ETE=1(无侵犯)、Location 均匀分布、T=0。占位保证 JSON
结构合规不丢分,但这三项本身的得分等同随机 —— **这是已知缺口,不是可接受的终态**,
分类头训好后必须由调用方显式传入真实预测。
"""

import math

# 甲状腺外侵犯:输出编码(规范 §五.4)。
ETE_NONE = 1
ETE_CAPSULE = 2
ETE_SOFT_TISSUE = 3
ETE_VESSEL = 4
ETE_CODES = (ETE_NONE, ETE_CAPSULE, ETE_SOFT_TISSUE, ETE_VESSEL)

# xlsx `extrathy_ext` 原始标签 → 输出编码。
# "无病灶(阴性)" 与 "NA/UNK" 在输出侧无对应值,统一归并到 1(无侵犯):阴性病例本就
# 不该有结节条目,未知则按最保守("无侵犯")处置——把未知报成有侵犯会虚增分期。
_ETE_LABEL_TO_CODE = {
    "甲状腺包膜": ETE_CAPSULE,
    "毗邻软组织": ETE_SOFT_TISSUE,
    "血管": ETE_VESSEL,
    "无": ETE_NONE,
    "无病灶（阴性）": ETE_NONE,
    "无病灶(阴性)": ETE_NONE,
    "NA/UNK": ETE_NONE,
}

# 主病灶位置:输出 `Location` 1..7 与 `LocationProbs[7]` 的**固定顺序**,取自
# 《数据格式说明》`lesion_loc` 的列举次序。顺序一旦写错,概率分布与类别标签系统性
# 错位,而 JSON 完全合法、本地永远测不出来 —— 故此处是单一权威定义,训练侧构造标签
# 必须 import 本常量,不得另抄一份。
LOCATION_LABELS = (
    "左上叶",   # 1
    "左下叶",   # 2
    "左中叶",   # 3
    "右上叶",   # 4
    "右下叶",   # 5
    "右中叶",   # 6
    "峡部",     # 7
)
NUM_LOCATIONS = len(LOCATION_LABELS)

# xlsx 里超出上述 7 类的两个值。它们不进概率分布,遇到时该样本不参与位置训练。
LOCATION_NULL_LABELS = ("无（阴性）", "无(阴性)", "NA/UNK")

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
    """xlsx 的 `lesion_loc` 中文标签 → 0-based 索引(用于构造训练标签)。

    Returns:
        int 0..6,或 `None` 表示该样本无位置标签(阴性/未知),**不应参与位置损失**。
        返回 None 而非默认 0:把未知样本当"左上叶"训练会污染整个分布。
    """
    if label is None:
        return None
    s = str(label).strip()
    if s in LOCATION_NULL_LABELS:
        return None
    try:
        return LOCATION_LABELS.index(s)
    except ValueError:
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
    位置分类头未训练前,调用方应显式传 `location=LOCATION_NONE`(0) 或文档化此
    局限,而不是依赖均匀分布的"中性"假象。
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


def build_nodule_entry(
    nodule_id: str,
    mask_file: str,
    detection_confidence: float,
    malignancy_confidence: float,
    tirads_class: int,
    ete_code: int = ETE_NONE,
    location_probs=None,
    location: int = None,
    t_clin_sig: int = 0,
    t_confidence: float = 0.0,
    malignancy_threshold: float = 0.5,
) -> dict:
    """组装单个结节条目,含全部必填字段的合规校验。

    Args:
        nodule_id: 如 `"N1"`。
        mask_file: mask 文件名(**仅文件名,不含路径** —— 规范的 MaskFiles 是文件名列表,
            路径由 `{StudyUid}/{SeriesUid}/` 目录结构隐含)。
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

    Raises:
        PredictionSchemaError: 任一字段越界或类型不合规。
    """
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
    if not (1 <= tirads_class <= 5):
        raise PredictionSchemaError(f"TI_RADS_Class 必须为 1..5,收到 {tirads_class}")

    t_clin_sig = int(t_clin_sig)
    if t_clin_sig not in (0, 1):
        raise PredictionSchemaError(f"T_ClinSig.T 必须为 0 或 1,收到 {t_clin_sig}")

    mal_conf = _clamp01(malignancy_confidence, "Malignancy.confidence")

    return {
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


def build_prediction_json(
    study_uid: str,
    nodules,
    processing_time_ms: int,
    interpretation: dict = None,
) -> dict:
    """组装完整 `prediction.json`。

    Args:
        study_uid: 检查唯一标识。
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

    for i, n in enumerate(nodules):
        path = f"Prediction.Nodules[{i}]"
        for key in ("NoduleID", "MaskFile", "Detection", "Malignancy",
                    "TI_RADS_Class", "ExtrathyroidalExtension", "Location",
                    "LocationProbs", "T_ClinSig"):
            if key not in n:
                raise PredictionSchemaError(f"{path} 缺必填字段: {key}")

        if not (1 <= n["TI_RADS_Class"] <= 5):
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
