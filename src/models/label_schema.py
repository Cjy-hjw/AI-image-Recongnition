"""官方类别定义的**单一权威** —— 位置 7 类 / 甲状腺外侵犯 4 码。

## 为什么放在 `src/models/` 而不是 `src/inference/`

这些常量原先定义在 `src/inference/prediction_json.py`。但 `.importlinter` 契约 2 规定
**`src.models` 是最底层，不得反向依赖 `src.inference`**，而位置头（`models/heads/location.py`）
必须按这份顺序构造标签 —— 常量留在 inference 会让 CI 的 `import-linter` job 直接红。

类别定义本就是**模型输出空间的定义**，属于底层；交付侧是它的消费方。故下沉到 models，
`prediction_json` 改为从这里 re-export（对外 import 路径不变，向后兼容）。

**改这里之前先想清楚**：训练侧与交付侧共用同一份定义是硬约束 —— 两边各抄一份、其中
一份漂移，会造成概率分布与类别标签系统性错位：JSON 完全合法、和仍为 1、测试仍过，
只是指标莫名偏低，本地永远查不出根因。
"""

# ── 甲状腺外侵犯:输出编码(规范 §五.4) ──────────────────────────────────────
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

# ── 主病灶位置:输出 `Location` 1..7 与 `LocationProbs[7]` 的**固定顺序** ────
#
# ⚠️ **顺序取自官方 xlsx 的实际单元格值,不是《数据格式说明》的文字列举次序** ——
# 二者不一致,照文档写会错 4/7 类。2026-08-01 实测官方标注表 `主病灶位置` 列的
# 单元格形如 `1左上叶` / `2左中叶` / `5右中叶`(编码与标签写在同一格),据此:
#     1=左上叶  2=左中叶  5=右中叶
# 而《数据格式说明》的列举次序是「左上叶、左下叶、左中叶、右上叶、右下叶、右中叶、
# 峡部」,按它推则 2=左下叶、5=右下叶,与实测**直接冲突**。唯一自洽的解释是真实编码
# 按解剖习惯「上→中→下」排列(文档只是在列举有哪些类别,并非在定义编码)。
#
# 该错位不会引发任何报错:概率分布与类别标签系统性错位,JSON 完全合法,指标只是
# "偏低",本地永远查不出根因。故此处是**单一权威定义**,训练侧构造标签必须
# `import` 本常量,不得另抄一份,更不得照抄文档列举顺序。
LOCATION_LABELS = (
    "左上叶",   # 1
    "左中叶",   # 2  ← 实测证据:xlsx 单元格 `2左中叶`
    "左下叶",   # 3
    "右上叶",   # 4
    "右中叶",   # 5  ← 实测证据:xlsx 单元格 `5右中叶`
    "右下叶",   # 6
    "峡部",     # 7
)
NUM_LOCATIONS = len(LOCATION_LABELS)

# xlsx 里超出上述 7 类的两个值。它们不进概率分布,遇到时该样本不参与位置训练。
LOCATION_NULL_LABELS = ("无（阴性）", "无(阴性)", "NA/UNK")

# ── TI-RADS 总分类:输出 `TI_RADS_Class` 1..5 ──────────────────────────────
#
# xlsx `ROI-TIRADS` 列的真实字面是**罗马数字+"类"**(`V类`),不带数字前缀
# (`docs/训练与架构/训练数据定稿.md` §1.7 实测);但同表另有列用"数字+空格+罗马"
# 混写(如 `5 V类`),故解析器必须两条路都走:先取数字前缀,无前缀再回退罗马数字匹配。
TIRADS_LABELS = ("I类", "II类", "III类", "IV类", "V类")
NUM_TIRADS_CLASSES = len(TIRADS_LABELS)
TIRADS_CODES = tuple(range(1, NUM_TIRADS_CLASSES + 1))  # (1, 2, 3, 4, 5)

# 罗马数字 token(已去除"类"/"级"尾缀、已归一化)→ 官方码 1..5。
# 🔴 必须精确查表(dict 精确命中),严禁子串 `in` 匹配:"IV" 含 "I"、"III" 含 "II"
# 和 "I",用 `in` 会把 IV/III 误判成 I。
# 同时收 ASCII(I/II/III/IV/V)与 Unicode 罗马数字字符(Ⅰ-Ⅴ,U+2160..U+2164)两套,
# 真实数据用哪种未定,两套都要能命中。
_TIRADS_ROMAN_TO_CODE = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,          # ASCII
    "Ⅰ": 1, "Ⅱ": 2, "Ⅲ": 3, "Ⅳ": 4, "Ⅴ": 5,  # Ⅰ Ⅱ Ⅲ Ⅳ Ⅴ
}

__all__ = [
    "ETE_NONE",
    "ETE_CAPSULE",
    "ETE_SOFT_TISSUE",
    "ETE_VESSEL",
    "ETE_CODES",
    "_ETE_LABEL_TO_CODE",
    "LOCATION_LABELS",
    "NUM_LOCATIONS",
    "LOCATION_NULL_LABELS",
    "TIRADS_LABELS",
    "NUM_TIRADS_CLASSES",
    "TIRADS_CODES",
    "_TIRADS_ROMAN_TO_CODE",
]
