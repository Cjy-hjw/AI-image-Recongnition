"""官方比赛数据(2026 医保影像AI 赛道七)训练加载器。

把官方标注(xlsx 标注表 + `.dcm`/`.nii.gz` 影像 + NIfTI mask)转成与
`ThyroidDataset`(`src/dataset/loader.py`)**同构**的训练样本
`(img[1,H,W]f32∈[0,1], mask[1,H,W]f32, labels)`,可无改动接入
`train_step.train_batch` 的 masked-loss。

数据格式见 `docs/训练与架构/训练数据定稿.md`。**代码本地写、取数运行在云端/A800**(官方数据禁本地),
本地用合成 fixture 单测(`tests/test_official_loader.py`)。

三件套(照 `pathology_dataset.py`):
    parse_annotation_xlsx(xlsx) -> (records, colmap)   按字段名解析,禁硬编码列号
    build_official_samples(root, xlsx) -> list[sample]  静动分派 + RoiName 对 mask + masked 判据
    OfficialThyroidDataset(samples, img_size, transform) 输出同 ThyroidDataset 契约

关键铁律:
- **标签常量硬依赖 `prediction_json`,禁重定义**(Plan M3):混写单元格 `2左中叶`/`4恶性`
  解析走 `location_label_to_index` / `ete_label_to_code`,顺序/映射是单一权威定义,
  重抄会静默错位、JSON 合法但指标偏低、本地永远查不出根因。
- **masked-loss 判据 = 该临床列是否为空**(空→valid=0、有值→valid=1),逐列逐行,
  不是"是否 M 行"(定稿 §2.2:非 M 行 ~13% 也有标注)。
- **按字段名解析,禁硬编码列号**(定稿 §4.2:各轮列号不一致)。
- 患者分组切分防泄漏(rules.md §5;复用 `patient_grouped_split`)。
"""

import os
import glob
import re
import random
import sys
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# 水平翻转时位置标签的左右置换(单一权威,禁自定义映射)
from src.models.heads import flip_location_index

# 复用:患者分组切分(防同患者跨集泄漏,rules.md §5 铁律)
from src.dataset.pathology_dataset import patient_grouped_split
# 复用:DICOM/位深→uint8 灰度(不整段调 load_input_image——训练路径无需其 PHI 匿名化
#       语义,且函数签名带 output_dir(现仅占位);只借 _to_gray 的通道/位深收敛,
#       与推理侧口径一致。2026-09-01 起该函数已不写盘,本处不复用的理由收窄为
#       "训练无需匿名化开销",设计不变。)
from src.inference.io import _to_gray
# 复用:NIfTI 读取(动图 + mask),内含 (X,Y,Z)↔(H,W) 转置约定(定稿最易错处,勿另抄)
from src.inference.nifti_io import load_nifti_frames
# 复用:标签常量与混写解析(单一权威,禁重定义;Plan M3)
from src.inference.prediction_json import (
    location_label_to_index,
    _ETE_LABEL_TO_CODE,
    _TIRADS_ROMAN_TO_CODE,
)

# 可选依赖探测:不在模块层 import(避免 Pyright possibly-unbound + 缺失即整模块崩),
# 实际 import 在用到的函数内、HAS_* 守卫之后。
import importlib.util as _ilu
HAS_PYDICOM = _ilu.find_spec("pydicom") is not None
HAS_PANDAS = _ilu.find_spec("pandas") is not None
# pandas.read_excel(.xlsx) 的真实解析后端是 openpyxl(pandas 本身不含 xlsx 解析器)。
# 装了 pandas 但没装 openpyxl 时,pd.read_excel 抛的是 pandas 内部的晦涩
# ImportError,不会指名道姓说"装 openpyxl"——这里单独探测给出直白报错。
HAS_OPENPYXL = _ilu.find_spec("openpyxl") is not None


# ── xlsx 字段名映射(按字段名解析,禁硬编码列号) ─────────────────────────────
# 每个逻辑字段给多个候选列名,命中 DataFrame 里第一个存在的列。官方表头为中文。
# ⚠️ 部分列名需对**真实 xlsx** 复核后锁定(数据禁本地,在云端/A800 验)。字段名式
#    解析意味着:只需增改这里的候选名即可适配,不动下方逻辑。已确认的列名有据可依:
#      图像性质 / ANNOTATION->SliceIndex(定稿 §4.2)、主病灶位置(prediction_json 注释)、
#      ROI良恶性 写"恶/良"(定稿 §1.7)。其余为合理候选,真实表复核后收敛。
_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "patient_id":     ("PatientId", "患者ID", "患者编号", "patient_id"),
    "series_uid":     ("SeriesUid", "SeriesInstanceUID", "序列UID"),   # 静态图=磁盘子目录名
    # 🔴 **可选列,不是必需列**。官方 74 列表头里没有「图像性质」(数据集概况.md 列0-73
    # 全量清单)。静动判据已改为探测磁盘扩展名,本列仅在存在时做一致性校验。保留候选名
    # 是为了"若某轮数据真有这列就顺手校验",不命中完全不影响训练。
    "image_property": ("图像性质", "ImageProperty", "图像类型"),        # 1=静态 2=动态
    "roi_name":       ("RoiName", "ROI名称", "roi_name"),
    "slice_index":    ("ANNOTATION->SliceIndex", "SliceIndex"),        # 1-based 帧号
    # 临床标签列。列号依据 `docs/8.1赛事公开资料/数据集概况.md`(74 列全量清单):
    # 52 良恶性 / 53 TI-RADS / 59 甲状腺外侵犯 / 60 主病灶位置 / 61 临床T分期。
    # ⚠️ 候选名的空格/全角/大小写差异已由 `_resolve_columns` 的归一化匹配兜住,
    #    **不需要**为 `甲状腺 外侵犯` 这类空格变体单独加一条候选名。
    "malignancy":     ("ROI->CLINICAL->良恶性", "ROI良恶性", "良恶性"),  # 恶/良/无
    "lesion_loc":     ("主病灶位置", "病灶位置", "lesion_loc"),          # 混写 2左中叶
    "extrathy_ext":   ("甲状腺外侵犯", "ETE", "extrathy_ext"),           # 混写 2甲状腺包膜
    "clin_t_stage":   ("临床T分期", "T分期", "clin_t_stage"),           # 混写 1有 / 0无
    # TI-RADS 总分类(非"详细评分子项",两者是不同粒度的列,定稿 §2.2)。
    # `ROI->CLINICAL->TI-RADS`(列53)放第一位。⚠️ 末位 `ROI->CLINICAL->TI-RDAS` 是官方表
    # 列54 的**拼写错误冗余列**(RDAS≠RADS,数据集概况.md 已标注),正常应命中列53;放在
    # 最后仅作兜底,防列53 缺失时彻底零监督。切勿把它排到前面——两列同信息时应优先正确列。
    "tirads":         ("ROI->CLINICAL->TI-RADS", "ROI-TIRADS", "TI-RADS",
                        "TIRADS", "TI_RADS", "TI-RADS分类", "TI-RADS总分类",
                        "ROI->CLINICAL->TI-RDAS"),
    # CBM 概念子标签(列名/编码映射待真实表复核;命中则按数字前缀解析,缺失→valid=0)
    "composition":    ("成分", "composition"),
    "echogenicity":   ("回声", "echogenicity"),
    "shape":          ("形状", "纵横比", "shape"),
    "margin":         ("边缘", "margin"),
}

# RoiName:主病灶标识 = "M";次病灶 = "病灶1".."病灶N"(定稿 §1.4)。
MAIN_LESION_ROI = "M"
_SECONDARY_ROI_RE = re.compile(r"^病灶\d+$")
# 关键帧/开始/结束行仅坐标、无临床标签,不构成分类/分割样本(定稿 §2.3),
# 由 _is_lesion_roi 只放行 M/病灶N 隐式排除。

# CBM 概念任务(与 loader.py / train_step 一致);官方列名/编码未锁前默认 valid=0。
_CONCEPT_TASKS = ("composition", "echogenicity", "shape", "margin")


# ── 单元格解析工具 ──────────────────────────────────────────────────────────
def _cell_str(v) -> str:
    """单元格 → 去空白字符串;NaN/None/空 → ""。"""
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("", "nan", "none") else s


def _is_empty(v) -> bool:
    """masked-loss 判据:该列是否为空(空→该任务 valid=0,不回传梯度)。"""
    return _cell_str(v) == ""


def _digit_prefix(v) -> Optional[int]:
    """取单元格的数字前缀(混写 `2左中叶`→2、`1有`→1);无数字前缀→None。"""
    m = re.match(r"^\s*(\d+)", _cell_str(v))
    return int(m.group(1)) if m else None


def _parse_malignancy(v) -> Optional[int]:
    """ROI良恶性 → 0良/1恶。定稿 §1.7:该列写"恶"/"良"(无数字前缀);兼容"1阳性/4恶性"。"""
    s = _cell_str(v)
    if not s:
        return None
    if "恶" in s or "阳" in s:
        return 1
    if "良" in s or "阴" in s:
        return 0
    d = _digit_prefix(s)          # 兜底:数字编码(极少见)
    return None if d is None else int(d != 0)


def _parse_t_stage(v) -> Optional[int]:
    """临床T分期 → 0无/1有。混写 `1有`/`0无`:数字前缀优先,回退中文。"""
    s = _cell_str(v)
    if not s:
        return None
    d = _digit_prefix(s)
    if d is not None:
        return int(d != 0)
    if "有" in s:
        return 1
    if "无" in s:
        return 0
    return None


def _parse_ete_code(v) -> Optional[int]:
    """甲状腺外侵犯 → 输出码 1..4(1无 2包膜 3软组织 4血管);空/未知/阴性→None。

    ⚠️ **不可回退 `ete_label_to_code`**:它对任何未知串按最保守 1"无"处置——那是
       **输出侧**策略(占位不丢分),训练侧套用会把真侵犯样本(如标注 `侵犯前纵膈`)当
       "无侵犯"学,污染梯度、违反"未知→valid=0"铁律(审查 S3)。

    ⚠️ **同样不可直接吃 `_ETE_LABEL_TO_CODE` 全表**(审查 S3 Finding1 / 与 S4 一致):
       该表含输出侧的 `NA/UNK`、`无病灶（阴性）`→码1 归并项;训练侧这些是"未知/阴性",
       必须 valid=0(阴性无结节→无侵犯可评,同位置/ETE 的 S4 决策),故先显式排除再匹配。

    解析优先级:① 显式排除 `阴性`/`无病灶`(**优先于数字前缀**——阴性行即便被打了 ETE
    数字码也不入侵犯训练,与 S4 一致)→None;② 数字前缀 1..4(混写 `2甲状腺包膜`,前缀即
    官方码);③ 剥前缀后 `NA`/`UNK` 等→None;④ 规范标签精确命中 `_ETE_LABEL_TO_CODE`
    才返码(含 0-based 变体 `0无`→中文"无"→1);⑤ 其余→None→valid=0(安全丢弃不污染)。

    @云TODO(真实 xlsx 到位复核):非规范措辞(`无侵犯`/`毗邻软组织侵犯` 这类含关键词但非
       规范短标签)当前 safe-drop 到 valid=0(仅欠采、不误标)。届时须按真实措辞重锁 Chinese
       matcher——但**禁激进子串匹配**(否定陷阱:`无血管侵犯` 含"血管"会误判血管侵犯)。
       digit-prefix 路径覆盖定稿确认的混写常态(`1无侵犯`/`3毗邻软组织` 等带前缀),是可靠主路。
    """
    if _is_empty(v):
        return None
    s_full = _cell_str(v)
    # 阴性/无病灶 → None,**优先于数字前缀**(S3 Finding1 + S4:阴性无结节→无侵犯可评)。
    if ("阴性" in s_full) or ("无病灶" in s_full):
        return None
    d = _digit_prefix(v)
    if d is not None and 1 <= d <= 4:
        return d                                # 混写:数字前缀即官方输出码 1..4
    # 剥数字前缀取中文/字母部分;NA/UNK 等未知记号 → None(不吃 _ETE_LABEL_TO_CODE 的归并项)。
    s = re.sub(r"^\s*\d+\s*", "", s_full).strip()
    if s.upper() in ("NA/UNK", "NA", "N/A", "UNK", "UNKNOWN"):
        return None
    return _ETE_LABEL_TO_CODE.get(s)            # 规范标签精确命中才认;未知措辞→None(valid=0)


def _parse_tirads(v) -> Optional[int]:
    """TI-RADS 总分类 → 输出码 1..5;空/未知/越界→None。

    定稿 §1.7:真实字面是**罗马数字+"类"**(`V类`,无数字前缀),但同表另有列用
    "数字+空格+罗马"混写(`5 V类`);两条路都要走。

    解析优先级:① 空→None;② 数字前缀 1..5(混写路径,前缀即官方码);③ 剥前缀后
    归一化(去空白/全角转半角/大写/去尾缀"类"|"级")在 `_TIRADS_ROMAN_TO_CODE`
    **精确查表**才返码;④ 其余(NA/UNK/未评估/越界数字)→None→valid=0。

    🔴 **禁子串 `in` 匹配**:"IV" 含 "I"、"III" 含 "II"/"I",子串匹配会把 IV/III
    误判成 I/II。精确 dict 查表天然避免这个坑(与 `_parse_ete_code` 同一教训)。
    """
    if _is_empty(v):
        return None
    s_full = _cell_str(v)
    d = _digit_prefix(v)
    if d is not None and 1 <= d <= 5:
        return d                                 # 混写:数字前缀即官方输出码 1..5
    # 剥数字前缀,归一化后在罗马数字表精确查找(不做子串匹配)。
    s = re.sub(r"^\s*\d+\s*", "", s_full).strip()
    s = unicodedata.normalize("NFKC", s)          # 全角→半角、兼容字符归一
    s = re.sub(r"(类|级)\s*$", "", s).strip().upper()
    if not s:
        return None
    return _TIRADS_ROMAN_TO_CODE.get(s)           # 精确命中才认;未知措辞→None(valid=0)


def _parse_concept(v) -> Optional[int]:
    """CBM 概念(成分/回声/形状/边缘)→ 0-based 类索引 = 数字前缀-1。无前缀→None。

    ⚠️ 编码映射待真实 xlsx 复核:此处仅做"数字前缀-1"的通用转换,真实列若非此编码
       需在此调整。缺列/空 → 上游置 valid=0,masked-loss 跳过,不喂错标签。
    """
    d = _digit_prefix(v)
    return None if d is None or d < 1 else d - 1


# ── xlsx 解析 ───────────────────────────────────────────────────────────────
def _normalize_col(name: str) -> str:
    """列名归一化:去所有空白 + 全角转半角 + 转小写。

    🔴 **不要退回精确匹配**。官方 xlsx 的表头存在同一字段的多种书写变体,精确匹配
    (`a in cols`)对下列任一差异都会静默漏掉,而漏掉临床列**不报错**、只是该头全程
    `valid_*=0` 零监督(最隐蔽的失败模式):
      - 空格变体:云端实测报告过 `甲状腺 外侵犯`(中间有空格),而 `docs/8.1赛事公开
        资料/数据集概况.md` 列59 记的是 `甲状腺外侵犯`(无空格)。两个来源矛盾,归一化
        后二者等价,不必赌哪个对。
      - 全角/半角:`ＴＩ－ＲＡＤＳ` vs `TI-RADS`、全角括号与减号。
      - 大小写:`PatientId` vs `patientid` vs `PATIENTID`。
    归一化只用于**匹配**,`colmap` 里存的仍是 DataFrame 的原始列名(取值时必须用原名)。
    """
    s = unicodedata.normalize("NFKC", str(name))
    return "".join(s.split()).lower()


def _resolve_columns(columns) -> Dict[str, Optional[str]]:
    """逻辑字段 → DataFrame 实际列名(候选里第一个存在的);找不到记 None。

    两轮匹配:先精确(最快、最不容易误命中),再归一化(容忍空格/全角/大小写变体)。
    返回的始终是**原始列名**,供 `row[col]` 取值。
    """
    cols = list(columns)
    norm_to_orig: Dict[str, str] = {}
    for c in cols:
        # 首个命中优先:同一归一化键有多列时(如冗余的拼写错误列),保留靠前的那个,
        # 与精确匹配"候选顺序即优先级"的语义一致。
        norm_to_orig.setdefault(_normalize_col(c), c)

    colmap: Dict[str, Optional[str]] = {}
    for logical, aliases in _FIELD_ALIASES.items():
        hit = next((a for a in aliases if a in cols), None)
        if hit is None:
            hit = next((norm_to_orig[k] for a in aliases
                        if (k := _normalize_col(a)) in norm_to_orig), None)
        colmap[logical] = hit
    return colmap


def parse_annotation_xlsx(
    xlsx_path: str, sheet_name: str = "ROI级别"
) -> Tuple[List[Dict], Dict[str, Optional[str]]]:
    """读官方标注表的 ROI 级别 sheet,**按字段名**抽列,返回逐 ROI 行 record。

    Returns:
        (records, colmap)
        records — list[dict],每 dict 键为逻辑字段名(见 _FIELD_ALIASES),值为原始单元格。
        colmap  — 逻辑字段 → 命中的真实列名(None 表示该列在表中缺失)。

    Raises:
        RuntimeError: pandas 或 openpyxl 未安装。
        KeyError:     关键列(patient_id / roi_name)全候选名皆缺失
                      —— 宁可显式报错,不静默产出空样本集。
                      ⚠️ `image_property` **不在**此列表(2026-08-14 实测:官方表无该列),
                      它缺失只走软降级,详见函数体内注释。
    """
    if not HAS_PANDAS:
        raise RuntimeError("pandas 未安装,无法读 .xlsx 官方标注表")
    if not HAS_OPENPYXL:
        raise RuntimeError(
            "openpyxl 未安装,无法读 .xlsx 官方标注表(pandas.read_excel 的实际"
            "解析后端)。训练环境请先 `pip install -r requirements-train.txt`。"
        )
    import pandas as pd
    # dtype=str 保留 `2左中叶` 原样,避免 pandas 把纯数字列转成 float 引入 `2.0`。
    # header=1:官方 xlsx 用合并单元格表头,真实列名(PatientId/ROI->CLINICAL->良恶性等)
    # 在第2行(0-based row=1),第1行是合并单元格的分组标题。云端 2026-08-13 用真实官方数据
    # 实测确认——header=0(默认)会把分组标题当列名,导致 _FIELD_ALIASES 全部候选名零命中、
    # 四头全程静默 valid=0(不报错,是最隐蔽的失败模式)。header=1 后列名匹配率 100%(5/5)。
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=1, dtype=str)
    colmap = _resolve_columns(df.columns)

    for key in ("patient_id", "roi_name"):
        if colmap[key] is None:
            raise KeyError(
                f"官方标注表缺关键列 `{key}`(候选名 {_FIELD_ALIASES[key]} 均未命中)。"
                f"表头实为 {list(df.columns)} —— 请据真实表更新 _FIELD_ALIASES。"
            )
    # 🔴 `image_property` **不在**必需列里(2026-08-14 云端真数据实测修正)。
    # 官方 74 列表头中根本没有「图像性质」这一列(`docs/8.1赛事公开资料/数据集概况.md`
    # 列 0-73 全量清单可查),此前把它列为必需列导致真数据一进来必 KeyError。
    # 根因不是列名写错,是**我方从未核实过该列存在**:证据链 14 轮里它只在第一轮出现过
    # 一次、原文写的是"推断为图像性质列,需 Excel 验证",却被当成已确认写进了代码注释。
    # 现在静动判据改为**探测磁盘文件扩展名**(见 build_official_samples),那是定稿 §1.3
    # 「.dcm=静 / .nii.gz=动,全量互斥 0 混合」的直接证据,也与推理侧按扩展名分发一致,
    # 比表里的字段更可靠。该列若真存在则作为**辅助校验**使用,不存在也不影响训练。

    records: List[Dict] = []
    for _, row in df.iterrows():
        records.append(
            {logical: (row[col] if col is not None else None)
             for logical, col in colmap.items()}
        )
    return records, colmap


# ── 样本构造 ────────────────────────────────────────────────────────────────
def _is_lesion_roi(roi_name: str) -> bool:
    """是否为构成分类/分割样本的病灶行(M 或 病灶N);关键帧/开始/结束行返回 False。"""
    s = _cell_str(roi_name)
    return s == MAIN_LESION_ROI or bool(_SECONDARY_ROI_RE.match(s))


def _mask_path_for_roi(img_dir: str, roi_name: str) -> Optional[str]:
    """按 RoiName 定位 mask 文件:M→`M_1_mask.nii.gz`、病灶N→`病灶N_N_mask.nii.gz`。

    用 glob 而非硬拼数字(定稿写 `病灶N_N_mask.nii.gz`,但第二段编号不完全确定),
    命中第一个;找不到返回 None。

    🔴 **两种命名都要收(2026-08-13 Gate2 加固)**:`{roi}_*_mask.nii.gz` 里的 `*`
    在 fnmatch 语义下**至少吃一个字符**,所以该模式结构性地匹配不上无编号的裸
    `M_mask.nii.gz`。定稿 §1.4/§2/§3 三处均写实测命名为 `M_1_mask.nii.gz`(带段编号),
    故当前不触发;但万一官方数据里存在无编号变体,漏匹配是**静默失败** ——
    mask_path=None → `valid_mask=0` → 主病灶分割监督消失,动态路线还会因定位不到
    非零帧而整例跳过(:379),全程不报错、不告警,只是分数莫名偏低。官方数据禁本地、
    上云只有一次机会,多 glob 一次的成本是零。
    排序后取第一个:带编号的 `M_1_mask` 天然排在 `M_mask` 之前('1'<'m'),
    即两者并存时仍优先用定稿实测的那种命名。
    """
    roi_s = _cell_str(roi_name)
    hits = sorted(
        glob.glob(os.path.join(img_dir, f"{roi_s}_*_mask.nii.gz"))
        + glob.glob(os.path.join(img_dir, f"{roi_s}_mask.nii.gz"))
    )
    return hits[0] if hits else None


def _build_labels(rec: Dict, patient_id: str, lesion_id: str = "") -> Dict:
    """从一条 ROI record 构造 labels dict(同 ThyroidDataset 键 + 三新维度 + valid_*)。

    masked-loss 判据:每个任务的 `valid_{task}` = 该临床列**非空**才 1。空 → 0 + 占位值,
    下游 masked-loss 零梯度(定稿 §2.2:判据是列是否为空,不是是否 M 行)。

    `is_main`/`lesion_id` 供验证侧复现官方评分口径(评审规则:三新维度"仅针对匹配成功的
    主病灶"评估;动态数据分类指标按**病灶级**汇总、每病灶只贡献一个样本)——只有 labels
    dict 会随 `__getitem__` 进 batch,加在 sample base 里到不了训练循环。
    """
    labels: Dict = {"patient_id": patient_id}
    roi = _cell_str(rec.get("roi_name"))
    labels["is_main"] = int(roi == MAIN_LESION_ROI)
    labels["lesion_id"] = lesion_id or f"{patient_id}::{roi}"

    # 良恶性
    # ⚠️ 审查 S4:"阴性"→良性(0)+valid=1(有效负样本),与位置/ETE 对阴性置 valid=0
    #    **刻意不一致**——阴性病例"确定良性"是恶性头需要的负例;而"无结节可定位、无侵犯
    #    可评"使位置/ETE 天然无监督对象。语义自洽,非 bug。
    #    @云TODO:官方数据里阴性是否真以 ROI 行出现待验(ROI=病灶区,阴性可能根本无 ROI
    #    行→此路径不触发,阴性走患者级空 prediction.json)。云端验证 official_loader 时确认。
    mal = _parse_malignancy(rec.get("malignancy"))
    labels["malignancy"] = 0 if mal is None else mal
    labels["valid_malignancy"] = int(mal is not None)

    # 位置(0..6,复用权威解析;None=阴性/未知→不参与位置损失)
    loc = location_label_to_index(rec.get("lesion_loc"))
    labels["lesion_loc"] = 0 if loc is None else int(loc)
    labels["valid_lesion_loc"] = int(loc is not None and not _is_empty(rec.get("lesion_loc")))

    # 甲状腺外侵犯:输出码 1..4 → 训练 0-based 类索引 (码-1)。空列→valid=0。
    ete_code = _parse_ete_code(rec.get("extrathy_ext"))
    labels["extrathy_ext"] = 0 if ete_code is None else int(ete_code - 1)
    labels["valid_extrathy_ext"] = int(ete_code is not None)

    # 临床T分期(0/1)
    t = _parse_t_stage(rec.get("clin_t_stage"))
    labels["clin_t_stage"] = 0 if t is None else t
    labels["valid_clin_t_stage"] = int(t is not None)

    # TI-RADS 总分类:输出码 1..5 → 训练 0-based 类索引(码-1)。指标"仅匹配成功的主
    # 病灶"、**不带**恶性限定(与 ETE/T 不同,评审规则原文无"且为恶性")。空列→valid=0。
    tirads_code = _parse_tirads(rec.get("tirads"))
    labels["tirads"] = 0 if tirads_code is None else int(tirads_code - 1)
    labels["valid_tirads"] = int(tirads_code is not None)

    # CBM 概念(成分/回声/形状/边缘):列名/编码未锁前,命中列则按数字前缀-1,否则 valid=0
    for task in _CONCEPT_TASKS:
        val = _parse_concept(rec.get(task))
        labels[task] = 0 if val is None else val
        labels[f"valid_{task}"] = int(val is not None)
    # foci 为 4 维多标签,官方列/编码未定 → 占位零 + valid=0(masked-loss 跳过)
    labels["foci"] = torch.zeros(4, dtype=torch.float32)
    labels["valid_foci"] = 0

    return labels


# 三头 + TI-RADS 训练监督的实际来源列;命中列名候选表(_FIELD_ALIASES)全数缺失
# 就等于该任务零监督——`_resolve_columns` 是精确字符串匹配、缺列不报错、只让
# valid_*=0 全程静默。上云后第一时间用这个诊断确认列名候选是否命中真实表头。
_CLINICAL_LOGICAL_FIELDS = (
    "malignancy", "lesion_loc", "extrathy_ext", "clin_t_stage", "tirads",
)


def _safe_print(msg: str) -> None:
    """编码安全的 print —— 本模块所有诊断输出都必须走这里。

    🔴 **不要退回裸 print**:本模块的诊断信息全是中文,而云端官方容器常见
    `LANG=C`/`POSIX` locale(`sys.stdout.encoding` 为 ascii),Windows 中文控制台
    则是 gbk。两者下 `print("临床列命中:...")` 都会 `UnicodeEncodeError` **直接
    崩掉训练**,而崩点恰好是"列名没命中、最需要看到告警"的时候。
    实测:`PYTHONIOENCODING=ascii python -c "print('中文')"` → UnicodeEncodeError。
    注意崩的是**中文本身**不是 emoji —— 只删 emoji 不解决问题。
    这里退化成 `?` 也远好过让训练崩在一行日志上。
    """
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(msg.encode(enc, "replace").decode(enc, "replace"))


def _warn_missing_clinical_columns(colmap: Dict[str, Optional[str]]) -> None:
    """`build_official_samples` 启动时打印临床列命中摘要,未命中项显式 warning。

    不 raise(真实表可能确实缺某列,不该阻塞训练启动),但必须在训练日志第一屏
    可见——否则某个逻辑字段的候选列名全部未命中真实表头,会零监督且全程无提示,
    直到指标离奇偏低才被发现(`_resolve_columns` 本身对此完全沉默)。
    """
    missing = [f for f in _CLINICAL_LOGICAL_FIELDS if colmap.get(f) is None]
    hit = {f: colmap[f] for f in _CLINICAL_LOGICAL_FIELDS if colmap.get(f) is not None}
    _safe_print(f"--> [official_loader] 临床列命中:{hit}")
    if missing:
        _safe_print(
            f"--> [official_loader] ⚠️ 以下逻辑字段未命中任何候选列名(该任务将零监督,"
            f"valid_*=0 全程静默):{missing}。核对 _FIELD_ALIASES 候选名与真实表头是否一致。"
        )


def build_official_samples(
    root: str, xlsx_path: str, sheet_name: str = "ROI级别"
) -> List[Dict]:
    """扫描官方标注 + 影像目录,构造样本列表(照 pathology_dataset 的 build_* 模式)。

    分派(定稿 §1.11):静态 `{pid}/{series_uid}/*.dcm`;动态 `{pid}/{pid}/{pid}.nii.gz`。
    mask 靠 RoiName。动态:一个非零 mask 帧 = 一个样本(共享该结节标签,@云TODO:是否
    覆盖全部非零帧、精确子路径待真实数据复核)。无影像/无标注 → 跳过。

    Returns:
        list[sample dict],每项:
          img_path, mask_path(可 None), frame_index(动态为原始 0-based 帧号;静态 None),
          is_dynamic(bool), patient_key(=pid,供 patient_grouped_split), label(=malignancy,
          供分层), labels(完整 labels dict)。
    """
    records, colmap = parse_annotation_xlsx(xlsx_path, sheet_name)
    _warn_missing_clinical_columns(colmap)

    by_patient: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        pid = _cell_str(rec.get("patient_id"))
        if pid:
            by_patient[pid].append(rec)

    samples: List[Dict] = []
    for pid, recs in by_patient.items():
        # 静动判据 = **磁盘文件扩展名**(定稿 §1.3:.dcm 静 / .nii.gz 动,全量互斥 0 混合)。
        # 不用表里的「图像性质」列:官方 74 列表头中没有该列(见 parse_annotation_xlsx 注释)。
        # 每患者探一次(静动是患者级属性),避免逐行重复 I/O。
        dyn_path = os.path.join(root, pid, pid, f"{pid}.nii.gz")
        patient_is_dynamic = os.path.exists(dyn_path)
        # 若表里恰好存在该列,取患者级非空值做**一致性校验**(不参与判定,只在冲突时告警),
        # 便于发现"磁盘与标注表不一致"这类上游数据问题。
        patient_prop = next((_digit_prefix(r.get("image_property"))
                             for r in recs if not _is_empty(r.get("image_property"))), None)
        if patient_prop is not None and (patient_prop == 2) != patient_is_dynamic:
            _safe_print(f"--> [official_loader] ⚠️ 患者 {pid} 静动判据冲突:"
                        f"表列图像性质={patient_prop}(1静/2动) 但磁盘 {'有' if patient_is_dynamic else '无'} "
                        f"{pid}.nii.gz —— 以磁盘为准继续。")

        for rec in recs:
            roi = _cell_str(rec.get("roi_name"))
            if not _is_lesion_roi(roi):
                continue  # 关键帧/开始/结束:仅坐标,不入分类/分割样本

            # 静动判据:患者级探磁盘为主,**行级 series 目录为兜底**。
            # 主判据依据 = 证据链第10轮全量扫描 5401 个 **PatientId 目录**:仅静 5057 /
            # 仅动 343 / 混合 0(5057+343=5400≈5401 全覆盖,单位就是患者,不是子目录)。
            # 但"0 混合"终究是一次性实测结论,而误判的后果是**静默**的(静态行的标签会被
            # 挂到动态 nii.gz 上,不报错、不跳过、照常训完),故不赌它:该行若明确写了
            # series_uid 且那个目录下真有 .dcm,就按静态处理——磁盘实物比统计结论硬。
            is_dynamic = patient_is_dynamic
            if is_dynamic:
                _series = _cell_str(rec.get("series_uid"))
                if _series and glob.glob(os.path.join(root, pid, _series, "*.dcm")):
                    is_dynamic = False
                    _safe_print(
                        f"--> [official_loader] ⚠️ 患者 {pid} 同时存在动态 {pid}.nii.gz 与静态"
                        f"序列目录 {_series}(含 .dcm)——本行按**静态**处理。这与证据链第10轮"
                        f"「患者级 0 混合」矛盾,请回报此 PatientId。"
                    )

            if is_dynamic:
                img_dir = os.path.join(root, pid, pid)
                img_path = os.path.join(img_dir, f"{pid}.nii.gz")
            else:
                series = _cell_str(rec.get("series_uid"))
                img_dir = os.path.join(root, pid, series) if series else os.path.join(root, pid)
                dcms = sorted(glob.glob(os.path.join(img_dir, "*.dcm")))
                img_path = dcms[0] if dcms else None

            if not img_path or not os.path.exists(img_path):
                continue  # 无对应影像,跳过(定稿:部分检查目录在 ROI 表无对应影像)

            mask_path = _mask_path_for_roi(img_dir, roi)
            # 病灶唯一键:同一病灶的多帧(动态)必须归为一个评测样本。用 mask 路径最稳
            # (同目录同 roi 唯一);无 mask 时退回 影像目录+roi —— 同患者多序列(静态)
            # 各有自己的 img_dir,不会互相撞。
            lesion_id = mask_path or f"{img_dir}::{roi}"
            labels = _build_labels(rec, pid, lesion_id=lesion_id)
            labels["valid_mask"] = int(mask_path is not None)
            base = {
                "img_path": img_path,
                "mask_path": mask_path,
                "is_dynamic": is_dynamic,
                "patient_key": pid,
                "label": labels["malignancy"],   # 分层键(patient_grouped_split)
                "labels": labels,
            }

            if is_dynamic:
                # 动态:从 mask 非零帧定位标注帧(主病灶 M 无 SliceIndex,定稿 §4.2 项6),
                # 每非零帧一样本。无 mask 则无法定位帧 → 跳过该动态病灶。
                if mask_path is None:
                    continue
                for fidx in _nonzero_mask_frames(mask_path):
                    s = dict(base)
                    s["frame_index"] = fidx
                    samples.append(s)
            else:
                s = dict(base)
                s["frame_index"] = None
                samples.append(s)

    return samples


# ── 影像 / mask 读取 ────────────────────────────────────────────────────────
def _read_dcm_gray(path: str) -> np.ndarray:
    """读静态 `.dcm` → uint8 灰度 (H,W)。复用 _to_gray(与推理侧口径一致)。"""
    if not HAS_PYDICOM:
        raise RuntimeError("pydicom 未安装,无法读官方静态 .dcm")
    import pydicom
    ds = pydicom.dcmread(path)
    return _to_gray(ds.pixel_array)


def _read_nifti_binmasks(path: str) -> Tuple[List[np.ndarray], List[int]]:
    """读 NIfTI mask 全帧 → (list[(H,W) uint8 {0,1}], frame_indices)。

    复用 load_nifti_frames 的 (X,Y,Z)↔(H,W) 转置约定(勿另抄,定稿最易错处);
    max_frames<=0 取全帧;任意非零→前景。读失败返回 ([], [])。
    """
    frames, idxs, _, err, _ = load_nifti_frames(path, max_frames=-1)
    if err or not frames:
        return [], []
    binf = [(np.asarray(f) > 0).astype(np.uint8) for f in frames]
    return binf, list(idxs)


def _nonzero_mask_frames(mask_path: str) -> List[int]:
    """动态 mask 的非零帧原始帧号(0-based)——主病灶标注帧定位口径(定稿 §4.2 项6)。"""
    binf, idxs = _read_nifti_binmasks(mask_path)
    return [idx for idx, m in zip(idxs, binf) if m.any()]


def _read_nifti_frame_gray(path: str, frame_index: int) -> np.ndarray:
    """读动态 `.nii.gz` 指定原始帧号的灰度帧 (H,W) uint8。"""
    frames, idxs, _, err, _ = load_nifti_frames(path, max_frames=-1)
    if err or not frames:
        raise RuntimeError(f"读动图失败 {path}: {err}")
    pos = idxs.index(frame_index) if frame_index in idxs else frame_index
    return np.asarray(frames[pos])


def resize_img_to(img: np.ndarray, size: int) -> np.ndarray:
    """灰度帧 → (size,size)。**训练侧 resize 口径的单一权威**,预抽取脚本必须复用本函数。

    🔴 别再抄一份:`scripts/data/preextract_dynamic.py` 把帧提前算好存 PNG,与
    `__getitem__` 现算是**同一次计算的两个时点**。两处各写一份 cv2.resize 调用,当前
    逐位一致(实测 5 种尺寸),但任何一方将来改插值方式/加抗锯齿,另一方不会跟着变 ——
    结果是缓存图与实时图悄悄不同,而 loss 照常下降、指标只是略低,本地永远查不出。
    """
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size))
    return img


def resize_mask_to(mask: np.ndarray, size: int) -> np.ndarray:
    """mask → (size,size),**必须 INTER_NEAREST**(双线性会在边界造出 0/1 之外的中间值,
    虽然下游统一 `>0` 二值化不会崩,但前景面积会系统性变化)。理由同 `resize_img_to`。"""
    if mask.shape[:2] != (size, size):
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return mask


def _read_cached_pair(img_png: str, msk_png: str) -> Tuple[np.ndarray, np.ndarray]:
    """读预抽取缓存的 (影像帧, mask 帧),均为 uint8 灰度 (H,W)。

    产出方是 `scripts/data/preextract_dynamic.py`,存的是 **resize 之后、增强之前**
    的那一张,与走 .nii.gz 的原路径数值等价。

    🔴 **缺文件必须抛异常,不得静默返回零图**:缓存缺失意味着预抽取没覆盖到这个样本,
    静默补零 = 拿全黑图 + 全零 mask 当真样本训(分割监督被污染、四头标签还在),
    loss 照常下降、本地永远查不出根因。宁可训练启动即崩。

    用 `np.fromfile + cv2.imdecode` 而非 `cv2.imread`:后者在非 ASCII 路径下静默
    返回 None(官方数据路径含中文的可能性不为零),错误信息与"路径编码"毫无关联。
    """
    out = []
    for p in (img_png, msk_png):
        if not p or not os.path.exists(p):
            raise RuntimeError(f"预抽取缓存缺失: {p}(重跑 scripts/data/preextract_dynamic.py)")
        a = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if a is None:
            raise RuntimeError(f"预抽取缓存解码失败(文件损坏?): {p}")
        out.append(a)
    return out[0], out[1]


# ── Dataset ────────────────────────────────────────────────────────────────
class OfficialThyroidDataset(Dataset):
    """官方数据集。输出 `(img[1,H,W]f32∈[0,1], mask[1,H,W]f32, labels)`,同 ThyroidDataset。

    labels 含现有键(composition/echogenicity/shape/margin/foci/malignancy/patient_id)
    + 三新维度(lesion_loc/extrathy_ext/clin_t_stage)+ 全部 `valid_{task}`/`valid_mask`。
    三新维度键"只产出、暂不消费"——train_step/损失接线在 Phase 2(当前忽略额外键,无害)。

    ⚠️ 性能:动态样本命中 `cache_img` 走 `_read_cached_pair`(预抽取 PNG),**不再重解码**
       `.nii.gz`;无 cache 键回落原路径(每样本全量解压影像+mask 两个体积)。预抽取见
       `scripts/data/preextract_dynamic.py`,缓存数值等价、见 `_read_cached_pair` 注释。
    """

    def __init__(self, samples: List[Dict], img_size: int = 512, transform=None,
                 hflip_prob: float = 0.0, seed: Optional[int] = None):
        self.samples = samples
        self.img_size = img_size
        self.transform = transform
        self.hflip_prob = float(hflip_prob)
        # 🔴 **生产路径必须 seed=None**(走全局 `random`,与 augmentation/transforms.py
        # 里所有增强同源):全局 RNG 由 DataLoader 的 `worker_init_fn`(seed_worker)按
        # `torch.initial_seed()` 播种,而 torch **每 epoch 创建 iterator 时重抽 base_seed**
        # → 各 worker、各 epoch 的翻转序列都不同,可复现性由主种子统一保证。
        #
        # 传固定 seed 只用于**测试确定性**。持久 `random.Random(seed)` 实例在生产会
        # 多样性塌缩:dataset 对象被 pickle 复制到每个 worker(状态相同),且
        # persistent_workers=False 时每 epoch 重建 worker 使序列重置 —— 实测三个 epoch
        # 的翻转序列**逐位相同**(不报错、loss 照常降,只是增强白做)。
        self._rng = random.Random(seed) if seed is not None else None

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_img(self, img: np.ndarray) -> np.ndarray:
        return resize_img_to(img, self.img_size)      # 口径单一权威,见该函数 docstring

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        return resize_mask_to(mask, self.img_size)

    def __getitem__(self, idx):
        s = self.samples[idx]

        if s["is_dynamic"] and s.get("cache_img"):
            # 预抽取缓存命中(scripts/data/preextract_dynamic.py 产物)。
            # 🔴 走这条就**不碰 .nii.gz**:原路径每样本要全量解压影像+mask 两个体积
            # (各 258MB 量级、~4s),1400 动态样本 × 35 epoch ≈ 24 小时纯解压,且
            # num_workers=12 时瞬时占用 ~8.4GB —— 首轮真训的 `invalid block type`
            # 与步耗时 22 倍抖动即由此而来。缓存图是 resize 之后、增强之前的那一张,
            # 与原路径**数值等价**(见预抽取脚本 docstring),下面的 _resize_* 对它是 no-op。
            img, mask = _read_cached_pair(s["cache_img"], s["cache_msk"])
        elif s["is_dynamic"]:
            img = _read_nifti_frame_gray(s["img_path"], s["frame_index"])
            mask = self._dynamic_mask_frame(s["mask_path"], s["frame_index"])
        else:
            img = _read_dcm_gray(s["img_path"])
            mask = self._static_mask(s["mask_path"])

        img = self._resize_img(img)
        mask = self._resize_mask(mask)

        # 增强在归一化前(img/mask 仍 uint8,与 ThyroidDataset/PathologyDataset 一致)
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # labels 浅拷贝 + foci 单独 clone:foci 是 _build_labels 建的**共享**零 tensor,
        # 浅拷贝会让多样本/多帧共享同一对象,下游(Phase 2 消费 foci)若原地改会跨样本
        # 污染,故克隆隔离。其余值均为不可变标量,浅拷贝安全。
        labels = dict(s["labels"])
        labels["foci"] = labels["foci"].clone()

        # 水平翻转:必须与位置标签联动。漏了会让位置头学到左右相反的监督,而 loss 照常
        # 下降(location.py:97-98 的警告)。**在 transform 之后、归一化之前**做,故本类
        # 的 transform 不得自带 HorizontalFlip(否则翻两次且标签只翻一次)。
        # RNG 取用见 `__init__`:生产走全局 random(每 worker/每 epoch 不同),
        # 仅测试传 seed 走独立实例(确定性)。
        rng = self._rng if self._rng is not None else random
        if self.hflip_prob > 0.0 and rng.random() < self.hflip_prob:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
            if labels.get("valid_lesion_loc", 0) == 1:
                # flip_location_index 内部 `perm[idx]` 取 idx.device —— python int 直传
                # 报 AttributeError,必须包成 tensor(0-d 进 0-d 出,取值正确)。
                labels["lesion_loc"] = int(
                    flip_location_index(torch.tensor(int(labels["lesion_loc"])))
                )

        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        mask_t = torch.tensor((mask > 0).astype(np.float32), dtype=torch.float32).unsqueeze(0)
        return img_t, mask_t, labels

    def _static_mask(self, mask_path: Optional[str]) -> np.ndarray:
        if mask_path is None:
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        binf, _ = _read_nifti_binmasks(mask_path)
        return binf[0] if binf else np.zeros((self.img_size, self.img_size), dtype=np.uint8)

    def _dynamic_mask_frame(self, mask_path: Optional[str], frame_index: int) -> np.ndarray:
        binf, idxs = _read_nifti_binmasks(mask_path) if mask_path else ([], [])
        if not binf:
            # 理论不达:动态样本仅在有 mask 且该帧非零时构造
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        pos = idxs.index(frame_index) if frame_index in idxs else frame_index
        return binf[pos]


def _load_samples_pkl(path: str) -> Tuple[List[Dict], Dict]:
    """加载预抽取产物 `samples.pkl` → (samples, meta)。

    格式由 `scripts/data/preextract_dynamic.py` 写出:`{"_meta": {...}, "samples": [...]}`。
    只认这一种带 `_meta` 的字典;裸 list 一律拒绝 —— 没有 meta 就无从校验 img_size,
    而 img_size 不一致是**静默掉画质**的那类 bug,不能靠"大概是对的"放行。
    """
    import pickle
    if not os.path.exists(path):
        raise RuntimeError(f"samples.pkl 不存在: {path}(先跑 scripts/data/preextract_dynamic.py)")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "samples" not in payload:
        raise RuntimeError(
            f"samples.pkl 格式不对(缺 'samples' 键): {path}。"
            f"须由 scripts/data/preextract_dynamic.py 生成")
    return payload["samples"], payload.get("_meta", {})


def build_official_datasets(
    root: str,
    xlsx_path: str,
    img_size: int = 512,
    val_frac: float = 0.2,
    seed: int = 42,
    train_transform=None,
    sheet_name: str = "ROI级别",
    hflip_prob: float = 0.0,
    samples_pkl: Optional[str] = None,
) -> Tuple["OfficialThyroidDataset", "OfficialThyroidDataset"]:
    """便捷入口:构样本 → 患者分组切分 → 建 train/val Dataset。返回 (train_ds, val_ds)。

    分组切分复用 `patient_grouped_split`(按 patient_key 分组、label 分层),防同患者
    跨集泄漏(rules.md §5)。val 集不加增强(含 hflip=0:翻转会改位置标签,验证侧必须
    用原始朝向)。DataLoader/Sampler 由调用方按训练脚本组装。

    ⚠️ `seed` **只喂给切分**,不传给 Dataset —— Dataset 的翻转 RNG 必须留 None 走全局
    random(由 DataLoader 的 worker_init_fn 每 worker/每 epoch 播种)。曾把这个 seed
    一并传下去,实测导致翻转序列跨 epoch 逐位相同,见 `OfficialThyroidDataset.__init__`。

    `samples_pkl`:预抽取产物 `<cache-dir>/samples.pkl`(见
    `scripts/data/preextract_dynamic.py`)。给了就**跳过整个扫描重建**——原路径要重扫
    xlsx 并为定位动态非零帧解压 343 个 mask,实测 10-25 分钟,每次启动都付一遍。
    pkl 里的动态样本自带 `cache_img`/`cache_msk`,`__getitem__` 随之走 PNG 而非 .nii.gz。
    切分仍在此处按同一 `seed`/`val_frac` 现算,故换 seed 不必重跑预抽取。
    """
    if samples_pkl:
        samples, meta = _load_samples_pkl(samples_pkl)
        if meta.get("partial"):
            # 🔴 硬拒而非告警:`--limit N` 的 pkl 只覆盖 N 个动态影像文件,其余动态样本
            # 没有缓存、会**静默回落**到全量解压的原路径 —— 既没提速又只验了一小部分,
            # 而表面上训练照跑、日志照出。这正是"合法但全错"那一类,必须堵死在入口。
            raise RuntimeError(
                f"{samples_pkl} 是 --limit 模式产出的残件(_meta.partial=True),不能用于训练。"
                f"去掉 --limit 重跑 scripts/data/preextract_dynamic.py 生成完整缓存")
        cached_size = meta.get("img_size")
        if cached_size is not None and int(cached_size) != int(img_size):
            # 🔴 2026-09-11 由 WARN 升级为 raise(分辨率实验教训,长期保留):
            # 旧实现只 WARN 不拦,命中旧分辨率缓存时 __getitem__ 会把缓存 PNG
            # 上采样插值到训练 img_size —— 训练照跑、日志照出、分辨率收益归零,
            # 属"合法但全错"。任何改 img_size 的实验下这都等于白烧整轮
            # (预抽取 45min+ + 训练 1h+)。
            # 该分支只在"缓存分辨率≠训练分辨率"时触发,同分辨率正常路径不受影响。
            # ⚠️ 仍剩一条绕过路径(结构性,见 preextract_dynamic 的 frame_key):
            # 复用旧 cache-dir 换 --img-size 重跑,旧 PNG 因 frame_key 不含分辨率
            # 被原样保留,新 samples.pkl 的 meta 会"声称"新分辨率 → 本 raise 不
            # 触发。故换分辨率必须用全新 --cache-dir(物理隔离是唯一可靠解)。
            raise RuntimeError(
                f"预抽取缓存 img_size={cached_size} != 训练 img_size={img_size},"
                f"缓存帧会被二次 resize(静默掉画质,分辨率实验收益归零)。"
                f"重跑预抽取: scripts/data/preextract_dynamic.py --img-size {img_size} "
                f"(换分辨率须用全新 --cache-dir —— frame_key 不含 img_size,复用旧 "
                f"frames/ 目录会静默保留旧分辨率 PNG 且 meta 谎报新分辨率)")
    else:
        samples = build_official_samples(root, xlsx_path, sheet_name)
    if not samples:
        raise RuntimeError(f"未在 {root} + {xlsx_path} 构造出任何样本,检查路径/表头")
    train_s, val_s = patient_grouped_split(samples, val_frac=val_frac, seed=seed)
    train_ds = OfficialThyroidDataset(train_s, img_size=img_size, transform=train_transform,
                                      hflip_prob=hflip_prob)
    val_ds = OfficialThyroidDataset(val_s, img_size=img_size, transform=None)
    return train_ds, val_ds
