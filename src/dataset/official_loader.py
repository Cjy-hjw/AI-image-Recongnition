"""官方比赛数据(2026 医保影像AI 赛道七)训练加载器。

把官方标注(xlsx 标注表 + `.dcm`/`.nii.gz` 影像 + NIfTI mask)转成与
`ThyroidDataset`(`src/dataset/loader.py`)**同构**的训练样本
`(img[1,H,W]f32∈[0,1], mask[1,H,W]f32, labels)`,可无改动接入
`train_step.train_batch` 的 masked-loss。

数据格式见 `docs/训练数据定稿.md`。**代码本地写、取数运行在云端/A800**(官方数据禁本地),
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
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# 复用:患者分组切分(防同患者跨集泄漏,rules.md §5 铁律)
from src.dataset.pathology_dataset import patient_grouped_split
# 复用:DICOM/位深→uint8 灰度(不整段调 load_input_image——那含 PHI 匿名化写盘副作用,
#       每样本写盘不可接受;只借 _to_gray 的通道/位深收敛,与推理侧口径一致)
from src.inference.io import _to_gray
# 复用:NIfTI 读取(动图 + mask),内含 (X,Y,Z)↔(H,W) 转置约定(定稿最易错处,勿另抄)
from src.inference.nifti_io import load_nifti_frames
# 复用:标签常量与混写解析(单一权威,禁重定义;Plan M3)
from src.inference.prediction_json import (
    location_label_to_index,
    _ETE_LABEL_TO_CODE,
)

# 可选依赖探测:不在模块层 import(避免 Pyright possibly-unbound + 缺失即整模块崩),
# 实际 import 在用到的函数内、HAS_* 守卫之后。
import importlib.util as _ilu
HAS_PYDICOM = _ilu.find_spec("pydicom") is not None
HAS_PANDAS = _ilu.find_spec("pandas") is not None


# ── xlsx 字段名映射(按字段名解析,禁硬编码列号) ─────────────────────────────
# 每个逻辑字段给多个候选列名,命中 DataFrame 里第一个存在的列。官方表头为中文。
# ⚠️ 部分列名需对**真实 xlsx** 复核后锁定(数据禁本地,在云端/A800 验)。字段名式
#    解析意味着:只需增改这里的候选名即可适配,不动下方逻辑。已确认的列名有据可依:
#      图像性质 / ANNOTATION->SliceIndex(定稿 §4.2)、主病灶位置(prediction_json 注释)、
#      ROI良恶性 写"恶/良"(定稿 §1.7)。其余为合理候选,真实表复核后收敛。
_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "patient_id":     ("PatientId", "患者ID", "患者编号", "patient_id"),
    "series_uid":     ("SeriesUid", "SeriesInstanceUID", "序列UID"),   # 静态图=磁盘子目录名
    "image_property": ("图像性质",),                                    # 1=静态 2=动态
    "roi_name":       ("RoiName", "ROI名称", "roi_name"),
    "slice_index":    ("ANNOTATION->SliceIndex", "SliceIndex"),        # 1-based 帧号
    # 临床标签列
    "malignancy":     ("ROI良恶性", "良恶性"),                          # 恶/良
    "lesion_loc":     ("主病灶位置", "病灶位置", "lesion_loc"),          # 混写 2左中叶
    "extrathy_ext":   ("甲状腺外侵犯", "ETE", "extrathy_ext"),           # 混写 2甲状腺包膜
    "clin_t_stage":   ("临床T分期", "T分期", "clin_t_stage"),           # 混写 1有 / 0无
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


def _parse_concept(v) -> Optional[int]:
    """CBM 概念(成分/回声/形状/边缘)→ 0-based 类索引 = 数字前缀-1。无前缀→None。

    ⚠️ 编码映射待真实 xlsx 复核:此处仅做"数字前缀-1"的通用转换,真实列若非此编码
       需在此调整。缺列/空 → 上游置 valid=0,masked-loss 跳过,不喂错标签。
    """
    d = _digit_prefix(v)
    return None if d is None or d < 1 else d - 1


# ── xlsx 解析 ───────────────────────────────────────────────────────────────
def _resolve_columns(columns) -> Dict[str, Optional[str]]:
    """逻辑字段 → DataFrame 实际列名(候选里第一个存在的);找不到记 None。"""
    cols = list(columns)
    colmap: Dict[str, Optional[str]] = {}
    for logical, aliases in _FIELD_ALIASES.items():
        colmap[logical] = next((a for a in aliases if a in cols), None)
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
        RuntimeError: pandas 未安装。
        KeyError:     关键列(patient_id / image_property / roi_name)全候选名皆缺失
                      —— 宁可显式报错,不静默产出空样本集。
    """
    if not HAS_PANDAS:
        raise RuntimeError("pandas 未安装,无法读 .xlsx 官方标注表")
    import pandas as pd
    # dtype=str 保留 `2左中叶` 原样,避免 pandas 把纯数字列转成 float 引入 `2.0`。
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype=str)
    colmap = _resolve_columns(df.columns)

    for key in ("patient_id", "image_property", "roi_name"):
        if colmap[key] is None:
            raise KeyError(
                f"官方标注表缺关键列 `{key}`(候选名 {_FIELD_ALIASES[key]} 均未命中)。"
                f"表头实为 {list(df.columns)} —— 请据真实表更新 _FIELD_ALIASES。"
            )

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
    """按 RoiName 定位 mask 文件:M→`M_*_mask.nii.gz`、病灶N→`病灶N_*_mask.nii.gz`。

    用 glob 而非硬拼数字(定稿写 `病灶N_N_mask.nii.gz`,但第二段编号不完全确定),
    命中第一个;找不到返回 None。
    """
    hits = sorted(glob.glob(os.path.join(img_dir, f"{_cell_str(roi_name)}_*_mask.nii.gz")))
    return hits[0] if hits else None


def _build_labels(rec: Dict, patient_id: str) -> Dict:
    """从一条 ROI record 构造 labels dict(同 ThyroidDataset 键 + 三新维度 + valid_*)。

    masked-loss 判据:每个任务的 `valid_{task}` = 该临床列**非空**才 1。空 → 0 + 占位值,
    下游 masked-loss 零梯度(定稿 §2.2:判据是列是否为空,不是是否 M 行)。
    """
    labels: Dict = {"patient_id": patient_id}

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

    # CBM 概念(成分/回声/形状/边缘):列名/编码未锁前,命中列则按数字前缀-1,否则 valid=0
    for task in _CONCEPT_TASKS:
        val = _parse_concept(rec.get(task))
        labels[task] = 0 if val is None else val
        labels[f"valid_{task}"] = int(val is not None)
    # foci 为 4 维多标签,官方列/编码未定 → 占位零 + valid=0(masked-loss 跳过)
    labels["foci"] = torch.zeros(4, dtype=torch.float32)
    labels["valid_foci"] = 0

    return labels


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
    records, _ = parse_annotation_xlsx(xlsx_path, sheet_name)

    by_patient: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        pid = _cell_str(rec.get("patient_id"))
        if pid:
            by_patient[pid].append(rec)

    samples: List[Dict] = []
    for pid, recs in by_patient.items():
        # 患者级 fallback:某些行 图像性质 可能空,用该患者任一非空值兜底(静动互斥,定稿 §1.3)。
        patient_prop = next((_digit_prefix(r.get("image_property"))
                             for r in recs if not _is_empty(r.get("image_property"))), None)

        for rec in recs:
            roi = _cell_str(rec.get("roi_name"))
            if not _is_lesion_roi(roi):
                continue  # 关键帧/开始/结束:仅坐标,不入分类/分割样本

            # 图像性质是**逐行**语义(定稿 §4.2 项4:该行走 Col15 还是 PatientId),
            # 优先按本行值分派;本行空则回退患者级。
            rp = _digit_prefix(rec.get("image_property"))
            is_dynamic = ((rp if rp is not None else patient_prop) == 2)

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
            labels = _build_labels(rec, pid)
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
    frames, idxs, _, err = load_nifti_frames(path, max_frames=-1)
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
    frames, idxs, _, err = load_nifti_frames(path, max_frames=-1)
    if err or not frames:
        raise RuntimeError(f"读动图失败 {path}: {err}")
    pos = idxs.index(frame_index) if frame_index in idxs else frame_index
    return np.asarray(frames[pos])


# ── Dataset ────────────────────────────────────────────────────────────────
class OfficialThyroidDataset(Dataset):
    """官方数据集。输出 `(img[1,H,W]f32∈[0,1], mask[1,H,W]f32, labels)`,同 ThyroidDataset。

    labels 含现有键(composition/echogenicity/shape/margin/foci/malignancy/patient_id)
    + 三新维度(lesion_loc/extrathy_ext/clin_t_stage)+ 全部 `valid_{task}`/`valid_mask`。
    三新维度键"只产出、暂不消费"——train_step/损失接线在 Phase 2(当前忽略额外键,无害)。

    ⚠️ 性能:动态样本每次 __getitem__ 重解码整段 .nii.gz(影像+mask)。正确性优先,
       Phase 4 训练前若成瓶颈再做帧缓存/预抽取。
    """

    def __init__(self, samples: List[Dict], img_size: int = 512, transform=None):
        self.samples = samples
        self.img_size = img_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_img(self, img: np.ndarray) -> np.ndarray:
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size))
        return img

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.shape[:2] != (self.img_size, self.img_size):
            mask = cv2.resize(
                mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST
            )
        return mask

    def __getitem__(self, idx):
        s = self.samples[idx]

        if s["is_dynamic"]:
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

        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        mask_t = torch.tensor((mask > 0).astype(np.float32), dtype=torch.float32).unsqueeze(0)
        # labels 浅拷贝 + foci 单独 clone:foci 是 _build_labels 建的**共享**零 tensor,
        # 浅拷贝会让多样本/多帧共享同一对象,下游(Phase 2 消费 foci)若原地改会跨样本
        # 污染,故克隆隔离。其余值均为不可变标量,浅拷贝安全。
        labels = dict(s["labels"])
        labels["foci"] = labels["foci"].clone()
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


def build_official_datasets(
    root: str,
    xlsx_path: str,
    img_size: int = 512,
    val_frac: float = 0.2,
    seed: int = 42,
    train_transform=None,
    sheet_name: str = "ROI级别",
) -> Tuple["OfficialThyroidDataset", "OfficialThyroidDataset"]:
    """便捷入口:构样本 → 患者分组切分 → 建 train/val Dataset。返回 (train_ds, val_ds)。

    分组切分复用 `patient_grouped_split`(按 patient_key 分组、label 分层),防同患者
    跨集泄漏(rules.md §5)。val 集不加增强。DataLoader/Sampler 由调用方按训练脚本组装。
    """
    samples = build_official_samples(root, xlsx_path, sheet_name)
    if not samples:
        raise RuntimeError(f"未在 {root} + {xlsx_path} 构造出任何样本,检查路径/表头")
    train_s, val_s = patient_grouped_split(samples, val_frac=val_frac, seed=seed)
    train_ds = OfficialThyroidDataset(train_s, img_size=img_size, transform=train_transform)
    val_ds = OfficialThyroidDataset(val_s, img_size=img_size, transform=None)
    return train_ds, val_ds
