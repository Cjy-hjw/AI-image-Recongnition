"""ThyUS2Path 病理良恶性数据集(figshare 27021604)。

用途:官方数据(2026-08-01)到位前,用真实病理金标准(术后组织学 histo_label)
**预训练 malignancy 分支 backbone**。原生 JPG,天然抗官方 JPG 域偏移。

数据形态(A800 已就地解压至 data/27021604_extract/):
    batch1_image/  batch1_image.csv        patient_name,path   (患者→图像文件名)
                   batch1_image_label.csv  patient_name,histo_label  (0=良,1=恶)
                   dataset/*.Jpg           实际图像(命名 {patient}_{seq}.Jpg)
    batch2_image/  同结构

关键约束(防数据泄漏,P0):
- 病理标签是**患者级**(术后一个病理结论),必须**广播到该患者所有图像**;
- train/val 划分必须**按患者分组**——同一患者的图像绝不能跨 train/val,
  否则同患者近似帧泄漏会虚高 AUC(超声连续帧高度相似)。
- batch1/batch2 患者 ID 可能数值重叠但**是不同人**,故内部 ID 加 batch 前缀
  (b1_/b2_)隔离,既防误并同 ID、又防跨 batch 泄漏。

输出与 ThyroidDataset(loader.py)完全同构:(img_tensor, mask_tensor, labels),
labels 只置 valid_malignancy=1,其余 valid_*=0 与 valid_mask=0——下游 masked
loss / label_validity 自动对无标签任务零梯度,无需改损失,可直接接现有训练循环。
"""

import os
import csv as _csv
from collections import defaultdict
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# 每个 batch 目录下 CSV / 图像子目录的命名约定(见模块 docstring)。
_BATCH_DIRS = ("batch1_image", "batch2_image")
_IMG_SUBDIR = "dataset"

# labels dict 中除 malignancy 外的概念任务(病理数据无这些标注 → valid=0)。
_OTHER_CONCEPT_TASKS = ("composition", "echogenicity", "shape", "margin", "foci")


def _read_csv_rows(path: str) -> List[List[str]]:
    """读 CSV(跳过表头),返回去空白的字段行。文件不存在则抛错。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"病理数据 CSV 缺失: {path}")
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = _csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:  # 表头 patient_name,path / patient_name,histo_label
                continue
            row = [c.strip() for c in row if c.strip() != ""]
            if len(row) >= 2:
                rows.append(row)
    return rows


def build_pathology_samples(root: str) -> List[Dict]:
    """扫描 root 下所有 batch,把患者级 histo_label 广播到图像级。

    Returns: list of dict {img_path, label(int 0/1), patient_key(str, 带 batch 前缀)}
    校验:image.csv 引用的患者必须在 label.csv 中有标签、图像文件必须真实存在,
    否则跳过并计数(返回前打印丢弃统计,避免静默吞样本)。
    """
    samples: List[Dict] = []
    n_missing_label = 0
    n_missing_file = 0

    # 跨 batch 患者编号说明:实测 batch1 patient_name 为 0..600、batch2 为 0..240,
    # 二者数值**完全重叠** —— 证实 patient_name 是「各 batch 内部顺序编号」而非全局
    # 患者 ID(两批是不同病人各自从 0 编号)。因此 b1_/b2_ 前缀隔离正是应对此情形的
    # 正确设计:加前缀后 b1_0 与 b2_0 被正确视为两个人,分组划分不会泄漏。
    # 此处仅打印重叠规模供人工确认「确系不同队列」,不再 abort(前缀已消除风险)。
    _batch_patients: Dict[str, set] = {}
    for batch_dir in _BATCH_DIRS:
        base = os.path.join(root, batch_dir)
        lab_csv = os.path.join(base, f"{batch_dir}_label.csv")
        if os.path.exists(lab_csv):
            _batch_patients[batch_dir] = {r[0] for r in _read_csv_rows(lab_csv)}
    if len(_batch_patients) == 2:
        b1p = _batch_patients.get("batch1_image", set())
        b2p = _batch_patients.get("batch2_image", set())
        overlap = b1p & b2p
        if overlap:
            print(
                f"[PathologyDataset] batch1/batch2 编号重叠 {len(overlap)} 个"
                f"(b1={len(b1p)} b2={len(b2p)});已按 b1_/b2_ 前缀隔离为不同患者,"
                "分组划分无泄漏。若二者实为同一队列请勿混用。",
                flush=True,
            )

    for batch_dir in _BATCH_DIRS:
        base = os.path.join(root, batch_dir)
        if not os.path.isdir(base):
            continue
        prefix = "b1_" if batch_dir.startswith("batch1") else "b2_"

        # 患者 → 标签
        label_rows = _read_csv_rows(os.path.join(base, f"{batch_dir}_label.csv"))
        patient_label: Dict[str, int] = {}
        for pname, hlabel in ((r[0], r[1]) for r in label_rows):
            hl = int(hlabel)
            if pname in patient_label and patient_label[pname] != hl:
                print(
                    f"[PathologyDataset] 警告:{batch_dir} 患者 {pname} 标签冲突 "
                    f"({patient_label[pname]}→{hl}),保留后者",
                    flush=True,
                )
            patient_label[pname] = hl

        # 患者 → 图像文件:直接扫描 dataset/ 目录,从文件名首段提取 patient_id。
        # 不依赖 image.csv 的文件枚举——batch1 的 image.csv 有大量序号跳空导致
        # ~43% 文件名对不上磁盘实际文件;直接扫目录可用到所有真实存在的图像。
        # 文件名格式: batch1 = {patient}_{seq}.Jpg
        #              batch2 = {patient}_{seq}_{timestamp}.Jpg
        # 两者首段(按 '_' 分割第 0 位)均为 patient_name。
        img_dir = os.path.join(base, _IMG_SUBDIR)
        if not os.path.isdir(img_dir):
            print(f"[PathologyDataset] {img_dir} 不存在,跳过", flush=True)
            continue
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(".jpg"):
                continue
            pname = fname.split("_")[0]
            if pname not in patient_label:
                n_missing_label += 1
                continue
            samples.append({
                "img_path": os.path.join(img_dir, fname),
                "label": patient_label[pname],
                "patient_key": prefix + pname,
            })

    if n_missing_label or n_missing_file:
        print(
            f"[PathologyDataset] 丢弃 {n_missing_label} 张(患者无标签) + "
            f"{n_missing_file} 张(文件缺失);保留 {len(samples)} 张",
            flush=True,
        )
    if not samples:
        raise RuntimeError(f"未在 {root} 找到任何有效病理样本,检查路径/解压完整性")
    return samples


def patient_grouped_split(
    samples: List[Dict],
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """按患者分组 + 按患者标签分层的 train/val 划分(防同患者跨集泄漏)。

    先把患者按 label 分层,各层内独立随机抽 val_frac 比例的**患者**入 val,
    再把患者的所有图像整体划入对应集合。保证:
      1. 同一患者所有图像同侧(无泄漏);
      2. train/val 的良恶性患者比例大致一致(分层)。
    """
    assert 0.0 < val_frac < 1.0, f"val_frac 须在 (0,1),当前 {val_frac}"

    # 患者 → 其 label(同患者所有图像 label 一致,取其一)
    patient_to_label: Dict[str, int] = {}
    patient_to_imgs: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        patient_to_label[s["patient_key"]] = s["label"]
        patient_to_imgs[s["patient_key"]].append(s)

    rng = np.random.RandomState(seed)
    val_patients = set()
    # 按 label 分层抽患者
    by_label: Dict[int, List[str]] = defaultdict(list)
    for p, lab in patient_to_label.items():
        by_label[lab].append(p)
    for lab, plist in by_label.items():
        plist = sorted(plist)  # 固定顺序,保证 seed 可复现
        rng.shuffle(plist)
        # n_val 上界压到 len-1:保证该 label 类**至少留 1 患者给 train**,否则
        # 单患者类(len==1)会被 max(1,·) 整个挖进 val→train 该类清空→class-weight
        # 除零/单类 batch 崩 loss。len==1 时 min(max(1,1), 0)=0,唯一患者留 train
        # (train 不缺类;val 缺一个稀有类可接受——该类本就无法有意义评估)。
        # len>=2 时 max(1,·) 保 val 至少 1;len=2→1,len=10&frac=.2→2。
        n_val = min(max(1, int(round(len(plist) * val_frac))), len(plist) - 1)
        val_patients.update(plist[:n_val])

    train, val = [], []
    for p, imgs in patient_to_imgs.items():
        (val if p in val_patients else train).extend(imgs)
    # 防泄漏自检(QA_VP §1.4 防火墙):同患者所有图像应严格同侧,train/val 患者
    # 互斥。此断言防未来有人误改成随机划分而不自知(见 memory/patient-leakage)。
    train_pids = {s["patient_key"] for s in train}
    val_pids = {s["patient_key"] for s in val}
    assert not (train_pids & val_pids), \
        f"患者跨 train/val 泄漏: {train_pids & val_pids}"
    return train, val


class PathologyDataset(Dataset):
    """病理良恶性数据集。返回 (img[1,H,W]f32∈[0,1], mask[1,H,W]全零, labels)。

    labels 结构与 ThyroidDataset 对齐,仅 malignancy 有效:
        malignancy: int(0/1)
        valid_malignancy: 1     # 参与 loss
        valid_mask: 0           # 无分割标注,防污染 seg loss
        valid_{其余概念}: 0      # 无 TI-RADS 标注,防污染 concept loss
        composition/echogenicity/shape/margin: 0 (占位)
        foci: tensor[4] 零 (占位)
    """

    def __init__(self, samples: List[Dict], img_size: int = 512, transform=None):
        self.samples = samples
        self.img_size = img_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["img_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"图像读取失败: {s['img_path']}")
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))

        # 无分割标注 → 全零 mask(uint8,供 transform 与 ThyroidDataset 一致处理)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # 增强在归一化前(img 仍是 uint8,满足 JPEG 增强断言)
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        labels: Dict = {
            "composition": 0,
            "echogenicity": 0,
            "shape": 0,
            "margin": 0,
            "foci": torch.zeros(4, dtype=torch.float32),
            "malignancy": int(s["label"]),
            "patient_id": s["patient_key"],
            # validity:仅 malignancy 有效,其余零梯度
            "valid_malignancy": 1,
            "valid_mask": 0,
        }
        for t in _OTHER_CONCEPT_TASKS:
            labels[f"valid_{t}"] = 0
        return img_tensor, mask_tensor, labels

    def class_counts(self) -> Dict[int, int]:
        """返回 {0: n_benign, 1: n_malignant},供 CE class-weight / 采样用。"""
        c = defaultdict(int)
        for s in self.samples:
            c[int(s["label"])] += 1
        return dict(c)
