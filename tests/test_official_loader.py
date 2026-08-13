"""official_loader.py 单测：合成 mini 官方数据集(运行时生成 .dcm/.nii.gz/xlsx,
不入库二进制),端到端验证 xlsx 混写解析 / 静动分派 / RoiName 对 mask / 输出契约 /
患者分组无泄漏。

依赖 pydicom / nibabel / pandas / openpyxl,缺任一则 skip(与容器/CI 一致,本地可能缺)。
"""
import os

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
nib = pytest.importorskip("nibabel")
pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

from src.dataset import official_loader as O
from src.inference.nifti_io import save_binary_mask


# ── 合成 fixture 生成器 ─────────────────────────────────────────────────────
def _make_dcm(path: str, arr: np.ndarray):
    """写最小可读 MONOCHROME2 8bit DICOM(pixel_array 能读回)。"""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    ds = Dataset()
    ds.Rows, ds.Columns = int(arr.shape[0]), int(arr.shape[1])
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.PixelData = arr.astype(np.uint8).tobytes()

    fm = FileMetaDataset()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.MediaStorageSOPClassUID = generate_uid()
    fm.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta = fm
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        ds.save_as(path, write_like_original=False)          # pydicom 2.x
    except TypeError:                                        # pydicom 3.x
        ds.save_as(path, enforce_file_format=True)


def _save_image_nifti_3d(path: str, vol_nhw: np.ndarray):
    """把 (n_frames,H,W) 存成 NIfTI,使 load_nifti_frames 读回同形。

    load_nifti_frames 对 3D 做 transpose(2,1,0),故存前先逆置成 (W,H,n)。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.transpose(vol_nhw.astype(np.uint8), (2, 1, 0))
    nib.save(nib.Nifti1Image(np.ascontiguousarray(data), np.eye(4)), path)


@pytest.fixture
def mini_root(tmp_path):
    """构造 2 静态患者 + 1 动态患者(2 个非零帧)的合成官方数据集 + mock xlsx。

    返回 (root_dir, xlsx_path)。
    """
    root = tmp_path / "annotation"
    H = W = 64

    # P1 静态:M 病灶,全标注
    p1_dir = root / "P1" / "S1"
    _make_dcm(str(p1_dir / "img1.dcm"), np.full((H, W), 120, np.uint8))
    m1 = np.zeros((H, W), np.uint8); m1[20:40, 20:40] = 1
    save_binary_mask(m1, str(p1_dir / "M_1_mask.nii.gz"))

    # P2 静态:M 病灶,ETE 列留空(→ valid_extrathy_ext=0)
    p2_dir = root / "P2" / "S2"
    _make_dcm(str(p2_dir / "img2.dcm"), np.full((H, W), 80, np.uint8))
    m2 = np.zeros((H, W), np.uint8); m2[10:30, 30:50] = 1
    save_binary_mask(m2, str(p2_dir / "M_1_mask.nii.gz"))

    # P3 动态:5 帧,mask 仅第 1、3 帧非零 → 2 个样本
    p3_dir = root / "P3" / "P3"
    vol = np.stack([np.full((H, W), 40 + 20 * i, np.uint8) for i in range(5)])
    _save_image_nifti_3d(str(p3_dir / "P3.nii.gz"), vol)
    dmask = np.zeros((5, H, W), np.uint8)
    dmask[1, 15:35, 15:35] = 1
    dmask[3, 25:45, 25:45] = 1
    save_binary_mask(dmask, str(p3_dir / "M_1_mask.nii.gz"))

    # mock xlsx(ROI级别 sheet),列名对齐 _FIELD_ALIASES;含 1 个应被跳过的清晰帧行
    rows = [
        # PatientId, SeriesUid, 图像性质, RoiName, 主病灶位置, 甲状腺外侵犯, 临床T分期, ROI良恶性, ANNOTATION->SliceIndex
        ["P1", "S1", "1", "M", "2左中叶", "2甲状腺包膜", "1有", "恶", ""],
        ["P2", "S2", "1", "M", "5右中叶", "", "0无", "良", ""],
        ["P3", "", "2", "M", "7峡部", "1无", "1有", "恶", ""],
        ["P3", "", "2", "清晰帧1", "", "", "", "", "2"],   # 应被跳过(非病灶行)
    ]
    df = pd.DataFrame(rows, columns=[
        "PatientId", "SeriesUid", "图像性质", "RoiName", "主病灶位置",
        "甲状腺外侵犯", "临床T分期", "ROI良恶性", "ANNOTATION->SliceIndex",
    ])
    xlsx = tmp_path / "labels.xlsx"
    df.to_excel(xlsx, sheet_name="ROI级别", index=False)
    return str(root), str(xlsx)


# ── 测试 ────────────────────────────────────────────────────────────────────
def test_parse_xlsx_mixed_cells(mini_root):
    """混写单元格按 prediction_json 权威常量解析;空列→valid=0。"""
    _, xlsx = mini_root
    records, colmap = O.parse_annotation_xlsx(xlsx)
    assert colmap["patient_id"] == "PatientId"
    assert len(records) == 4

    p1 = records[0]
    lab = O._build_labels(p1, "P1")
    assert lab["lesion_loc"] == 1 and lab["valid_lesion_loc"] == 1        # 2左中叶→idx1
    assert lab["extrathy_ext"] == 1 and lab["valid_extrathy_ext"] == 1    # 码2-1=1
    assert lab["clin_t_stage"] == 1 and lab["malignancy"] == 1

    p2 = records[1]
    lab2 = O._build_labels(p2, "P2")
    assert lab2["valid_extrathy_ext"] == 0                                # ETE 列空
    assert lab2["lesion_loc"] == 4 and lab2["malignancy"] == 0            # 5右中叶→idx4, 良


def test_build_samples_dispatch_and_skip(mini_root):
    """静动分派正确、清晰帧行被跳过、动态按非零帧展开。"""
    root, xlsx = mini_root
    samples = O.build_official_samples(root, xlsx)
    # P1 静 + P2 静 + P3 动(2 非零帧) = 4;清晰帧行不计
    assert len(samples) == 4
    static = [s for s in samples if not s["is_dynamic"]]
    dynamic = [s for s in samples if s["is_dynamic"]]
    assert len(static) == 2 and len(dynamic) == 2
    assert sorted(s["frame_index"] for s in dynamic) == [1, 3]           # mask 非零帧
    # 静态样本 img 指向 .dcm、mask 靠 RoiName 命中
    assert all(s["img_path"].endswith(".dcm") for s in static)
    assert all(s["mask_path"] and s["mask_path"].endswith("M_1_mask.nii.gz") for s in static)


def test_output_contract_static(mini_root):
    """静态 __getitem__ 输出契约:img[1,S,S]∈[0,1]、mask 二值、labels 含 valid_*。"""
    root, xlsx = mini_root
    samples = O.build_official_samples(root, xlsx)
    ds = O.OfficialThyroidDataset(samples, img_size=64)
    static_idx = next(i for i, s in enumerate(samples) if not s["is_dynamic"])
    img, mask, labels = ds[static_idx]
    assert tuple(img.shape) == (1, 64, 64)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert tuple(mask.shape) == (1, 64, 64)
    assert set(np.unique(mask.numpy())).issubset({0.0, 1.0})
    for k in ("valid_malignancy", "valid_lesion_loc", "valid_extrathy_ext",
              "valid_clin_t_stage", "valid_mask"):
        assert k in labels
    assert labels["valid_mask"] == 1


def test_output_contract_dynamic(mini_root):
    """动态 __getitem__ 取到正确非零帧,契约同静态。"""
    root, xlsx = mini_root
    samples = O.build_official_samples(root, xlsx)
    ds = O.OfficialThyroidDataset(samples, img_size=64)
    dyn_idx = next(i for i, s in enumerate(samples) if s["is_dynamic"])
    img, mask, labels = ds[dyn_idx]
    assert tuple(img.shape) == (1, 64, 64)
    assert tuple(mask.shape) == (1, 64, 64)
    assert mask.numpy().sum() > 0                     # 非零帧,mask 有前景
    assert labels["lesion_loc"] == 6                  # 7峡部→idx6


def test_patient_grouped_split_no_leak(mini_root):
    """患者分组切分:train/val 患者无交集(rules.md §5 铁律)。"""
    root, xlsx = mini_root
    samples = O.build_official_samples(root, xlsx)
    train, val = O.patient_grouped_split(samples, val_frac=0.34, seed=0)
    train_pids = {s["patient_key"] for s in train}
    val_pids = {s["patient_key"] for s in val}
    assert train_pids & val_pids == set()
    assert train_pids | val_pids == {"P1", "P2", "P3"}


def test_ete_parse_edge_cases():
    """ETE 解析:纯中文回退、越界数字不产生负类索引、1无→码1(sonnet 审查 A)、
    未知但存在的串→None不静默当"无"(审查 S3)。"""
    assert O._parse_ete_code("甲状腺包膜") == 2      # 纯中文→权威映射
    assert O._parse_ete_code("2甲状腺包膜") == 2     # 混写→数字前缀
    assert O._parse_ete_code("1无") == 1
    assert O._parse_ete_code("4血管") == 4
    assert O._parse_ete_code("") is None
    # 越界数字前缀(0-based 风格 `0无`)不得直接返回 0(否则 码-1=-1 非法类索引):回退→1(无)
    assert O._parse_ete_code("0无") == 1
    # 审查 S3:未知但存在的 ETE 串(真标注,不在权威表)→ None,绝不静默回退"无侵犯"valid=1
    assert O._parse_ete_code("侵犯前纵膈") is None
    assert O._parse_ete_code("3毗邻软组织") == 3     # 混写数字前缀仍优先
    # 审查 S3 Finding1(/code-review):输出侧归并项 NA/UNK、无病灶（阴性）在训练侧是
    # "未知/阴性",必须 → None(valid=0),不得当硬"无侵犯"负例(与 S4 阴性→valid=0 一致)
    assert O._parse_ete_code("NA/UNK") is None
    assert O._parse_ete_code("无病灶（阴性）") is None
    assert O._parse_ete_code("无病灶(阴性)") is None
    assert O._parse_ete_code("1无病灶（阴性）") is None   # 带前缀但 0/非1..4 也不放行阴性
    # safe-drop:非规范措辞(含关键词但非规范短标签)当前 →None(欠采不误标),digit 前缀路径兜常态
    assert O._parse_ete_code("毗邻软组织侵犯") is None    # @云TODO 真实措辞到位重锁
    assert O._parse_ete_code("3毗邻软组织侵犯") == 3      # 带前缀仍稳
    # 上游 _build_labels:未知 ETE → valid=0,不进侵犯损失
    lab = O._build_labels({"extrathy_ext": "侵犯前纵膈"}, "P")
    assert lab["valid_extrathy_ext"] == 0
    lab2 = O._build_labels({"extrathy_ext": "NA/UNK"}, "P")
    assert lab2["valid_extrathy_ext"] == 0


def test_location_pure_chinese_and_negative():
    """位置:纯中文回退命中、阴性/未知→valid=0 不参与位置损失(绝不默认 0=左上叶)。"""
    lab = O._build_labels({"lesion_loc": "左中叶"}, "P")   # 无数字前缀→中文回退
    assert lab["lesion_loc"] == 1 and lab["valid_lesion_loc"] == 1
    neg = O._build_labels({"lesion_loc": "无（阴性）"}, "P")
    assert neg["valid_lesion_loc"] == 0
    unk = O._build_labels({"lesion_loc": ""}, "P")
    assert unk["valid_lesion_loc"] == 0


def test_labels_carry_is_main_and_lesion_id(mini_root):
    """`is_main`/`lesion_id` 随 labels 产出:验证侧按官方口径筛主病灶 + 按病灶聚合。"""
    root, xlsx = mini_root
    samples = O.build_official_samples(root, xlsx)
    for s in samples:
        lab = s["labels"]
        assert lab["is_main"] == 1                      # 合成集只有 M 主病灶行
        assert isinstance(lab["lesion_id"], str) and lab["lesion_id"]
    # 同一动态病灶的多帧共享同一 lesion_id(病灶级聚合的前提:帧维度不得各成一样本)
    dyn = [s for s in samples if s["is_dynamic"]]
    assert len({s["labels"]["lesion_id"] for s in dyn}) == 1
    # 不同患者的静态病灶 lesion_id 必须互异(否则跨患者被误并成一个病灶)
    static = [s for s in samples if not s["is_dynamic"]]
    assert len({s["labels"]["lesion_id"] for s in static}) == len(static)
    # 次要病灶(RoiName=M1)标 is_main=0
    assert O._build_labels({"roi_name": "M1"}, "P")["is_main"] == 0


def test_hflip_syncs_location_label(mini_root):
    """hflip_prob=1.0 时图像/mask 镜像**且**位置标签同步置换(左右反了 loss 照常降)。"""
    root, xlsx = mini_root
    samples = O.build_official_samples(root, xlsx)
    idx = next(i for i, s in enumerate(samples)
               if not s["is_dynamic"] and s["labels"]["lesion_loc"] == 1)

    plain = O.OfficialThyroidDataset(samples, img_size=64)
    flipped = O.OfficialThyroidDataset(samples, img_size=64, hflip_prob=1.0, seed=0)
    img0, mask0, lab0 = plain[idx]
    img1, mask1, lab1 = flipped[idx]

    assert lab0["lesion_loc"] == 1 and lab1["lesion_loc"] == 4      # 左中叶→右中叶
    assert np.allclose(img1.numpy(), img0.numpy()[:, :, ::-1])
    assert np.allclose(mask1.numpy(), mask0.numpy()[:, :, ::-1])
    # 原样本 labels 不被就地改写(dict(s["labels"]) 浅拷贝隔离),重复取样不累积翻转
    assert samples[idx]["labels"]["lesion_loc"] == 1
    assert flipped[idx][2]["lesion_loc"] == 4


def test_hflip_skips_invalid_location(mini_root):
    """valid_lesion_loc=0 时不得置换位置标签(无监督样本的占位值置换毫无意义且掩盖 bug)。"""
    root, xlsx = mini_root
    samples = O.build_official_samples(root, xlsx)
    idx = next(i for i, s in enumerate(samples) if not s["is_dynamic"])
    samples[idx]["labels"] = dict(samples[idx]["labels"])
    samples[idx]["labels"]["valid_lesion_loc"] = 0
    orig = samples[idx]["labels"]["lesion_loc"]

    ds = O.OfficialThyroidDataset(samples, img_size=64, hflip_prob=1.0, seed=0)
    img, _, lab = ds[idx]
    assert lab["lesion_loc"] == orig                    # 标签不动
    plain = O.OfficialThyroidDataset(samples, img_size=64)
    assert np.allclose(img.numpy(), plain[idx][0].numpy()[:, :, ::-1])   # 图像仍翻


def test_val_dataset_has_no_hflip(mini_root):
    """build_official_datasets:train 吃 hflip,val 必须原始朝向(翻转会改位置标签)。"""
    root, xlsx = mini_root
    train_ds, val_ds = O.build_official_datasets(
        root, xlsx, img_size=64, val_frac=0.34, seed=0, hflip_prob=0.5
    )
    assert train_ds.hflip_prob == 0.5
    assert val_ds.hflip_prob == 0.0 and val_ds.transform is None


def test_build_datasets_does_not_freeze_hflip_rng(mini_root):
    """🔴 `build_official_datasets` 的 `seed` **只喂切分**,不得传给 Dataset 的翻转 RNG。

    曾经传了(`OfficialThyroidDataset(..., seed=seed)`),后果是**多样性塌缩**:持久
    `random.Random(seed)` 实例被 pickle 复制进每个 worker(状态相同),且
    persistent_workers=False 时每 epoch 重建 worker 使序列重置 —— 实测三个 epoch 的
    翻转序列**逐位相同**,增强等于白做。不报错、loss 照常降,只有靠这条断言能发现。

    生产必须 `_rng is None` → 走全局 `random`(由 DataLoader 的 worker_init_fn 按
    `torch.initial_seed()` 播种,torch 每 epoch 重抽 base_seed → 每 worker/每 epoch 都不同)。
    """
    root, xlsx = mini_root
    train_ds, val_ds = O.build_official_datasets(
        root, xlsx, img_size=64, val_frac=0.34, seed=0, hflip_prob=0.5
    )
    assert train_ds._rng is None, (
        "train_ds 持有固定种子的 RNG 实例 —— 翻转序列会跨 worker/跨 epoch 完全相同。"
        "seed 只能喂 patient_grouped_split,不能传给 Dataset。"
    )
    assert val_ds._rng is None
    # 显式传 seed 仍走独立 RNG(测试确定性路径不受影响)
    assert O.OfficialThyroidDataset([], seed=0)._rng is not None


def test_dynamic_no_mask_skipped(mini_root):
    """动态病灶无 mask → 无法定位标注帧,整例跳过(不产出坏样本;sonnet 审查覆盖缺口)。"""
    root, xlsx = mini_root
    os.remove(os.path.join(root, "P3", "P3", "M_1_mask.nii.gz"))
    samples = O.build_official_samples(root, xlsx)
    assert all(not s["is_dynamic"] for s in samples)       # P3 动态被跳过
    assert len(samples) == 2                               # 仅剩 P1/P2 静态
