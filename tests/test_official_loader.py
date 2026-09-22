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

import torch

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
        # PatientId, SeriesUid, 图像性质, RoiName, 主病灶位置, 甲状腺外侵犯, 临床T分期, ROI良恶性, ANNOTATION->SliceIndex, ROI-TIRADS
        ["P1", "S1", "1", "M", "2左中叶", "2甲状腺包膜", "1有", "恶", "", "V类"],
        ["P2", "S2", "1", "M", "5右中叶", "", "0无", "良", "", "IV类"],
        ["P3", "", "2", "M", "7峡部", "1无", "1有", "恶", "", ""],
        ["P3", "", "2", "清晰帧1", "", "", "", "", "2", ""],   # 应被跳过(非病灶行)
    ]
    df = pd.DataFrame(rows, columns=[
        "PatientId", "SeriesUid", "图像性质", "RoiName", "主病灶位置",
        "甲状腺外侵犯", "临床T分期", "ROI良恶性", "ANNOTATION->SliceIndex", "ROI-TIRADS",
    ])
    xlsx = tmp_path / "labels.xlsx"
    # startrow=1:真实官方 xlsx 是合并单元格表头,列名在第2行(0-based row=1)、
    # 第1行是分组标题——parse_annotation_xlsx 现在按 header=1 读(2026-08-13 云端
    # 真实数据实测确认),mock 必须复刻同一双行结构,否则测的是已不存在的单行表头格式。
    df.to_excel(xlsx, sheet_name="ROI级别", index=False, startrow=1)
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
    assert lab["tirads"] == 4 and lab["valid_tirads"] == 1                # V类→码5-1=4

    p2 = records[1]
    lab2 = O._build_labels(p2, "P2")
    assert lab2["valid_extrathy_ext"] == 0                                # ETE 列空
    assert lab2["lesion_loc"] == 4 and lab2["malignancy"] == 0            # 5右中叶→idx4, 良
    assert lab2["tirads"] == 3 and lab2["valid_tirads"] == 1              # IV类→码4-1=3

    p3 = records[2]
    lab3 = O._build_labels(p3, "P3")
    assert lab3["valid_tirads"] == 0                                      # TI-RADS 列空


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


def test_tirads_parse_edge_cases():
    """TI-RADS:罗马数字精确查表(禁子串误判 IV/III→I)、混写数字前缀、
    Unicode 罗马数字字符、越界/空/未知→None。"""
    assert O._parse_tirads("V类") == 5
    assert O._parse_tirads("5 V类") == 5          # 混写:数字前缀优先
    assert O._parse_tirads("IV类") == 4           # 🔴 不得被子串匹配误判成 I(=1)
    assert O._parse_tirads("III类") == 3          # 🔴 不得被子串匹配误判成 I 或 II
    assert O._parse_tirads("II类") == 2
    assert O._parse_tirads("I类") == 1
    assert O._parse_tirads("Ⅳ类") == 4            # Unicode 罗马数字字符(U+2163)
    assert O._parse_tirads("Ⅴ类") == 5            # Unicode 罗马数字字符(U+2164)
    assert O._parse_tirads("") is None
    assert O._parse_tirads(None) is None
    assert O._parse_tirads("NA/UNK") is None
    assert O._parse_tirads("未评估") is None       # 未知措辞 → None(不静默当某一级)
    # 🔴 越界数字前缀分叉(branch-2 + branch-3 不对称):`脱前缀再罗马查表` 只处理
    # 归一化后的串,"6类"/"0类" 不含罗马字符 → get(None);detach 了整个数字。
    # 分支1: `6类` → `6 V 类` 混写(`6 V...`)会返回 6 → valid=0(None over clamps)。
    assert O._parse_tirads("6类") is None          # 越界数字前缀
    assert O._parse_tirads("0类") is None          # 越界数字前缀(0 非法码)
    # 两段数字前缀 + 无空格混写(真实边界值):`4IV类` → 前缀 4 → 码 4,与 `5 V类`
    # 混写分叉到分支3(空格版剥前缀→罗马查表)不同——空格才是进入分支3的开关。
    assert O._parse_tirads("4IV类") == 4
    # 上游 _build_labels:TI-RADS 码 → 0-based(码-1);未知 → valid=0
    lab = O._build_labels({"tirads": "V类"}, "P")
    assert lab["tirads"] == 4 and lab["valid_tirads"] == 1
    lab2 = O._build_labels({"tirads": ""}, "P")
    assert lab2["tirads"] == 0 and lab2["valid_tirads"] == 0


def test_tirads_parse_known_gaps_pinned_by_regression():
    """TI-RADS:数字前缀分支不校验后缀、罗马数字表不含越界形态 —— 当前设计行为,钉住防漂移。

    Gate2 审查(2026-08-13,角度1-2)发现的两个边界:
    ① 数字前缀 1..5 命中即返回,不校验后缀内容(`"5 NA"`→5)——**当前对真实数据无害**:
       `_FIELD_ALIASES["tirads"]` 只映射到总分类列,该列全表实测(第12轮云端核实)永远是
       纯罗马数字、从不带数字前缀,此分支目前无真实种子会触发;若未来候选列名改指向带
       数字前缀的编号列,需重新评估是否要加后缀校验。
    ② `"VI"`(6)/`"Vi"`(大小写混排空格)不在 `_TIRADS_ROMAN_TO_CODE` 精确表内 → None,
       与"越界数字前缀→None"同一保守策略,不静默出错。
    这条测试只钉当前行为,不代表这是理想行为——若后缀校验/罗马数字表扩容,预期一起改本测试。
    """
    # ① 数字前缀命中即返回,不看后缀(已知设计缺口,当前列映射下无真实数据会触发)
    assert O._parse_tirads("5 NA") == 5
    assert O._parse_tirads("5 UNK") == 5
    # ② 罗马数字越界/非常规空格形态 → None(未知措辞的保守兜底,不臆测成某一级)
    assert O._parse_tirads("VI类") is None
    assert O._parse_tirads("Vi级") is None
    # ③ 真实官方数据 value_counts 首见 `IV类`/`V类`/`III类`/`I类` 全部是纯罗马
    #    无前缀,打印观察(不 assert,只做确认性输出)
    for v in ("IV类", "V类", "III类", "II类", "I类"):
        assert O._parse_tirads(v) is not None, f"官方 value_counts 值 {v} 解析失败"


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


def test_mask_glob_matches_both_namings(tmp_path):
    """mask 定位必须同时兼容 `M_1_mask.nii.gz`(定稿实测)与裸 `M_mask.nii.gz`。

    2026-08-13 Gate2 加固:原模式只有 `{roi}_*_mask.nii.gz`,而 fnmatch 的 `*`
    **至少吃一个字符**,结构性地匹配不上无编号命名。漏匹配是**静默失败**——
    mask_path=None → valid_mask=0 → 主病灶分割监督消失、动态整例跳过,全程不报错。
    定稿三处均写实测为带编号命名,故当前不触发;本测试钉死"两种都收"这个加固,
    防止后人"简化"回单一模式。
    """
    d = tmp_path / "case"
    d.mkdir()

    # ① 只有带段编号的命名(定稿实测形态)
    (d / "M_1_mask.nii.gz").write_bytes(b"x")
    got = O._mask_path_for_roi(str(d), "M")
    assert got is not None and got.endswith("M_1_mask.nii.gz")

    # ② 只有裸命名(无段编号)——加固前这里必然 None
    (d / "M_1_mask.nii.gz").unlink()
    (d / "M_mask.nii.gz").write_bytes(b"x")
    got = O._mask_path_for_roi(str(d), "M")
    assert got is not None and got.endswith("M_mask.nii.gz"), (
        "裸 M_mask.nii.gz 未被匹配 —— `{roi}_*_mask` 的 `*` 至少吃一个字符,"
        "必须额外 glob 一次无编号形态,否则主病灶 mask 会静默丢失。"
    )

    # ③ 两者并存 → 优先带编号那个('1' < 'm',sorted 天然靠前)
    (d / "M_1_mask.nii.gz").write_bytes(b"x")
    got = O._mask_path_for_roi(str(d), "M")
    assert got is not None and got.endswith("M_1_mask.nii.gz")

    # ④ 中文次病灶同样两种命名都认
    (d / "病灶1_mask.nii.gz").write_bytes(b"x")
    got = O._mask_path_for_roi(str(d), "病灶1")
    assert got is not None and got.endswith("病灶1_mask.nii.gz")

    # ⑤ 无任何匹配 → None(不抛异常)
    assert O._mask_path_for_roi(str(d), "病灶9") is None


# ── 列名解析:真实表头变体(2026-08-14 云端真数据实测反推)────────────────────
def test_resolve_columns_real_header_variants():
    """归一化匹配必须吃下真实 xlsx 的列名变体,否则临床列静默零监督。

    背景(必须保留,否则后人会把归一化改回精确匹配):
    2026-08-14 云端第一次真训直接 KeyError,暴露两类问题——
      ① 「图像性质」列在官方 74 列表头里**根本不存在**(`docs/8.1赛事公开资料/
         数据集概况.md` 列 0-73 全量清单),而它曾被列为必需列 → 必崩;
      ② 「甲状腺外侵犯」云端实测报告为 `甲状腺 外侵犯`(中间有空格),与数据集概况.md
         记录的无空格版矛盾。空格差异在精确匹配下**不报错**、只是 valid_ete=0 全程
         零监督,是最隐蔽的失败模式。
    故本测试钉死:临床列的空格/全角/大小写变体都必须命中。
    """
    cols = [
        "PatientId", "SeriesUid", "RoiName",
        "ROI->CLINICAL->良恶性",        # 列52
        "ROI->CLINICAL->TI-RADS",       # 列53
        "甲状腺 外侵犯",                 # 列59:云端实测带空格
        "主病灶位置",                    # 列60
        "临床T分期",                     # 列61
    ]
    cm = O._resolve_columns(cols)
    assert cm["malignancy"] == "ROI->CLINICAL->良恶性"
    assert cm["tirads"] == "ROI->CLINICAL->TI-RADS"
    assert cm["lesion_loc"] == "主病灶位置"
    assert cm["clin_t_stage"] == "临床T分期"
    assert cm["extrathy_ext"] == "甲状腺 外侵犯", (
        "带空格的『甲状腺 外侵犯』未命中 —— 归一化匹配被改回精确匹配了?"
        "这会让 ETE 头全程 valid=0 零监督且不报任何错。"
    )
    # 返回值必须是**原始列名**(含空格),否则 row[col] 取值会 KeyError
    assert cm["extrathy_ext"] in cols
    # image_property 不存在也不该抛,只是记 None(它已不是必需列)
    assert cm["image_property"] is None


def test_resolve_columns_normalizes_fullwidth_and_case():
    """全角/半角、大小写差异同样要吃下(同一字段的常见书写变体)。"""
    cm = O._resolve_columns(["ＰａｔｉｅｎｔＩｄ", "ｒｏｉｎａｍｅ", "ＴＩ－ＲＡＤＳ"])
    assert cm["patient_id"] == "ＰａｔｉｅｎｔＩｄ"
    assert cm["roi_name"] == "ｒｏｉｎａｍｅ"
    assert cm["tirads"] == "ＴＩ－ＲＡＤＳ"


def test_resolve_columns_prefers_correct_over_misspelled_tirads():
    """官方列54 `TI-RDAS` 是拼写错误冗余列;两列并存时必须优先正确的列53。"""
    cm = O._resolve_columns([
        "PatientId", "RoiName",
        "ROI->CLINICAL->TI-RDAS",    # 列54 拼写错误(RDAS)
        "ROI->CLINICAL->TI-RADS",    # 列53 正确
    ])
    assert cm["tirads"] == "ROI->CLINICAL->TI-RADS", (
        "拼写错误列被优先命中 —— 候选名顺序被改动了?正确列必须排在 TI-RDAS 之前。"
    )
    # 只有错误列时才用它兜底(总比零监督好)
    cm2 = O._resolve_columns(["PatientId", "RoiName", "ROI->CLINICAL->TI-RDAS"])
    assert cm2["tirads"] == "ROI->CLINICAL->TI-RDAS"


def test_missing_image_property_column_does_not_crash(mini_root):
    """官方表无「图像性质」列时,静动判据靠磁盘扩展名,训练样本照常构建。

    这是 2026-08-14 云端真训 KeyError 的直接回归测试:该列曾是必需列 → 必崩。
    """
    root, xlsx = mini_root
    df = pd.read_excel(xlsx, sheet_name="ROI级别", header=1, dtype=str)
    df = df.drop(columns=["图像性质"])              # 复刻官方真实表:压根没有这列
    xlsx2 = os.path.join(os.path.dirname(xlsx), "labels_no_imgprop.xlsx")
    df.to_excel(xlsx2, sheet_name="ROI级别", index=False, startrow=1)

    samples = O.build_official_samples(root, xlsx2, "ROI级别")
    assert samples, "无『图像性质』列时构不出样本 —— 静动判据没改成探磁盘?"
    # P3 是动态(磁盘有 P3/P3/P3.nii.gz),必须仍被识别为动态并按帧展开
    p3 = [s for s in samples if s["patient_key"] == "P3"]
    assert p3 and all(s["is_dynamic"] for s in p3), (
        "动态患者未被识别 —— 磁盘扩展名判据失效。"
    )
    # P1/P2 静态
    p12 = [s for s in samples if s["patient_key"] in ("P1", "P2")]
    assert p12 and not any(s["is_dynamic"] for s in p12)


# ── L2 union 监督(plan 任务 L2,2026-09-22)────────────────────────────────────
# 预注册不变式四类:K=1 图逐位一致 / K=2 静态 OR / K=2 动态同帧·异帧 / pkl 硬拒。
class TestUnionMaskL2:
    """union_mask=True 的样本构造 + Dataset 像素合成契约。"""

    def _build_union_root(self, tmp_path):
        """3 静态患者(M 单灶 / M+病灶1 双灶 / M 无 mask) + 1 动态双病灶(同帧)。"""
        root = tmp_path / "annotation"
        H = W = 64

        # PA 静态单病灶:K=1,union 输出须与逐病灶读取逐位一致
        pa = root / "PA" / "SA"
        _make_dcm(str(pa / "img.dcm"), np.full((H, W), 120, np.uint8))
        ma = np.zeros((H, W), np.uint8); ma[20:40, 20:40] = 1
        save_binary_mask(ma, str(pa / "M_1_mask.nii.gz"))

        # PB 静态双病灶:M + 病灶1(部分重叠,验证 OR 而非相加/覆盖)
        pb = root / "PB" / "SB"
        _make_dcm(str(pb / "img.dcm"), np.full((H, W), 100, np.uint8))
        mb = np.zeros((H, W), np.uint8); mb[10:30, 10:30] = 1
        sb = np.zeros((H, W), np.uint8); sb[20:40, 20:40] = 1   # 与 M 重叠 10×10
        save_binary_mask(mb, str(pb / "M_1_mask.nii.gz"))
        save_binary_mask(sb, str(pb / "病灶1_1_mask.nii.gz"))

        # PC 静态:行有 M 但 mask 文件缺失 → union 列表不含、valid_mask=0 语义不变
        pc = root / "PC" / "SC"
        _make_dcm(str(pc / "img.dcm"), np.full((H, W), 90, np.uint8))

        # PD 动态双病灶:5 帧,M 非零于帧 1;病灶1 非零于帧 1、3(同帧 OR + 异帧分立)
        pd_dir = root / "PD" / "PD"
        vol = np.stack([np.full((H, W), 40 + 10 * i, np.uint8) for i in range(5)])
        _save_image_nifti_3d(str(pd_dir / "PD.nii.gz"), vol)
        dm = np.zeros((5, H, W), np.uint8); dm[1, 10:30, 10:30] = 1
        save_binary_mask(dm, str(pd_dir / "M_1_mask.nii.gz"))
        ds = np.zeros((5, H, W), np.uint8); ds[1, 20:40, 20:40] = 1; ds[3, 30:50, 30:50] = 1
        save_binary_mask(ds, str(pd_dir / "病灶1_1_mask.nii.gz"))

        rows = [
            ["PA", "SA", "1", "M", "2左中叶", "2甲状腺包膜", "1有", "恶", "", "V类"],
            ["PB", "SB", "1", "M", "5右中叶", "1无", "0无", "良", "", "IV类"],
            ["PB", "SB", "1", "病灶1", "5右中叶", "1无", "0无", "良", "", "IV类"],
            ["PC", "SC", "1", "M", "4右上叶", "1无", "1有", "恶", "", "V类"],
            ["PD", "", "2", "M", "7峡部", "1无", "1有", "恶", "", ""],
            ["PD", "", "2", "病灶1", "7峡部", "1无", "1有", "良", "", ""],
        ]
        df = pd.DataFrame(rows, columns=[
            "PatientId", "SeriesUid", "图像性质", "RoiName", "主病灶位置",
            "甲状腺外侵犯", "临床T分期", "ROI良恶性", "ANNOTATION->SliceIndex", "ROI-TIRADS",
        ])
        xlsx = tmp_path / "labels_union.xlsx"
        df.to_excel(xlsx, sheet_name="ROI级别", index=False, startrow=1)
        return str(root), str(xlsx)

    def test_union_off_by_default_no_key(self, tmp_path):
        """默认 union_mask=False:样本无 union_mask_paths 键(零回归)。"""
        root, xlsx = self._build_union_root(tmp_path)
        samples = O.build_official_samples(root, xlsx)
        assert samples, "empty samples"
        assert all("union_mask_paths" not in s for s in samples)

    def test_union_k1_identical_to_single(self, tmp_path):
        """K=1 图(单病灶):union 路径 mask 与非 union 逐位一致(关键不变式)。"""
        root, xlsx = self._build_union_root(tmp_path)
        plain = [s for s in O.build_official_samples(root, xlsx)
                 if s["patient_key"] == "PA"][0]
        uni = [s for s in O.build_official_samples(root, xlsx, union_mask=True)
               if s["patient_key"] == "PA"][0]
        ds_plain = O.OfficialThyroidDataset([plain], img_size=64)
        ds_uni = O.OfficialThyroidDataset([uni], img_size=64)
        _, m_plain, _ = ds_plain[0]
        _, m_uni, _ = ds_uni[0]
        assert torch.equal(m_plain, m_uni)

    def test_union_static_or_semantics(self, tmp_path):
        """K=2 静态:union mask == M|病灶1(重叠区不重复计,值域 {0,1})。"""
        root, xlsx = self._build_union_root(tmp_path)
        uni = [s for s in O.build_official_samples(root, xlsx, union_mask=True)
               if s["patient_key"] == "PB"]
        assert len(uni) == 2
        for s in uni:  # 两样本同图 → 同一 union 键
            assert s["union_mask_paths"] == sorted([
                os.path.join(root, "PB", "SB", "M_1_mask.nii.gz"),
                os.path.join(root, "PB", "SB", "病灶1_1_mask.nii.gz")])
        ds = O.OfficialThyroidDataset(uni, img_size=64)
        masks = [ds[i][1].numpy()[0] for i in range(2)]     # [1,H,W] → [H,W]
        m1 = np.zeros((64, 64), np.uint8); m1[10:30, 10:30] = 1
        m2 = np.zeros((64, 64), np.uint8); m2[20:40, 20:40] = 1
        expect = (m1 | m2).astype(np.float32)
        for m in masks:
            assert set(np.unique(m)) <= {0.0, 1.0}
            assert np.array_equal(m, expect)

    def test_union_no_mask_row_not_in_list(self, tmp_path):
        """无 mask 的行(PC)不进 union 列表;valid_mask=0 语义不变。"""
        root, xlsx = self._build_union_root(tmp_path)
        uni = [s for s in O.build_official_samples(root, xlsx, union_mask=True)
               if s["patient_key"] == "PC"]
        assert len(uni) == 1
        assert uni[0]["union_mask_paths"] == []
        assert uni[0]["labels"]["valid_mask"] == 0

    def test_union_dynamic_same_frame_or(self, tmp_path):
        """K=2 动态同帧(帧1):M|病灶1 OR;异帧(帧3)只含病灶1。"""
        root, xlsx = self._build_union_root(tmp_path)
        uni = [s for s in O.build_official_samples(root, xlsx, union_mask=True)
               if s["patient_key"] == "PD"]
        frames = sorted(s["frame_index"] for s in uni)
        # 每病灶自己的非零帧一样本(union 不改样本结构):M→帧1;
        # 病灶1→帧1、3 → 帧序列 [1,1,3](两个帧1 样本同帧、共享同一 union mask)。
        assert frames == [1, 1, 3]
        ds = O.OfficialThyroidDataset(uni, img_size=64)
        by_f = {s["frame_index"]: ds[i][1].numpy()[0] for i, s in enumerate(uni)}
        m_m = np.zeros((64, 64), np.uint8); m_m[10:30, 10:30] = 1
        m_s = np.zeros((64, 64), np.uint8); m_s[20:40, 20:40] = 1
        assert np.array_equal(by_f[1], (m_m | m_s).astype(np.float32))
        m_s3 = np.zeros((64, 64), np.uint8); m_s3[30:50, 30:50] = 1
        assert np.array_equal(by_f[3], m_s3.astype(np.float32))

    def test_union_dynamic_labels_unchanged(self, tmp_path):
        """union 不动样本结构:四头标签/is_main/lesion_id/样本数与 plain 一致。"""
        root, xlsx = self._build_union_root(tmp_path)
        plain = O.build_official_samples(root, xlsx)
        uni = O.build_official_samples(root, xlsx, union_mask=True)
        assert len(plain) == len(uni)
        for a, b in zip(plain, uni):
            for k in a["labels"]:
                va, vb = a["labels"][k], b["labels"][k]
                if isinstance(va, torch.Tensor):
                    assert torch.equal(va, vb), k
                else:
                    assert va == vb, k
            assert a["mask_path"] == b["mask_path"]
            assert a["frame_index"] == b["frame_index"]
            assert a["patient_key"] == b["patient_key"]


    # ── 路B审查 P0-1/P2-1/P1-1 的区分力测试(2026-09-22)─────────────────────
    def test_union_pkl_accepted_with_flag(self, tmp_path):
        """P0-1 区分力:union 版 pkl + union_mask=True 必须被**接受**。
        首版 gate 拒一切 pkl(union 版也被拒)= 云端链第 0 步死,本测试杀该变异。"""
        import pickle as _pkl
        root, xlsx = self._build_union_root(tmp_path)
        uni = O.build_official_samples(root, xlsx, union_mask=True)
        pkl_p = tmp_path / "union.pkl"
        pkl_p.write_bytes(_pkl.dumps(
            {"_meta": {"img_size": 64, "union_mask": True}, "samples": uni}))
        train_ds, val_ds = O.build_official_datasets(
            root, xlsx, img_size=64, samples_pkl=str(pkl_p), union_mask=True)
        assert train_ds.samples or val_ds.samples

    def test_old_pkl_rejected_with_union_flag(self, tmp_path):
        """旧模式 pkl + union_mask=True → raise(静默回落防线)。"""
        import pickle as _pkl
        root, xlsx = self._build_union_root(tmp_path)
        plain = O.build_official_samples(root, xlsx)
        pkl_p = tmp_path / "plain.pkl"
        pkl_p.write_bytes(_pkl.dumps({"_meta": {"img_size": 64}, "samples": plain}))
        with pytest.raises(RuntimeError, match="union_mask_paths"):
            O.build_official_datasets(root, xlsx, img_size=64,
                                      samples_pkl=str(pkl_p), union_mask=True)

    def test_union_pkl_rejected_without_union_flag(self, tmp_path):
        """P2-1:union 版 pkl + 未传 union_mask → raise(半开态防线)。"""
        import pickle as _pkl
        root, xlsx = self._build_union_root(tmp_path)
        uni = O.build_official_samples(root, xlsx, union_mask=True)
        pkl_p = tmp_path / "union2.pkl"
        pkl_p.write_bytes(_pkl.dumps(
            {"_meta": {"img_size": 64, "union_mask": True}, "samples": uni}))
        with pytest.raises(RuntimeError, match="半开态"):
            O.build_official_datasets(root, xlsx, img_size=64,
                                      samples_pkl=str(pkl_p), union_mask=False)

    def test_drift_filter_covers_union_paths(self, tmp_path):
        """P1-1:union 路径文件消失 → 样本被剔除(不留错误负监督)。"""
        import pickle as _pkl
        root, xlsx = self._build_union_root(tmp_path)
        uni = O.build_official_samples(root, xlsx, union_mask=True)
        pkl_p = tmp_path / "u3.pkl"
        pkl_p.write_bytes(_pkl.dumps(
            {"_meta": {"img_size": 64, "union_mask": True}, "samples": uni}))
        # 删掉 PB 的次病灶 mask(9-14 官方删 21 mask 的同型漂移)
        os.remove(os.path.join(root, "PB", "SB", "病灶1_1_mask.nii.gz"))
        # PB 两样本的 union 路径缺一 → 剔除;fixture 里 2/7=28.6% 超 5% 阈值
        # → 同轮 raise 兜底(防"过滤掩盖真事故")。两道防线都断言:
        with pytest.raises(RuntimeError, match="文件缺失样本占比"):
            O.build_official_datasets(
                root, xlsx, img_size=64, samples_pkl=str(pkl_p), union_mask=True)
        # 低比例路径:只删 1 个样本的 union 路径(PA 单样本),1/7≈14%>5% 仍 raise;
        # 构造 5% 内场景:再用一个大 pkl(副本 ×5)稀释 → 只验证"剔除不炸、不静默"。
        import pickle as _pkl2
        uni5 = uni + [dict(s) for s in uni for _ in range(4)]  # 7→35 条
        pkl5 = tmp_path / "u5.pkl"
        pkl5.write_bytes(_pkl2.dumps(
            {"_meta": {"img_size": 64, "union_mask": True}, "samples": uni5}))
        os.remove(os.path.join(root, "PA", "SA", "M_1_mask.nii.gz"))
        with pytest.raises(RuntimeError, match="占比|half"):  # 5/35≈14% 仍超,
            O.build_official_datasets(
                root, xlsx, img_size=64, samples_pkl=str(pkl5), union_mask=True)
