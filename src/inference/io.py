"""Image I/O with DICOM PHI anonymization (HIPAA Safe Harbor fields)."""

import cv2
import numpy as np

# HIPAA Safe Harbor 相关的 DICOM PHI 标签 → 匿名化目标值(**标量**类字段)。
# 覆盖患者身份、就诊、机构、设备及操作者标识等可重识别字段。
# 注意:此字典只放**标量 VR**(PN/LO/SH/DA/TM/AS 等)可 setattr 字符串的字段;
# 序列类(SQ)或应整体删除的字段放 _PHI_DELETE_TAGS,不能给 SQ setattr 字符串
# (会破坏数据结构)。补全依据:R2 审查 + HIPAA Safe Harbor 18 类标识符——
# 原清单只列 21 项。🔴 2026-09-01 起匿名化副本不再落盘(save_as 已删,见
# load_input_image docstring),此清单仅作内存清洗防御性保留:若未来重新引入
# 落盘,清单外 PHI 标签会原样写进副本——故保持完整,勿裁剪。
_PHI_FIELDS = {
    "PatientName": "ANONYMIZED",
    "PatientID": "ANONYMIZED",
    "PatientBirthDate": "00000101",
    "PatientBirthTime": "",
    "PatientSex": "",
    "PatientAge": "",
    "PatientAddress": "ANONYMIZED",
    "PatientTelephoneNumbers": "",
    "OtherPatientIDs": "",
    "OtherPatientNames": "",
    "PatientMotherBirthName": "",
    "IssuerOfPatientID": "",
    "EthnicGroup": "",
    "PatientReligiousPreference": "",
    "PatientInstitutionResidence": "",
    "Occupation": "",
    "PatientComments": "",
    "AdditionalPatientHistory": "",
    "MilitaryRank": "",
    "ReferringPhysicianName": "ANONYMIZED",
    "PerformingPhysicianName": "ANONYMIZED",
    "RequestingPhysician": "ANONYMIZED",
    "OperatorsName": "ANONYMIZED",
    "InstitutionName": "ANONYMIZED",
    "InstitutionAddress": "ANONYMIZED",
    "InstitutionalDepartmentName": "",
    "StationName": "ANONYMIZED",
    "DeviceSerialNumber": "ANONYMIZED",
    "AccessionNumber": "ANONYMIZED",
    "StudyID": "ANONYMIZED",
    "StudyDate": "00000101",
    "SeriesDate": "00000101",
    "AcquisitionDate": "00000101",
    "ContentDate": "00000101",
    "StudyTime": "",
    "SeriesTime": "",
    "AcquisitionTime": "",
    "ContentTime": "",
}

# 应**整体删除**(del 而非 setattr)的高危 PHI 标签。绝大多数是序列类(SQ VR)——
# setattr 字符串会破坏 SQ 结构;末尾两项 ResponsiblePerson(PN)/ResponsibleOrganization(LO)
# 是标量,放这里并非标错类型,而是与序列共用同一条 del 通道(语义同"整体抹除",del
# 对存在标量与 SQ 同样正确)。**别据此把普通标量 PHI 也塞进删除表**——那是给"可整体
# 抹除"的标签用的,常规标量走上面 _PHI_FIELDS 的 setattr 匿名值。
_PHI_DELETE_TAGS = (
    "OtherPatientIDsSequence",
    "ReferringPhysicianIdentificationSequence",
    "PerformingPhysicianIdentificationSequence",
    "RequestingPhysicianIdentificationSequence",
    "OperatorIdentificationSequence",
    "PatientInsurancePlanCodeSequence",
    "IssuerOfPatientIDQualifiersSequence",
    "ResponsiblePerson",        # 标量(PN),非 SQ——见表头注释,共用 del 通道
    "ResponsibleOrganization",  # 标量(LO),非 SQ——见表头注释,共用 del 通道
)


def _anonymize_phi(ds):
    """就地匿名化 DICOM 数据集的 PHI 标签(静/动两路径共用,口径一致)。

    标量 PHI(_PHI_FIELDS)→ setattr 匿名值;序列/删除类(_PHI_DELETE_TAGS)→ del。
    只处理**已存在**的标签,不新增。成像必需标签(PixelData/Rows/Columns/
    PhotometricInterpretation/BitsAllocated 等)不在清单内,绝不触碰。
    """
    for tag, anon_value in _PHI_FIELDS.items():
        if tag in ds:
            setattr(ds, tag, anon_value)
    for tag in _PHI_DELETE_TAGS:
        if tag in ds:
            del ds[tag]

try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def _rescale_to_uint8(arr: np.ndarray) -> np.ndarray:
    """单帧 min-max 归一化到 8-bit 灰度(统一静/动态图 DICOM 的位深处理)。"""
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    return ((arr - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8)


def _to_gray(frame: np.ndarray, channel_order: str = "rgb") -> np.ndarray:
    """统一转单通道灰度 uint8(覆盖所有通道形态,杜绝 3D 漏过)。

    彩色多普勒等多通道帧若不转灰度,送入 preprocess_image 会 unsqueeze 成
    5 维张量 (B,1,H,W,C) 触发模型前向 `too many values to unpack` 崩溃。
    静(io)/动(video_io)两条路径共用本函数,口径一致。

    Args:
        channel_order: 3 通道彩色帧的通道序。DICOM 像素为 RGB("rgb",默认);
            OpenCV `cap.read()` 解码的视频帧为 BGR("bgr")。用错会令多普勒
            红/蓝血流的灰度亮度系统性偏移(R/B 加权对调),故按来源显式区分。

    收敛规则(按 ndim/通道数):
        (H,W)             → 直通
        (H,W,1)           → 去掉单通道轴
        (H,W,3)           → cvtColor(按 channel_order 选 RGB/BGR→GRAY)
        (H,W,4)           → 取前 3 通道(丢 alpha)再按 channel_order 转灰度
        (H,W,C) 其它 C    → 取首通道兜底(罕见,如多光谱),避免静默返回 3D
        (C,H,W) channel-first(C∈{1,3,4}) → 转置为 (H,W,C) 后按上述处理
    非 uint8(如 16-bit 灰度 DICOM)→ min-max 归一化到 8-bit。
    """
    rgb2gray = cv2.COLOR_RGB2GRAY if channel_order == "rgb" else cv2.COLOR_BGR2GRAY

    # channel-first (C,H,W) → (H,W,C):仅当首轴是通道数(1/3/4)且非方阵歧义时
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[2] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.ndim == 3:
        c = frame.shape[2]
        if c == 1:
            frame = frame[:, :, 0]
        elif c == 3:
            frame = cv2.cvtColor(frame, rgb2gray)
        elif c == 4:
            frame = cv2.cvtColor(frame[:, :, :3], rgb2gray)
        else:
            frame = frame[:, :, 0]  # 未知通道数兜底,绝不返回 3D
    elif frame.ndim != 2:
        raise ValueError(f"_to_gray 不支持的维度 ndim={frame.ndim} shape={frame.shape}")

    if frame.dtype != np.uint8:
        frame = _rescale_to_uint8(frame)
    return frame


def load_input_image(path: str, output_dir: str):
    """Load a B-mode ultrasound image (PNG/JPG/DCM).

    For DICOM inputs, anonymizes HIPAA Safe Harbor PHI fields in memory
    (see _anonymize_phi). 🔴 2026-09-01 起不再把匿名化副本落盘:官方 V2.1 评分器
    对预测目录做白名单校验(只允许 .nii/.nii.gz/.json),写入 .dcm 即整批判零。
    `output_dir` 参数仅为调用方签名兼容保留,本函数不再向其写任何文件
    (回归锁见 tests/test_video.py::TestZeroFileLeakRedline)。

    Returns:
        (img_np, is_dicom, error_msg) — img_np is None on failure.
    """
    is_dicom = path.lower().endswith((".dcm", ".dicom"))

    if is_dicom:
        if not HAS_PYDICOM:
            return None, True, "DICOM image provided but pydicom is not installed."
        try:
            print("Reading PACS DICOM metadata and pixel data...")
            ds = pydicom.dcmread(path)

            # 统一经 _to_gray:彩色多普勒单帧 DICOM 的 pixel_array 为 (H,W,3),
            # 必须转灰度,否则下游 preprocess_image 产出 5 维张量令模型前向崩溃。
            # _to_gray 内部对非 uint8(如 16-bit)走 min-max 归一化,等价原 rescale。
            img_np = _to_gray(ds.pixel_array)

            # PHI metadata anonymization (HIPAA Safe Harbor) — 标量匿名 + 序列删除,
            # 见 _anonymize_phi(静/动共用,只碰已存在的 PHI 标签,不动成像参数)。
            _anonymize_phi(ds)
        except Exception as e:
            return None, True, f"Error reading/anonymizing DICOM file: {e}"
    else:
        img_np = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img_np is None:
        return None, is_dicom, f"Could not read image from {path}"

    return img_np, is_dicom, None
