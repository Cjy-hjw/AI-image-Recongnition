"""生成可被 pydicom.dcmread() 真实解析的最小合法 DICOM 文件（单帧/多帧 cine）。

与 `tests/test_entrypoint_dispatch.py::_write_dicom` 的区别：那个只写 128 字节
preamble + "DICM" 魔数，供 bash 分发路由测试用（entrypoint.sh 只嗅探 offset-128
魔数，不需要真能被 pydicom 解析）。本脚本构造完整合法的 DICOM 数据集（含
file_meta、TransferSyntaxUID、像素相关标签、PixelData），能被
`src/inference/video_io.py::_load_dicom_cine` 真实读出 `pixel_array`——用于
docker-build-smoke CI job 里验证"DICOM 单帧/多帧 cine 经真实容器完整跑通"，
而不仅仅是"entrypoint.sh 分发路由正确"这一层。

用法（CI host runner 调用，仅需 pydicom + numpy，不依赖仓库其它模块）：
    python scripts/verify/gen_synthetic_dicom.py --out /tmp/single.dcm --frames 1
    python scripts/verify/gen_synthetic_dicom.py --out /tmp/cine.dcm --frames 4 --rows 32 --cols 32

测试侧复用（tests/test_smoke_entrypoints.py）：
    from scripts.verify.gen_synthetic_dicom import build_dicom
    ds = build_dicom(rows=64, cols=64, n_frames=4)
    ds.save_as(path, enforce_file_format=True)  # pydicom>=3 显式要求
"""
import argparse

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def build_dicom(rows: int = 64, cols: int = 64, n_frames: int = 1) -> FileDataset:
    """构造一个最小合法的灰度 DICOM 数据集（8-bit MONOCHROME2）。

    n_frames == 1 时省略 NumberOfFrames 标签（等价单帧 DICOM，pixel_array
    为 2D）；n_frames > 1 时显式写 NumberOfFrames（等价 cine，pixel_array
    为 3D (frames,H,W)）——两种形态对应 `_load_dicom_cine` 里
    `arr.ndim == 2` 的单帧兼容分支与原生多帧分支。
    """
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(
        None, {}, file_meta=file_meta, preamble=b"\x00" * 128
    )
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientName = "SYNTHETIC^TEST"
    ds.PatientID = "SYNTH0001"
    ds.Modality = "US"

    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0

    if n_frames > 1:
        ds.NumberOfFrames = n_frames
        pixel_shape = (n_frames, rows, cols)
    else:
        pixel_shape = (rows, cols)

    rng = np.random.default_rng(seed=0)
    pixels = rng.integers(0, 256, size=pixel_shape, dtype=np.uint8)
    ds.PixelData = pixels.tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, required=True, help="输出 .dcm 路径")
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--cols", type=int, default=64)
    p.add_argument("--frames", type=int, default=1, help="1=单帧 DICOM,>1=多帧 cine")
    args = p.parse_args()

    ds = build_dicom(rows=args.rows, cols=args.cols, n_frames=args.frames)
    ds.save_as(args.out, enforce_file_format=True)
    print(f"--> Wrote synthetic DICOM ({args.frames} frame(s), {args.rows}x{args.cols}) to {args.out}")


if __name__ == "__main__":
    main()
