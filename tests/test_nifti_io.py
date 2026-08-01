"""`src/inference/nifti_io.py` 回归测试。

**本文件的核心防线是维度约定**(见 `TestDimensionConvention`):NIfTI header 的
`dim` 是 (X,Y,Z) 列主序,numpy 图像是 (H,W) 行主序,前两轴互为转置。若写出时漏转置,
本地"写出→读回"用同一套错误约定看不出任何异常(转置两次等于没转),但评测脚本按
header 读形状会判定与原图尺寸不符 —— 整例失分且本地永远复现不了。故所有形状断言
一律用**非方阵**(H≠W),方阵会让转置错误完全隐形。
"""

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from src.inference.nifti_io import (  # noqa: E402
    is_nifti_path,
    load_nifti_frames,
    save_binary_mask,
    scatter_frames_to_full_length,
)

# 非方阵:H=7, W=11。任何形状断言都必须能区分 (7,11) 与 (11,7)。
_H, _W = 7, 11


def _make_2d_mask(fill=1):
    m = np.zeros((_H, _W), np.uint8)
    m[2:5, 3:9] = fill
    return m


class TestDimensionConvention:
    """写出的 NIfTI header 必须是 (W,H[,n]),读入必须还原成 (H,W)。"""

    def test_2d_header_is_width_height(self, tmp_path):
        out = save_binary_mask(_make_2d_mask(), str(tmp_path / "m.nii.gz"))
        assert nib.load(out).shape == (_W, _H), "header dim 应为 (X,Y)=(W,H),漏转置会得到 (H,W)"

    def test_3d_frame_axis_lands_on_z(self, tmp_path):
        n = 4
        m = np.zeros((n, _H, _W), np.uint8)
        m[1, 2:5, 3:9] = 1
        out = save_binary_mask(m, str(tmp_path / "m3.nii.gz"))
        assert nib.load(out).shape == (_W, _H, n), "帧轴必须落在 NIfTI 的 Z(第三轴)"

    def test_3d_roundtrip_preserves_content(self, tmp_path):
        """内容而非仅形状:转置方向写反时形状可能碰巧对,前景位置必错。"""
        n = 4
        m = np.zeros((n, _H, _W), np.uint8)
        m[1, 2:5, 3:9] = 1
        out = save_binary_mask(m, str(tmp_path / "m3.nii.gz"))
        back = np.transpose(np.asarray(nib.load(out).dataobj), (2, 1, 0))
        assert np.array_equal(back, m)

    def test_load_returns_height_width(self, tmp_path):
        """读入路径独立验证:直接用 nibabel 造 (X,Y,Z) 文件,不经 save_binary_mask,
        避免"写读两侧同错互相抵消"这一自欺场景。"""
        vol = np.zeros((_W, _H, 3), np.int16)
        for z in range(3):
            vol[:, :, z] = (z + 1) * 100
        p = str(tmp_path / "v.nii.gz")
        nib.save(nib.Nifti1Image(vol, np.eye(4)), p)

        frames, idxs, is_dicom, err = load_nifti_frames(p)
        assert err is None
        assert len(frames) == 3
        assert frames[0].shape == (_H, _W), "读入应还原成 numpy 的 (H,W)"
        assert idxs == [0, 1, 2]
        assert is_dicom is False


class TestBinaryValueRange:
    """规范要求 `1`=结节 `0`=背景。写出 255 时文件完全合法,但评测按 `==1` 取前景
    会得到全空 mask —— 静默零分,是本模块最危险的失败模式。"""

    def test_255_input_written_as_1(self, tmp_path):
        out = save_binary_mask(_make_2d_mask(fill=255), str(tmp_path / "m.nii.gz"))
        vals = set(np.asarray(nib.load(out).dataobj).flatten().tolist())
        assert vals == {0, 1}

    def test_arbitrary_nonzero_treated_as_foreground(self, tmp_path):
        m = np.zeros((_H, _W), np.float32)
        m[1, 1] = 0.3
        m[2, 2] = 7.9
        out = save_binary_mask(m, str(tmp_path / "m.nii.gz"))
        data = np.asarray(nib.load(out).dataobj)
        assert set(data.flatten().tolist()) == {0, 1}
        assert data.sum() == 2

    def test_dtype_is_uint8(self, tmp_path):
        out = save_binary_mask(_make_2d_mask(), str(tmp_path / "m.nii.gz"))
        assert np.asarray(nib.load(out).dataobj).dtype == np.uint8

    def test_all_zero_mask_is_written(self, tmp_path):
        """全零 mask 合法(该结节被完全滤除仍需占位文件),不得与空数组混为一谈。"""
        out = save_binary_mask(np.zeros((_H, _W), np.uint8), str(tmp_path / "m.nii.gz"))
        assert np.asarray(nib.load(out).dataobj).sum() == 0


class TestSaveGuards:
    @pytest.mark.parametrize(
        "arr",
        [
            np.zeros(5, np.uint8),  # 1D
            np.zeros((2, 3, 4, 5), np.uint8),  # 4D
        ],
        ids=["1d", "4d"],
    )
    def test_bad_ndim_raises(self, arr, tmp_path):
        with pytest.raises(ValueError, match="2D|3D|ndim"):
            save_binary_mask(arr, str(tmp_path / "x.nii.gz"))

    def test_empty_array_raises(self, tmp_path):
        with pytest.raises(ValueError, match="空"):
            save_binary_mask(np.zeros((0, 5), np.uint8), str(tmp_path / "x.nii.gz"))

    def test_creates_parent_dirs(self, tmp_path):
        """官方输出路径 answer/{eval_id}/{StudyUid}/{SeriesUid}/ 是推理时动态创建的
        (规范 §二.2 明写 evaluation_id 目录须由参赛者代码创建),父目录不存在必须自建。"""
        out = save_binary_mask(
            _make_2d_mask(), str(tmp_path / "a" / "b" / "c" / "m.nii.gz"))
        assert nib.load(out).shape == (_W, _H)


class TestLoadGuards:
    def test_missing_file_returns_error_not_raise(self):
        """与 video_io 一致的错误契约:返回错误信息而非抛异常,由入口层兜成 ERROR JSON。"""
        frames, idxs, _, err = load_nifti_frames("/nonexistent/x.nii.gz")
        assert frames is None and idxs == [] and err

    def test_2d_nifti_treated_as_single_frame(self, tmp_path):
        p = str(tmp_path / "s.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros((_W, _H), np.int16), np.eye(4)), p)
        frames, idxs, _, err = load_nifti_frames(p)
        assert err is None and len(frames) == 1 and frames[0].shape == (_H, _W)

    def test_4d_with_real_z_axis_rejected(self, tmp_path):
        """Z>1 的 4D 是真 3D+时间序列,本项目不支持——明确报错好过静默取第一层。"""
        p = str(tmp_path / "q.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros((_W, _H, 5, 3), np.int16), np.eye(4)), p)
        frames, _, _, err = load_nifti_frames(p)
        assert frames is None and "4D" in err

    def test_max_frames_subsamples(self, tmp_path):
        p = str(tmp_path / "long.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros((_W, _H, 20), np.int16), np.eye(4)), p)
        frames, idxs, _, err = load_nifti_frames(p, max_frames=5)
        assert err is None and len(frames) == 5 == len(idxs)

    def test_negative_values_normalized_to_uint8(self, tmp_path):
        """NIfTI 常见 int16/float32 且可含负值,须经 min-max 归一化到 uint8,
        与 DICOM/视频两条既有路径口径一致。"""
        vol = np.full((_W, _H, 2), -100, np.int16)
        vol[0, 0, 0] = 500
        p = str(tmp_path / "neg.nii.gz")
        nib.save(nib.Nifti1Image(vol, np.eye(4)), p)
        frames, _, _, err = load_nifti_frames(p)
        assert err is None
        assert frames[0].dtype == np.uint8
        assert frames[0].min() >= 0 and frames[0].max() <= 255


class TestPathDetection:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("a.nii.gz", True),
            ("a.nii", True),
            ("A.NII.GZ", True),  # 大小写不敏感
            ("a.mp4", False),
            ("a.dcm", False),
            ("a.gz", False),  # 仅 .gz 不是 NIfTI
        ],
    )
    def test_extension_detection(self, path, expected):
        assert is_nifti_path(path) is expected


class TestLoaderSignatureAlignment:
    """签名必须与 `video_io.load_video_frames(path, output_dir, max_frames)` 逐位对齐。

    入口层按扩展名选 loader 后统一传三个参数;若 NIfTI 侧缺 `output_dir`,output_dir
    字符串会绑到 max_frames,`uniform_sample_indices(n, "data/output")` 抛 TypeError,
    所有 .nii.gz 样本零预测,且报错信息与"参数错位"毫无关联,极难定位。
    """

    def test_positional_three_args_matches_video_loader(self, tmp_path):
        from src.inference.video_io import load_video_frames

        import inspect
        nifti_params = list(inspect.signature(load_nifti_frames).parameters)
        video_params = list(inspect.signature(load_video_frames).parameters)
        assert nifti_params[:3] == video_params[:3] == ["path", "output_dir", "max_frames"]

    def test_three_positional_args_work(self, tmp_path):
        p = str(tmp_path / "v.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros((_W, _H, 20), np.int16), np.eye(4)), p)
        frames, idxs, _, err = load_nifti_frames(p, str(tmp_path), 5)
        assert err is None and len(frames) == 5 == len(idxs)


class TestScatterToFullLength:
    """规范 §五.2:动态 mask 的 Z 轴必须对应**全视频帧序列**(必须输出所有帧)。

    loader 默认只抽 32 帧。直接堆叠抽样帧的 mask 会让 Z 长度与原视频不符 ——
    评测按 `SliceIndex`(原始帧号)取标注帧时会取到**错误切片**,或判尺寸不一致。
    两种失败都不报错、产物完全合法,是最隐蔽的零分路径。
    """

    def test_sampled_masks_land_on_original_frame_positions(self):
        idxs = [0, 5, 10, 14, 19]
        masks = [np.ones((_H, _W), np.uint8) for _ in idxs]
        full = scatter_frames_to_full_length(masks, idxs, 20)

        assert full.shape == (20, _H, _W)
        assert np.flatnonzero(full.any(axis=(1, 2))).tolist() == idxs

    def test_unsampled_frames_are_background(self):
        full = scatter_frames_to_full_length([np.ones((4, 6), np.uint8)], [3], 10)
        assert full.sum() == 24  # 只有第 3 帧有前景
        assert full[0].sum() == 0 and full[9].sum() == 0

    def test_output_is_binary_uint8(self):
        full = scatter_frames_to_full_length([np.full((4, 6), 255, np.uint8)], [1], 5)
        assert full.dtype == np.uint8
        assert set(full.flatten().tolist()) == {0, 1}

    def test_roundtrip_through_save_preserves_full_length(self, tmp_path):
        idxs = [0, 3, 7]
        masks = [np.ones((_H, _W), np.uint8) for _ in idxs]
        full = scatter_frames_to_full_length(masks, idxs, 8)
        out = save_binary_mask(full, str(tmp_path / "m.nii.gz"))
        assert nib.load(out).shape == (_W, _H, 8), "写出的 Z 长度必须是全帧数"

    @pytest.mark.parametrize(
        "masks,idxs,n_total,desc",
        [
            ([np.ones((4, 6), np.uint8)] * 3, [0, 1], 10, "长度不等"),
            ([], [], 10, "空列表"),
            ([np.ones((4, 6), np.uint8)], [99], 10, "帧号越界"),
            ([np.ones((4, 6), np.uint8)], [-1], 10, "帧号为负"),
            ([np.ones((4, 6), np.uint8), np.ones((2, 2), np.uint8)], [0, 1], 10, "尺寸不一致"),
            ([np.ones((4, 6), np.uint8)], [0], 0, "n_total=0"),
        ],
    )
    def test_guards(self, masks, idxs, n_total, desc):
        with pytest.raises(ValueError):
            scatter_frames_to_full_length(masks, idxs, n_total)
