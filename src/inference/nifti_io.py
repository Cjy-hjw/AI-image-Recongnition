"""NIfTI(.nii/.nii.gz) 读写:动态图输入解码 + 官方规范要求的结节 mask 落盘。

两个方向都是官方规范硬约束(2026-08-01《赛事开发规范 V1.0》§五.2 +《数据格式说明》):

**写出** — 每个预测结节输出一个 NIfTI 二值 mask,`1` 表示结节、`0` 为背景,
Height × Width **必须与对应原始输入超声图像严格一致**。🔴 2026-09-01 实测(第 5 次
跑分破案):官方 GT **静态 mask 也是 3D**——(Columns, Rows, 1) 单帧 Z=1(训练集
病例 0030d3c5:Rows=726/Columns=1068 → GT=(1068,726,1)),纯 2D 输出会被官方
shape 校验判「differs from every GT mask」整批判零。故 2D 输入也统一写出
(W, H, 1)。动态图为 3D 且 Z 轴对应**全视频帧序列**(必须预测并输出所有帧,
评测只抽标注帧计分但缺帧的帧无法计分)。

**读入** — 《数据格式说明》明确动图"影像原始数据为.nii.gz格式"(静态图才是 DICOM),
而 `video_io.load_video_frames` 只认 `.mp4/.avi/.mov/.mkv` + DICOM,遇 `.nii.gz`
直接返回 "Unsupported video format" ⇒ 该样本零预测。本模块补上这条路径。

维度约定(与 nibabel 的 Fortran 序对齐,是本模块最容易出错的地方):
    NIfTI 的 `shape` 语义是 (X, Y, Z) 列主序,而 numpy 图像数组是 (H, W) 行主序,
    二者的前两轴**互为转置**。若直接把 (H, W) 数组塞进 `Nifti1Image`,写出的文件
    header 里 dim 会变成 (W, H) —— 评测脚本按 header 读形状即与原图不符,判定尺寸
    不一致而整例失分,**且本地用同样的错误约定读回来看不出任何问题**(转置两次等于
    没转)。故写出时统一 `np.asarray(mask).T`,读入时同样转置回来。
"""

import os

import numpy as np

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:  # pragma: no cover - 依赖缺失路径由调用方兜底
    HAS_NIBABEL = False

_NIFTI_EXTS = (".nii", ".nii.gz")


def is_nifti_path(path: str) -> bool:
    """扩展名判定。`.nii.gz` 是双段后缀,`os.path.splitext` 只会切掉 `.gz`,
    故直接用 `endswith` 而非 splitext。"""
    return path.lower().endswith(_NIFTI_EXTS)


def _require_nibabel():
    if not HAS_NIBABEL:
        raise RuntimeError(
            "nibabel 未安装,无法读写 NIfTI。官方规范要求结节 mask 必须为 .nii/.nii.gz "
            "格式(§五.2),且动态图原始数据也是 .nii.gz —— 缺此依赖等于该输入类型全程零分。"
            "请确认 requirements.txt 的 nibabel 已随容器安装。"
        )


def save_binary_mask(mask: np.ndarray, out_path: str, affine=None) -> str:
    """把二值 mask 写成 NIfTI,值域强制 `{0, 1}`,dtype uint8。

    Args:
        mask: 2D `(H, W)` 静态图 或 3D `(n_frames, H, W)` 动态图。**接受任意
            非零表示前景**(含内部惯用的 0/255 `mask_binary`),内部统一二值化——
            规范要求的是 `1` 而非 `255`,直接写 255 会让评测按 `== 1` 取前景时
            得到全空 mask,而文件本身完全合法、看不出错。
            🔴 2026-09-01 起 2D 输入也写出 **3D (W, H, 1)**(单帧 Z=1):官方 GT
            静态 mask 实测为 3D 带单帧(见函数体注释),纯 2D 会被官方 shape
            校验整批判零。
        out_path: 目标路径,`.nii.gz` 按扩展名自动 gzip 压缩(nibabel 内部处理)。
        affine: 可选 4x4 仿射矩阵。超声无标准空间信息,默认单位阵——评测按体素
            重叠度匹配主病灶,不依赖世界坐标,单位阵是安全且明确的选择。

    Returns:
        实际写出的路径(便于调用方直接填进 prediction.json 的 MaskFile)。

    Raises:
        ValueError: 维度非 2D/3D,或数组为空。
    """
    _require_nibabel()

    arr = np.asarray(mask)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"save_binary_mask 只接受 2D (H,W) 或 3D (n_frames,H,W),收到 ndim={arr.ndim} "
            f"shape={arr.shape}"
        )
    if arr.size == 0:
        raise ValueError(f"save_binary_mask 收到空数组 shape={arr.shape}——拒绝写出空 mask")

    binary = (arr != 0).astype(np.uint8)

    # (H,W) → (W,H,1) / (n,H,W) → (W,H,n):见模块 docstring 的维度约定。
    # 🔴 2026-09-01 修复(第 5 次跑分破案):静态 mask 必须带 Z=1 单帧维。
    # 官方 GT 静态 mask 实测为 3D (Columns, Rows, 1)(云端训练集病例 0030d3c5:
    # Rows=726/Columns=1068 → GT shape=(1068,726,1)),此前 2D 路径写纯 2D
    # (W,H),官方校验 mask shape 与 GT 逐维比对 → 「shape differs from every
    # GT mask」整批判零(2026-09-01 跑分回报)。转置方向本身是对的(前两维
    # =(W,H) 与 GT 一致),缺的只有单帧 Z 维。
    # 训练侧读回不受影响:load_nifti_frames 对 ndim==3 走 transpose(2,1,0)
    # → (1,H,W),binf[0] 取帧口径不变。
    if binary.ndim == 2:
        data = binary.T[..., None]
    else:
        data = np.transpose(binary, (2, 1, 0))

    if affine is None:
        affine = np.eye(4, dtype=np.float64)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    img = nib.Nifti1Image(np.ascontiguousarray(data), affine)
    # set_data_dtype 只改 header 声明,不转换实际数据;数据已是 uint8,二者一致。
    img.set_data_dtype(np.uint8)
    nib.save(img, out_path)
    return out_path


def load_nifti_frames(path: str, output_dir: str = None, max_frames: int = 32):
    """读取 NIfTI 动态图,均匀抽至多 max_frames 帧。

    **签名与 `video_io.load_video_frames(path, output_dir, max_frames)` 逐位对齐**,
    入口层可直接按扩展名选 loader 再统一传参,无需分支:

        loader = load_nifti_frames if is_nifti_path(p) else load_video_frames
        frames, idxs, is_dcm, err, n_total = loader(p, output_dir, max_frames)

    `output_dir` 本身**不被使用**(NIfTI 无 DICOM 那样的 PHI 元数据,无需落匿名化
    副本),但必须占住第二个位置参数 —— 否则上面那行统一调用会把 output_dir 字符串
    绑到 max_frames,`uniform_sample_indices(n, "data/output")` 抛 TypeError,所有
    .nii.gz 样本零预测,而错误信息与"路径参数错位"毫无关联,极难定位。

    Returns:
        (frames, frame_indices, is_dicom, error_msg, n_total_frames)
        frames          — list[np.ndarray] 灰度 uint8,失败为 None
        frame_indices   — 每帧对应的**原始帧号**(0-based;注意官方 xlsx 的
                          `ANNOTATION->SliceIndex` 是 **1-based**,跨界时需 ±1)
        is_dicom        — 恒 False(NIfTI 不是 DICOM,无 PHI 脱敏需求)
        error_msg       — 失败原因,成功为 None
        n_total_frames  — 原始体数据的**总帧数**(Z 轴长度);供
                          `scatter_frames_to_full_length` 把抽样帧的 mask 回填成
                          全帧序列 3D mask(规范 §五.2)。失败分支返回 0。

    ⚠️ **抽样与规范的冲突(接入前必须处理)**:规范 §五.2 要求动态数据"必须预测并
    输出所有帧"的 3D mask(Z 轴对应全视频帧序列)。本函数默认只抽 32 帧,故调用方
    在写出 mask 前**必须把抽样帧的预测回填到原始帧位**(用 `frame_indices` 作下标,
    未抽中的帧填 0),否则 mask 的 Z 长度与原体积不符,评测按 SliceIndex 取标注帧
    会取到错误切片、或直接判尺寸不一致 → 该结节 mask 项零分。
    传 `max_frames<=0` 可取全部帧,规避该问题(代价是长视频显存/耗时上升)。

    与 `video_io` 一致的错误契约:失败返回错误信息而非抛异常,由入口层统一兜成
    `status=ERROR` 的 JSON。
    """
    del output_dir  # 见 docstring:仅为签名对齐占位,NIfTI 无需落匿名化副本
    from src.inference.io import _to_gray
    from src.inference.video_aggregate import uniform_sample_indices

    if not HAS_NIBABEL:
        return None, [], False, (
            "NIfTI 输入但 nibabel 未安装(动态图原始数据为 .nii.gz,见《数据格式说明》)"
        ), 0
    try:
        img = nib.load(path)
        arr = np.asarray(img.dataobj)

        # NIfTI (X,Y,Z) → numpy (Z,H,W)=(n_frames,H,W):与 save_binary_mask 互逆。
        if arr.ndim == 2:
            arr = arr.T[None, ...]  # 单帧 (X,Y) → (1,H,W)
        elif arr.ndim == 3:
            arr = np.transpose(arr, (2, 1, 0))
        elif arr.ndim == 4:
            # (X,Y,Z,T) 罕见形态:超声 cine 若被存成第四维时间轴,压掉长度为 1 的
            # Z 轴;Z>1 属真 3D+时间,本项目不支持,明确报错好过静默取第一层。
            if arr.shape[2] != 1:
                return None, [], False, (
                    f"4D NIfTI 且 Z 轴长度 {arr.shape[2]} != 1(真 3D+时间序列),本项目不支持"
                ), 0
            arr = np.transpose(arr[:, :, 0, :], (2, 1, 0))
        else:
            return None, [], False, f"不支持的 NIfTI 维度 ndim={arr.ndim} shape={arr.shape}", 0

        n_total = arr.shape[0]
        if n_total == 0:
            return None, [], False, f"NIfTI 无有效帧: {path}", 0

        idxs = uniform_sample_indices(n_total, max_frames)
        # _to_gray 统一位深/通道:NIfTI 常见 int16/float32,须 min-max 归一化到 uint8,
        # 与 DICOM/视频两条既有路径口径一致(单帧独立归一化)。
        frames = [_to_gray(arr[i]) for i in idxs]
        return frames, list(idxs), False, None, n_total
    except Exception as e:  # noqa: BLE001 — 守卫契约,失败返回错误信息不抛
        return None, [], False, f"Error reading NIfTI: {type(e).__name__}: {e}", 0


def scatter_frames_to_full_length(frame_masks, frame_indices, n_total_frames):
    """把**抽样帧**的 mask 回填成**全帧序列** 3D mask(规范 §五.2 硬约束)。

    规范原文:动态数据"每个结节输出 3D 二值掩码,Z 轴对应全视频帧序列(**必须预测
    并输出所有帧**)"。而 `load_nifti_frames` / `load_video_frames` 为控制显存与耗时
    默认只均匀抽 32 帧 —— 若直接把 32 帧 mask 堆叠写出,Z 长度与原视频不符:
    评测按 `ANNOTATION->SliceIndex`(原始帧号)取标注帧时会取到**错误切片**,或直接
    判尺寸不一致,该结节 mask 项零分。且两种失败都不会报错,产物完全合法。

    未被抽中的帧填 0(背景)。规范另有一条兜底:"评测时仅抽取标注帧计算指标,未标注
    帧不计分" —— 故未抽中帧填 0 的代价,仅限于该帧恰好是标注帧的情形;均匀抽样下
    这一概率随 max_frames 增大而降低,必要时传 `max_frames<=0` 取全帧彻底规避。

    Args:
        frame_masks: list[np.ndarray],每项为 `(H, W)` 二值 mask,顺序与
            `frame_indices` 一一对应。
        frame_indices: list[int],每帧对应的**原始帧号**(0-based),即 loader 返回的
            第二个元素。
        n_total_frames: 原始视频/体数据的总帧数。

    Returns:
        `(n_total_frames, H, W)` 的 np.uint8 数组,值域 `{0,1}`。

    Raises:
        ValueError: 两个列表长度不等、为空、帧号越界,或各帧 mask 尺寸不一致。
    """
    if len(frame_masks) != len(frame_indices):
        raise ValueError(
            f"frame_masks({len(frame_masks)}) 与 frame_indices({len(frame_indices)}) "
            f"长度不等 —— 二者必须一一对应,否则 mask 会被写到错误的帧位"
        )
    if not frame_masks:
        raise ValueError("frame_masks 为空,无法确定 (H, W);阴性病例应直接跳过 mask 写出")

    first = np.asarray(frame_masks[0])
    if first.ndim != 2:
        raise ValueError(f"frame_masks 每项须为 2D (H,W),收到 ndim={first.ndim}")
    h, w = first.shape

    if n_total_frames <= 0:
        raise ValueError(f"n_total_frames 必须为正,收到 {n_total_frames}")

    out = np.zeros((n_total_frames, h, w), dtype=np.uint8)
    for m, idx in zip(frame_masks, frame_indices):
        m = np.asarray(m)
        if m.shape != (h, w):
            raise ValueError(
                f"帧 {idx} 的 mask 尺寸 {m.shape} 与首帧 {(h, w)} 不一致"
            )
        if not (0 <= idx < n_total_frames):
            raise ValueError(
                f"帧号 {idx} 越界 [0, {n_total_frames}) —— 抽样下标与总帧数不匹配,"
                f"回填会静默丢帧或 IndexError"
            )
        out[idx] = (m != 0).astype(np.uint8)
    return out
