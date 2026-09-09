"""整图分割 mask → 逐结节实例。

官方规范(2026-08-01《赛事开发规范 V1.0》§五)要求**每个检出结节**单独一条记录
(`NoduleID` / `MaskFile` / 各属性),而本项目模型输出的是**单通道整图 mask**
(`multi_task_model` 的 `seg_head` 是 `Conv2d(20,1,1)`,无实例分割概念)。本模块用
连通域分析补上这一层:整图二值 mask → 有序的实例列表,每个实例带独立 mask、bbox、
面积、质心,供逐结节裁剪分类 + 逐结节 `.nii.gz` 落盘。

## 两个关键设计决策

**① 在原图尺寸上拆,不在 512×512 模型分辨率上拆。**
规范要求 mask 的 H×W 与原始输入图严格一致。若先在 512×512 拆分再逐实例 resize,
每个实例各自经历一次最近邻插值,相邻结节的边界可能在还原后重叠或产生缝隙,且累计
误差不可控。直接在最终尺寸上做连通域,实例 mask 天然满足尺寸约束且互不重叠。

**② 面积阈值用相对比例而非绝对像素。**
超声图尺寸从 ~400×300 到 ~1024×768 不等,固定像素阈值在小图上会滤掉真结节、在
大图上放过噪声点。默认 `min_area_ratio=1e-4`(万分之一)对 512×512 约合 26 px,
对 1024×768 约合 78 px,量级随图缩放。

## 动态图的 3D 连通性

3D 模式(`(n_frames, H, W)`)用 `scipy.ndimage.label` 的 26-邻域,时间上连续出现的
同一结节自然连成一个实例,直接满足"每个结节输出 3D mask,Z 轴对应全帧序列"。
**已知局限**:若分割在中间某帧漏检导致结节断开,会被拆成两个实例(NoduleID 重复
计数)。缓解手段是先做时间维闭运算(`close_gap_frames`),默认开启跨 1 帧的缝合。
"""

import numpy as np

try:
    import cv2

    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False

try:
    from scipy import ndimage as ndi

    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False


# 默认最小面积占比(万分之一)。取值依据:临床可报告的甲状腺结节一般 ≥3mm,在典型
# 超声视野(~4-6cm 宽)下约占图像宽度的 5-7%,面积占比远高于 1e-4;该阈值只滤明显
# 的分割噪点(孤立几像素),不会误伤真结节。
DEFAULT_MIN_AREA_RATIO = 1e-4

# 单例上限。防止分割崩坏(如全图噪声)时产出上千个实例撑爆 prediction.json 与磁盘。
# 临床单次超声切面极少超过 10 个可报告结节,20 是宽松上界。超出部分按面积降序截断。
DEFAULT_MAX_NODULES = 20


def split_nodules(
    mask: np.ndarray,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
    max_nodules: int = DEFAULT_MAX_NODULES,
    close_gap_frames: int = 1,
):
    """整图二值 mask → 按面积降序的实例列表。

    Args:
        mask: 2D `(H, W)` 静态图,或 3D `(n_frames, H, W)` 动态图。**任意非零视作
            前景**(兼容内部 0/255 的 `mask_binary`),内部统一二值化。
        min_area_ratio: 最小面积占比(相对单帧 H×W)。小于此值的连通域视为噪声丢弃。
            传 0 表示不过滤。
        max_nodules: 保留的实例数上限,按面积降序截断。
        close_gap_frames: 仅 3D 有效。时间维闭运算的缝合跨度(帧),弥合分割在个别帧
            漏检导致的同一结节断裂。0 表示不缝合。

    Returns:
        list[dict],按 `area` 降序,每项:
            `mask`     — **2D**: np.uint8 实例掩码,值域 `{0,1}`,shape 与输入一致。
                         **3D**: 稀疏存储(2026-09-09 内存治理),键为 `mask_sparse`
                         `{array:(nz,bh,bw) uint8 bbox 裁剪块, origin:(z0,y0,x0) 全尺寸
                         坐标系偏移}`——全尺寸视图按需 `restore_instance_mask` 重建
                         (每实例 (n_frames,H,W) 全尺寸 ≈0.9GB × ≤20 例 ≈17GB 峰值,
                         实际每结节只在几十帧出现,全尺寸驻留浪费 90%+;
                         bbox 块典型 <1MB)。**2D 路径不变**(`mask` 键全尺寸,小图无此问题)。
            `area`     — int 前景体素数(3D 为全帧累计)
            `bbox`     — 2D: `(x, y, w, h)`;3D: `(x, y, w, h, z0, z1)`,`z1` 为**闭区间**
            `centroid` — 2D: `(cx, cy)`;3D: `(cx, cy, cz)`,float
            `frame_indices` — 仅 3D,该结节出现的帧下标 list[int](对应输入数组的第 0 轴)

        输入无前景(阴性病例)时返回**空列表** —— 规范明确阴性输出空文件夹/空数组,
        空列表是合法结果而非错误。

    Raises:
        ValueError: 维度非 2D/3D。
        RuntimeError: 所需依赖缺失(2D 需 cv2,3D 需 scipy)。
    """
    arr = np.asarray(mask)
    if arr.ndim == 2:
        return _split_2d(arr, min_area_ratio, max_nodules)
    if arr.ndim == 3:
        return _split_3d(arr, min_area_ratio, max_nodules, close_gap_frames)
    raise ValueError(
        f"split_nodules 只接受 2D (H,W) 或 3D (n_frames,H,W),收到 ndim={arr.ndim} "
        f"shape={arr.shape}"
    )


def _split_2d(arr, min_area_ratio, max_nodules):
    if not HAS_CV2:
        raise RuntimeError("split_nodules(2D) 需要 opencv,但 cv2 未安装")

    binary = (arr != 0).astype(np.uint8)
    h, w = binary.shape
    min_area = max(1, int(min_area_ratio * h * w))

    # connectivity=8:超声结节边界在离散网格上常呈斜向阶梯,4-邻域会把单个结节沿
    # 对角切成多块,虚增结节数。
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    nodules = []
    for i in range(1, n_labels):  # 0 是背景
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        nodules.append({
            "mask": (labels == i).astype(np.uint8),
            "area": area,
            "bbox": (
                int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
            ),
            "centroid": (float(centroids[i][0]), float(centroids[i][1])),
        })

    nodules.sort(key=lambda d: d["area"], reverse=True)
    return nodules[:max_nodules]


def _split_3d(arr, min_area_ratio, max_nodules, close_gap_frames):
    if not HAS_SCIPY:
        raise RuntimeError("split_nodules(3D) 需要 scipy.ndimage.label,但 scipy 未安装")

    binary = (arr != 0).astype(np.uint8)
    n_frames, h, w = binary.shape
    # 阈值口径:与 `area`(全帧累计体素)比较,而非逐帧峰值。
    #
    # 这是**刻意选择**,不是把 2D 逻辑照搬到 3D 的疏漏:动态图噪声多为单帧瞬时斑点
    # (探头抖动/增益突变/伪影),真结节必然跨帧持续。按单帧阈值过滤会误杀"每帧
    # 20px、连续 10 帧"的真结节(单帧 20 < 26),而按累计阈值它得 200px 稳稳保留。
    #
    # 已知代价:只在 1 帧出现但面积超阈值的斑点会被保留。这通常正是"真结节只在
    # 关键帧清晰"的正常情形——规范的动态 AP 按病灶级汇总(每病灶只以唯一
    # confidence 进入排序一次),漏检真结节的代价大于多报一个低置信疑似结节。
    min_area = max(1, int(min_area_ratio * h * w))

    labeled_src = binary
    if close_gap_frames > 0 and n_frames > 1:
        # 只在时间轴(第 0 轴)做闭运算:结构元 shape (2k+1,1,1) 仅沿帧方向膨胀/腐蚀,
        # **不改变任何帧内的空间形态**。目的是让"第 5 帧有、第 6 帧漏检、第 7 帧又有"
        # 的同一结节保持连通,而不是把两个空间上分离的结节粘起来。
        # 注意:闭运算结果**只用于确定标签归属**,最终实例 mask 仍回落到原始 binary
        # (见下方 `& binary`),避免把补出来的虚构帧写进交付 mask。
        #
        # 不能直接用 `ndi.binary_closing`:其腐蚀步骤默认 `border_value=0`(数组外
        # 视作背景),会把**贴着首帧/末帧边界**的前景整个吃掉——实测 (帧0,帧2) 有、
        # 帧1 漏检的结节,默认 closing 后变成只剩 (帧1,帧2),帧 0 凭空消失。结节出现
        # 在视频首帧或末帧极其常见,该行为等于系统性丢帧。故显式拆成 dilation +
        # erosion(border_value=1,数组外视作前景),使边界不参与腐蚀。
        span = 2 * close_gap_frames + 1
        structure = np.ones((span, 1, 1), dtype=bool)
        dilated = ndi.binary_dilation(binary.astype(bool), structure=structure)
        labeled_src = ndi.binary_erosion(dilated, structure=structure, border_value=1)

    # 26-邻域:帧内 8-邻域 + 跨帧对角,与 2D 的 connectivity=8 口径一致。
    labels, n_labels = ndi.label(labeled_src, structure=np.ones((3, 3, 3), dtype=bool))

    nodules = []
    for i in range(1, n_labels + 1):
        inst = ((labels == i) & (binary != 0)).astype(np.uint8)
        area = int(inst.sum())
        if area < min_area:
            continue

        frame_idx = np.flatnonzero(inst.reshape(n_frames, -1).any(axis=1))
        if frame_idx.size == 0:
            # 闭运算补出的连通体在原始 binary 上完全为空:该实例纯属虚构,丢弃。
            continue
        ys, xs = np.nonzero(inst.any(axis=0))
        zs = frame_idx

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        z0, z1 = int(zs.min()), int(zs.max())
        # 🔴 2026-09-09 内存治理:实例改为 bbox 裁剪稀疏存储(见 split_nodules
        # docstring 的定量依据)。全尺寸 `inst` 仅在本循环体内临时存在,算完
        # bbox/centroid 即拷出裁剪块并 del——**峰值 = 1 个临时全尺寸 + 已存
        # bbox 块**,不再随实例数累积(~17GB → ~几十MB)。全尺寸视图由消费方
        # `restore_instance_mask` 按需重建(逐实例用完即弃)。
        sparse = {
            "array": inst[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1].copy(),
            "origin": (z0, y0, x0),   # 在 (n_frames,H,W) 全尺寸坐标系中的偏移
        }
        del inst
        nodules.append({
            "mask_sparse": sparse,
            "area": area,
            "bbox": (
                x0, y0,
                x1 - x0 + 1, y1 - y0 + 1,
                z0, z1,
            ),
            "centroid": (
                float(np.nonzero(sparse["array"])[2].mean() + x0),
                float(np.nonzero(sparse["array"])[1].mean() + y0),
                float(np.nonzero(sparse["array"])[0].mean() + z0),
            ),
            "frame_indices": [int(z) for z in zs],
        })

    nodules.sort(key=lambda d: d["area"], reverse=True)
    return nodules[:max_nodules]


def restore_instance_mask(sparse: dict, full_shape: tuple) -> np.ndarray:
    """稀疏存储 → 全尺寸 uint8 实例 mask(值域 `{0,1}`,shape == full_shape)。

    写盘(`save_binary_mask`)与 IoU/逐帧消费需要全尺寸/规范形状;bbox 裁剪只用于
    **驻留内存**。逐实例调用方须用完即弃(全尺寸 ≈0.9GB/例,重建是手段不是常态)。

    回归锚:`tests/test_nodule_split.py` 的往返测试断言 restore 结果与旧全尺寸
    实现**逐位一致** + 前景体素总数 == `area`(shape 断言抓不住 origin 偏移错、
    内容错位,体素数才能抓——审查 P2-5)。
    """
    z0, y0, x0 = sparse["origin"]
    arr = np.asarray(sparse["array"])
    out = np.zeros(full_shape, dtype=np.uint8)
    out[z0:z0 + arr.shape[0],
        y0:y0 + arr.shape[1],
        x0:x0 + arr.shape[2]] = arr
    return out


def crop_nodule(image: np.ndarray, bbox, pad_ratio: float = 0.25):
    """按 bbox 从原图裁出结节 patch,四周留 `pad_ratio` 比例的上下文边距。

    逐结节分类需要 patch 而非整图。留边距的理由:TI-RADS 的**边缘**(margin)与
    **腺体外侵犯**(ETE)判据本质上依赖结节与周围组织的**交界形态**,贴边裁剪会把
    判据本身切掉。0.25 是超声结节分类的常用取值(约等于各方向外扩 1/4 结节尺寸)。

    Args:
        image: 2D `(H, W)` 灰度图。
        bbox: `(x, y, w, h)`,或 3D 的 `(x, y, w, h, z0, z1)`(后两项忽略)。
        pad_ratio: 相对 bbox 宽/高的外扩比例,按各自维度独立计算(细长结节不会被
            短边的边距撑成方形)。

    Returns:
        `(patch, (x0, y0, x1, y1))` — patch 为裁剪后的子图(已按图像边界钳位),
        第二项是实际生效的裁剪框,便于把 patch 坐标映射回原图。

    Raises:
        ValueError: image 非 2D,或 bbox 越界到图外导致裁剪结果为空。
    """
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"crop_nodule 只接受 2D 灰度图,收到 ndim={arr.ndim}")

    h_img, w_img = arr.shape
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    pad_x = int(round(w * pad_ratio))
    pad_y = int(round(h * pad_ratio))

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w_img, x + w + pad_x)
    y1 = min(h_img, y + h + pad_y)

    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"crop_nodule: bbox={bbox} 与图像 {arr.shape} 无交集,裁剪结果为空"
        )
    return arr[y0:y1, x0:x1], (x0, y0, x1, y1)


def detection_confidence(prob_mask: np.ndarray, instance_mask: np.ndarray) -> float:
    """结节级检测置信度 = 实例区域内分割概率的均值。

    规范要求每个结节有 `Detection.confidence ∈ [0,1]`,且**动态数据代表整个序列的
    置信度**(病灶级汇总:每个预测病灶仅以其唯一 confidence 进入全测试集排序一次,
    避免同一病灶因多帧被重复计数)。区域内概率均值天然满足:值域 [0,1],3D 实例
    自动是全帧加权平均。

    用**均值**而非最大值:最大值几乎恒为 1.0(任何结节内部都有高置信像素),失去区分
    度,而 AP 排序依赖 confidence 的相对次序 —— 全 1.0 等于随机排序。

    Args:
        prob_mask: 分割**概率**图(sigmoid 后,非二值),shape 需与 instance_mask 一致。
        instance_mask: 实例掩码,非零为前景。

    Returns:
        float ∈ [0,1]。实例为空时返回 0.0。

    Raises:
        ValueError: 两者 shape 不一致(静默广播会得到无意义的数值),或区域内概率
            含 NaN/Inf(见下)。
    """
    p = np.asarray(prob_mask, dtype=np.float64)
    m = np.asarray(instance_mask) != 0
    if p.shape != m.shape:
        raise ValueError(
            f"detection_confidence: prob_mask{p.shape} 与 instance_mask{m.shape} "
            f"形状不一致"
        )
    if not m.any():
        return 0.0

    mean = p[m].mean()
    # NaN 硬拦(QA_VP §1.7「概率/置信度必须有限」)。**不能只靠 np.clip** ——
    # `np.clip(nan, 0, 1)` 返回 nan 而非钳位值,NaN 会一路流进 prediction.json,
    # 而 `json.dumps` 把它写成字面量 `NaN`(非标准 JSON),Go/Rust/Jackson 等严格
    # 解析器会拒绝**整份**文件 ⇒ 该样本零输出,比预测错更严重。
    # 下游 `build_nodule_entry` 虽也会拦,但源头不拦等于依赖调用方永远走那条路径。
    if not np.isfinite(mean):
        raise ValueError(
            "detection_confidence: 实例区域内概率均值非有限(NaN/Inf)——"
            "分割头 logits 数值异常或权重损坏,拒绝导出无效置信度"
        )
    return float(np.clip(mean, 0.0, 1.0))
