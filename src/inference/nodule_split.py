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

import math
import os

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

# 零训练四件套④(2026-09-10):静态路径的最小面积占比可经 env 放宽(AP 单调性——
# 尾部追加低 conf 小实例,检测 AP 只涨不跌,Dice 不受影响)。**只作用于静态**:
# 动态放宽会增多 3D 实例,写盘耗时与内存都是 8h 时限下的风险面,动态生产调用点
# 显式传 DEFAULT_MIN_AREA_RATIO 钉死不走 env(见 v21_output._write_dynamic)。
_ENV_MIN_AREA_RATIO = "AICOMP_MIN_AREA_RATIO"

# 阶段1(2026-09-13,Opus 路线方向A):预测结节数上限 env 化。None 哨兵→读 env
# 模式(照 _ENV_MIN_AREA_RATIO 先例),调用点(静态生产/动态生产/诊断镜像静/动
# ——4 个非测试调用点)默认同源——只改 v21_output 调用点会让诊断镜像静默停在
# 20,离线测不到生产行为。
# 🔴 前置条件:n_pred==cap 截断饱和率(analyze_diag_sidecar)证明截断在绑才设 env,
# 盲开 = 零收益正成本(动态写盘 O(全尺寸体素),本地实测最坏动态例
# 750×802×1442 约 13.6s/实例,20→50 每例 +400s 量级,数百饱和例对 8h 硬限
# 是数小时级风险)。不饱和的池设了也只在 n_pred≥cap 时才生效,天然零影响。
# 🔴 单一 env 静/动两池同时生效(无法分池独立)——设之前必须确认**两池**的
# n_pred 饱和率(analyze_diag_sidecar 分池报),任一侧不饱和则不开。
# 诊断扫参轴 `max_nodules` 显式传参优先于 env(扫参进程必须确定性)。
_ENV_MAX_NODULES = "AICOMP_MAX_NODULES"
# env 上界(审查 P2:1e308 是有限正整数、通过全部合法性检查,nodules[:1e308]
# 不炸 → 单个 env 即静默关掉 DEFAULT_MAX_NODULES 的"防分割崩坏撑爆磁盘"兜底;
# ceiling=100 ≈ 最坏写入量已验证量级,放开决策的候选值 30/50 远在其下)。
_MAX_NODULES_CEILING = 100

# ── Z2 实例级收缩(2026-09-22,新架构线零训练第二波,默认关)────────────────
# 设计规格(设计文档已删,规格要点=3×3 腐蚀×k 预注册;实现记录见总纲 §6.4 Z2 条):
#   算法 = 形态学腐蚀 3×3 ×k(k∈{1,2} 预注册扫参域)。
#   为什么选腐蚀而非概率域截尾/实例级阈值重切(另外两个预注册候选):
#   - 阈值重切在 hysteresis 恒等(grow 已判死默认关)下与全局 {soft>thr} 连通域
#     逐位等价 = 踩"全局阈值扫参已判死"边界(阈值-AP 耦合),排除;
#   - 截尾需要 soft 概率进 _split_2d(签名加参+生产调用点改),违反设计文档
#     "消费端 v21_output._write_static 不重写"挂钩约束;腐蚀只需 binary,
#     照 _clean_2d/_ENV_MIN_AREA_RATIO 先例纯 env 驱动,消费端零改动自动生效。
#   判死线(预注册):Δ匹配≥+0.012 且 ΔAP≥+0.008 vs ep16(0.8478/0.8250);
#   Dice 必须同看(收缩直接改 mask 尺寸,预检 r>1 占 61.4% 但集中 0.8~1.2 带)。
# 🔴 边界(勿重蹈,设计文档 Z2 节):不碰 hysteresis grow 路径(滞回生长已判死)、
#   不动二值化阈值(全局阈值扫参已判死)、只做**实例层面**收缩。
_ENV_SHRINK = "AICOMP_STATIC_SHRINK"
_ENV_SHRINK_K = "AICOMP_STATIC_SHRINK_K"
_SHRINK_KERNEL = np.ones((3, 3), np.uint8)


def _shrink_enabled() -> bool:
    """AICOMP_STATIC_SHRINK 三态开关(畸形回关——未验证路径的安全侧=关)。

    与 tiling/_clean_2d 同款解析:未设/"0"/垃圾串一律关,判死回退安全侧。
    """
    raw = os.environ.get(_ENV_SHRINK, "0").strip().lower()
    return raw in ("1", "true", "on", "yes")


def _env_shrink_k() -> int:
    """开关开时的 K 解析:未设=1(最小腐蚀);垃圾串回 1;数值 ∉{1,2} fail-loud。

    域 {1,2} 是预注册扫参域(设计文档 Z2 算法规格 2),越界=临场发明,
    照 tiling tile≠512 先例 raise 不静默钳(静默钳会让未注册配方溜上测评)。
    """
    raw = os.environ.get(_ENV_SHRINK_K)
    if raw is None:
        return 1
    try:
        k = int(raw)
    except (TypeError, ValueError):
        return 1
    if k not in (1, 2):
        raise ValueError(
            f"AICOMP_STATIC_SHRINK_K 须在 {{1, 2}}(预注册扫参域,设计文档 Z2 节),"
            f"收到 {k}——越域值须先扩预注册规格,不静默钳")
    return k

# ── 静态 2D 形态学清理(Fable 低成本点,2026-09-12;🔴 默认关,离线双升才开)──
# 预测 mask 上的两类噪声:① 孤立小连通域(碎片)→ 变成假阳性结节拖检测 AP;
# ② 结节内部孔洞(概率中间塌陷)→ 蚀 Dice 边缘。清理 = 填孔 + 小碎片剔除。
# **默认关的依据**:① 碎片其实已被 min_area_ratio 过滤(1e-4 ≈ 26px@512²,
#   碎片要活过过滤得先有 26px,而 26px 的"碎片"可能本身就是低置信小结节,
#   剔它 = 丢 AP 尾部);② 填孔会扩 mask 边界,Dice 在欠分割侧才赚、过分割侧
#   反亏——方向未经验证。Fable 条款:离线回放 Dice/AP **双升**才开。
# env 三态:"0"=关(默认)/"1"=开/其它值=关(垃圾配置回安全值)。
_ENV_CLEAN_2D = "AICOMP_CLEAN_2D"
# 闭运算迭代数(kernel 恒 3×3):iterations=1 只缝 1-2px 缝。⚠️ 该值是
# **迭代数不是半径**:设 2 等效 5×5 膨胀,会把独立小结节粘到主结节上
# (虚减结节数,AP 直接掉),别动。
_CLEAN_2D_ITER = 1


def _clean_2d_enabled() -> bool:
    raw = os.environ.get(_ENV_CLEAN_2D)
    return raw is not None and raw.strip() == "1"


def _env_min_area_ratio() -> float:
    """env 解析静态 min_area_ratio;未设/非法一律回默认(不静默采信垃圾配置)。

    🔴 非 finite 必须挡(A路审查P2):inf 会经 int(ratio*h*w) 抛 OverflowError
    炸掉整例;nan 在 >=0.0 比较下侥幸回默认但属巧合非设计。
    """
    raw = os.environ.get(_ENV_MIN_AREA_RATIO)
    if raw is None:
        return DEFAULT_MIN_AREA_RATIO
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_MIN_AREA_RATIO
    if not math.isfinite(v) or v < 0.0:
        return DEFAULT_MIN_AREA_RATIO
    return v


def _env_max_nodules() -> int:
    """env 解析 max_nodules;未设/非法/非正整数/超上界一律回默认(不静默采信垃圾配置)。

    - 非整数(20.5)回默认是**策略选择**(与扫参轴的报错策略对齐),不是
      TypeError 防护——防 TypeError 的是 int() 转换本身;
    - 🔴 0 必须并入非法集:0 对 min_area 是合法值(不过滤),对 max_nodules
      是**归零实例数**的灾难值(nodules[:0]==[] → 每例零结节 → 静默全零分,
      且空 Nodules 数组是合规阴性输出、不报错);
    - 上界 _MAX_NODULES_CEILING:防 1e3/1e308 类合法大整数静默关掉
      DEFAULT_MAX_NODULES 的磁盘兜底(见常量注释)。
    """
    raw = os.environ.get(_ENV_MAX_NODULES)
    if raw is None:
        return DEFAULT_MAX_NODULES
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_NODULES
    if (not math.isfinite(v) or v < 1.0 or v != int(v)
            or v > _MAX_NODULES_CEILING):
        return DEFAULT_MAX_NODULES
    return int(v)


def hysteresis_binarize(soft: np.ndarray, thr_seed: float,
                        thr_grow: float) -> np.ndarray:
    """滞回阈值二值化(阶段3 手段4,2026-09-16):高阈值成核 + 低阈值描边。

    尺度规则:soft 是**任意尺寸**的概率图,函数不 resize。生产在模型分辨率
    (512²)上调用(与 `postprocess.decode_predictions` 的既有二值化点同源),
    之后由 `v21_output._resize_binary`(INTER_NEAREST)升到原图 —— **不要**
    挪到原图尺寸上做:那会与 `soft_fullres` 的 INTER_LINEAR 插值产生二次误差,
    且 conf 消费的是另一条链路。

    与"全局降阈值"的本质区别(这是本函数存在的唯一理由):
    - 全局降阈值:`soft > thr_low` 让新的低概率区域**独立成核**,实例候选集变大;
    - 本函数:`seed = soft > thr_seed` 决定**谁能成核**,`grow = soft > thr_grow`
      只负责把已成核实例的轮廓外延;**不含任何 seed 的 grow 连通域整块丢弃**
      —— 低概率区域永远无法独立成核,实例候选集与 `soft > thr_seed` 逐位相同。

    跨实例接触面切割(🔴 实现要点,不是可选优化):
    同一 grow 连通域可能含**多个 seed 实例**(低阈值区域把两个高阈值核桥连)。
    仅按"最近 seed 归属"给像素打标签**不解决问题** —— 输出的仍是二值图,
    归属后两核在 8/26 邻域下依然相邻,下游 `split_nodules` 照样合并成一个
    实例(实测:20px 一行两个 seed + 全桥接,归属后连通域数仍为 1)。
    正解是把接触面上的**非 seed 像素**删掉,使两核在二值图上断开;seed 像素
    **恒保留**(接触面判据显式排除 seed),故即使核只有 1px 宽也不会被削空。

    Args:
        soft: 2D `(H, W)` 或 3D `(n_frames, H, W)` 分割**概率**图(sigmoid 后)。
        thr_seed: 成核阈值。`soft > thr_seed` 的像素为种子。
        thr_grow: 生长阈值,须 ≤ thr_seed 才有效(生长语义)。
            **`thr_grow >= thr_seed` 时逐位退化为 `soft > thr_seed`** —— 这既是
            默认关闭态(`postprocess._grow_threshold` 默认返回 thr_seed),也是
            钉死零回归的恒等路径。⚠️ 该分支在 scipy 检查**之前**返回,故默认
            路径不引入 scipy 依赖。

    Returns:
        np.uint8 数组,shape 与输入一致,值域 `{0, 255}`(与
        `postprocess.py:148` 原契约一致;下游只看非零性,但 `_resize_binary`
        走 INTER_NEAREST,值域不一致会埋雷)。

    Raises:
        ValueError: ndim 非 2/3。
        RuntimeError: 生长路径需要 scipy,而 scipy 未安装(恒等路径不触发)。
    """
    arr = np.asarray(soft)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"hysteresis_binarize 只接受 2D (H,W) 或 3D (n_frames,H,W),"
            f"收到 ndim={arr.ndim} shape={arr.shape}")

    # 恒等退化:默认关闭态即走这里。**必须在 scipy 检查之前**(见 docstring)。
    if thr_grow >= thr_seed:
        return (arr > thr_seed).astype(np.uint8) * 255

    if not HAS_SCIPY:
        raise RuntimeError(
            "hysteresis_binarize 的生长路径需要 scipy.ndimage,但 scipy 未安装")

    seed = arr > thr_seed
    grow = arr > thr_grow
    # 2D 用 (3,3)=8 邻域、3D 用 (3,3,3)=26 邻域,与 `_split_3d:302` 的
    # connectivity 口径一致(2D 侧 `_split_2d` 用 cv2 connectivity=8,等价)。
    struct = np.ones((3,) * arr.ndim, dtype=bool)

    lab_grow, _ = ndi.label(grow, structure=struct)
    out = np.zeros(arr.shape, dtype=bool)
    # find_objects 给每域的 bbox 切片:EDT 只在"多 seed 域"的 bbox 内做,
    # 而非全图 —— 若未来在 3D 大体积(如 750×512×512 离线工具)上调用,全尺寸
    # EDT 是秒级成本;单 seed 域(真实数据绝大多数)完全跳过 EDT。
    # ⚠️ 生产路径静/动都是**逐帧 2D**调用(case_pipeline 逐帧 decode),
    # 3D 分支仅供测试与未来离线工具;3D 大体积下 lab_grow int32 全尺寸
    # 数组本身即是内存大头,须先做预算再启用。
    for i, sl in enumerate(ndi.find_objects(lab_grow), start=1):
        sub_grow = lab_grow[sl] == i
        # thr_grow < thr_seed(恒等分支已拦)⇒ seed ⊆ grow 逐位成立,无需再交。
        sub_seed = seed[sl]
        if not sub_seed.any():
            # 不含种子的低概率连通域 —— 整块丢弃。这正是本函数与"全局降阈值"
            # 的唯一分野,对应测试的"孤立低概率区域不成核"核心断言。
            continue
        lab_seed, n_seed = ndi.label(sub_seed, structure=struct)
        if n_seed == 1:
            out[sl] |= sub_grow
            continue

        # 多 seed 桥连:最近种子归属 + 切开接触面。
        _, inds = ndi.distance_transform_edt(~sub_seed, return_indices=True)
        owner = lab_seed[tuple(inds)]
        # EDT 对全切片每像素都给最近 seed 索引 → 非 grow 像素也会拿到标签,
        # 必须显式清零,否则边界判据会把域外像素算进来。
        owner = np.where(sub_grow, owner, 0)

        # 接触面 = 邻域内出现不同 owner 的 grow 像素。
        # maximum_filter 直接可用(0 天然最小);minimum_filter 须把 0 换成
        # sentinel 大值,否则域外/背景的 0 会把 mn 拉到 0,使**每个**非 seed
        # 像素都满足 mx!=mn → 整个域的外轮廓被当接触面删光(实测该误删会让
        # 输出 mask 收缩一圈甚至消失)。
        pos_max = owner
        pos_min = np.where(owner > 0, owner, np.iinfo(np.int32).max)
        mx = ndi.maximum_filter(pos_max, size=3, mode="constant", cval=0)
        mn = ndi.minimum_filter(pos_min, size=3, mode="constant",
                                cval=np.iinfo(np.int32).max)
        boundary = (mx != mn) & sub_grow & ~sub_seed
        out[sl] |= (sub_grow & ~boundary)

    return out.astype(np.uint8) * 255


def split_nodules(
    mask: np.ndarray,
    min_area_ratio=None,
    max_nodules=None,
    close_gap_frames: int = 1,
):
    """整图二值 mask → 按面积降序的实例列表。

    Args:
        mask: 2D `(H, W)` 静态图,或 3D `(n_frames, H, W)` 动态图。**任意非零视作
            前景**(兼容内部 0/255 的 `mask_binary`),内部统一二值化。
        min_area_ratio: 最小面积占比(相对单帧 H×W)。小于此值的连通域视为噪声丢弃。
            传 0 表示不过滤。**None(默认)时走 `AICOMP_MIN_AREA_RATIO` env**
            (件套④:仅静态生产调用方靠默认参吃 env;动态生产调用方显式传
            `DEFAULT_MIN_AREA_RATIO` 不受 env 影响)。显式传参时 env 不生效
            ——诊断脚本扫参/测试靠这一层优先级保持确定性。
        max_nodules: 保留的实例数上限,按面积降序截断。**None(默认)时走
            `AICOMP_MAX_NODULES` env**(阶段1方向A:三调用点默认同源,显式传参
            优先于 env——扫参确定性同上)。
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
    # 件套④三层优先级:显式传参 > env(None 时) > DEFAULT(env 未设时)。
    if min_area_ratio is None:
        min_area_ratio = _env_min_area_ratio()
    if max_nodules is None:
        max_nodules = _env_max_nodules()
    if arr.ndim == 2:
        return _shrink_instances_2d(_split_2d(arr, min_area_ratio, max_nodules))
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

    # 形态学清理(env 开关,默认关,见 _ENV_CLEAN_2D 注释):
    # ① 3×3 闭运算填 1px 孔/缝;② 边界 flood-fill 填孔。
    # 🔴 填孔种子必须**四角都试**(Gate2 审查 P1,2026-09-12 实测反例:前景竖带
    #   贯通上下边时,(0,0) 所在背景与右侧背景不连通 → flood 未达的整块右侧
    #   被当"孔洞"灌成前景,单实例面积 53% 图幅;左缘条带+真结节场景更炸成
    #   98% 图幅的单个巨型 blob,Dice/AP 双灭。四角任一命中背景即从它灌;
    #   四角全是前景(极端)跳过填孔,宁少填不误扩)。
    # 🔴 holes 面积护栏:填孔后前景占比超 0.5 → 放弃清理回退原始 mask
    #   (flood 静默失败的失败模式是"整图变前景"而非"清理没生效",变异实测
    #   floodFill no-op 后实心圆 4096/4096 全前景;占比护栏是这条无界失败
    #   面的硬兜底,超过 min(seg_over_half 护栏口径, 0.5) 必然是异常)。
    if _clean_2d_enabled():
        k = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k,
                                  iterations=_CLEAN_2D_ITER)
        h_, w_ = binary.shape[:2]
        seed = next(((y, x) for y, x in ((0, 0), (0, w_ - 1), (h_ - 1, 0),
                                         (h_ - 1, w_ - 1))
                     if binary[y, x] == 0), None)
        if seed is not None:
            ff_mask = np.zeros((h_ + 2, w_ + 2), np.uint8)
            flood = binary.copy()
            cv2.floodFill(flood, ff_mask, seed, 1)
            holes = (flood == 0)            # flood 未达 = 孔洞
            cleaned = ((binary != 0) | holes).astype(np.uint8)
            if float(cleaned.mean()) <= 0.5:
                binary = cleaned
            else:
                print("--> [WARNING] mask 清理填孔后前景占比>0.5(异常 flood),"
                      "回退原始 mask", flush=True)

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


def _shrink_instances_2d(nodules: list) -> list:
    """Z2 实例级收缩(2026-09-22,默认关):对已成实例的 mask 做 3×3 腐蚀 ×k。

    **实例化之后腐蚀**(而非 _split_2d 入口先腐蚀再连通域)的理由:
    - 入口腐蚀会把贴邻的两实例接触面一起削、可能改变连通域数(n_pred 变),
      而 Z2 的对症假设是"已检出实例尺寸偏大",实例数不应变——扫参对拍
      ΔAP 时 n_pred 漂移会污染归因;
    - 实例级腐蚀天然满足"1px 实例收缩后不得为空"(设计文档测试规格 3):
      腐蚀后 area==0 的实例整例回退不腐蚀(空实例=丢检测=AP 负涨);
    - 动态路径零触及:本函数只被 2D 消费链调用(设计文档"动态不做")。

    与已判死手段的边界:不碰 hysteresis grow(滞回生长已判死)、不动
    decode 的二值化阈值(全局阈值扫参已判死)——本函数在 split 之后、
    纯实例层面。bbox/area/centroid 全部按收缩后 mask 重算(bbox 不外扩,
    面积单调不增,设计文档测试规格 2)。
    """
    if not _shrink_enabled():
        return nodules
    k = _env_shrink_k()
    out = []
    for inst in nodules:
        # 🔴 入口压 {0,1}(路B审查 P2-1,2026-09-22):上游 `_split_2d` 出口
        # 恒 {0,1}(当前不可达),但本函数是模块公共符号,未来复用方喂 0/255
        # mask 时:①正常分支——erode 是 min 滤波,255 值会原样保留,`area`
        # 键与前景数差 255 倍、`detection_confidence` 的 `mask != 0` 区域不
        # 变但 sum 语义崩;②回退分支——255 透传同型。入口一次压域(每实例
        # O(H×W) 一次比较,零成本),两分支同时免疫。
        m = (np.asarray(inst["mask"]) != 0).astype(np.uint8)
        shrunk = cv2.erode(m, _SHRINK_KERNEL, iterations=k)
        if int(shrunk.sum()) == 0:
            # 腐蚀到空 = 丢该检测(AP 负涨路径,设计文档测试规格 3 的负例):
            # 回退不腐蚀,保住实例(宁可偏大不丢检测)。mask 已在入口压域。
            out.append({**inst, "mask": m})
            continue
        ys, xs = np.nonzero(shrunk)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        out.append({
            "mask": shrunk.astype(np.uint8),
            "area": int(shrunk.sum()),
            "bbox": (x0, y0, x1 - x0 + 1, y1 - y0 + 1),
            "centroid": (float(xs.mean()), float(ys.mean())),
        })
    # 面积降序须按收缩后 area 重排(排序截断消费面积序,原序会让大实例的
    # 残影占位;_split_2d 出口契约就是面积降序,保持不变量)。
    # (max_nodules 截断不在此重复:收缩只减面积不减实例数,截断集不变。)
    out.sort(key=lambda d: d["area"], reverse=True)
    return out


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
