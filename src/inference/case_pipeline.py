"""单病例推理核心(静态图 + 动态图),从 `predict.py`/`video_predict.py` 抽出。

## 为什么抽这一层

V2.1 交付把形态从「entrypoint.sh 单文件离线 CLI」改为「HTTP 常驻批量服务」
(`src/service/`)。服务层 runner 要对一个目录里的**批量病例**逐个推理,不能复用
CLI 的 `if __name__=="__main__"` 主函数。故把两个 CLI `main()` 里「输入加载完成 →
落盘之前」那段**纯推理核心**抽到此处,CLI 与 service 共用同一份诊断逻辑,只在
两端各自落盘(CLI 落 V2.0 `result.json`,service 落 V2.1 `prediction.json`)。

## 边界:什么在这里,什么留在调用方

**留在调用方(main/runner)**:
- `build_model` / `TemperatureScaler` 构造 —— 有构建成本,service 要跨病例复用一次;
- 输入加载(`load_input_image`/`load_video_frames`/`load_nifti_frames`)—— 含 DICOM
  内存 PHI 匿名化(2026-09-01 起不写盘,见 io.py docstring),且失败要落 rejection
  JSON(控制流属调用方);
- gatekeeper 审计 —— 有副作用(逐帧审计、写 rejection),且 hard 模式的一票否决
  return 语义属调用方控制流;
- 落盘(V2.0 `save_prediction_artifacts` / V2.1 `write_case_v21`)与 Grad-CAM(CLI 专属)。

**在这里(纯核心,无副作用、无落盘)**:
- 静态:`preprocess → run_forward → decode → mask resize → tirads breakdown`;
- 动态:逐帧诊断(坏帧隔离)+ 聚合 + 关键帧选择 + 关键帧张量防错位。

## 无有效结果的两个分支

动态图两种「无法产出任何诊断」的情形(全帧数值失败 / 无有效关键帧)在原 `main()`
里是 `save_rejection_json` + `return`。抽出后本层不落盘,改为 `raise DynamicCaseError`
携带诊断字段,由调用方转成对应 rejection —— 保持与原行为逐字节一致(reviewer P)。
"""

import os

import cv2
import torch

from src.inference.model_runner import preprocess_image, run_forward
from src.inference.postprocess import decode_predictions, build_tirads_breakdown
from src.inference.video_aggregate import aggregate_frame_predictions
# NonFiniteProbabilityError 逐帧兜底信号;AssertionError 为帧形状非法(preprocess 校验)。
from src.inference.postprocess import NonFiniteProbabilityError

# --- 动态批量前向(2026-09-09) ---
# 全帧前向(检测 AP 药方,`_DEFAULT_MAX_FRAMES=0`)后单动态例可达 750 帧,逐帧 batch=1
# 前向 = 750 次 GPU 调用,kernel 启动开销占大头。攒帧批量前向后动态侧提速 2-5 倍。
_ENV_VIDEO_BATCH = "AICOMP_VIDEO_BATCH"
_DEFAULT_VIDEO_BATCH = 8   # 最小部署面安全值;云端 32G 卡实测单帧激活峰值后可经 env 调 16/32

# --- 静态 hflip TTA(D2,2026-09-12,Fable 收官路线) ---
# 只对静态图水平翻转一次再前向,只平均**分割概率图**;分类/位置/ETE/T/TI-RADS
# 一律不融合(各头对翻转不变性的假设未经数据验证,位置头更是 🔴 **绝不能 flip**
# —— 左右叶语义随翻转直接互换,融合会把 9.36 分的位置 AUC 打崩)。
# 动态图不做 TTA(全帧前向本来就贵,翻倍前向撞 8h 时限风险)。
# 默认开(env 设 "0"/"false" 关闭):本地+云端对拍通过后就是交付形态,默认关
# 会让"忘了设 env"静默退化回无 TTA,与零训练四件套①"改代码默认值而非设 env"
# 的部署纪律同理——**测评容器不重建、env 分栏改不了**。
_ENV_TTA = "AICOMP_STATIC_TTA"


def _static_tta_enabled() -> bool:
    """解析 AICOMP_STATIC_TTA(畸形值回默认 True,与 _seg_threshold 同模式)。"""
    raw = os.environ.get(_ENV_TTA, "1").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    return True


def _resolve_video_batch_size():
    """解析 AICOMP_VIDEO_BATCH(畸形回默认;"0"/负数钳到 1 = 显式回退逐帧旧行为,对拍用)。

    🔴 合法解析的 "<=0" **钳到 1 而非回默认**:batch=1 必须是显式的"我要旧行为"开关,
    回默认会让 `AICOMP_VIDEO_BATCH=1` 的对拍静默失效(跑的还是批量)。
    """
    raw = os.environ.get(_ENV_VIDEO_BATCH, str(_DEFAULT_VIDEO_BATCH))
    try:
        b = int(raw)
    except ValueError:
        return _DEFAULT_VIDEO_BATCH
    return max(1, b)


# OOM 判定:torch>=2.4 的 CUDA OOM 专用异常类型 + 消息子串兜底(cuDNN/CUBLAS 抛的裸
# RuntimeError 形态)。两个方向的误判都有兜底:误判"OOM"实为真故障 → 对半递归到 B=1 时
# 任何 RuntimeError 无条件冒泡,故障必然浮出;误判"非 OOM"实为显存不足 → 立刻冒泡成
# 整例 ERROR,与旧代码(逐帧 B=1 也可能 OOM)行为一致,批量化没有更糟。
# (A 路审查 P2:不含裸 "oom" 子串——它是 room/bloom 的子串,消息含这些词的
# RuntimeError 会被误判进重试;标准 OOM 文案已被前两判据覆盖,该兜底收益近零。)
_OOM_MARKERS = ("out of memory", "not enough memory", "cuda error: oom")


def _is_oom(exc):
    if isinstance(exc, getattr(torch, "OutOfMemoryError", ())):
        return True
    msg = str(exc).lower()
    return any(m in msg for m in _OOM_MARKERS)


def _slice_outputs(outputs, i):
    """批量 outputs → 第 i 个单样本视图 dict(dim>0 的键切 [i:i+1],零拷贝)。

    `decode_predictions` 全文硬取 `[0]`(单样本语义,不动它——改它是最大风险源),
    本切片器是唯一适配层:切出的 [1,...] 形状与 B=1 前向逐位一致(mask [1,1,H,W]→[0,0]、
    *_probs [1,C]→[0]、t_conf [1]→[0])。`spin_l1` 是 dim==0 标量,天然不切、原样共享,
    decode 不消费它;ModelOutputs 全部键均为 tensor,无 None/int 值(审查逐键核过)。
    """
    return {k: (v[i:i + 1] if torch.is_tensor(v) and v.dim() > 0 else v)
            for k, v in outputs.items()}


def _forward_frames(model, img_tensors, precision, device):
    """一窗帧张量([1,1,H,W] × N)→ 逐样本切片后的 outputs dict 列表(与输入等长同序)。

    OOM 且 N>1:empty_cache 后对半递归;N==1 或非 OOM 的任何 RuntimeError 原样冒泡
    (真故障交调用方转 ERROR)。B=1 兜底 = 降级的终止保证 + 故障不吞保证,二者合一。
    torch.cat 是拷贝:每帧 img_tensor 独立分配,批前向后原张量完好,kf 跟踪直接引用。
    """
    if not img_tensors:
        return []
    try:
        outputs = run_forward(model, torch.cat(img_tensors, dim=0),
                              precision, device)
        return [_slice_outputs(outputs, i) for i in range(len(img_tensors))]
    except RuntimeError as e:
        if len(img_tensors) == 1 or not _is_oom(e):
            raise
        # 🔴 递归前必须解开异常引用(A 路审查 P1,2026-09-09):`except ... as e` 的
        # e.__traceback__ 持有失败调用栈各帧的 f_locals,把整批中间激活钉成
        # "使用中"——empty_cache 只能归还无引用的 cached blocks,不先解引用,
        # 降级递归会在失败批的显存还占着的情况下再叠重试批,最坏连环失败后
        # 以 B=1 OOM 冒泡成整例 ERROR(而旧逐帧代码同场景能跑)。
        e.__traceback__ = None
        del e
        if device.type == "cuda":
            # OOM 后 allocator 的 cached blocks 不释放,直接对半重试可能仍撞同一堵墙
            torch.cuda.empty_cache()
        mid = len(img_tensors) // 2
        print(f"--> [WARNING] 批量前向 OOM(batch={len(img_tensors)}),"
              f"降级 {mid}+{len(img_tensors) - mid} 重试", flush=True)
        return (_forward_frames(model, img_tensors[:mid], precision, device)
                + _forward_frames(model, img_tensors[mid:], precision, device))


def _decode_frame_sample(outputs, frame_np, calibrator=None, *, dynamic=False):
    """单样本切片 outputs → decode_predictions(测试注入点)。

    `frame_np` 生产不消费,仅供测试指纹反查帧号(延续 `_fake_diagnose` 的设计手法,
    显式声明的测试缝)——bad-frame 注入测试依赖它定位"第几帧"。
    `dynamic` 透传给 `decode_predictions` 选 `AICOMP_SEG_THRESHOLD_DYNAMIC`
    (件套①:动态帧的分割阈值独立于静态,生产动态帧循环恒传 True)。
    """
    return decode_predictions(outputs, calibrator=calibrator, dynamic=dynamic)


class DynamicCaseError(Exception):
    """动态图无有效诊断结果(全帧失败 / 无有效关键帧)。

    携带 `reason` 与 `diagnostic`(n_frames_* / failed_* 字段),供调用方原样落
    rejection JSON —— 本层不落盘,把「怎么落」的控制流交回 main/runner(与原
    `video_predict.main()` 的两处 `save_rejection_json`+`return` 逐字节等价)。
    """

    def __init__(self, reason, diagnostic):
        super().__init__(reason)
        self.reason = reason
        self.diagnostic = diagnostic


def diagnose_static_case(model, img_np, config, calibrator, precision, device):
    """单张静态图的推理核心(输入已加载+已过 gatekeeper)。

    Args:
        model: 已 `build_model` 的模型(调用方持有,service 跨病例复用)。
        img_np: 已加载(且必要时匿名化)的 uint8 灰度图 HxW。
        config: 解析后的配置 dict(内部取 `data.img_size`)。
        calibrator: `TemperatureScaler` 或 None(全 T=1.0 时逐位等同无校准)。
        precision: "fp32"/"fp16"/"bf16"。
        device: torch.device。

    Returns:
        dict:
          - `preds`: `decode_predictions` 全量输出(含软 `pred_mask`、`mask_binary`、
            各头 `*_idx`/`*_prob`、`mal_prob`/`mal_probs`);
          - `img_tensor`: `preprocess_image` 产出的普通张量(供调用方在 inference_mode
            之外重跑 Grad-CAM 复用);
          - `mask_resized`: `mask_binary` 用 `INTER_NEAREST` 缩放回原图 HxW(V2.0 落盘用);
          - `breakdown`: `build_tirads_breakdown` 的 TI-RADS 分项。

    Raises:
        NonFiniteProbabilityError: 概率/mask 含 NaN/Inf(`decode_predictions` 硬拦),
            由调用方转 ERROR —— 与原 `predict.main()` 的 except Exception 兜底一致。
    """
    img_size = config["data"]["img_size"]
    img_tensor = preprocess_image(img_np, img_size, device)
    outputs = run_forward(model, img_tensor, precision, device)

    # --- 静态 hflip TTA(D2,2026-09-12) ---
    # 翻转前向 → sigmoid 概率 → 逆 flip 回原坐标 → 与原图概率**逐像素平均**。
    # 🔴 平均的是概率不是 logits(sigmoid(平均logits) ≠ 平均(sigmoid(logits)),
    # 且 sigmoid 非线性下前者无概率语义);🔴 逆变换先于平均(直接平均翻转概率
    # = 病灶左右镜像错位叠加)。分类各头**不融合**:全部走原图 outputs(下方
    # decode 的分类路径读的就是这份 outputs,与无 TTA 时逐位一致)。
    # 回退策略(Gate2 两路审查 F1/F2 修正,2026-09-12):
    #   ① 只在翻转前向**这一步**罩异常,sigmoid/平均留在 try 外——那两行纯张量
    #     运算无独立故障面,罩进去只会把"值异常"也吞成静默回退;
    #   ② except 收窄到 OOM(`_is_oom`,与 `_forward_frames` 的既有策略同源:非 OOM
    #     的 RuntimeError 是真故障必须冒泡,不能静默劣化成"无 TTA 照常跑"——那
    #     会让整批处于半 TTA 状态且日志只有 warning,D2-3 的 GO/ROLLBACK 对拍
    #     判据随之失真);
    #   ③ NonFiniteProbabilityError 不拦:翻转前向抛 NaN 说明数值层面已异常,
    #     静默回退会掩盖它,按既有硬拦哲学冒泡交调用方落 ERROR。
    mask_prob_override = None
    if _static_tta_enabled():
        try:
            outputs_flip = run_forward(
                model, torch.flip(img_tensor, dims=[3]), precision, device)
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            print(f"--> [WARNING] TTA 翻转前向 OOM,回退单前向({e})", flush=True)
            outputs_flip = None
        if outputs_flip is not None:
            prob_orig = torch.sigmoid(outputs["mask"].float())[0, 0]
            prob_flip = torch.flip(
                torch.sigmoid(outputs_flip["mask"].float())[0, 0], dims=[1])
            mask_prob_override = (prob_orig + prob_flip) / 2.0

    preds = decode_predictions(outputs, calibrator=calibrator,
                               mask_prob_override=mask_prob_override)
    mask_resized = cv2.resize(
        preds["mask_binary"], (img_np.shape[1], img_np.shape[0]),
        interpolation=cv2.INTER_NEAREST)
    breakdown = build_tirads_breakdown(
        preds["comp_idx"], preds["echo_idx"],
        preds["shape_idx"], preds["margin_idx"], preds["foci_prob"])
    return {
        "preds": preds,
        "img_tensor": img_tensor,
        "mask_resized": mask_resized,
        "breakdown": breakdown,
    }


def _diagnose_frame(model, frame_np, precision, device, img_size, calibrator=None,
                    *, dynamic=False):
    """单帧前向 → 解码,返回 (preds, img_tensor)。img_tensor 留作关键帧 Grad-CAM 复用。

    与原 `video_predict.diagnose_frame` 同义(该函数保留为向后兼容 shim)。
    `dynamic` 透传 decode(件套①);唯一生产调用方 kf 一致性锁重算分支传 True
    与动态主循环同阈值(其 preds 本就只取张量,传值是为语义一致)。
    """
    img_tensor = preprocess_image(frame_np, img_size, device)
    outputs = run_forward(model, img_tensor, precision, device)
    return decode_predictions(outputs, calibrator=calibrator,
                              dynamic=dynamic), img_tensor


def diagnose_dynamic_case(model, frames, frame_indices, config, calibrator,
                          precision, device, *, batch_size=None):
    """动态图(cine)的推理核心:逐帧诊断 + 聚合 + 关键帧(输入已抽帧+已过 gatekeeper)。

    坏帧隔离(与原 `video_predict.main()` 逐字节一致):`NonFiniteProbabilityError`
    (概率/mask NaN/Inf)与 `AssertionError`(帧形状/维度非法)**只丢该帧不拖垮整段**;
    `RuntimeError`(CUDA OOM/权重维度不符)、`cv2.error` 等真故障**冒泡**由调用方转 ERROR。

    关键帧张量防错位(与原实现一致):`img_tensor`(GPU 张量)只保留**当前最恶性帧**
    那一个(边扫边留 argmax 帧,显存从 N 帧降到 1 帧,防长视频显存/耗时 OOM);判据用
    严格 `>`,与 `aggregate_frame_predictions` 的 `max`(相等取首个)严格同序。

    Args:
        model/config/calibrator/precision/device: 同 `diagnose_static_case`。
        frames: 已抽样的帧列表(uint8 灰度 HxW),gatekeeper 已在调用方完成。
        frame_indices: 各帧的**原始帧号**(与 frames 等长同序)。
        batch_size: 批量前向的窗大小(keyword-only)。None = 读
            `AICOMP_VIDEO_BATCH`(默认 8);1 = 完全回退逐帧旧行为(对拍用)。

    Returns:
        dict:
          - `per_frame`: 每帧 `{frame_index, mal_prob, preds, frame_np}`(preds 含软
            `pred_mask`,供 V2.1 逐帧 mask 落盘);
          - `agg`: `aggregate_frame_predictions` 结果(mal_prob_max/mean、keyframe_*);
          - `kf_preds`/`kf_frame_np`/`kf_tensor`: 关键帧的解码结果/原图/张量;
          - `failed_frames`: `[{frame_index, reason}, ...]` 被跳过的坏帧(按帧号升序);
          - `breakdown`: 关键帧 TI-RADS 分项;
          - `mask_resized`: 关键帧 `mask_binary` 缩放回原图 HxW(V2.0 落盘用)。

    Raises:
        DynamicCaseError: 全帧数值失败 或 无有效关键帧 —— 携带诊断字段,调用方落 ERROR。

    ## 批量前向的三段式坏帧隔离(2026-09-09,替代逐帧 `_diagnose_frame` 循环)

    ① 逐帧预处理:`AssertionError`(帧形状非 2D)在**组批前**抛,天然可定位;
    ② 批量前向:一窗帧 `torch.cat` 后一次 `run_forward`,OOM 对半降级在内;
    ③ 逐样本切片+decode:`NonFiniteProbabilityError`(概率/mask NaN/Inf)在**逐样本
    decode 上**捕获——绝不能整批校验,否则一帧 NaN 拖垮整批(旧 bug 回归)。
    `_diagnose_frame` 保留:唯一真实调用方是下方 kf 一致性锁的重算分支
    (video_predict.py 的 shim 是独立复刻函数体,并不调它——2026-09-09 审查订正)。
    ⚠️ 异常归属的隐含收窄:模型 forward 内部的 AssertionError(如 img_size 断言)
    旧代码被丢帧、新代码整例冒泡——实测不可达(preprocess 恒 resize 到 img_size 且
    模型与 config 同源),记录为语义备忘。
    """
    img_size = config["data"]["img_size"]
    bs = batch_size if batch_size is not None else _resolve_video_batch_size()
    print(f"动态逐帧诊断(批量前向 batch_size={bs})...", flush=True)

    # --- 三段式坏帧隔离(见 docstring) ---
    per_frame = []
    failed_frames = []
    kf_tensor = None          # 仅当前最恶性帧的张量(Grad-CAM 复用)
    kf_img_tensor_pos = -1    # 它在 per_frame 里的下标
    pairs = list(zip(frame_indices, frames))
    for w in range(0, len(pairs), bs):
        # ① 逐帧预处理:形状非法的帧组批前拦截(与旧 `assert ndim==2` 同源)
        ok = []
        for orig_idx, frame_np in pairs[w:w + bs]:
            try:
                ok.append((orig_idx, frame_np,
                           preprocess_image(frame_np, img_size, device)))
            except AssertionError as e:
                failed_frames.append({"frame_index": int(orig_idx), "reason": str(e)})
                print(f"--> [WARNING] frame {orig_idx} 异常已跳过"
                      f"(AssertionError: {e});整段继续", flush=True)
        if not ok:
            continue
        # ② 批量前向(OOM 对半降级在内);③ 逐样本 decode:NaN 只丢该样本
        per_outputs = _forward_frames(model, [t for _, _, t in ok],
                                      precision, device)
        for (orig_idx, frame_np, img_tensor), outputs_i in zip(ok, per_outputs):
            try:
                preds = _decode_frame_sample(outputs_i, frame_np,
                                             calibrator=calibrator, dynamic=True)
            except NonFiniteProbabilityError as e:
                failed_frames.append({"frame_index": int(orig_idx), "reason": str(e)})
                print(f"--> [WARNING] frame {orig_idx} 异常已跳过"
                      f"({type(e).__name__}: {e});整段继续", flush=True)
                continue
            cur = {
                "frame_index": int(orig_idx),
                "mal_prob": float(preds["mal_prob"]),
                "preds": preds,
                "frame_np": frame_np,
            }
            # 严格 `>`:只在**真正更恶性**时刷新保留帧,平分保留首帧,与聚合器 max 同序。
            # kf 张量来自 ① 段逐帧独立分配的 img_tensor(torch.cat 是拷贝,批前向不碰它)
            if kf_tensor is None or cur["mal_prob"] > per_frame[kf_img_tensor_pos]["mal_prob"]:
                kf_tensor = img_tensor
                kf_img_tensor_pos = len(per_frame)
            per_frame.append(cur)

    # ①段失败先于③段失败入列,同窗跨段可能乱序(如帧5形状坏+帧2NaN→[5,2]);
    # frame_indices 单调递增,按帧号排序恢复旧全局时序(failed_frames 逐字节可比)。
    failed_frames.sort(key=lambda f: f["frame_index"])

    # 全帧失败 → 无任何可用诊断。抛 DynamicCaseError 由调用方落 ERROR(原为
    # save_rejection_json+return,字段逐字节一致)。
    if not per_frame:
        reason = f"All {len(frames)} sampled frames failed numeric validation"
        print(f"[FATAL] {reason}")
        raise DynamicCaseError(reason, {
            "n_frames_sampled": len(frames),
            "n_frames_diagnosed": 0,
            "n_frames_failed": len(failed_frames),
            "failed_frame_indices": [f["frame_index"] for f in failed_frames],
            "failed_frames": failed_frames,
        })

    # --- 聚合 + 关键帧 ---
    agg = aggregate_frame_predictions(
        [{"frame_index": f["frame_index"], "mal_prob": f["mal_prob"]} for f in per_frame])
    # 双重守卫:keyframe_index<0 绝不能拿去索引(Python 负索引静默取末帧致错位)。
    if agg["keyframe_index"] < 0:
        raise DynamicCaseError(
            f"no valid keyframe (keyframe_index={agg['keyframe_index']})", {
                "n_frames_sampled": len(frames),
                "n_frames_diagnosed": agg["n_frames"],
                "n_frames_failed": len(failed_frames),
                "failed_frame_indices": [f["frame_index"] for f in failed_frames],
                "failed_frames": failed_frames,
            })
    kf = per_frame[agg["keyframe_index"]]
    kf_preds = kf["preds"]
    kf_frame_np = kf["frame_np"]
    # 一致性锁:保留帧下标须等于聚合关键帧下标,不等说明两条 max 判据分叉(近乎
    # 不可能),此时不静默用错帧张量——打 warning 并重算该关键帧张量。
    if kf_img_tensor_pos != agg["keyframe_index"]:
        print(f"--> [WARNING] 关键帧张量下标({kf_img_tensor_pos})与聚合关键帧"
              f"({agg['keyframe_index']})不一致,Grad-CAM 改用重算张量", flush=True)
        _, kf_tensor = _diagnose_frame(
            model, kf_frame_np, precision, device, img_size, calibrator=calibrator,
            dynamic=True)

    mask_resized = cv2.resize(
        kf_preds["mask_binary"], (kf_frame_np.shape[1], kf_frame_np.shape[0]),
        interpolation=cv2.INTER_NEAREST)
    breakdown = build_tirads_breakdown(
        kf_preds["comp_idx"], kf_preds["echo_idx"],
        kf_preds["shape_idx"], kf_preds["margin_idx"], kf_preds["foci_prob"])

    return {
        "per_frame": per_frame,
        "agg": agg,
        "kf_preds": kf_preds,
        "kf_frame_np": kf_frame_np,
        "kf_tensor": kf_tensor,
        "failed_frames": failed_frames,
        "breakdown": breakdown,
        "mask_resized": mask_resized,
    }
