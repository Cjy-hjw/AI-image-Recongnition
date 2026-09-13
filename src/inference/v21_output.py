"""V2.1 单病例输出落盘:`{case_dir}/{subdir_uid}/nodule_NNN.nii.gz` + `{case_dir}/prediction.json`。

本模块是**唯一的 V2.1 写盘入口**。它把 `case_pipeline` 的推理结果(模型输出尺寸的
软/二值 mask、良恶性概率、TI-RADS 分项)转成官方《赛事开发规范 V2.1》§五要求的
逐结节交付形态:

  1. 软 `pred_mask`(sigmoid)与 `mask_binary` 都先 resize 回**原始输入图 H×W**——规范
     硬约束 mask 尺寸必须与原图严格一致。软 mask 用 `INTER_LINEAR`(概率平滑),二值
     mask 用 `INTER_NEAREST`(不产生 0/255 之外的中间值,保住二值语义)。
  2. `nodule_split.split_nodules` 在原图尺寸的二值 mask 上做连通域分析,得到逐结节
     实例(2D 静态 / 3D 动态跨帧)。
  3. 每结节 `nifti_io.save_binary_mask` 落一个 `.nii.gz`(值域强制 {0,1});
     检测置信度:静态 `detection_confidence(软图, 实例 mask)`(区域内概率均值);
     动态 2026-09-08 起走 `_dynamic_det_conf`(实例出现帧的逐帧概率均值,旧口径
     是 scatter 二值化后的帧覆盖率、恒≈1.0 无排序能力,已废弃)。
  4. 每结节 `build_nodule_entry`(良恶性/TI-RADS/ETE/位置/临床T **全部为模型真实
     预测**——四头已于 2026-08-17 由 backlog #116b 接线,此前是 ETE=1/Location 均匀/
     T=0 的占位)→ `build_prediction_json` → `validate_prediction_json` →
     **原子写**(先 `.tmp` 再 `os.replace`)。

阴性病例(`split_nodules` 出 0 个结节)是合法输出:`Nodules`/`MaskFiles` 均为空数组,
仍写一份结构合规的 `prediction.json`(规范:阴性输出空数组,不建空子目录)。

## 两个最容易出错、且本地测不出来的点(写死在本模块,勿改)

- **维度约定**:`save_binary_mask` 内部已做 `(H,W)→(W,H)` / `(n,H,W)→(W,H,n)` 的
  NIfTI 列主序转置,本模块**只传 numpy 行主序数组**,不再额外转置(转两次等于没转,
  评测按 header 读形状会判尺寸不符)。
- **mask_file 必须是相对病例一级目录的路径** `{subdir_uid}/nodule_NNN.nii.gz`,
  禁绝对路径、禁 `..`(`_validate_relative_mask_path` 在 build/validate 双侧拦截)。
  本模块用 `os.path.join(subdir_uid, name)` 拼盘内绝对路径,用
  `f"{subdir_uid}/{name}"`(**正斜杠**)拼写进 JSON 的相对路径——Windows 上
  `os.path.join` 会产 `\\`,虽然校验放行,但官方评测在 Linux 按 `/` 解析,反斜杠会让
  文件定位失败。

## 动态跨帧结节关联(@云TODO)

动态图官方要求每个结节输出 **3D mask,Z 轴对齐全视频帧序列**(评估只抽标注帧计分,
但缺帧的帧无法计分)。本模块的取舍:

  - 各抽样帧逐帧 resize 回原图后,用 `scatter_frames_to_full_length(帧mask, 帧号,
    n_total_frames)` 回填成 `(n_total_frames, H, W)` 全帧 3D mask(未抽中帧填 0);
  - 在此 3D mask 上跑 `split_nodules`(3D 模式,26-邻域 + 时间维闭运算)——**时间上
    连续出现的同一结节自然连成一个 3D 实例**,这就是本期的"跨帧关联"。

**这是有意的最简实现,不是完备的多目标跨帧跟踪**:若分割在中间某帧漏检导致同一
结节断开,会被拆成两个实例(NoduleID 重复计数);两个空间相邻的结节也可能被闭运算
粘连。`nodule_split._split_3d` 的 docstring 已记录该局限。真正的跨帧实例关联
(tracking-by-detection / ReID)依赖官方数据校正,标 `@云TODO` 待训练阶段4后处理。

`n_total_frames` 由调用方(runner)在 case_result 里提供(A1 已让 loader 返回并一路
透传);**取不到时硬失败(`V21OutputError`),不再回退猜测**——原回退是「最大帧号+1」,
但尾部若有未抽中帧,Z 长度就短于原视频,而评审规则 `:205` 对此的罚则是「判定为格式
错误,该病例所有指标不予计分」(整例清零),回退却只打一条 warning、输出看着正常。
宁可该病例 ERROR(日志可见)也不交一份注定被判格式错误的 mask。
"""

import json
import os

import numpy as np

try:
    import cv2

    HAS_CV2 = True
except ImportError:  # pragma: no cover - 依赖缺失路径由调用方兜底
    HAS_CV2 = False

from src.inference import prediction_json as pj
from src.inference.nifti_io import save_binary_mask, scatter_frames_to_full_length
from src.inference.nodule_split import DEFAULT_MIN_AREA_RATIO
from src.inference.nodule_split import restore_instance_mask
from src.inference.nodule_split import detection_confidence, split_nodules

__all__ = ["write_case_v21"]


class V21OutputError(RuntimeError):
    """V2.1 落盘失败(输入契约不满足 / 写盘异常)。

    独立于 `PredictionSchemaError`:schema 错是「组装逻辑 bug」,本异常是「case_result
    缺关键键 / 维度不符 / 依赖缺失」这类运行期故障,由 runner 的病例级隔离捕获转
    ERROR 占位,不拖垮整批。
    """


def _tirads_class_from_breakdown(breakdown) -> int:
    """从 `build_tirads_breakdown` 的 `total_points` 推 1..5 的 TI_RADS_Class。

    🔴 **2026-08-17 起已无任何调用方**(backlog #116b):生产 `_shared_nodule_context`
    改用 `tirads_head` 的真实 softmax。本函数走的是 CBM 概念头的 ACR 积分反推,
    那条通路**没有专属训练监督**(概念头梯度对 TI-RADS 标签恒 0)、期望加权
    Kappa≈0,而 `tirads_head` 实测 0.6887。
    ⚠️ **保留理由 = 可回退 + 最小 diff,不是"CLI 还在用"** —— 实测
    `predict.py`/`video_predict.py` **都不调用它**(它们走 `build_probability_export`,
    该函数不含 TI-RADS 类别);唯一引用来自 `tests/test_v21_output.py`。
    (此处曾误写"CLI 调试路径仍在用",系未核实即写进注释,已由 Gate2 角度 7-8 实测纠正。)

    复刻 `report_generator.calculate_tirads_points` 内置的分级阈值(同一份 ACR 规则,
    单一事实源在那里的 tr_level 字符串里;此处只需 int 类别,故独立实现阈值映射,
    不解析字符串——字符串是给人看的,解析它会随措辞改动而脆断):
        0 分→1(TR1),≤2→2(TR2),==3→3(TR3),≤6→4(TR4),≥7→5(TR5)。
    """
    pts = int(breakdown["total_points"])
    if pts <= 0:
        return 1
    if pts <= 2:
        return 2
    if pts == 3:
        return 3
    if pts <= 6:
        return 4
    return 5


def _static_orig_hw(case_result) -> tuple:
    """静态病例的原图 (H, W)。

    唯一可靠来源是 `case_result["orig_hw"]`(调用方在加载输入图时记下 `img_np.shape`
    并透传)。`img_tensor` 是 `preprocess_image` resize 到 `img_size`(如 512×512)后的
    模型输入张量,**其 H×W 不是原图尺寸**,绝不能拿来当原图——故缺 `orig_hw` 时直接
    报错,不静默用模型输入尺寸冒充原图(那会让 mask 尺寸与原图不符,评测判尺寸不一致
    而整例失分,且本地不易察觉)。
    """
    if "orig_hw" in case_result and case_result["orig_hw"] is not None:
        h, w = case_result["orig_hw"]
        return int(h), int(w)
    raise V21OutputError(
        "静态 case_result 缺 `orig_hw`(原图 H×W)。调用方必须在加载输入图时透传 "
        "`orig_hw=img_np.shape[:2]`——img_tensor 是 resize 到 img_size 的模型输入,"
        "不能用来推原图尺寸")


def _dynamic_orig_hw(case_result) -> tuple:
    """动态病例的原图 (H, W)。

    动态链路每帧的 `frame_np` 就是已加载的原图灰度帧(uint8 HxW,未经模型 resize),
    故最权威来源是 `kf_frame_np.shape`(关键帧原图)。这比静态的 img_tensor 回退可靠。
    仍保留 `orig_hw` 透传优先,便于调用方显式覆盖。
    """
    if "orig_hw" in case_result and case_result["orig_hw"] is not None:
        h, w = case_result["orig_hw"]
        return int(h), int(w)
    kf_frame = case_result.get("kf_frame_np")
    if kf_frame is not None:
        arr = np.asarray(kf_frame)
        if arr.ndim >= 2:
            return int(arr.shape[0]), int(arr.shape[1])
    # 最后回退:任一 per_frame 的 frame_np。
    per_frame = case_result.get("per_frame") or []
    if per_frame and per_frame[0].get("frame_np") is not None:
        arr = np.asarray(per_frame[0]["frame_np"])
        return int(arr.shape[0]), int(arr.shape[1])
    raise V21OutputError("动态 case_result 既无 orig_hw 也无可用 frame_np,无法确定原图 H×W")


def _resize_soft(soft_mask: np.ndarray, hw: tuple) -> np.ndarray:
    """软概率 mask resize 回原图,`INTER_LINEAR` 保概率平滑。返回 float 概率图(H,W)。"""
    h, w = hw
    out = cv2.resize(
        np.asarray(soft_mask, dtype=np.float32), (w, h),
        interpolation=cv2.INTER_LINEAR)
    return out


def _resize_binary(binary_mask: np.ndarray, hw: tuple) -> np.ndarray:
    """二值 mask resize 回原图,`INTER_NEAREST` 保二值语义(不出中间值)。返回 {0,1} uint8。"""
    h, w = hw
    out = cv2.resize(
        np.asarray(binary_mask), (w, h), interpolation=cv2.INTER_NEAREST)
    # save_binary_mask/split_nodules 接受任意非零为前景,但此处显式压成 {0,1},
    # 避免把内部 0/255 一路带进逐结节实例(实例 mask 值域本就 {0,1},统一更干净)。
    return (out != 0).astype(np.uint8)


def _mask_filename(index: int) -> str:
    """结节 mask 文件名(1-based,三位零填充):`nodule_001.nii.gz`。"""
    return f"nodule_{index:03d}.nii.gz"


def _atomic_write_json(path: str, payload: dict) -> None:
    """原子写 JSON(先写 `.{pid}.tmp` 再 `os.replace`),防半写文件。

    **刻意不复用 `export._write_json_atomic`**:那份用 `json.dump(payload, f, indent=2)`
    无 `ensure_ascii=False`,会把中文转义成 `\\uXXXX`;而 V2.1 的 `Interpretation.Conclusion`
    (CBM 报告/拒绝原因)是中文,必须 `encoding="utf-8"` + `ensure_ascii=False` 原样保留。
    两处算法相同(tmp+os.replace)但序列化参数不同,合并会破坏中文输出——勿"好心统一"。
    `.tmp` 名带 pid 防并发/重入撞名。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _dynamic_det_conf(inst: dict, soft_by_frame: dict, mask3d) -> float:
    """动态实例的检测置信度(2026-09-08 conf 语义修复)。

    = 实例**实际出现的帧**上,区域内 float 概率均值的算术平均。与旧口径
    (scatter 后 uint8 上的 detection_confidence = "实例体素落在有预测帧的比例")
    的区别:旧值对全程存在的结节恒趋近 1.0、排序能力近零(检测诊断报告实锤
    动态 conf 直方图挤 ~1.0);新值保留连续概率的区分度。

    数值契约与 `detection_confidence` 对齐:非有限值 raise(同源 NaN 硬拦理由,
    见其 docstring);实例帧全部缺概率图(理论上不可达,实例帧来自抽样帧)时
    回退 0.0。

    2026-09-09 内存治理:签名加 `mask3d` 参数(3D 实例已改稀疏存储,全尺寸
    mask 由调用方逐实例重建、用完即弃——本函数不再从 `inst["mask"]` 取)。
    """
    frame_indices = inst.get("frame_indices") or []
    probs = []
    mask3d = np.asarray(mask3d)
    for fi in frame_indices:
        soft = soft_by_frame.get(int(fi))
        if soft is None:
            # 部分缺帧静默少平均(生产构造下不可达:实例帧来自有预测的抽样帧,
            # 见 A 路审查 V3c;若未来构造变化让缺帧可达,应改 raise 而非 continue)。
            continue
        m = mask3d[int(fi)] != 0
        if m.any():
            # B 路审查 B-P2(实测 5.8×):先掩膜后 mean(子集 ≤ 万级像素,float32
            # 精度足够),不做整帧 float64 拷贝——750 帧病态例该项从 ~30s 降到 ~5s。
            probs.append(float(np.asarray(soft)[m].mean()))
    if not probs:
        return 0.0
    mean = float(np.mean(probs))
    if not np.isfinite(mean):
        raise ValueError(
            "_dynamic_det_conf: 实例逐帧概率均值非有限(NaN/Inf)——"
            "分割头数值异常,拒绝导出无效置信度(与 detection_confidence 同判)")
    return float(np.clip(mean, 0.0, 1.0))


def _build_entries_and_save_masks(instances, soft_fullres, case_dir, subdir_uid,
                                  nodule_base_context, soft_by_frame=None,
                                  full_shape=None):
    """公共尾段:逐结节落 mask + 组装 build_nodule_entry。

    Args:
        instances: `split_nodules` 的实例列表。**2D**:含 `mask`(全尺寸);**3D**:
            含 `mask_sparse`(bbox 裁剪稀疏存储,2026-09-09 内存治理)与
            `frame_indices`(conf 消费)。
        soft_fullres: 与实例 mask 同 shape 的软概率图(2D 原图 / 3D 全帧),供
            `detection_confidence` 取区域内概率均值。🔴 3D 时它是 scatter 后的
            uint8 二值图(帧覆盖率语义,排序能力近零),仅当 `soft_by_frame` 缺席
            时才作为回退消费(静态路径不受影响,soft_fullres 本就是 float)。
        soft_by_frame: 2026-09-08 conf 修复新增。动态路径的 **{原始帧号: float
            概率图}**,优先于 soft_fullres —— det_conf 改为"实例出现帧的逐帧
            概率均值再平均",保留连续概率区分度。静态调用方不传(缺省 None)。
        full_shape: 2026-09-09 内存治理新增。3D 全尺寸 `(n_frames,H,W)`,
            稀疏实例 `restore_instance_mask` 重建全尺寸 mask 用。🔴 动态路径
            **必传**(3D 实例无 `mask` 键,缺它无法重建写盘);静态/2D 不传。
            🔴 **须在调用方 `del binary_3d` 之前捕获 shape**(del 后取会 NameError,
            审查 P2-1)。
        case_dir / subdir_uid: 落盘根与相对子目录。
        nodule_base_context: dict,含 `malignancy_confidence` / `tirads_class` /
            `ete_code` / `location_probs` / `location` / `t_clin_sig` / `t_confidence`
            等每结节共享的预测值(2026-08-17 起**全部为四头真实预测**,不再有占位);
            另含可选的 #116a 对冲键 `tirads_probs` / `ete_probs`(缺席则不写出)。

    Returns:
        nodule_entries — 供 build_prediction_json;阴性时为空列表。
    """
    entries = []
    for i, inst in enumerate(instances, start=1):
        name = _mask_filename(i)
        abs_path = os.path.join(case_dir, subdir_uid, name)
        # 写进 JSON 的相对路径强制正斜杠(官方 Linux 评测按 `/` 解析,见模块 docstring)。
        rel_path = f"{subdir_uid}/{name}"

        if "mask_sparse" in inst:
            # 3D 稀疏实例(2026-09-09 内存治理):全尺寸 mask 逐实例重建,
            # 写盘 + det_conf 两个消费点共用,用完即弃(全尺寸 ≈0.9GB/例,
            # 峰值恒为 1 份,不再随实例数累积)。
            if full_shape is None:
                raise V21OutputError(
                    "3D 稀疏实例需传 full_shape 才能 restore_instance_mask"
                    "(动态调用方漏传——见 _write_dynamic 的构造)")
            full_mask = restore_instance_mask(inst["mask_sparse"], full_shape)
            try:
                # 落盘:实例 mask(值域 {0,1})。save_binary_mask 内部做 NIfTI 转置。
                save_binary_mask(full_mask, abs_path)
                if soft_by_frame is not None:
                    det_conf = _dynamic_det_conf(inst, soft_by_frame, full_mask)
                else:
                    det_conf = detection_confidence(soft_fullres, full_mask)
            finally:
                del full_mask   # 异常路径也不许全尺寸 mask 存活跨实例
        else:
            # 2D 实例:全尺寸本来就小,原样直落。
            save_binary_mask(inst["mask"], abs_path)
            if soft_by_frame is not None:
                det_conf = _dynamic_det_conf(inst, soft_by_frame, inst["mask"])
            else:
                det_conf = detection_confidence(soft_fullres, inst["mask"])

        entry = pj.build_nodule_entry(
            nodule_id=f"N{i}",
            mask_file=rel_path,
            detection_confidence=det_conf,
            malignancy_confidence=nodule_base_context["malignancy_confidence"],
            tirads_class=nodule_base_context["tirads_class"],
            ete_code=nodule_base_context.get("ete_code", pj.ETE_NONE),
            location_probs=nodule_base_context.get("location_probs"),
            location=nodule_base_context.get("location"),
            t_clin_sig=nodule_base_context.get("t_clin_sig", 0),
            t_confidence=nodule_base_context.get("t_confidence", 0.0),
            # #116a 对冲键。用 `.get()` 而非 `[]`:CLI 调试等不产四头概率的
            # 调用方拿到 None → build_nodule_entry 不写出该键,向后兼容。
            tirads_probs=nodule_base_context.get("tirads_probs"),
            ete_probs=nodule_base_context.get("ete_probs"),
        )
        entries.append(entry)
    return entries


def _qwk_round(probs, lo=1):
    """加权 Kappa 指标下的决策规则:期望取整 round(Σ j·p_j),编码 = idx + lo。

    (2026-09-10 零训练四件套③,替换 TI-RADS/ETE 的 argmax 决策)
    二次加权 Kappa 的损失核是 (i-j)²,给定概率 p,最小化期望损失的点估计是
    期望 E[j]=Σ j·p_j,取整后作为输出编码。概率质量向高分段偏移时,期望取整
    比 argmax 更早给出高编码 —— argmax 在不确定时恒取峰值类,Kappa"方向对了
    按距离给分"的口径下这白丢分;期望取整让 0.4/0.3/0.2 这类双峰分布落在中间。
    ⚠️ 官方只写"加权Kappa"未公布二次/线性形态:期望取整在二次权重下最优,
    线性权重下最优是中位数,期望取整介于 argmax 与中位数之间 —— 均为合理
    近似而非两种口径下严格最优(此措辞经 Gate2 审查订正,勿再写强断言)。
    ⚠️ 训练侧 `metrics_official` 验证指标仍是 argmax 口径,推理侧改期望取整后
    两侧决策规则脱节(训练已关账,仅诊断性影响,不影响生产)。
    🔴 `round` 是 Python **银行家舍入**(半整数取向偶数:E=0.5→0、E=1.5→2),
    不是四舍五入——半整数处相邻两类到均值等距、Kappa 损失完全相同,属纯
    tie-break,无系统性偏差;但照本 docstring 在别处复刻时若写 floor(x+0.5)
    会得到不同编码(E=0.5 处 1 vs 0),且两种都合法、本地查不出。复刻须用
    同款 `int(round(...))`(np.round 同为银行家舍入)。
    """
    return int(round(float(np.dot(np.arange(len(probs)), probs)))) + lo


def _shared_nodule_context(preds) -> dict:
    """从 preds 提取每结节共享的预测上下文(**四头已于 2026-08-17 接真实输出**)。

    backlog #116b:此前 ETE 恒 1 / Location 均匀分布 / T 恒 0 / TI-RADS 走 ACR 概念
    积分反推,四项对官方评分等同随机(目标四 40% 三项 + 目标三 TI-RADS 一项)。
    云端训练已把四头训出真实能力(loc_auc 0.7519 / t_auc 0.7441 /
    tirads_kappa 0.6887 / ete_kappa 0.2086),此处直接消费其 softmax 概率。

    ## 编码换算(本函数最易错处,错了**不报错**)

    `code = 决策(idx) + 1` 对 **ETE / TI-RADS / Location** 三项成立;其中
    **ETE / TI-RADS 自 2026-09-10 件套③起决策 = `_qwk_round`(期望取整)而非
    argmax**(加权 Kappa 下的最优决策,见该函数 docstring),**Location 仍是
    argmax**(宏 AUC 吃分布,argmax 是其天然点估计)。
    (`label_schema.py`:ETE_CODES=(1,2,3,4) / TIRADS_CODES=(1..5) /
     LOCATION_LABELS[0]="左上叶"→码 1)。
    🔴 **`t_clin_sig` 是唯一不 +1 的** —— 官方 `T∈{0,1}` 与 0-based 下标同值
    (`heads/clinical_t.py:110-112`)。写成 `+1` 会得 2,被
    `prediction_json.py:296` 拦成整例 ERROR(至少会炸);而其余三项若漏 `+1`
    得到的是**另一个合法码**(如把"包膜侵犯"写成"无侵犯"),越界校验放行、
    JSON 完全合法、只是指标偏低,本地永远查不出根因。

    ## 位置 7 维顺序不得重排

    `label_schema.py:44-58` 明确警告:真实编码是"上→中→下"(实测 xlsx
    `2左中叶`/`5右中叶`),照《数据格式说明》的列举次序写会**错 4/7 类**。
    头输出顺序已由 `heads/location.py:65-76` 的 import 期断言锁死,直接透传即可;
    2026-08-17 逐字读评审规则 `:325-340` 的官方表格,与该顺序**完全一致**。

    `breakdown` 参数已于 2026-08-17 **从签名移除**:TI-RADS 改由 `tirads_head` 直供后
    它在本函数体内零引用,留着只会让下一个改它的人以为还有用(Gate2 两路独立指出)。
    ACR 反推路径见 `_tirads_class_from_breakdown`(**现已无任何调用方**,保留理由见该函数)。
    """
    ete_probs = preds["ete_probs"]
    tirads_probs = preds["tirads_probs"]
    loc_probs = [float(x) for x in preds["location_probs"]]
    t_probs = preds["t_probs"]
    return {
        "malignancy_confidence": float(preds["mal_prob"]),
        # 零训练四件套③(2026-09-10):TI-RADS/ETE 走期望取整(_qwk_round),
        # 替换 argmax —— 加权 Kappa 按类间距离惩罚,期望取整是不确定分布下的
        # 最优点估计;location/T 不动(位置是宏AUC吃分布、T是AUC吃t_conf排序,
        # 均非 Kappa 决策路径)。
        "tirads_class": _qwk_round(tirads_probs),          # idx0→码1(I类)
        "ete_code": _qwk_round(ete_probs),                 # idx0→码1(无侵犯)
        "location_probs": loc_probs,
        "location": int(np.argmax(loc_probs)) + 1,          # idx0→码1(左上叶)
        "t_clin_sig": int(np.argmax(t_probs)),              # 🔴 0/1,**不 +1**
        "t_confidence": float(preds["t_conf"]),             # = P(T=1)
        # --- #116a 对冲键(官方 schema 无概率位,评审规则 :185 又要求"无概率计0分") ---
        # ⚠️ 评审规则 :251 正面强制"概率键名必须与下表完全一致",而表里这两项
        #    只有 int 位 → 自拟键名必然对不上,**拿分概率≈0,纯防御性下注**。
        #    净收益仍 ≥0(int 字段照常合规、零损失),故做但别指望得分。
        #    决策见 docs/待官方答复清单.md Q4。
        "tirads_probs": [float(x) for x in tirads_probs],
        "ete_probs": [float(x) for x in ete_probs],
    }


def _lesion_level_head_preds(per_frame, kf_preds) -> dict:
    """动态病例:把逐帧四头概率按**算术平均**汇总成病灶级的一份预测。

    ## 为什么是均值而不是关键帧

    官方评审规则 `:215` 原文:「结节检测AP、良恶性AUC-PR、TI-RADS加权Kappa、
    侵犯分级加权Kappa、主病灶位置加权宏平均AUC、T分级AUC均采用**病灶级汇总,
    每个预测病灶仅贡献一个样本**(Kappa在全测试集一次计算)。帧维度仅用于匹配
    判定和Dice系列指标的计算。」
    且**与训练侧验证口径一致** —— `metrics_official.aggregate_by_lesion`
    (`src/training/metrics_official.py:43-50`)就是按 `lesion_id` 对 probs 取均值。
    🔴 **该函数的 docstring 明写「若推理侧改用取最大/中位数,这里要同步改,否则
    验证指标与交付表现脱节」—— 两侧算法互为约束,改一侧必须同步改另一侧。**
    两边同口径才不会出现"本地指标好、线上差"的盲区。
    (⚠️ 但**不该合并成一个函数**:训练侧要按 `lesion_id` 分桶聚合多病灶,推理侧
     四头是整图级、单病例只有一份,数据形态不同。约束的是算法,不是实现。)

    ## 一个必须知道的前提(不是本次引入的问题,但别误以为已解决)

    官方只说"病灶级汇总",**没规定汇总方法**(均值/关键帧/最大置信帧都可能)。
    严格说"帧均值 = 病灶级"只在"每帧恰好一个结节实例"时成立;多结节时这是
    **帧粒度**汇总而非实例粒度。之所以只能这样做:四头输出本就是**整图级**
    (`stage4_spatial` 全局池化,`heads/*.py` 无实例维度),当前架构下不存在
    "每实例一份四头概率",实例级汇总无从谈起。这是既有架构性质,不是接线引入的。

    ## 两个实现细节

    - `per_frame` **只含成功帧**(`case_pipeline.py:144-163`:异常帧
      `failed_frames.append` 后 `continue`,在 append 进 per_frame **之前**),
      故均值分母天然就是成功帧数,此处不必再过滤。该前提有回归测试钉住。
    - **不重归一化**:各帧概率各自和为 1,均值的和恒为 1(实测偏差 2.2e-16,
      远小于 `validate_prediction_json` 的 1e-6 容差)。`location_probs` 仍会在
      `build_nodule_entry` 内过 `normalize_location_probs`,是无害的防御。

    Args:
        per_frame: `diagnose_dynamic_case` 的逐帧列表,每项含 `preds`。
        kf_preds: 关键帧解码结果。**空 per_frame 时的兜底**(理论不可达 ——
            `diagnose_dynamic_case` 在 per_frame 为空时已抛 `DynamicCaseError` ——
            但此处不赌上游契约,除零会崩在离根因很远的地方)。

    Returns:
        dict:**全部分类概率**(四头 + 良恶性)取均值,其余键沿用 `kf_preds`。

        🔴 **`mal_prob` 也必须均值,别只均值四头**(2026-08-17 终审 P0.1 修正):
        官方 `:215` 那句话把「良恶性AUC-PR」和四头**并列**写在同一个"病灶级汇总"
        清单里,不是只管四头;训练侧 `metrics_official.py:8/26-29` 同样把良恶性
        列进按 `lesion_id` 取均值的范围。此前本函数只均值四头、`mal_prob` 留在
        关键帧口径,构成**口径分裂** —— 关键帧是「最恶性帧」(argmax),系统性
        高于病灶级均值,而良恶性 AUC-PR 是目标三里我方最强的一项(0.90),
        偏差直接打在它上面。
        ⚠️ 本 docstring 曾写「别顺手把良恶性也改了」,**那句是错的**,已删。

        `breakdown`(TI-RADS 概念分项)仍用关键帧:它只喂 CBM 报告文本
        (`Interpretation`),初赛不计分,且没有"多帧平均一份概念分项"的语义。
    """
    if not per_frame:
        return kf_preds
    out = dict(kf_preds)
    for key in ("ete_probs", "location_probs", "t_probs", "tirads_probs"):
        out[key] = np.mean(
            np.stack([np.asarray(f["preds"][key], dtype=np.float64)
                      for f in per_frame]), axis=0)
    out["t_conf"] = float(np.mean([float(f["preds"]["t_conf"]) for f in per_frame]))
    # 良恶性同口径:`mal_probs` 与 `mal_prob` 必须同源(prediction_json 只消费
    # `mal_prob`,但两者不一致会让 CLI 调试路径与生产各说各话)。
    out["mal_probs"] = np.mean(
        np.stack([np.asarray(f["preds"]["mal_probs"], dtype=np.float64)
                  for f in per_frame]), axis=0)
    out["mal_prob"] = float(out["mal_probs"][1])
    return out


def _write_static(case_dir, patient_id, subdir_uid, case_result,
                  processing_ms, interpretation):
    preds = case_result["preds"]
    hw = _static_orig_hw(case_result)

    soft_fullres = _resize_soft(preds["pred_mask"], hw)
    binary_fullres = _resize_binary(preds["mask_binary"], hw)

    # 件套④:静态不传 → split_nodules 默认 None → 吃 AICOMP_MIN_AREA_RATIO env。
    # max_nodules 同理不传(None→AICOMP_MAX_NODULES env→20,阶段1方向A)。
    instances = split_nodules(binary_fullres)
    context = _shared_nodule_context(preds)
    entries = _build_entries_and_save_masks(
        instances, soft_fullres, case_dir, subdir_uid, context)

    payload = pj.build_prediction_json(
        study_uid=patient_id,
        nodules=entries,
        processing_time_ms=processing_ms,
        interpretation=interpretation,
    )
    pj.validate_prediction_json(payload)
    _atomic_write_json(os.path.join(case_dir, "prediction.json"), payload)


def _write_dynamic(case_dir, patient_id, subdir_uid, case_result,
                   processing_ms, interpretation):
    if not HAS_CV2:
        raise V21OutputError("动态 V2.1 落盘需要 cv2(resize),但 cv2 未安装")

    kf_preds = case_result["kf_preds"]
    per_frame = case_result["per_frame"]
    hw = _dynamic_orig_hw(case_result)

    # --- 全帧 3D mask:抽样帧逐帧 resize 回原图 → scatter 回填到视频全长 ---
    frame_indices = [int(f["frame_index"]) for f in per_frame]
    n_total = case_result.get("n_total_frames")
    if not n_total or int(n_total) <= 0:
        # 🔴 硬失败而非回退(2026-08-17 第三轮 review lane3 P2-1)。
        # 原实现回退为 `max(frame_indices)+1`,但**尾部若有未抽中帧,Z 长度就会短于
        # 原视频** —— 评审规则 :205 对此的罚则是「判定为格式错误,该病例所有指标
        # 不予计分」,即**整例清零**,而回退只打一条 warning、输出看着完全正常。
        # 宁可该病例 ERROR(至少能在日志里发现),也不交一份注定被判格式错误的 mask。
        #
        # 生产可达性:`runner.py:329` 恒注入本键,且 `load_nifti_frames` 失败时返回
        # 0 → `runner.py:283-284` 提前 raise。故本分支在 service 路径不可达,
        # 它拦的是「未来新增调用方忘了注入」这种回归。
        raise V21OutputError(
            f"动态 case_result 缺有效 n_total_frames(收到 {n_total!r})——无法确定 "
            f"3D mask 的 Z 轴长度。官方要求 z 轴帧数必须与原视频完全一致,"
            f"猜测长度会导致该病例所有指标不予计分(评审规则 :205)。"
            f"调用方须从 loader 透传 n_total_frames。"
        )
    n_total = int(n_total)

    binary_frames = []
    soft_frames = []
    for f in per_frame:
        p = f["preds"]
        binary_frames.append(_resize_binary(p["mask_binary"], hw))
        soft_frames.append(_resize_soft(p["pred_mask"], hw))

    binary_3d = scatter_frames_to_full_length(binary_frames, frame_indices, n_total)
    # 🔴 顺序陷阱(审查 P2-1):full_shape 必须在 del binary_3d **之前**捕获
    # (3D 实例已稀疏化,写盘前 restore 需要全尺寸 shape;del 后取会 NameError)。
    full_shape = binary_3d.shape
    # 🔴 2026-09-08 conf 语义修复 + soft_3d 死分配删除(B 路审查 B-P3):
    # conf 走 soft_by_frame(逐帧 float 概率均值),soft_3d/soft_frames 不再
    # 全量驻留。原 `soft_frames` 全帧列表(resize 回原图 float32 ×750 ≈3.5GB)
    # 已删除:实例帧 ⊆ 抽样帧,`_dynamic_det_conf` 消费时按帧即时取用。
    soft_by_frame = {int(fi): sf for fi, sf in zip(frame_indices, soft_frames)}
    del binary_frames, soft_frames  # soft 已进 soft_by_frame;scatter 消费完即弃

    # --- 3D 连通域 = 跨帧实例关联 ---
    # 🔴 **缝合跨度必须按实际抽样间隔算,不能用默认的 1**(2026-08-17 终审 D 路 P1)。
    # 抽样后 scatter 回填,**未抽中的帧全是 0** —— 帧间隔 = ceil(n_total / 抽样帧数)。
    # 若 `close_gap_frames`(默认 1)小于该间隔,26-邻域在时间维**根本连不上**,
    # 同一个结节会被沿时间轴切成一串碎片。
    #
    # 实测(`split_nodules` 默认参数,单个全程存在的结节):
    #   原视频 50 帧(间隔1)→ 1 个实例 ✅
    #   原视频 100 帧(间隔3)→ **7 个**
    #   原视频 128/200/400 帧(间隔4/6/12)→ **各 20 个**(撞 DEFAULT_MAX_NODULES 上限)
    #
    # 2026-09-08 动态默认改全帧后(`_DEFAULT_MAX_FRAMES=0`),间隔恒 1、gap 恒 2,
    # 下方公式自然退化,无需分支;但 AICOMP_MAX_FRAMES 回退抽样时本公式仍必要,
    # 故保留原样(对全帧输入 n_sampled==n_total → gap=2,闭运算仅弥合单帧漏检,
    # 语义正确:全帧下"某帧漏检"就是真实的单帧空隙)。
    # 取 `间隔 + 1` 而非恰好等于间隔:留一帧余量,兼顾"某个抽样帧本身漏检"的情况
    # (那会让实际空隙变成 2 倍间隔;+1 不足以覆盖全部但成本为零)。
    # ⚠️ 副作用已实测排除:闭运算是 **3D** 的,时空分离的两个结节即使 gap 调到 20
    # 仍正确保持 2 个实例(空间不重叠就连不上),不会把不同结节粘连。
    n_sampled = len(frame_indices)
    gap = max(1, -(-n_total // n_sampled) + 1) if n_sampled else 1
    # 件套④:动态显式传 DEFAULT_MIN_AREA_RATIO —— split_nodules 默认参已改
    # None→env(静态专用),这里不显式传会让动态意外吃到 AICOMP_MIN_AREA_RATIO
    # (动态放宽会增多 3D 实例,写盘耗时与内存都是 8h 时限下的风险面,故钉死)。
    instances = split_nodules(binary_3d, min_area_ratio=DEFAULT_MIN_AREA_RATIO,
                              close_gap_frames=gap)
    # max_nodules 不显式传(默认 None→AICOMP_MAX_NODULES env→20)。🔴 该 env
    # 与静态 :489 共用同一键,静/动两池**同时**生效,无法分池独立——设之前
    # 必须同时确认两池的 n_pred 饱和率(analyze_diag_sidecar 分池报);动态侧
    # 写盘 O(全尺寸体素)是 8h 时限风险面,饱和例多时慎开(与 min_area 钉死
    # 的差异:max_nodules 是阶段1扫参的决策对象,故不吃钉死、走 env)。
    # B 路审查 B-P3:3D 连通域已消费完 binary_3d,后续只有 mask 写盘(逐实例
    # 自持数据),全帧下该数组 ≈0.87GB/例,尽早释放(worker=2 双动态并发时
    # 峰值 -1.7GB)。诊断脚本 run_dynamic_inference_from_result 早有同款 del,
    # 生产此前没有。
    del binary_3d
    # 四头 + 良恶性走**病灶级均值**(官方 :215「每个预测病灶仅贡献一个样本」),
    # 详见 `_lesion_level_head_preds` docstring。
    context = _shared_nodule_context(
        _lesion_level_head_preds(per_frame, kf_preds))
    entries = _build_entries_and_save_masks(
        instances, None, case_dir, subdir_uid, context,
        soft_by_frame=soft_by_frame, full_shape=full_shape)

    payload = pj.build_prediction_json(
        study_uid=patient_id,
        nodules=entries,
        processing_time_ms=processing_ms,
        interpretation=interpretation,
    )
    pj.validate_prediction_json(payload)
    _atomic_write_json(os.path.join(case_dir, "prediction.json"), payload)


def write_case_v21(case_dir, patient_id, subdir_uid, case_result, kind,
                   processing_ms, interpretation=None) -> None:
    """把单病例推理结果落成官方 V2.1 交付形态(逐结节 .nii.gz + prediction.json)。

    Args:
        case_dir: 该病例输出目录(已含 patient_id 层,即 `{output_dir}/{PatientId}`)。
            调用方保证父目录存在;本函数负责建 `{case_dir}/{subdir_uid}/` 子目录
            (由 `save_binary_mask` 内部 `makedirs` 完成)。
        patient_id: 填 `prediction.json` 的 `StudyUID`(V2.1 起其值即 PatientId)。
        subdir_uid: mask 相对路径的子目录名(静态=SeriesUid,动态=AccessionNumber)。
        case_result: `diagnose_static_case` / `diagnose_dynamic_case` 的返回 dict
            (A2 runner 会额外注入 `orig_hw`/`n_total_frames`/`gatekeeper_audit` 等键,
            本函数按需读取,缺省有回退)。
        kind: `"static"` | `"dynamic"`。
        processing_ms: 推理总耗时(毫秒),透传进 `ProcessingTime_ms`(int)。
        interpretation: 可选 CBM 报告 `{"Conclusion": str, "KeyFindings": [str]}`。

    Returns:
        None。产物写在磁盘上。

    Raises:
        V21OutputError: case_result 契约不满足 / 依赖缺失 / 维度不符。
        PredictionSchemaError: 组装出的 payload 不合规(应由 build/validate 自拦)。
    """
    if not HAS_CV2:
        raise V21OutputError("V2.1 落盘需要 cv2(mask resize 回原图),但 cv2 未安装")

    # 拒绝/无诊断语义由调用方(runner `_write_case_error` / `_rejection_result`)兜底,
    # 本函数只处理「有真实诊断结果」的病例。`preds is None` 是 rejection 占位,不应到此处。
    if case_result.get("rejected"):
        raise V21OutputError(
            "case_result 标记为 rejected(gatekeeper 拒绝),不应走 V2.1 正常落盘路径;"
            "请由 runner 的 ERROR/REJECTED 兜底处理")

    os.makedirs(case_dir, exist_ok=True)

    if kind == "static":
        _write_static(case_dir, patient_id, subdir_uid, case_result,
                      processing_ms, interpretation)
    elif kind == "dynamic":
        _write_dynamic(case_dir, patient_id, subdir_uid, case_result,
                       processing_ms, interpretation)
    else:
        raise V21OutputError(f"未知 kind={kind!r},只支持 'static'/'dynamic'")
