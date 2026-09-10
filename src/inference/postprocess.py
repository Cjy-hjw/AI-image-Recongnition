"""Decode model outputs and compute TI-RADS breakdown."""

import os

import numpy as np
import torch

# 四头的类别数常量:用于 `decode_predictions` 的维度契约硬校验。
# 单一权威在 `src.models` 侧,此处只读不再抄一份 —— 若模型改类别数而这里抄了个
# 旧数字,校验就会把正确输出judge成错误(或反过来放行真错误)。
# 依赖方向合规:`.importlinter` 契约禁的是 `src.models` 依赖 `src.inference`,反向可以。
from src.models.heads.clinical_t import NUM_CLIN_T_CLASSES
from src.models.label_schema import ETE_CODES, NUM_LOCATIONS, NUM_TIRADS_CLASSES
from src.utils.report_generator import calculate_tirads_points


class NonFiniteProbabilityError(ValueError):
    """概率含 NaN/Inf 的专用异常(`ValueError` 子类)。

    独立子类的理由:`video_predict.py` 逐帧循环只应把「这一帧数值异常、丢帧不丢
    整段」的错当可跳帧信号,但 `ValueError` 这个**类型名**本身太宽——`cv2.resize`
    对 0 尺寸/单像素畸形帧抛 `cv2.error`、numpy 在 exotic dtype 帧上也抛
    `ValueError`,这些按 QA_VP §1.7 属「输入/解码故障」应整段 ERROR,不该被逐帧
    兜底吞成"少几帧 PASSED"。入口层 `except NonFiniteProbabilityError` 即可精确
    区分「概率无效(可丢帧)」与「其它 ValueError(故障,须冒泡)」。
    """


_ENV_SEG_THR = "AICOMP_SEG_THRESHOLD"
_ENV_SEG_THR_DYNAMIC = "AICOMP_SEG_THRESHOLD_DYNAMIC"


def _seg_threshold(dynamic: bool) -> float:
    """分割二值化阈值(零训练四件套①,2026-09-10)。默认 0.5 = 历史行为逐位不变。

    真实病灶只占 512² 的 0.1~3%,极端不平衡下 Dice/BCE 训出的概率系统性欠分割,
    最优阈值通常低于 0.5(云端 val 扫参定值)。静/动两条 env 分开:动态匹配是
    各标注帧 IoU 的算术平均、某帧全空按官方口径记 0,空帧对动态毁灭性更强,
    动态阈值应比静态更低。解析失败/越界一律钳回 0.5(与 `case_pipeline.
    _resolve_video_batch_size` 同模式——垃圾配置不静默采信也不崩,回退安全值)。
    """
    raw = os.environ.get(_ENV_SEG_THR_DYNAMIC if dynamic else _ENV_SEG_THR, "0.5")
    try:
        t = float(raw)
    except (TypeError, ValueError):
        return 0.5
    return t if 0.0 < t < 1.0 else 0.5


def decode_predictions(outputs: dict, calibrator=None, *, dynamic: bool = False) -> dict:
    """Extract all prediction fields from raw model outputs.

    Args:
        outputs: 原始模型输出(各头 logits)。
        calibrator: 可选 `TemperatureScaler`。每个分类头在 softmax/sigmoid **之前**
            先经 `calibrator.apply(task, logits)` 做温度缩放。为 None 或全 T=1.0 时
            **无任何行为变化**(逐位等同无校准)。argmax 预测类别不受 T 影响。

    Returns dict with keys:
        pred_mask, mask_binary, comp_idx, echo_idx, shape_idx, margin_idx,
        comp_prob, echo_prob, shape_prob, margin_prob, foci_prob,
        mal_prob, mal_probs,
        **ete_probs, location_probs, t_probs, t_conf, tirads_probs**(官方四头,
        backlog #116b 于 2026-08-17 接入;各头概率均为模型内部已做过的 fp32 softmax)

    `*_prob` 为各头**完整概率分布**(softmax 头和为 1;foci 为多标签 sigmoid,不求和),
    `mal_probs` 为良恶性二类分布 `[p_benign, p_malignant]`。官方硬约束"所有分类/分级/
    分型任务必须输出功率分布或置信度,否则计0分"(赛前动员会 2026-07-28),概率必须
    随 `*_idx` 一并导出到 result.json,不得只留 argmax。
    """
    def _cal(task, logits):
        return calibrator.apply(task, logits) if calibrator is not None else logits

    def _assert_finite(task, prob):
        """概率含 NaN/Inf 即硬报错,拒绝导出"看起来正常"的无效分布。

        两个后果都不可接受(2026-07-29 /code-review 实测复现):
        ① `torch.argmax` 对全 NaN 向量**静默返回 0** —— logits=[0.2, nan, 1.5, -0.3]
           时真实最大值在 idx=2,却报 idx=0,报告会打印一个错误的 TI-RADS 分项;
        ② `json.dumps` 把 NaN 写成字面量 `NaN`,这是 **非标准 JSON**,严格解析器
           (Go/Rust/Jackson 默认)会拒绝**整个** result.json —— 等于该样本零输出,
           比预测错更严重。
        与 `report_generator.generate_report` 对 malignancy_prob 的既定处置同范式
        (硬拦截、交上层 except 转 ERROR),但覆盖面扩到全部分类头——generate_report
        只校验恶性标量,管不到四个概念头。**2026-07-30 起连 mask 也罩**(sigmoid 后):
        NaN logits 经 sigmoid 得 NaN,`(NaN > 0.5)` 恒 False 会静默产出全零分割图,
        而 status=PASSED、报告正常——比报错更隐蔽,评测只看 JSON 察觉不到分割已废。

        校验的是**激活后的概率**而非输入 logits:`-inf` 经 softmax/sigmoid 得 0,
        分布仍有限、仍和为 1(数学含义即"该类概率为 0"),属合法输出不该拦。

        抛 `NonFiniteProbabilityError`(`ValueError` 子类):视频逐帧循环靠这个**专属
        类型**区分「概率无效可丢帧」与「其它 ValueError 是故障须冒泡」——若抛基类
        ValueError,cv2/numpy 在畸形帧上抛的同类型异常会被误当可丢帧吞掉。
        """
        if not torch.isfinite(prob).all():
            raise NonFiniteProbabilityError(
                f"decode_predictions: {task} 概率含 NaN/Inf(logits 数值异常或权重损坏)"
                f"——拒绝导出无效分布(argmax 会静默返回 0,NaN 还会写出非标准 JSON)"
            )
        return prob

    def _softmax_head(task, logits):
        """→ (完整概率向量 np.ndarray, argmax 索引 int)。

        argmax 取自概率向量而非 logits:softmax 严格单调,二者等价,少一次前向。
        """
        prob = _assert_finite(task, torch.softmax(_cal(task, logits.float()), dim=1)[0])
        return prob.cpu().numpy(), int(torch.argmax(prob).item())

    # mask 为分割,不走温度缩放(分割校准是另一套,本期不碰)。
    # 但 NaN 校验不能省:分类头有 `_assert_finite` 硬拦,mask 若漏拦,NaN logits
    # 经 sigmoid 得 NaN → `(NaN > 0.5)` 恒 False → **静默产出全零分割图**,而
    # status=PASSED、报告正常——人工/评测只看 JSON 完全察觉不到分割已废(比报错更
    # 隐蔽)。故对激活后的 pred_mask 同样硬拦,交上层处置(单图 ERROR / 视频丢帧)。
    pred_mask = torch.sigmoid(outputs["mask"].float())[0, 0]
    _assert_finite("mask", pred_mask)
    pred_mask = pred_mask.cpu().numpy()
    # 件套①:阈值经 _seg_threshold(dynamic) 解析,env 未设时恒 0.5(逐位等于旧
    # 硬编码)。下游全链(case_pipeline/nodule_split/v21_output/诊断脚本)只读本
    # mask_binary 的非零性 → 此处一改即全链生效,无第二个分割阈值。
    mask_binary = (pred_mask > _seg_threshold(dynamic)).astype(np.uint8) * 255

    # argmax 对温度缩放不变(T>0 单调),保留校准以与概率刻度一致、便于上层取置信度。
    # 各头统一 .float():AMP fp16 下温度除法 logits/T 在半精度有舍入损失,先升 fp32 再缩放/softmax,
    # 与 malignancy 的精度路径对齐(reviewer 建议),不改 argmax 结果。
    comp_prob, comp_idx = _softmax_head("composition", outputs["composition"])
    echo_prob, echo_idx = _softmax_head("echogenicity", outputs["echogenicity"])
    shape_prob, shape_idx = _softmax_head("shape", outputs["shape"])
    margin_prob, margin_idx = _softmax_head("margin", outputs["margin"])
    foci_prob = _assert_finite(
        "foci", torch.sigmoid(_cal("foci", outputs["foci"].float()))[0]).cpu().numpy()
    mal_probs = _assert_finite(
        "malignancy",
        torch.softmax(_cal("malignancy", outputs["malignancy"].float()), dim=1)[0]).cpu().numpy()
    mal_prob = float(mal_probs[1])

    # --- 官方四头(位置/ETE/临床T/TI-RADS)真实输出,backlog #116b(2026-08-17) ---
    # 🔴 **不再二次 softmax**:各头已在模型内部做过 fp32 softmax
    #    (`heads/location.py:184-189` / `extension.py:192` / `clinical_t.py:230-236`
    #     / `tirads.py:166`),这里重算会把分布压平(softmax(softmax(x)) ≠ softmax(x))。
    # 🔴 **不走 calibrator**:`CALIBRATED_TASKS`(calibration.py:25-32)不含这四项,
    #    温度缩放对未标定的头是伪精度。
    # 🔴 **键缺失即硬报 KeyError,不给默认值**:四头是 `MultiTaskThyroidModel.forward`
    #    的无条件输出(multi_task_model.py:349-358)。若缺席,说明加载的是**不含四头
    #    结构的旧模型**——此时静默兜底会让 prediction.json 写出确定性占位值,而目标四
    #    40% 三项 + 目标三 TI-RADS 全部静默失分,且本地看不出来(JSON 完全合法)。
    # 🔴 **长度也要校验**(2026-08-17 第三轮 review lane4 L3,实测反例):
    #    下游 `v21_output._shared_nodule_context` 一律 `argmax+1`,长度错时若 argmax
    #    恰好落在合法值域内,就会**静默产出错误编码**——JSON 完全合法、该维度按错值
    #    计分(Kappa/AUC 被污染),本地永远查不出来。
    #    实测 8 个长度错场景:ETE/TI-RADS/Location 各自被下游的对冲键长度校验或
    #    `LocationProbs` 长度校验**侧面**拦下(那是巧合不是设计),**唯独 `t_probs`
    #    全程无人校验** —— 4 维且 argmax=idx1 时一路静默通过写出 T=1。
    #    故在源头统一按类别数硬校验,不依赖下游的偶然拦截。
    def _head_probs(key, n_expect):
        v = _assert_finite(key, outputs[f"{key}_probs"].float()[0]).cpu().numpy()
        if v.shape != (n_expect,):
            raise ValueError(
                f"{key}_probs 维度契约不符:期望 ({n_expect},),实得 {v.shape}。"
                f"说明加载的模型类别数与交付 schema 不一致,或上游取错了头——"
                f"继续下去会 argmax 出错误编码且不报错。"
            )
        return v

    ete_probs = _head_probs("ete", len(ETE_CODES))
    location_probs = _head_probs("location", NUM_LOCATIONS)
    t_probs = _head_probs("t", NUM_CLIN_T_CLASSES)
    tirads_probs = _head_probs("tirads", NUM_TIRADS_CLASSES)
    # t_conf 是 [B] 标量(= P(T=1)),非分布,无长度可校验;仍过有限性校验。
    # (NaN 会在此抛 NonFiniteProbabilityError —— 早于下游 `_clamp01`,报错点更靠近根因。)
    t_conf = float(_assert_finite("t_conf", outputs["t_conf"].float()[0]).item())

    return {
        "pred_mask": pred_mask,
        "mask_binary": mask_binary,
        "comp_idx": comp_idx,
        "echo_idx": echo_idx,
        "shape_idx": shape_idx,
        "margin_idx": margin_idx,
        "comp_prob": comp_prob,
        "echo_prob": echo_prob,
        "shape_prob": shape_prob,
        "margin_prob": margin_prob,
        "foci_prob": foci_prob,
        "mal_prob": mal_prob,
        "mal_probs": mal_probs,
        # --- 官方四头(#116b),消费方 v21_output._shared_nodule_context ---
        "ete_probs": ete_probs,
        "location_probs": location_probs,
        "t_probs": t_probs,
        "t_conf": t_conf,
        "tirads_probs": tirads_probs,
    }


def build_tirads_breakdown(comp_idx, echo_idx, shape_idx, margin_idx, foci_prob):
    """Compute TI-RADS point breakdown using calculate_tirads_points."""
    return calculate_tirads_points(
        comp_idx, echo_idx, shape_idx, margin_idx, foci_prob
    )


def build_probability_export(preds: dict) -> dict:
    """把 `decode_predictions` 的概率字段整成 JSON 可序列化的导出结构。

    官方硬约束(赛前动员会 2026-07-28):"所有分类/分级/分型任务必须输出功率分布或
    置信度,否则计0分"。静态图(`predict.py`)与动态图(`video_predict.py`)共用本函数,
    保证两条链路的 result.json schema 完全一致——新增维度只需改这一处。

    类别顺序即索引顺序,与 `src/utils/report_generator.py` 的 `COMPOSITION_MAP` /
    `ECHOGENICITY_MAP` / `SHAPE_MAP` / `MARGIN_MAP` / `FOCI_MAP` 的键严格对应。
    `composition`/`echogenicity`/`shape`/`margin` 为 softmax 分布(和为 1);
    `foci` 是**多标签 sigmoid**,各元素独立、不求和。
    """
    return {
        "concept_probabilities": {
            "composition": preds["comp_prob"].tolist(),
            "echogenicity": preds["echo_prob"].tolist(),
            "shape": preds["shape_prob"].tolist(),
            "margin": preds["margin_prob"].tolist(),
            "foci": preds["foci_prob"].tolist(),
        },
        "concept_predicted_index": {
            "composition": preds["comp_idx"],
            "echogenicity": preds["echo_idx"],
            "shape": preds["shape_idx"],
            "margin": preds["margin_idx"],
        },
        "malignancy_probabilities": preds["mal_probs"].tolist(),
    }
