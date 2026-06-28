"""动态图(cine/视频)推理的纯逻辑:抽帧采样 + 帧级预测聚合。

与模型/IO 解耦,便于单测。`video_predict.py` 负责把帧喂给单帧模型,
再调用本模块聚合。聚合策略对齐"癌症检测"语义:恶性概率取**最大帧**做最终
判定(任一帧可疑即可疑,宁可召回不漏),同时报均值供参考。
"""

from typing import List, Dict


def uniform_sample_indices(n_total: int, max_frames: int) -> List[int]:
    """从 n_total 帧里均匀取至多 max_frames 个下标(含首尾,去重升序)。

    n_total <= max_frames 时返回全部下标。max_frames<=0 视为不限。
    """
    if n_total <= 0:
        return []
    if max_frames <= 0 or n_total <= max_frames:
        return list(range(n_total))
    if max_frames == 1:
        return [0]  # 单帧:取首帧(避免 step 除零)
    # 均匀分布在 [0, n_total-1],含端点
    step = (n_total - 1) / (max_frames - 1)
    idxs = sorted({int(round(i * step)) for i in range(max_frames)})
    return idxs


def aggregate_frame_predictions(per_frame: List[Dict]) -> Dict:
    """聚合逐帧预测。

    Args:
        per_frame: 每帧一个 dict,至少含 "mal_prob"(float);可含 "frame_index"。

    Returns:
        dict: {
            n_frames, mal_prob_mean, mal_prob_max, mal_prob_min,
            keyframe_index(最恶性帧在列表中的位置),
            keyframe_frame_index(该帧的原始帧号,缺省回退为 keyframe_index),
            final_is_malignant(按 max>=0.5),
        }
        per_frame 为空时返回 n_frames=0 的安全空结果。
    """
    n = len(per_frame)
    if n == 0:
        return {
            "n_frames": 0,
            "mal_prob_mean": 0.0,
            "mal_prob_max": 0.0,
            "mal_prob_min": 0.0,
            "keyframe_index": -1,
            "keyframe_frame_index": -1,
            "final_is_malignant": 0,
        }
    probs = [float(f["mal_prob"]) for f in per_frame]
    max_pos = max(range(n), key=lambda i: probs[i])
    kf_frame = int(per_frame[max_pos].get("frame_index", max_pos))
    return {
        "n_frames": n,
        "mal_prob_mean": sum(probs) / n,
        "mal_prob_max": probs[max_pos],
        "mal_prob_min": min(probs),
        "keyframe_index": max_pos,
        "keyframe_frame_index": kf_frame,
        "final_is_malignant": int(probs[max_pos] >= 0.5),
    }
