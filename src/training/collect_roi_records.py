# -*- coding: utf-8 -*-
"""per-lesion(ROI 范式)验证记录收集器 —— P0 尺子先行的落点之一(新架构任务书 #20)。

## 为什么存在(改造背景,`docs/新架构文件级改动清单_20260925.md` §F 根因 #3)

`validate_step.collect_official_records` 按**整图**收集记录:一图一条
(`B = outputs["malignancy"].shape[0]`)。整图模型下恰好正确;**ROI 模型一图 K 条
记录(每病灶一条)**,若仍走整图收集,后续 `metrics_official` 收到的 records
数量错、病灶级聚合的分组也错 —— **指标静默失真、选模挑错 checkpoint**。

## 口径铁律

**与整图链路逐位兼容**:per-lesion 记录的**字段集与语义**完全复刻
`collect_official_records` 产出的记录(逐键同名同义),保证下游
`metrics_official.aggregate_by_lesion` / 各头指标函数**不改行为**就能吃 K 条/图。
特别地:

- 病灶级聚合**单源复用** `metrics_official.aggregate_by_lesion`(经由
  `metrics_official.roi_metrics` → `compute_official_metrics`):本模块不重实现
  聚合口径 —— 那是官方口径的病灶级聚合(概率均值/标签取首),重写一份必然
  漂移成两把尺子。
- lesion_id 缺失时造进程内单调占位键(整图链路同款策略,`__unaggregatable__`
  前缀),**绝不复用 `id()`**(CPython 地址回收复用,跨 batch 撞名会把不同病灶
  错聚成一个)。复用整图链路的 `_placeholder_counter`(同一进程内两个收集器
  共享同一命名空间,占位键永不撞车)。
- `mal_prob` 用 **torch [K,2] softmax** 取正类 —— 与整图链路
  `torch.softmax(outputs["malignancy"].float(), dim=1)[:, 1]` 走**同一条浮点
  路径**(K=1 对拍逐位一致的基础);不用 python math 手算(float64 算术会在
  末位与 float32 torch 路径分叉,破坏逐位一致)。

## 输入契约(ROI 模型 forward 的 outputs 形态)

`outputs["roi_outputs"]`:list[dict],长度 K = 本批展开后的病灶数。每个 dict
含一病灶的四头输出与元数据:

- `lesion_id` (str):病灶级聚合键。缺失 → 占位键。
- `is_main` (int 0/1 或 0 维张量):主病灶标志(官方筛选条件,所有指标只取
  is_main==1)。缺省 1(与整图链路 `label_list("is_main", 1)` 对齐)。
- `patient_id` (str,可选):病灶归属患者。
- 四头键(与整图 outputs 同名):`malignancy` [2] logits / `location_probs`
  [7] / `ete_probs` [4] / `ete_logits` [4] / `t_conf` 标量 / `tirads_probs` [5]。

`labels`(per-lesion 标签 list,与 `roi_outputs` 逐条对齐):每条 dict 含
`gt_malignancy` / `gt_lesion_loc` / `gt_extrathy_ext` / `gt_clin_t_stage` /
`gt_tirads` 及 `valid_*` 标志。键缺失 → 默认值与整图链路对齐:gt 缺省 0 /
valid 缺省 0(对应 `targets.get(..., torch.zeros(B))`);唯 `valid_malignancy`
缺省 1(对应 `label_list("valid_malignancy", 1)`)。

张量或 python 标量均可(自动 `.detach().cpu()` / float)。
"""

import torch

from src.training.loop.validate_step import _placeholder_counter

__all__ = ["collect_roi_records", "roi_records_from_batch", "roi_metrics"]


def _to_py(v):
    """张量/标量 → python 标量或 list[float]。0 维张量 → float,1 维 → list。"""
    if isinstance(v, torch.Tensor):
        v = v.detach().float().cpu()
        return v.item() if v.dim() == 0 else v.tolist()
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return float(v)


def _norm_probs(v, n_classes, key):
    """概率/logit 向量规整:list/tuple/1维张量 → list[float] 且长度 == n_classes。

    长度是下游 `ete_monitor_metrics` 的**硬 assert** 消费面(错长在那里抛
    AssertionError);此处提前校验,把失败点拉近数据源,报错指向 roi_outputs
    的生产方(ROI 模型头)而非指标层。
    """
    out = _to_py(v)
    if not isinstance(out, list) or len(out) != n_classes:
        raise ValueError(
            f"roi_outputs[{key}] 应为长度 {n_classes} 的向量,得到 "
            f"{type(v).__name__}"
            f"{' 长度 ' + str(len(out)) if isinstance(out, list) else ' 非向量'}"
            " —— 检查 ROI 头输出通道数"
        )
    return out


def _label_val(entry, key, default):
    """per-lesion 取值:dict.get → 0 维张量/标量 → int;缺失给 default。

    🔴 只接受**标量**值:batch 张量([B] 形)在这里是"labels 仍是整图 tensor dict
    而非 per-lesion list"的误接线指纹,必须 fail-loud(取 v[0] 会静默把整批标签
    当成单病灶标签,错得无声)。
    """
    v = entry.get(key, default) if isinstance(entry, dict) else default
    if isinstance(v, torch.Tensor):
        if v.numel() != 1:
            raise ValueError(
                f"per-lesion 字段 {key} 必须是标量,得到 shape {tuple(v.shape)}"
                " —— labels/roi dict 疑似整图 batch 张量误接 per-lesion 链路"
            )
        v = v.reshape(()).item()
    elif isinstance(v, (list, tuple)):
        if len(v) != 1:
            raise ValueError(
                f"per-lesion 字段 {key} 必须是标量,得到长度 {len(v)} 的序列"
            )
        v = v[0]
    return int(v)


def collect_roi_records(roi_outputs, labels):
    """从 ROI 模型的 per-lesion outputs + labels 建 K 条记录。

    Returns:
        list[dict]:每病灶一条,字段集与 `collect_official_records` 的逐样本记录
        **逐键一致**(lesion_id / patient_id / is_main / mal_prob / gt_malignancy /
        valid_malignancy / location_probs / gt_lesion_loc / valid_lesion_loc /
        ete_probs / ete_logits / gt_extrathy_ext / valid_extrathy_ext / t_conf /
        gt_clin_t_stage / valid_clin_t_stage / tirads_probs / gt_tirads /
        valid_tirads)。K = len(roi_outputs)。

    Raises:
        ValueError: `labels` 与 `roi_outputs` 长度不齐、良恶性 logits 长度非 2、
        概率向量长度错(与 `ete_monitor_metrics` 的 assert 同消费面,提前到收集
        层)、或 per-lesion 字段误接 batch 张量。
    """
    if len(roi_outputs) != len(labels):
        raise ValueError(
            f"roi_outputs({len(roi_outputs)}) 与 labels({len(labels)})"
            " 长度不齐 —— per-lesion 输出与标签必须逐条对齐"
        )
    # 🔴 防误接整图 tensor dict(Gate2 A 路 P1-1 实弹):K 恰等于 dict 键数时
    # 长度校验放行,zip 迭代 dict 产 str 键 → _label_val 全走 default → 标签静默
    # 变 0/valid 全 default。整图 labels 是 dict(迭代产 str),per-lesion 必是
    # list[dict]。K=0 空批例外(损失侧同型早退,合法)。
    if labels and hasattr(labels, "keys"):
        raise ValueError(
            f"labels 是 {type(labels).__name__} —— 疑似误接整图 tensor dict"
            "(迭代产 str 键,per-lesion 标签必须是 list[dict])"
        )
    if labels and not isinstance(list(labels)[0], dict):
        raise ValueError(
            "labels 应为 list[dict](per-lesion 标签),首元素是 "
            f"{type(list(labels)[0]).__name__} —— 疑似误接整图形态"
        )

    from src.models.heads import NUM_ETE_CLASSES, NUM_TIRADS_CLASSES
    from src.models.label_schema import NUM_LOCATIONS

    # 良恶性 [K,2] softmax(与整图链路同一浮点路径,见模块 docstring)。
    # K=0 空批(Gate2 A 路 P2-1):损失侧同型早退支持空批,这里跳过 softmax
    # 避免 torch.tensor([]) 的 (0,) 形状在 [:, 1] 上 IndexError(Dimension
    # out of range)——报错指向这里而非数据源,掩盖真实问题。
    mal_p_all: list = []
    if roi_outputs:
        mal_rows = []
        for roi in roi_outputs:
            row = _to_py(roi["malignancy"])
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError(
                    f"roi_outputs['malignancy'] 应为长度 2 的 logits,得到 {row!r}"
                    " —— 良恶性头是二分类"
                )
            mal_rows.append(row)
        mal_p_all = torch.softmax(
            torch.tensor(mal_rows, dtype=torch.float32), dim=1
        )[:, 1].tolist()

    records = []
    for i, (roi, lab) in enumerate(zip(roi_outputs, labels)):
        lid = roi.get("lesion_id")
        if lid is None:
            # 整图链路同款:宁可不聚合,不可错聚合(进程内单调占位)。
            lid = f"__unaggregatable__{next(_placeholder_counter)}"

        # 🔴 E1-F1 硬前置②(2026-09-25):实例级聚合键。CCL 会把同一 GT 病灶
        # split 成多实例(推理侧 `_run_roi_static/_run_roi_dynamic` 逐实例独立
        # 前向),训练尺子必须把"同 lesion 的不同实例"区分开,否则同帧双实例
        # 0.9975/0.0025 被 `aggregate_by_lesion` 摊平成 0.5(本地 val 系统性
        # 乐观,判线对拍对不上)。取值优先级 = roi(模型输出侧,生产方)>
        # labels(meta 桥透传)> 缺省 0(单实例语义:GT-mask ROI 流的既有
        # 行为,聚合退化与 `aggregate_by_lesion` 逐位等价,有对拍测试钉死)。
        # batch 张量([K] 形)是"meta 仍是整图 tensor dict"的误接指纹,
        # 同 `_label_val` 纪律 fail-loud(静默取 v[0] 会把整批当单实例)。
        if "instance_idx" in roi:
            inst = _label_val(roi, "instance_idx", 0)
        else:
            inst = _label_val(lab, "instance_idx", 0)

        records.append({
            "lesion_id": lid,
            "instance_idx": inst,
            "patient_id": roi.get("patient_id"),
            "is_main": _label_val(roi, "is_main", 1),
            "mal_prob": mal_p_all[i],
            "gt_malignancy": _label_val(lab, "gt_malignancy", 0),
            "valid_malignancy": _label_val(lab, "valid_malignancy", 1),
            "location_probs": _norm_probs(
                roi["location_probs"], NUM_LOCATIONS, "location_probs"),
            "gt_lesion_loc": _label_val(lab, "gt_lesion_loc", 0),
            "valid_lesion_loc": _label_val(lab, "valid_lesion_loc", 0),
            "ete_probs": _norm_probs(
                roi["ete_probs"], NUM_ETE_CLASSES, "ete_probs"),
            "ete_logits": _norm_probs(
                roi["ete_logits"], NUM_ETE_CLASSES, "ete_logits"),
            "gt_extrathy_ext": _label_val(lab, "gt_extrathy_ext", 0),
            "valid_extrathy_ext": _label_val(lab, "valid_extrathy_ext", 0),
            "t_conf": float(_to_py(roi["t_conf"])),
            "gt_clin_t_stage": _label_val(lab, "gt_clin_t_stage", 0),
            "valid_clin_t_stage": _label_val(lab, "valid_clin_t_stage", 0),
            "tirads_probs": _norm_probs(
                roi["tirads_probs"], NUM_TIRADS_CLASSES, "tirads_probs"),
            "gt_tirads": _label_val(lab, "gt_tirads", 0),
            "valid_tirads": _label_val(lab, "valid_tirads", 0),
        })
    return records


def roi_records_from_batch(outputs, labels):
    """validate_batch 的 ROI 分支入口:outputs 含 `roi_outputs` 键时收集 per-lesion 记录。

    本函数**不嵌进** `collect_official_records` 函数体(保持整图函数可读性,
    两路各自独立演化)。`labels` 在 ROI 范式下是 per-lesion 标签 list
    (list[dict]),与 `outputs["roi_outputs"]` 逐条对齐;旧形态(tensor dict)
    会因长度不齐或 batch 张量校验 fail-loud,不会静默错。
    """
    return collect_roi_records(outputs["roi_outputs"], labels)


def roi_metrics(roi_records):
    """ROI 记录 → 官方口径指标(per-lesion 聚合后走既有 `compute_official_metrics`)。

    单独包装而不让调用方直连 metrics_official,是为了把"ROI 记录形态"与
    "指标入口"的耦合收在 collect_roi_records 一处(测试与后续训练侧消费
    统一 import 本模块)。空 records 透传(空口径由下游 dict 返回)。
    """
    from src.training.metrics_official import (
        aggregate_by_roi_instance, compute_official_metrics)

    # 🔴 E1-F1 硬前置②:ROI 尺子的聚合必须升**实例级**(aggregate_by_roi_instance),
    # 与推理侧逐实例独立前向同构。缺省 aggregate_by_lesion 会把同帧双实例
    # 0.9975/0.0025 均值摊平成 0.5 —— 本地 val 系统性乐观(E1 review F1 实锤)。
    # 整图链不受影响:它的调用点(trainer.py)不带 aggregation,逐位走旧口径。
    return compute_official_metrics(
        roi_records, aggregation=aggregate_by_roi_instance)
