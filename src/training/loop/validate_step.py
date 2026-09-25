import itertools

import torch


# lesion_id 缺失时的占位键计数器(进程内单调,绝不复用)。见 collect_official_records。
_placeholder_counter = itertools.count()


# 官方新增维度头(位置/ETE/临床T/TI-RADS)的 targets 组装规则与既有概念任务不同
# (无 mixup、validity 顶层直传),见 `train_step.py` 同名注释。验证期无 mixup,
# 故这里直接组装。
_OFFICIAL_HEAD_TASKS = ("lesion_loc", "extrathy_ext", "clin_t_stage", "tirads")


def _official_head_targets(labels, batch_size, device):
    """把 loader 给的官方头标签/validity 转成 targets 片段。键缺席 → 返回空 dict。"""
    out = {}
    for task in _OFFICIAL_HEAD_TASKS:
        if task not in labels:
            continue
        out[task] = labels[task].to(device).long()
        v = labels.get(f"valid_{task}")
        out[f"valid_{task}"] = (
            v.float().to(device)
            if isinstance(v, torch.Tensor)
            else torch.ones(batch_size, dtype=torch.float32, device=device)
        )
    return out


def collect_official_records(outputs, labels, targets):
    """逐样本抽出官方指标所需的全部字段,返回 list[dict](CPU、python 标量)。

    为什么按样本而非按批汇总:官方分类指标是**病灶级**的(动态数据同一病灶的多帧
    只算一个样本),聚合键 `lesion_id` 只能逐样本携带;且 ETE/T 的筛选条件里有
    "金标准为恶性"这一项,批级汇总后无从回溯。指标计算本身在
    `src/training/metrics_official.py`,本函数只负责搬运、不做筛选。

    三头键缺席(legacy 数据集)→ 返回空 list,调用方跳过官方指标。

    🔴 **ROI 范式分支(2026-09-25 P0 尺子先行)**:outputs 含 `roi_outputs` 键
    (list[dict],一图 K 条,每 dict 一病灶)时**不走本函数的整图路径**,由
    `collect_roi_records.roi_records_from_batch` 走 per-lesion 收集(分流逻辑
    见 `validate_batch` 内调用点,本函数保持整图语义不嵌 ROI 分支)。无该键时
    逐位走本函数原路径 —— **零回归铁律**。
    """
    # 🔴 ROI 分支(2026-09-25 P0):outputs 含 `roi_outputs` 键 → per-lesion 收集
    # (一图 K 条记录,每病灶一条)。无该键 → 逐位走下方整图路径(零回归铁律:
    # 整图模型的 val 指标不得因本次改造改变任何一位数字)。
    # 分流判据用**键存在性**而非真值:roi_outputs 恒为非空 list(ROI 范式下模型
    # 必产 per-lesion 输出;空 list 属上游契约违规,交给 collect_roi_records 的
    # 长度对齐校验去 fail-loud,不在这里静默回退整图路径 —— 那会把"ROI 模型
    # 却产出 0 记录"伪装成"整图指标")。
    if "roi_outputs" in outputs:
        from src.training.collect_roi_records import roi_records_from_batch

        return roi_records_from_batch(outputs, labels)

    if "lesion_loc" not in targets:
        return []

    B = outputs["malignancy"].shape[0]
    mal_probs = torch.softmax(outputs["malignancy"].float(), dim=1)[:, 1]

    def cpu_list(t):
        return t.detach().float().cpu().tolist()

    def label_list(key, default=0):
        v = labels.get(key)
        if v is None:
            return [default] * B
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().tolist()
        return list(v) if isinstance(v, (list, tuple)) else [v] * B

    loc_probs = cpu_list(outputs["location_probs"])
    ete_probs = cpu_list(outputs["ete_probs"])
    # 🔴 **logits 必须单独采集,不能从 probs 反推**(backlog #128 的 4 项监控之一
    # `val_logit_mean_per_class` 要用):softmax 有**平移不变性**,`logits` 与
    # `logits+c` 给出同一个 probs,故从 probs 反推不出 logits 的绝对量级 ——
    # 而 DR 的预警判据恰恰是绝对量级(`z̄3 − z̄0 > 0` = 对稀有类过度偏移)。
    ete_logits = cpu_list(outputs["ete_logits"])
    t_conf = cpu_list(outputs["t_conf"])
    tirads_probs = cpu_list(outputs["tirads_probs"])
    mal_p = cpu_list(mal_probs)

    # lesion_id / patient_id 经 default_collate 是 list[str];缺失时退化为逐样本
    # 唯一占位,绝不让不同病灶被并成一个(宁可不聚合,不可错聚合)。
    lesion_ids = label_list("lesion_id", None)
    patient_ids = label_list("patient_id", None)
    is_main = label_list("is_main", 1)
    gt_loc = cpu_list(targets["lesion_loc"])
    gt_ete = cpu_list(targets["extrathy_ext"]) if "extrathy_ext" in targets else [0] * B
    gt_t = cpu_list(targets["clin_t_stage"]) if "clin_t_stage" in targets else [0] * B
    gt_tirads = cpu_list(targets["tirads"]) if "tirads" in targets else [0] * B
    gt_mal = cpu_list(targets["malignancy"])
    v_loc = cpu_list(targets["valid_lesion_loc"])
    v_ete = cpu_list(targets.get("valid_extrathy_ext", torch.zeros(B)))
    v_t = cpu_list(targets.get("valid_clin_t_stage", torch.zeros(B)))
    v_tirads = cpu_list(targets.get("valid_tirads", torch.zeros(B)))
    v_mal = label_list("valid_malignancy", 1)

    records = []
    for i in range(B):
        lid = lesion_ids[i]
        if lid is None:
            # lesion_id 缺失(legacy 数据集)→ 造不可聚合的唯一占位。
            # 🔴 **不要用 `id(outputs)`**:CPython 的对象地址在前一个 dict 被回收后
            # 会复用(实测同地址),跨 batch 撞名会把不同病灶错误聚成一个 —— 恰好
            # 违反"宁可不聚合,不可错聚合"。用进程内单调计数器。
            lid = f"__unaggregatable__{next(_placeholder_counter)}"
        records.append({
            "lesion_id": lid,
            "patient_id": patient_ids[i],
            "is_main": int(is_main[i]),
            "mal_prob": mal_p[i],
            "gt_malignancy": int(gt_mal[i]),
            "valid_malignancy": int(v_mal[i]),
            "location_probs": loc_probs[i],
            "gt_lesion_loc": int(gt_loc[i]),
            "valid_lesion_loc": int(v_loc[i]),
            "ete_probs": ete_probs[i],
            "ete_logits": ete_logits[i],
            "gt_extrathy_ext": int(gt_ete[i]),
            "valid_extrathy_ext": int(v_ete[i]),
            "t_conf": t_conf[i],
            "gt_clin_t_stage": int(gt_t[i]),
            "valid_clin_t_stage": int(v_t[i]),
            "tirads_probs": tirads_probs[i],
            "gt_tirads": int(gt_tirads[i]),
            "valid_tirads": int(v_tirads[i]),
        })
    return records


def validate_batch(
    model, images, masks, labels, criterion, device, stage, batch_idx, epoch,
    seg_prob_out=None,
):
    """
    单个批次的验证逻辑。

    Args:
        seg_prob_out: 可选 dict。传入时**原地累加**分割 `pred_prob` 的统计量
            (`sum`/`count`/`max`/`over_half`),供 `validate_epoch` 汇总后写进 JSONL。
            🔴 **为什么需要(2026-08-16,勿删)**:`pred_prob` 此前**只在 `batch_idx==0`
            打一行 stdout**,JSONL 里完全没有。而 2026-08-15 那轮之所以能定位到
            "分割全程冻结"这个根因,靠的正是这个数(全域 0.4425~0.4463 < 0.5 →
            `pred_mask` 恒全零)。分割改自训后,它是"分割正在升温"的**唯一先行指标**
            —— hard `val_dice` 在前景 logit 越过 0 之前一直是 0.0,而 `pred_prob`
            的均值/最大值/过半像素占比会先动。没有它,云端 agent 在护栏触发前
            只能对着 `dice=0.000` 盲等。
            用出参而非加返回值:本函数返回 8 元组,`trainer.py:117` 按位解包,
            加宽会破坏既有调用点。
        labels: loader 给的 labels dict(collate 后)。**ROI 范式下形态不同**:
            当模型 outputs 含 `roi_outputs` 键时,本函数把 labels 原样转交
            `collect_roi_records`(期望 per-lesion 标签 list,与 roi_outputs
            逐条对齐;形态不符会在那里 fail-loud)。整图路径下 labels 仍是
            tensor dict,本函数的 mal_correct/mal_targets 等**整图统计字段照旧
            从 `outputs["malignancy"]` / `targets["malignancy"]` 计算**(ROI 模型
            仍须产出整图级 malignancy logits 供这些统计,该契约由模型侧保证)。
    Returns:
        loss: 损失
        loss_dict: 损失字典
        dice_sum: Dice 系数总和
        mal_correct: 恶性分类正确数
        batch_size: 批次大小
        mal_probs: 恶性概率 (CPU tensor)
        mal_targets: 恶性标签 (CPU tensor)
        official_records: list[dict] 官方指标字段。**两条口径**:
            整图模型(无 `roi_outputs` 键)→ 逐样本一条(三头键缺席时为空 list);
            ROI 模型(有键)→ per-lesion 一条(一图 K 条,字段集与整图记录
            逐键一致,由 `collect_roi_records` 保证)。
    """
    images = images.to(device)
    masks = masks.to(device)

    targets = {
        "mask": masks,
        "composition": labels["composition"].to(device),
        "echogenicity": labels["echogenicity"].to(device),
        "shape": labels["shape"].to(device),
        "margin": labels["margin"].to(device),
        "foci": labels["foci"].to(device),
        "malignancy": labels["malignancy"].to(device),
    }
    # 🔴 valid_mask 顶层直传(2026-09-13 修复,2026-09-14 Gate2 收紧为与
    # train_step **同存在性闸门**):此前验证路径漏组这个键,导致
    # compute_segmentation_loss 永远走 legacy 分支——权重全 0 时无副作用
    # (掩盖至今),seg-only 阶段2 加 boundary/lovasz 显式 raise 后才暴露。
    # ⚠️ 与 train_step.py 同条件(labels 有 valid_* 键才写,缺席走 legacy):
    # 若无条件写全 1 兜底,labels 无该键的数据集上 train 走 legacy、val 走
    # masked,两侧不同式且 seg_only 的 raise 只在 train 侧触发,报错指向误导。
    # 官方 loader 恒写 valid_mask(loader.py:96 / official_loader.py:552 /
    # pathology_dataset.py:250),官方路径两侧均走 masked 分支。
    # ⚠️ 口径注:本修复使官方 val_loss 从 legacy 换到 masked 路径(此前 train
    # masked/val legacy 本就不同式),r5/r6/r7 轮 JSONL 的 val_loss 与新轮不可比。
    if "valid_composition" in labels:
        v_mask = labels.get("valid_mask")
        targets["valid_mask"] = (
            v_mask.float().to(device) if isinstance(v_mask, torch.Tensor)
            else torch.full((images.size(0),), float(v_mask) if v_mask is not None else 1.0,
                            dtype=torch.float32, device=device)
        )
    targets.update(_official_head_targets(labels, images.size(0), device))
    # L1 悬崖权重(plan 任务L1,2026-09-23):val 侧全 1 中性权重——cliff 是
    # 训练期 hard-example mining 信号,val 指标不消费它;但权重>0 时损失
    # 的 fail-loud 要求键存在,这里补全 1(数学上 = 不加权的 val loss,语义
    # 与"val 不做加权"一致;seg_cliff 分项照写,JSONL 可观测 val 侧口径)。
    if getattr(criterion, "seg_cliff_weight", 0.0) > 0.0:
        targets["cliff_weights"] = torch.ones(
            images.size(0), dtype=torch.float32, device=device)
    # ROI 键注入(债1/债4,2026-09-25 训练侧接线;与 train_step 同型):整图
    # val 走 K=0 空批(ROI 范式的 val 指标由独立 validate_roi_epoch 算,主线
    # 拍板双 loader;整图 outputs 不产 roi_outputs → outputs 侧同补空 list,
    # 见下方 outputs 组装)。val 的 ROI 观测键只进 epoch 级 extra(roi_* 前缀),
    # 不进本函数的 loss/records。
    if getattr(criterion, "roi_head_weight", 0.0) > 0.0:
        targets["roi_labels"] = targets.get("roi_labels") or []

    # 验证期固定 gamma_prior=1.0(与最终部署一致),使各 epoch 的 val 指标在相同
    # 物理先验强度下计算,保证选模/早停判据的跨 epoch 可比性。训练期仍按 epoch 退火。
    outputs = model(images, gamma_prior=1.0)
    # ROI outputs 键(债4,与 train_step 同型):整图前向不产 roi_outputs,权重>0
    # 时损失层 fail-loud——补 **K=0 空 list**(合法早退)。🔴 疑虑"空 list 触发
    # collect_official_records 的 ROI 分流"在此不成立:分流在**本函数末尾**的
    # collect_official_records(outputs, ...) 消费 outputs——见下方,ROI 范式
    # 的 val 收集走 validate_roi_epoch 独立链,本处 outputs 注空 list 后
    # collect_official_records 会走 per-lesion 分支产 0 records → 整图官方指标
    # 消失!故**在此处把键摘掉再交收集**(收集层读的 outputs 是本函数局部
    # 变量;摘键 = 保持整图收集语义,损失层已算完)。
    _roi_injected = False
    if getattr(criterion, "roi_head_weight", 0.0) > 0.0:
        outputs = {**outputs, "roi_outputs": []}
        _roi_injected = True
    loss, loss_dict = criterion(
        outputs, targets, update_famo_weights=False, enable_mal_loss=(stage > 1)
    )
    if _roi_injected:
        outputs = {k: v for k, v in outputs.items() if k != "roi_outputs"}

    batch_size = images.size(0)

    # Seg Dice with diagnostic
    pred_prob = torch.sigmoid(outputs["mask"])
    pred_mask = (pred_prob > 0.5).float()

    # Diagnostic: log if predictions are degenerate (all-zeros or all-ones)
    if batch_idx == 0:
        pred_mean = pred_prob.mean().item()
        pred_std = pred_prob.std().item()
        print(
            f"  [Val Diag] pred_prob: mean={pred_mean:.4f}, std={pred_std:.4f}, min={pred_prob.min().item():.4f}, max={pred_prob.max().item():.4f}",
            flush=True,
        )

    # 分割概率分布统计(全批累加,不只第一批)——供 JSONL 观测分割升温。
    # 语义见本函数 docstring 的 `seg_prob_out` 段。`over_half` 是**过阈值像素占比**:
    # hard dice 只有在这个数离开 0 之后才可能非零,故它比 dice 更早给出信号。
    if seg_prob_out is not None:
        n = pred_prob.numel()
        seg_prob_out["sum"] = seg_prob_out.get("sum", 0.0) + pred_prob.sum().item()
        seg_prob_out["count"] = seg_prob_out.get("count", 0) + n
        seg_prob_out["max"] = max(seg_prob_out.get("max", 0.0), pred_prob.max().item())
        seg_prob_out["over_half"] = seg_prob_out.get("over_half", 0) + int(pred_mask.sum().item())
        # L1 cliff 分项累加(2026-09-24 审查 P1 修):val 全 1 注入使 val_loss
        # 含 cliff 项 ≈ 主项量级增量,与旧轮不可比且无字段可扣除。同 dict 顺带
        # 累加 loss_dict 的 seg_cliff 分项(权重 0 时 loss_dict 恒有 0.0 键,
        # 零成本),validate_epoch 汇总后写 JSONL,val 行即可对拍扣除。
        if "cliff_sum" not in seg_prob_out:
            seg_prob_out["cliff_sum"] = 0.0
            seg_prob_out["cliff_n"] = 0
        seg_prob_out["cliff_sum"] += float(
            loss_dict.get("seg_cliff", 0.0))
        seg_prob_out["cliff_n"] += 1

    intersection = (pred_mask * targets["mask"]).sum(dim=(2, 3))
    union = pred_mask.sum(dim=(2, 3)) + targets["mask"].sum(dim=(2, 3))
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    dice_sum = dice.sum().item()

    # Classification
    pred_mal = torch.argmax(outputs["malignancy"], dim=1)
    mal_correct = (pred_mal == targets["malignancy"]).sum().item()
    mal_probs = torch.softmax(outputs["malignancy"].float(), dim=1)[:, 1].detach().cpu()
    mal_targets = targets["malignancy"].detach().cpu()

    official_records = collect_official_records(outputs, labels, targets)

    return (
        loss,
        loss_dict,
        dice_sum,
        mal_correct,
        batch_size,
        mal_probs,
        mal_targets,
        official_records,
    )
