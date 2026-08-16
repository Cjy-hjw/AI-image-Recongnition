import torch

from src.dataset.augmentation import apply_mixup_or_cutmix
from src.training.loop.slr import compute_slr_weights
from src.training.loop.vclub import vclub_learning_step, compute_vclub_mi_bound
from src.training.stages import compute_gamma_prior

# 官方新增维度头任务名(位置/ETE/临床T/TI-RADS)。与 `validate_step._OFFICIAL_HEAD_TASKS`
# 同一份语义,两侧都只是遍历顺序,不构成跨模块契约,故各自持有(避免 loop 子模块间横向依赖)。
_OFFICIAL_HEAD_TASKS = ("lesion_loc", "extrathy_ext", "clin_t_stage", "tirads")


def build_official_head_targets(labels, mix_info, batch_size, device):
    """组装官方新增维度头(位置/ETE/临床T/TI-RADS)的 targets 片段。键缺席 → 返回空 dict。

    提成模块级函数(而非内联在 `train_batch`)是为了可单测:这里的每条规则错了都
    **不会报错**,只会让指标系统性偏低 —— 正是 `compute_gamma_prior` 当年被提取的
    同一理由(崩塌复盘时公式内联、无法单测)。测试见
    `tests/test_train_step_head_targets.py`。

    Args:
        labels: loader 给的 labels dict(collate 后)。
        mix_info: `apply_mixup_or_cutmix` 的返回,None 表示本批未混合。
        batch_size / device: 用于缺键时造默认张量。

    ## 三条规则

    **① 必须在 mixup/cutmix 之后调用**。`apply_mixup_or_cutmix` 原样返回 labels,
    既有概念任务靠 `lam` 做**软标签混合**(concepts.py),而三头是硬标签 masked CE,
    没有对应的软化路径。若在 mixup 前组装,图像被混了而位置/ETE/T 标签仍是
    recipient 的 → 静默错监督、loss 照常降。

    混合后的标签归属按 DR-CutMix 已算好的 `lam` 判定(mixup.py:50-68),不另发明规则:
      - `lam==0` → 100% 继承 donor(`index` 指向的样本)标签,须 gather;
      - `lam==1` → 100% 保留 recipient 标签;
      - 其余(含标准 MixUp 的软 lam)→ 归属不明确,validity 置 0 屏蔽掉。
    再乘 `mask_weight`(DR-CutMix 判定归属不明或无结节时已置 0)。

    **② validity 走顶层直传键**,不进 `label_validity` —— 后者只被 malignancy.py /
    concepts.py 消费,三头损失各自接 [B] validity 参数、根本不读它。塞错地方
    不报错,但 invalid 样本的垃圾标签会当真监督(审查 F2,钉在
    `tests/test_official_head_losses.py::test_validity_must_be_toplevel_key...`)。

    **③ 🔴 三头只训主病灶**。评审规则原文:「甲状腺外侵犯、主病灶位置、T临床显著性
    仅对主病灶标注,评估时仅针对匹配成功的主病灶」;总纲 §1 已把"损失加权与验证
    指标都只按主病灶统计"定为 2.4 接线的显式要求。`official_loader` 对 `病灶N`
    次病灶也建样本(约 13% 带临床标注),它们的分割/概念监督仍有价值,但三头标签
    **在官方口径里根本不存在** —— 放进损失等于拿非评分对象训评分头。
    `is_main` 键缺席(legacy 数据集)时全当主病灶,与接线前行为一致。
    """
    out = {}
    if not any(t in labels for t in _OFFICIAL_HEAD_TASKS):
        return out

    is_main = labels.get("is_main")
    if isinstance(is_main, torch.Tensor):
        main_w = is_main.float().to(device)
    else:
        main_w = torch.ones(batch_size, dtype=torch.float32, device=device)

    donor = hard = idx = None
    if mix_info is not None:
        lam = mix_info["lam"].to(device)
        idx = mix_info["index"].to(device)
        # 🔴 主导权阈值(2026-08-14 修真 bug,勿改回 `== 0.0` / `== 1.0`)
        #
        # 修复前用的是**精确相等**:`donor = lam == 0.0`、`hard = donor | (lam == 1.0)`。
        # CutMix 分支会把 `lams` 逐样本判成 0.0/1.0(DR-CutMix 按 donor 占比 0.65/0.35),
        # 所以精确相等对它成立;但 **MixUp 分支的 `lams` 是全批同一个
        # `lam ~ Beta(0.2,0.2)`,严格落在 (0,1)** —— 于是 MixUp 一触发,`hard` 全 False,
        # **四头(位置/ETE/临床T/TI-RADS)的 valid 被整批乘成 0**。
        #   实测后果:`apply_mixup_or_cutmix(prob=0.3)` 且 CutMix/MixUp 各半 →
        #   约 15% 的 batch 四头零监督,不报错、只是没学。这正是 2026-08-14 首次真训
        #   loss 在 ~5/~0.8 双峰跳动的根因(低值行四头分项全 0.0)。
        #
        # 改用与 DR-CutMix **同一套阈值**(0.65/0.35)判主导方:MixUp 里 donor 占比 = 1-lam,
        # 故 `lam <= 0.35` 判 donor 主导、`lam >= 0.65` 判 recipient 主导,中间才屏蔽。
        # ⚠️ **对 CutMix 完全等价**(其 lams 已是 0.0/1.0,`0.0<=0.35`、`1.0>=0.65` 恒成立),
        #    行为不变;中间档 CutMix 靠 `mask_weight=0` 屏蔽,这条路径也未动。
        # ⚠️ **刻意只改这里、不动 `mixup.py`**:`lam` 还被 CBM 概念损失
        #    (`losses/components/concepts.py:39-105` 软插值)与 SLR(`loop/slr.py:35`)
        #    当**软加权系数**消费,在 mixup 侧硬化成 0/1 会连带改掉旧任务的数学;
        #    而本处的闸门只服务四头。`mask_weight` 同理不动(它还门控概念损失)。
        donor = lam <= 0.35                     # donor 占比 >= 0.65 → 继承 donor 标签
        hard = donor | (lam >= 0.65)            # 归属明确(硬继承)才留监督
        # is_main 跟着标签走:donor 位继承 donor 的主病灶属性,否则会把次病灶
        # 的标签当主病灶训(标签换了、屏蔽没换)
        main_w = torch.where(donor, main_w[idx], main_w)

    for task in _OFFICIAL_HEAD_TASKS:
        if task not in labels:
            continue
        y = labels[task].to(device).long()
        v = labels.get(f"valid_{task}")
        v = (
            v.float().to(device)
            if isinstance(v, torch.Tensor)
            else torch.ones(batch_size, dtype=torch.float32, device=device)
        )
        if mix_info is not None:
            v = torch.where(donor, v[idx], v)   # donor 样本的 validity 也要跟着换
            y = torch.where(donor, y[idx], y)
            v = v * hard.float() * mix_info["mask_weight"].to(device)
        out[task] = y
        out[f"valid_{task}"] = v * main_w
    return out


def train_batch(
    model,
    images,
    masks,
    labels,
    optimizer,
    vclub_optimizer,
    criterion,
    scaler,
    device,
    config,
    running_losses_history,
    epoch,
    stage,
    update_famo_weights,
    prev_z_latent,
    prev_concept_vector,
    slr_beta,
    slr_threshold,
    label_smoothing,
    spin_l1_weight,
):
    """
    单个批次的训练逻辑。

    Returns:
        loss: 总损失
        loss_dict: 损失字典
        prev_z_latent: 缓存的 detached z_latent（用于下一批次 vCLUB）
        prev_concept_vector: 缓存的 detached concept_vector（用于下一批次 vCLUB）
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

    # Label validity mask (per-sample per-task): 1=valid, 0=absent
    # If labels contain valid_* keys (new datasets), build label_validity dict.
    # If absent (old data / no valid_* keys), skip → loss components use legacy path.
    _validity_tasks = ("composition", "echogenicity", "shape", "margin", "foci", "malignancy")
    if f"valid_{_validity_tasks[0]}" in labels:
        _B = masks.shape[0]

        def _to_validity_tensor(key):
            v = labels.get(key)
            if v is None:
                return torch.ones(_B, dtype=torch.float32, device=device)
            if isinstance(v, torch.Tensor):
                return v.float().to(device)
            # scalar (int/float) shared across batch
            return torch.full((_B,), float(v), dtype=torch.float32, device=device)

        targets["label_validity"] = {
            task: _to_validity_tensor(f"valid_{task}") for task in _validity_tasks
        }
        # Segmentation validity
        targets["valid_mask"] = _to_validity_tensor("valid_mask")

    # Apply MixUp / CutMix
    images, masks, targets, mix_info = apply_mixup_or_cutmix(
        images, masks, targets, alpha=0.2, prob=0.3
    )
    targets["mask"] = masks
    # label_validity is NOT mixed — validity flags are batch-level metadata,
    # not feature-space signals. We keep the original batch's validity unchanged.
    # (A mixed sample still has the same label availability as its primary sample.)

    # 官方三头 targets —— **必须在 mixup 之后**,规则与理由见
    # `build_official_head_targets` docstring(硬标签无软化路径 / validity 顶层直传 /
    # 只训主病灶)。三头键缺席时返回空 dict → legacy train.py 路径零回归。
    targets.update(
        build_official_head_targets(labels, mix_info, images.shape[0], device)
    )

    # vCLUB 独立优化：使用前一批次的 detached 输出，在 autocast 外执行独立 backward
    vclub_learning_step(model, vclub_optimizer, prev_z_latent, prev_concept_vector)

    optimizer.zero_grad()

    gamma_prior = compute_gamma_prior(epoch)
    amp_enabled = config["train"].get("amp", True)
    with torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=torch.bfloat16):
        outputs = model(images, gamma_prior=gamma_prior)

        # SLR 鲁棒标签校正
        if slr_beta < 1.0:
            slr_weights, _sample_losses = compute_slr_weights(
                outputs,
                targets,
                mix_info,
                running_losses_history,
                slr_beta,
                slr_threshold,
            )

            loss, loss_dict = criterion(
                outputs,
                targets,
                slr_weights=slr_weights,
                mix_info=mix_info,
                update_famo_weights=update_famo_weights,
                spin_l1_weight=spin_l1_weight,
                label_smoothing=label_smoothing,
                enable_mal_loss=(stage > 1),
            )
        else:
            loss, loss_dict = criterion(
                outputs,
                targets,
                mix_info=mix_info,
                update_famo_weights=update_famo_weights,
                spin_l1_weight=spin_l1_weight,
                label_smoothing=label_smoothing,
                enable_mal_loss=(stage > 1),
            )

        # vCLUB 互信息极小化约束
        vclub_mi_bound = compute_vclub_mi_bound(model, outputs)
        loss = loss + 0.02 * vclub_mi_bound

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    if hasattr(criterion, "mal_loss_fn"):
        criterion.mal_loss_fn.step_dual_variable("pre")

    params_to_clip = [p for group in optimizer.param_groups for p in group["params"]]
    torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
    scaler.step(optimizer)

    if hasattr(criterion, "mal_loss_fn"):
        criterion.mal_loss_fn.step_dual_variable("post")

    scaler.update()

    # Cache detached outputs for next batch's vCLUB learning loss
    prev_z_latent = outputs["z_latent"].detach()
    prev_concept_vector = outputs["concept_vector"].detach()

    return loss, loss_dict, prev_z_latent, prev_concept_vector
