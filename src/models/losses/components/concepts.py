import torch
import torch.nn.functional as F

from src.models.losses.concepts import (
    symmetric_label_regression_loss,
    symmetric_label_regression_loss_bce,
)


def resolve_mix_params(outputs, mix_info):
    """Resolve mixing parameters for mixup/cutmix augmentation."""
    batch_size = outputs["mask"].size(0)
    device = outputs["mask"].device

    if mix_info is not None:
        lam = mix_info["lam"]
        if isinstance(lam, float) or lam.dim() == 0:
            lam = torch.full((batch_size,), float(lam), device=device)
        index = mix_info["index"]
        mask_weight = mix_info.get("mask_weight", torch.ones(batch_size, device=device))
    else:
        lam = torch.ones(batch_size, device=device)
        index = torch.arange(batch_size, device=device)
        mask_weight = torch.ones(batch_size, device=device)

    return lam, index, mask_weight


def compute_concept_losses(
    outputs, targets, lam, index, mask_weight, label_smoothing, tirads_matrix
):
    """Compute TI-RADS concept SLR losses and clinical prior regularization.

    Returns dict with keys: comp, echo, shape, margin, foci, concepts_ce, prior
    """
    slr_alpha = 0.1
    slr_beta = 1.0

    loss_comp = lam * symmetric_label_regression_loss(
        outputs["composition"],
        targets["composition"],
        num_classes=4,
        alpha=slr_alpha,
        beta=slr_beta,
        label_smoothing=label_smoothing,
    ) + (1.0 - lam) * symmetric_label_regression_loss(
        outputs["composition"],
        targets["composition"][index],
        num_classes=4,
        alpha=slr_alpha,
        beta=slr_beta,
        label_smoothing=label_smoothing,
    )
    loss_echo = lam * symmetric_label_regression_loss(
        outputs["echogenicity"],
        targets["echogenicity"],
        num_classes=4,
        alpha=slr_alpha,
        beta=slr_beta,
        label_smoothing=label_smoothing,
    ) + (1.0 - lam) * symmetric_label_regression_loss(
        outputs["echogenicity"],
        targets["echogenicity"][index],
        num_classes=4,
        alpha=slr_alpha,
        beta=slr_beta,
        label_smoothing=label_smoothing,
    )
    loss_shape = lam * symmetric_label_regression_loss(
        outputs["shape"],
        targets["shape"],
        num_classes=2,
        alpha=slr_alpha,
        beta=slr_beta,
        label_smoothing=label_smoothing,
    ) + (1.0 - lam) * symmetric_label_regression_loss(
        outputs["shape"],
        targets["shape"][index],
        num_classes=2,
        alpha=slr_alpha,
        beta=slr_beta,
        label_smoothing=label_smoothing,
    )
    loss_margin = lam * symmetric_label_regression_loss(
        outputs["margin"],
        targets["margin"],
        num_classes=4,
        alpha=slr_alpha,
        beta=slr_beta,
        label_smoothing=label_smoothing,
    ) + (1.0 - lam) * symmetric_label_regression_loss(
        outputs["margin"],
        targets["margin"][index],
        num_classes=4,
        alpha=slr_alpha,
        beta=slr_beta,
        label_smoothing=label_smoothing,
    )
    loss_foci = lam * symmetric_label_regression_loss_bce(
        outputs["foci"],
        targets["foci"],
        alpha=slr_alpha,
        beta=slr_beta,
        label_smoothing=label_smoothing,
    ) + (1.0 - lam) * symmetric_label_regression_loss_bce(
        outputs["foci"],
        targets["foci"][index],
        alpha=slr_alpha,
        beta=slr_beta,
        label_smoothing=label_smoothing,
    )

    loss_concepts_ce = loss_comp + loss_echo + loss_shape + loss_margin + loss_foci

    # ACR TI-RADS clinical prior
    prob_comp = F.softmax(outputs["composition"], dim=1)
    prob_echo = F.softmax(outputs["echogenicity"], dim=1)
    prob_shape = F.softmax(outputs["shape"], dim=1)
    prob_margin = F.softmax(outputs["margin"], dim=1)
    prob_foci = torch.sigmoid(outputs["foci"])
    pred_concepts = torch.cat(
        [prob_comp, prob_echo, prob_shape, prob_margin, prob_foci], dim=1
    )

    gt_comp = F.one_hot(targets["composition"], num_classes=4).float()
    gt_echo = F.one_hot(targets["echogenicity"], num_classes=4).float()
    gt_shape = F.one_hot(targets["shape"], num_classes=2).float()
    gt_margin = F.one_hot(targets["margin"], num_classes=4).float()
    gt_foci = targets["foci"].float()
    gt_concepts = torch.cat([gt_comp, gt_echo, gt_shape, gt_margin, gt_foci], dim=1)

    gt_comp_perm = F.one_hot(targets["composition"][index], num_classes=4).float()
    gt_echo_perm = F.one_hot(targets["echogenicity"][index], num_classes=4).float()
    gt_shape_perm = F.one_hot(targets["shape"][index], num_classes=2).float()
    gt_margin_perm = F.one_hot(targets["margin"][index], num_classes=4).float()
    gt_foci_perm = targets["foci"][index].float()
    gt_concepts_perm = torch.cat(
        [gt_comp_perm, gt_echo_perm, gt_shape_perm, gt_margin_perm, gt_foci_perm], dim=1
    )

    score_pred = torch.mv(pred_concepts, tirads_matrix)
    score_gt = lam * torch.mv(gt_concepts, tirads_matrix) + (1.0 - lam) * torch.mv(
        gt_concepts_perm, tirads_matrix
    )
    loss_prior = F.huber_loss(score_pred, score_gt, delta=1.0)

    # ── Masked Task-Specific Loss (label validity) ──────────────────────────
    # When targets contains "label_validity", apply per-sample per-task masking
    # and use sum / (valid_count + eps) to avoid diluting the denominator with
    # masked-out (=0) samples. Falls back to original .mean() path if absent.
    _eps = 1e-8
    if "label_validity" in targets:
        lv = targets["label_validity"]  # dict of [B] float tensors

        def _masked_reduce(per_sample_loss, task):
            """sum(loss * validity * mask_weight) / (sum(validity * mask_weight) + eps)"""
            v = lv[task]
            w = per_sample_loss * v * mask_weight
            return w.sum() / ((v * mask_weight).sum() + _eps)

        loss_comp_r = _masked_reduce(loss_comp, "composition")
        loss_echo_r = _masked_reduce(loss_echo, "echogenicity")
        loss_shape_r = _masked_reduce(loss_shape, "shape")
        loss_margin_r = _masked_reduce(loss_margin, "margin")
        loss_foci_r = _masked_reduce(loss_foci, "foci")

        # concepts_ce: average of per-concept masked losses
        loss_concepts_ce_r = (
            loss_comp_r + loss_echo_r + loss_shape_r + loss_margin_r + loss_foci_r
        )

        # Prior: use composition validity as proxy (TI-RADS prior requires all
        # five concept labels; composition is the anchor concept in ACR TI-RADS
        # and also most commonly annotated). If composition is invalid the prior
        # score is unreliable, so we gate by composition validity.
        v_prior = lv["composition"]
        loss_prior_r = (loss_prior * v_prior * mask_weight).sum() / (
            (v_prior * mask_weight).sum() + _eps
        )

        return {
            "comp": loss_comp_r,
            "echo": loss_echo_r,
            "shape": loss_shape_r,
            "margin": loss_margin_r,
            "foci": loss_foci_r,
            "concepts_ce": loss_concepts_ce_r,
            "prior": loss_prior_r,
        }

    # ── Legacy path (no label_validity) — behaviour unchanged ───────────────
    loss_concepts_ce = loss_concepts_ce * mask_weight
    loss_prior = loss_prior * mask_weight

    return {
        "comp": loss_comp,
        "echo": loss_echo,
        "shape": loss_shape,
        "margin": loss_margin,
        "foci": loss_foci,
        "concepts_ce": loss_concepts_ce,
        "prior": loss_prior,
    }
