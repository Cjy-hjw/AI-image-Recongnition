import torch


def compute_malignancy_losses(
    mal_loss_fn, supcon_loss_fn, outputs, targets, mask_weight, slr_weights
):
    """Compute malignancy AUCM loss and supervised contrastive loss.

    When targets contains "label_validity", masks out invalid samples before
    feeding into the AUCM loss function (which is inherently batch-level, not
    per-sample). Only valid samples (validity=1) contribute to AUCM.

    supcon loss is NOT masked per-sample: supervised contrastive loss operates
    on the full batch embedding structure; masking individual anchors would
    collapse the contrastive pairs. If the entire batch has no valid malignancy
    labels, supcon is skipped (returns 0).

    Backward compatibility: when "label_validity" is absent, behaviour is
    identical to the original code path.
    """
    mal_logits = outputs["malignancy"]
    mal_targets = targets["malignancy"]

    # ── Masked path (label validity) ─────────────────────────────────────────
    if "label_validity" in targets:
        v_mal = targets["label_validity"]["malignancy"]  # [B] float tensor
        valid_idx = v_mal > 0.5  # boolean mask [B]

        if valid_idx.sum() < 0.5:
            # No valid malignancy labels in this batch → zero loss, no gradient
            loss_mal = torch.zeros((), device=mal_logits.device, requires_grad=False)
            loss_supcon = torch.zeros((), device=mal_logits.device, requires_grad=False)
        else:
            # Compute AUCM only on valid samples
            valid_logits = mal_logits[valid_idx]
            valid_targets = mal_targets[valid_idx]
            valid_mw = mask_weight[valid_idx]

            loss_mal = mal_loss_fn(valid_logits, valid_targets) * valid_mw
            if slr_weights is not None:
                loss_mal = loss_mal * slr_weights[valid_idx]
            # AUCM returns scalar; mask_weight broadcast → [n_valid]; reduce
            loss_mal = loss_mal.mean()

            loss_supcon = supcon_loss_fn(outputs["z_contrastive"], mal_targets)
            if loss_supcon.dim() > 0:
                loss_supcon = loss_supcon.mean()
    else:
        # ── Legacy path — behaviour unchanged ────────────────────────────────
        loss_mal = mal_loss_fn(mal_logits, mal_targets) * mask_weight
        if slr_weights is not None:
            loss_mal = loss_mal * slr_weights
        loss_mal = loss_mal.mean()
        loss_supcon = supcon_loss_fn(outputs["z_contrastive"], mal_targets)
        if loss_supcon.dim() > 0:
            loss_supcon = loss_supcon.mean()

    if torch.isnan(loss_mal):
        print(
            f"  [NaN Guard] loss_mal is NaN! mal_logits range=[{mal_logits.min().item():.4f}, {mal_logits.max().item():.4f}]",
            flush=True,
        )
    if torch.isnan(loss_supcon):
        print("  [NaN Guard] loss_supcon is NaN!", flush=True)

    return loss_mal, loss_supcon
