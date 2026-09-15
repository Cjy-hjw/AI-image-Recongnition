import torch
import torch.nn as nn

from src.models.famo import FAMOScheduler
from src.models.losses.segmentation import DiceLoss
from src.models.losses.classification import AUCMLossApprox, SupervisedContrastiveLoss
from src.models.losses.components import (
    compute_segmentation_loss,
    resolve_mix_params,
    compute_concept_losses,
    compute_malignancy_losses,
    apply_famo_weighting,
    compute_regularizers,
    compute_official_head_losses,
)


# 任务名 → loss_dict key 映射（多源异构标注的 task_mask 屏蔽路径参考）。
# 说明：per-sample 真屏蔽已由本模块 forward 的 valid_mask 注入 + components 内部
# label_validity 处理实现（无效样本不进入计算图，真正零梯度）；此表仅作为
# 「任务名对应哪些 loss 分项」的文档性常量，供上层构建 task_mask / 调试查阅。
# （原 masked_multitask.py 的后处理缩放 wrapper 已删除，其缩放法只改量级不屏蔽梯度。）
TASK_TO_LOSS_KEYS = {
    'mask': ['seg'],
    'composition': ['comp', 'concepts_ce', 'prior'],
    'echogenicity': ['echo', 'concepts_ce', 'prior'],
    'shape': ['shape', 'concepts_ce', 'prior'],
    'margin': ['margin', 'concepts_ce', 'prior'],
    'foci': ['foci', 'concepts_ce', 'prior'],
    'malignancy': ['malignancy', 'supcon'],
}


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss function.
    Balances segmentation BCE+Dice, TI-RADS concepts (via Symmetric Label Regression), and cost-sensitive malignancy classification
    using Homoscedastic Uncertainty Weighting and ACR TI-RADS prior regularization.
    """

    def __init__(self, num_tasks=5, alpha_prior=0.1, fn_weight=1.5,
                 official_head_config=None):
        super().__init__()
        self.alpha_prior = alpha_prior  # Clinical prior regularization factor
        self.mal_loss_fn = AUCMLossApprox(margin=1.0, imratio=0.33)
        self.famo = FAMOScheduler(num_tasks=num_tasks, beta=0.05, gamma=0.001)
        self.supcon_loss = SupervisedContrastiveLoss(temperature=0.07)

        self.bce_seg = nn.BCEWithLogitsLoss()
        self.dice_seg = DiceLoss()

        # ACR TI-RADS weights matrix
        tirads_weights = [
            0.0,
            0.0,
            1.0,
            2.0,  # Composition
            0.0,
            1.0,
            2.0,
            3.0,  # Echogenicity
            0.0,
            3.0,  # Shape
            0.0,
            0.0,
            2.0,
            3.0,  # Margin
            0.0,
            1.0,
            2.0,
            3.0,  # Foci
        ]
        self.register_buffer(
            "tirads_matrix", torch.tensor(tirads_weights, dtype=torch.float32)
        )
        self.register_buffer("lambda_latent", torch.tensor(0.01))
        self.delta_budget = 0.1

        # ── 官方新增维度头(位置/ETE/临床T/TI-RADS)的固定 λ 外挂配置 ─────────────
        # 由 `scripts/train_official.py` 从 config 组装后传入;为 None 时用默认值,
        # 且只要 targets 里没有对应头的键就整体跳过(legacy train.py 路径零回归)。
        # ETE 的 class_weights / log_prior 是 [C] 张量,须随模型 .to(device) 迁移,
        # 故走 register_buffer 而非普通属性(普通属性留在 CPU 会在损失里报设备不匹配)。
        cfg = dict(official_head_config or {})
        self.lambda_loc = float(cfg.get("loc_loss_weight", 1.0))
        self.lambda_ete = float(cfg.get("ete_loss_weight", 1.0))
        self.lambda_clin_t = float(cfg.get("clin_t_loss_weight", 1.0))
        self.lambda_tirads = float(cfg.get("tirads_loss_weight", 1.0))
        # Stage2 seg-only 微调(2026-09-13):Boundary/Lovász 权重透传到
        # compute_segmentation_loss。0(默认)= 老路径逐位一致,零回归。
        # 🔴 config 驱动 + `.get(k,0.0)` 缺键静默 0 = r7 事故同型的雷
        # (空微调被误读"损失无效"),故 train_official 的 --seg-only 启动门
        # 会打印配方横幅;权重在 config.train 下(seg_boundary_weight/
        # seg_lovasz_weight)。
        self.seg_boundary_weight = float(cfg.get("seg_boundary_weight", 0.0))
        self.seg_lovasz_weight = float(cfg.get("seg_lovasz_weight", 0.0))
        self.ete_focal_gamma = float(cfg.get("ete_focal_gamma", 2.0))
        self.ete_la_tau = float(cfg.get("ete_la_tau", 0.0))
        # backlog #128 LA-SoftQWK-OD:0(默认)= 走 CB-Focal 老路径,零回归。
        self.ete_qwk_weight = float(cfg.get("ete_qwk_weight", 0.0))
        self.ete_dist_weight = float(cfg.get("ete_dist_weight", 0.2))
        self.clin_t_label_smoothing = float(cfg.get("clin_t_label_smoothing", 0.0))
        self.tirads_label_smoothing = float(cfg.get("tirads_label_smoothing", 0.0))
        # 默认 0(关闭):见 `losses/components/tirads.py` docstring —— 没有真实数据
        # 无法调参,先证后改;权重为 0 时对 loss 精确无贡献。
        self.tirads_ordinal_weight = float(cfg.get("tirads_ordinal_weight", 0.0))
        self.register_buffer("ete_class_weights", cfg.get("ete_class_weights"))
        self.register_buffer("ete_log_prior", cfg.get("ete_log_prior"))
        self.register_buffer("clin_t_class_weights", cfg.get("clin_t_class_weights"))
        self.register_buffer("tirads_class_weights", cfg.get("tirads_class_weights"))

    def forward(
        self,
        outputs,
        targets,
        slr_weights=None,
        mix_info=None,
        update_famo_weights=True,
        spin_l1_weight=0.0,
        label_smoothing=0.0,
        enable_mal_loss=True,
        task_mask=None,
    ):
        """
        Args:
            task_mask: dict of binary masks (B,) indicating label availability
                {
                    'mask': Tensor[B],       # 1 if has segmentation mask, 0 otherwise
                    'composition': Tensor[B],
                    'echogenicity': Tensor[B],
                    'shape': Tensor[B],
                    'margin': Tensor[B],
                    'foci': Tensor[B],
                    'malignancy': Tensor[B],
                }
                If None, assumes all labels available (backward compat)

                Note: Segmentation already supports valid_mask via targets['valid_mask'],
                      so we pass task_mask['mask'] -> targets['valid_mask']
        """
        # 注入task_mask到targets（利用现有valid_mask支持）
        if task_mask is not None:
            if 'mask' in task_mask and 'valid_mask' not in targets:
                targets = {**targets, 'valid_mask': task_mask['mask'].float()}

        # 1. Segmentation Loss (自动应用valid_mask;Boundary/Lovász 可选外挂)
        loss_seg, seg_extra = compute_segmentation_loss(
            self.bce_seg, self.dice_seg, outputs, targets,
            boundary_weight=self.seg_boundary_weight,
            lovasz_weight=self.seg_lovasz_weight,
        )

        # 2. Concept losses
        lam, index, mask_weight = resolve_mix_params(outputs, mix_info)
        concept_losses = compute_concept_losses(
            outputs,
            targets,
            lam,
            index,
            mask_weight,
            label_smoothing,
            self.tirads_matrix,
        )

        # Store for external access
        self.loss_concepts_ce = concept_losses["concepts_ce"].mean()
        self.loss_prior = concept_losses["prior"].mean()

        # 3. Malignancy losses
        loss_mal, loss_supcon = compute_malignancy_losses(
            self.mal_loss_fn,
            self.supcon_loss,
            outputs,
            targets,
            mask_weight,
            slr_weights,
        )

        if not enable_mal_loss:
            loss_mal = loss_mal * 0.0
            loss_supcon = loss_supcon * 0.0

        # 4. FAMO 5-channel dynamic loss weighting
        total_loss = apply_famo_weighting(
            self.famo,
            loss_seg,
            self.loss_concepts_ce,
            self.loss_prior,
            loss_mal,
            loss_supcon,
            update_famo_weights,
            self.training,
        )

        # 4.5 官方三头(位置/ETE/临床T)固定 λ 外挂 —— **不进 FAMO**,理由见
        # `components/official_heads.py` docstring(checkpoint 兼容 + masked 0 值
        # 会被 FAMO 误读为"已收敛")。插在 FAMO 之后、正则之前:λ 项是任务损失,
        # 不该被 L2/spin-L1 的正则语义裹挟。targets 无三头键 → 整体跳过。
        head_total, head_detail = compute_official_head_losses(
            outputs,
            targets,
            lambda_loc=self.lambda_loc,
            lambda_ete=self.lambda_ete,
            lambda_clin_t=self.lambda_clin_t,
            lambda_tirads=self.lambda_tirads,
            ete_class_weights=self.ete_class_weights,
            ete_log_prior=self.ete_log_prior,
            ete_focal_gamma=self.ete_focal_gamma,
            ete_la_tau=self.ete_la_tau,
            ete_qwk_weight=self.ete_qwk_weight,
            ete_dist_weight=self.ete_dist_weight,
            clin_t_class_weights=self.clin_t_class_weights,
            clin_t_label_smoothing=self.clin_t_label_smoothing,
            tirads_class_weights=self.tirads_class_weights,
            tirads_label_smoothing=self.tirads_label_smoothing,
            tirads_ordinal_weight=self.tirads_ordinal_weight,
        )
        total_loss = total_loss + head_total

        # 5. Regularization
        total_loss, loss_l2 = compute_regularizers(
            total_loss,
            outputs,
            self.lambda_latent,
            self.delta_budget,
            spin_l1_weight,
        )

        loss_dict = {
            "total": total_loss,
            "seg": loss_seg,
            "concepts_ce": self.loss_concepts_ce,
            "prior": self.loss_prior,
            "comp": concept_losses["comp"].mean(),
            "echo": concept_losses["echo"].mean(),
            "shape": concept_losses["shape"].mean(),
            "margin": concept_losses["margin"].mean(),
            "foci": concept_losses["foci"].mean(),
            "malignancy": loss_mal,
            "supcon": loss_supcon,
            "latent_l2": loss_l2,
            "spin_l1": outputs["spin_l1"],
        }
        # 三头监控项(未乘 λ 的原始量级)。targets 无三头键时 head_detail 为空,
        # loss_dict 与 legacy 完全一致。
        loss_dict.update(head_detail)
        # seg 分项监控(Gate2 F3,2026-09-14):seg_boundary/seg_lovasz 的
        # 实际量级必须进 loss_dict → JSONL——config 写着"先观察量级再定权重",
        # 合并进 "seg" 后操作者无从观察;分项恒写出(权重=0 时为 0.0),
        # 接线被删时 JSONL 立刻可见(不自证)。
        loss_dict.update(seg_extra)

        return total_loss, loss_dict
