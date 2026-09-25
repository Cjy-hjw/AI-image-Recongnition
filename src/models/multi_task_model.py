from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.models.blocks import ConvBlock, TaskAlignmentAdaptor
from src.models.mask_features import nodule_scale_from_mask, physical_diameter_from_mask
from src.models.physical_prior import DifferentiablePhysicalPriorExtractor
from src.models.cbm_head import ConceptEmbeddingCBMHead, VariationalCLUBMean
from src.models.heads import ClinicalTHead, ExtensionHead, LocationHead, TiradsHead
from src.models.pretrained_loader import load_local_safetensors
from src.models.roi_heads import (
    RoiClinicalTHead,
    RoiConceptHead,
    RoiExtensionHead,
    RoiLocationHead,
    RoiTiradsHead,
)


class ModelOutputs(TypedDict):
    """Single source of truth for MultiTaskThyroidModel.forward() output keys.

    Downstream consumers (train_step, validate_step, vclub, losses, evaluate,
    predict, postprocess, export) all index these keys; renaming any key here
    flags every consumer site.
    """

    mask: torch.Tensor
    composition: torch.Tensor
    echogenicity: torch.Tensor
    shape: torch.Tensor
    margin: torch.Tensor
    foci: torch.Tensor
    malignancy: torch.Tensor
    concept_vector: torch.Tensor
    latent_res: torch.Tensor
    z_latent: torch.Tensor
    z_contrastive: torch.Tensor
    physical_vector: torch.Tensor
    spin_l1: torch.Tensor
    # --- Phase 2 三头(位置/侵犯/临床T)接口 ---
    # 命名即契约:带 `_detached` 后缀的键**已经 detach**,消费方拿到即无梯度。
    stage4_spatial: torch.Tensor
    soft_mask_detached: torch.Tensor
    nodule_scale_detached: torch.Tensor
    # --- Phase 2.4 三头输出 ---
    # 训练走 `*_logits`(原始未校正),交付/评估走 `*_probs`(fp32 softmax,和为 1)。
    location_side_logits: torch.Tensor
    location_level_logits: torch.Tensor
    location_probs: torch.Tensor
    ete_logits: torch.Tensor
    ete_probs: torch.Tensor
    ete_band_ratio: torch.Tensor
    t_logits: torch.Tensor
    t_probs: torch.Tensor
    t_conf: torch.Tensor
    # --- backlog #124:TI-RADS 总分类头输出 ---
    tirads_logits: torch.Tensor
    tirads_probs: torch.Tensor
    # --- 深监督辅助 logits(plan 任务D 首攻扩展,2026-09-20)---
    # 与 "mask" 同 shape [B,1,H,W](forward 内已上采样);训练侧
    # compute_segmentation_loss 按 seg_deepsup_weight 消费,推理侧不读。
    seg_aux2: torch.Tensor
    seg_aux3: torch.Tensor


class MultiTaskThyroidModel(nn.Module):
    """
    基于双分支特征解耦与可微物理先验注入的2D甲状腺超声多任务一体化诊断系统
    """

    CLS_STAGE3_CHANNELS = 88
    CLS_STAGE4_CHANNELS = 208

    STAGE1_FROZEN_CBM_SUBMODULES = ("classifier_direct", "latent_mlp")
    # 官方头前缀(位置/ETE/临床T/TI-RADS,四头):legacy 三阶段调度(stages.py)一律
    # 不训,由 schedule_official.py 接管。加载旧 checkpoint 时 missing_keys 的
    # 合法白名单也是这个集合(新头名一旦加进这里,warm-start/冻结/可训集自动生效,
    # 无需再改 `load_warm_start`/`schedule_official.py` —— 两处都动态读本元组)。
    # 🔴 深监督辅助头(2026-09-20)**有意不进本元组**:进这里 = warm-start 加载
    # r5/ep16 时 missing_keys 合法化 + 调度自动放行,但辅助头是"训练期梯度管道"
    # 而非"计分头",它的 missing 应走 load_warm_start 的非头键 fail-loud 分支
    # (旧 checkpoint 缺 aux 键 = 结构不匹配,静默随机初始化的辅助头会往解码器
    # 中层注入噪声梯度)。实际放行在 train_official.load_warm_start 的 aux 白
    # 名单(无条件放行 + n_aux_missing 分口径日志:旧权重热启动深监督实验是
    # 标准路径);推理侧 strict 校验天然拦截旧权重(🔴 代码/权重成对耦合:
    # cloud_infer 树的模型代码更新后,回退旧权重必须连 src 一起回退)。
    OFFICIAL_HEAD_PREFIXES = (
        "location_head", "extension_head", "clinical_t_head", "tirads_head",
    )
    # 深监督辅助头前缀(训练侧放行/白名单用;与 OFFICIAL_HEAD_PREFIXES 分离,
    # 理由见上)。
    SEG_AUX_PREFIXES = ("seg_aux_head_2", "seg_aux_head_3")
    STAGE3_FROZEN_PREFIXES = (
        "seg_backbone",
        "cls_backbone",
        "up_stages",
        "dec_stages",
        "seg_head",
        "prior_extractor",
        "prior_mlp",
        "supcon_projector",
        "adaptor_stage3",
        "adaptor_stage4",
    )

    def __init__(
        self,
        backbone_name="efficientnet_b2",
        pretrained=True,
        config=None,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()

        self.img_size = 512
        if config is not None and "data" in config:
            # int() 强转(A路审查):yaml 里 img_size 误写成引号字符串 "640" 时,
            # 不强转会让 torch.arange(str) 在物理先验处炸出 TypeError,报错
            # 指向 physical_prior 而非 config——错误定位误导。强转后与
            # verify/runner.py 的消费语义一致(单侧强转的不对称已消除)。
            self.img_size = int(config["data"].get("img_size", 512))

        self.current_epoch = 120

        self.seg_backbone = timm.create_model(
            "convnext_nano",
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=1,
        )
        if pretrained:
            seg_pretrained_path = "checkpoints/convnext_nano_pretrained.safetensors"
            loaded = load_local_safetensors(
                self.seg_backbone,
                seg_pretrained_path,
                in_chans=1,
                stem_key="stem.0.weight",
            )
            if not loaded:
                print(
                    "--> Local ConvNeXt-Nano weights not found; falling back to timm online pretrained."
                )
                online_seg = timm.create_model(
                    "convnext_nano",
                    pretrained=True,
                    features_only=True,
                    out_indices=(0, 1, 2, 3),
                    in_chans=1,
                )
                self.seg_backbone.load_state_dict(online_seg.state_dict(), strict=False)

        self.cls_backbone = timm.create_model(
            backbone_name, pretrained=False, features_only=False, in_chans=1
        )
        if backbone_name == "efficientnet_b2" and pretrained:
            load_local_safetensors(
                self.cls_backbone,
                "checkpoints/efficientnet_b2_pretrained.safetensors",
                stem_key="conv_stem.weight",
            )

        # LoRA 注入：在预训练权重加载完成后，替换目标 1x1 Conv2d 层
        if use_lora:
            from src.models.lora_utils import apply_lora
            n_cls = apply_lora(self.cls_backbone, r=lora_r, alpha=lora_alpha)
            n_seg = apply_lora(self.seg_backbone, r=lora_r, alpha=lora_alpha)
            print(f"--> LoRA 注入: cls_backbone {n_cls} 层, seg_backbone {n_seg} 层 (r={lora_r}, alpha={lora_alpha})")
            # 冻结所有参数（含 LoRA A/B）：模型以"LoRA 未激活"状态构建，
            # 训练脚本在 FT 阶段显式解冻 lora_A/lora_B（见 train_thyroidxl.py）。
            # 若不冻结 LoRA 参数，LP 阶段它们不在 optimizer 内、梯度永不清零，
            # 会跨 batch 累积直至数值溢出（§1.1 教训：LP/FT 参数集合必须与 requires_grad 严格对齐）。
            for p in self.cls_backbone.parameters():
                p.requires_grad = False
            for p in self.seg_backbone.parameters():
                p.requires_grad = False

        seg_channels = list(self.seg_backbone.feature_info.channels())
        self.seg_stage_channels = seg_channels
        self.cls_stage3_channels = self.CLS_STAGE3_CHANNELS
        self.cls_stage4_channels = self.CLS_STAGE4_CHANNELS
        self.adaptor_stage3 = TaskAlignmentAdaptor(
            seg_channels[1], self.cls_stage3_channels
        )
        self.adaptor_stage4 = TaskAlignmentAdaptor(
            seg_channels[2], self.cls_stage4_channels
        )

        self.up_stages = nn.ModuleList(
            [
                nn.ConvTranspose2d(seg_channels[3], 320, kernel_size=2, stride=2),
                nn.ConvTranspose2d(320, 160, kernel_size=2, stride=2),
                nn.ConvTranspose2d(160, 80, kernel_size=2, stride=2),
                nn.ConvTranspose2d(80, 40, kernel_size=2, stride=2),
                nn.ConvTranspose2d(40, 20, kernel_size=2, stride=2),
            ]
        )
        self.dec_stages = nn.ModuleList(
            [
                ConvBlock(320 + seg_channels[2], 320),
                ConvBlock(160 + seg_channels[1], 160),
                ConvBlock(80 + seg_channels[0], 80),
                ConvBlock(40, 40),
                ConvBlock(20, 20),
            ]
        )
        self.seg_head = nn.Conv2d(20, 1, kernel_size=1)

        # ── 多尺度深监督辅助头(plan 任务D 首攻扩展,2026-09-20)──────────────────
        # DR §3.1 首攻方案第三组件:对解码器 stage2/stage3 输出挂 1×1 辅助分割头,
        # 辅助 logits 上采样到全分辨率后与主头同式损失(BCE+Dice per-sample),
        # 权重 config.train.seg_deepsup_weight 控制(默认 0 = 零回归)。
        # 🔴 设计约束(Fable 终审机制跳跃禁令的反面——本组件只加强现有语义
        # logits 的中浅层监督,不换范式不换接口):
        # - 辅助头是**纯推理旁路**:forward 恒算(结构稳定性,防"训练/推理前向
        #   不一致"类 bug),但推理侧消费端(case_pipeline/v21_output)只读
        #   outputs["mask"],aux 键多出无影响;损失侧 weight=0 时逐位零贡献。
        # - 通道数动态取 dec_stages 输出(stage2→40ch@128²、stage3→20ch@256²,
        #   512 输入下),不硬编码——分辨率改 640 时 spatial 尺寸随动。
        # - 命名前缀 "seg_aux_head_*" 加入 SEG_TRAIN_DECODER 语义域(训练侧
        #   requires_grad 由调度按前缀放行,见 schedule_official.py 同轮改动)。
        self.seg_aux_head_2 = nn.Conv2d(40, 1, kernel_size=1)   # dec_stages[3] 出
        self.seg_aux_head_3 = nn.Conv2d(20, 1, kernel_size=1)   # dec_stages[4] 出


        self.prior_extractor = DifferentiablePhysicalPriorExtractor(
            img_size=self.img_size
        )
        self.prior_mlp = nn.Sequential(
            nn.Linear(3, 16), nn.LayerNorm(16), nn.ReLU(inplace=True), nn.Linear(16, 16)
        )

        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        # 分类分支 pooled 特征维度(EfficientNet-B2 最后 block 输出通道)与概念总维,
        # cbm_head / vclub / supcon_projector 三者必须共用同一 in_features,否则静默错配。
        self.CLS_POOLED_DIM = 352
        self.NUM_CONCEPTS = 18
        self.cbm_head = ConceptEmbeddingCBMHead(
            in_features=self.CLS_POOLED_DIM, num_concepts=self.NUM_CONCEPTS
        )
        self.vclub = VariationalCLUBMean(
            x_dim=self.CLS_POOLED_DIM, y_dim=self.NUM_CONCEPTS, hidden_size=64
        )
        self.supcon_projector = nn.Sequential(
            nn.Linear(self.CLS_POOLED_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

        # ── 官方新增维度头(位置/ETE/临床T/TI-RADS)────────────────────────────
        # in_channels 一律取 CLS_POOLED_DIM(352):四头都吃 blocks[6] 之后的
        # stage4 空间图,208 是 adaptor_stage4 的**输入**通道,写错要到真训练才崩。
        self.location_head = LocationHead(in_channels=self.CLS_POOLED_DIM)
        self.extension_head = ExtensionHead(in_channels=self.CLS_POOLED_DIM)
        self.clinical_t_head = ClinicalTHead(in_channels=self.CLS_POOLED_DIM)
        self.tirads_head = TiradsHead(in_channels=self.CLS_POOLED_DIM)

        # ── ROI 范式波3·模型面(2026-09-25,两阶段 ROI 范式第二阶段)─────────
        # ROI 头与整图头并存:整图 forward 一律不触 ROI 头(零回归铁律);
        # ROI 训练/推理只走 `forward_roi`。loc 头结构同构 LocationHead
        # (state_dict 可互载,零训练搬运载体);ete/tirads 新结构;clin_t 同构
        # (scale 语义 [d_mm, log d_mm])。ROI 头的调度/冻结由 schedule_official
        # 按 `ROI_HEAD_PREFIXES` 前缀接管(与 OFFICIAL_HEAD_PREFIXES 同语义)。
        self.roi_location_head = RoiLocationHead(in_channels=self.CLS_POOLED_DIM)
        self.roi_extension_head = RoiExtensionHead(in_channels=self.CLS_POOLED_DIM)
        self.roi_clinical_t_head = RoiClinicalTHead(in_channels=self.CLS_POOLED_DIM)
        self.roi_tirads_head = RoiTiradsHead(in_channels=self.CLS_POOLED_DIM)
        self.roi_concept_head = RoiConceptHead(in_channels=self.CLS_POOLED_DIM)

    # ROI 头前缀(调度/冻结/白名单消费;与 OFFICIAL_HEAD_PREFIXES 同语义)。
    # 🔴 **不进 OFFICIAL_HEAD_PREFIXES**:旧 checkpoint 无这些键,加进那里会让
    # `load_warm_start` 把 missing 合法化——但 ROI 头缺键 = 随机初始化是**预期
    # 行为**(冷启动新头),合法化点在 train_official 的 ROI 白名单(训练侧
    # 挂),不在模型常量。推理侧 build_model 的 strict 校验天然拦截"旧权重+
    # ROI 头"组合(代码/权重成对耦合红线,同 aux 头的 Z1 教训)。
    ROI_HEAD_PREFIXES = (
        "roi_location_head", "roi_extension_head",
        "roi_clinical_t_head", "roi_tirads_head", "roi_concept_head",
    )

    def is_param_trainable(self, name: str, stage: int) -> bool:
        if "vclub" in name:
            return True
        # 三头只由 `schedule_official.py` 的 A/B/C 调度训练,legacy 三阶段
        # (stages.py 的 1-40/41-100/101-120)一律冻结——那套调度的 Stage2 起点
        # 有 lr 100 倍跳变的已知崩塌缺陷,不得让三头搭上这条线。
        if any(name.startswith(pfx) for pfx in self.OFFICIAL_HEAD_PREFIXES):
            return False
        if stage == 1:
            return not any(sub in name for sub in self.STAGE1_FROZEN_CBM_SUBMODULES)
        if stage == 2:
            return True
        return not any(name.startswith(pfx) for pfx in self.STAGE3_FROZEN_PREFIXES)

    def set_training_stage(self, stage: int) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = self.is_param_trainable(name, stage)

    def _forward_cls_backbone(
        self, x: torch.Tensor, seg_features: list
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EfficientNet-B2 classification branch with task-alignment injection.

        Returns:
            pooled:  [B, CLS_POOLED_DIM]  —— 全局平均池化后的向量(既有契约)
            spatial: [B, CLS_POOLED_DIM, H/32, W/32] —— **pool 前**的空间特征图,
                尺寸跟随 config img_size(512²→[B,352,16,16],640²→[B,352,20,20],
                2026-09-11 起默认 640)。位置头需要空间结构(CoordConv 要
                坐标网格),pool 后拿不到。

        ⚠️ `spatial` **有意不 detach**:位置头梯度回流 `cls_backbone` 是设计意图
        (调度阶段 B 要 LLRD 解冻 Top-2 层)。这不会污染 `seg_backbone` —— 下方
        `adaptor_stage3/4` 两个调用点已对 `seg_features` 各自 `.detach()`,
        `gate8_adaptor_isolation` 的精确零断言前提不受影响。
        """
        B = x.shape[0]
        x_cls = self.cls_backbone.conv_stem(x)
        x_cls = self.cls_backbone.bn1(x_cls)
        x_cls = self.cls_backbone.blocks[0](x_cls)
        x_cls = self.cls_backbone.blocks[1](x_cls)
        x_cls = self.cls_backbone.blocks[2](x_cls)
        x_cls = self.cls_backbone.blocks[3](x_cls)
        x_cls = self.adaptor_stage3(seg_features[1].detach(), x_cls)
        x_cls = self.cls_backbone.blocks[4](x_cls)
        x_cls = self.cls_backbone.blocks[5](x_cls)
        x_cls = self.adaptor_stage4(seg_features[2].detach(), x_cls)
        # blocks[6] 把 CLS_STAGE4_CHANNELS(208,adaptor_stage4 的输出=blocks[6] 的输入)
        # 映射到 CLS_POOLED_DIM(352)。位置头第一层 in_channels 必须取 352,不是 208。
        x_cls = self.cls_backbone.blocks[6](x_cls)
        return self.cls_pool(x_cls).view(B, -1), x_cls

    def forward(self, x, gamma_prior=1.0) -> ModelOutputs:
        B, C, H, W = x.shape
        assert (
            H == self.img_size and W == self.img_size
        ), f"Input shape mismatch: expected {self.img_size}x{self.img_size}, got {H}x{W}"

        seg_features = self.seg_backbone(x)
        d = self.up_stages[0](seg_features[-1])
        d = self.dec_stages[0](torch.cat([d, seg_features[2]], dim=1))
        d = self.up_stages[1](d)
        d = self.dec_stages[1](torch.cat([d, seg_features[1]], dim=1))
        d = self.up_stages[2](d)
        d = self.dec_stages[2](torch.cat([d, seg_features[0]], dim=1))
        d = self.up_stages[3](d)
        d2 = self.dec_stages[3](d)          # 40ch @ H/4 —— 深监督 aux_2 挂点
        d = self.up_stages[4](d2)
        d3 = self.dec_stages[4](d)          # 20ch @ H/2 —— 深监督 aux_3 挂点
        logits_mask = self.seg_head(d3)
        soft_mask = torch.sigmoid(logits_mask)

        # 深监督辅助 logits:1×1 头后双线性上采样到全分辨率(与主头同 shape)。
        # 恒算(推理旁路):结构稳定优先——训练/推理前向不一致是比算力更贵的 bug。
        # aux 权重=0 时 loss 侧零消费,推理侧无人读这两个键。
        aux2_logits = self.seg_aux_head_2(d2)
        aux3_logits = self.seg_aux_head_3(d3)
        if aux2_logits.shape[-2:] != logits_mask.shape[-2:]:
            aux2_logits = F.interpolate(
                aux2_logits, size=logits_mask.shape[-2:], mode="bilinear",
                align_corners=False)
        if aux3_logits.shape[-2:] != logits_mask.shape[-2:]:
            aux3_logits = F.interpolate(
                aux3_logits, size=logits_mask.shape[-2:], mode="bilinear",
                align_corners=False)


        v_phys = self.prior_extractor(soft_mask, x)
        e_prior = self.prior_mlp(v_phys) * gamma_prior

        x_cls, x_cls_spatial = self._forward_cls_backbone(x, seg_features)
        assert x_cls.shape[1] == self.CLS_POOLED_DIM, (
            f"cls pooled dim mismatch: expected {self.CLS_POOLED_DIM}, "
            f"got {x_cls.shape[1]} — cbm_head/vclub/supcon in_features will misalign"
        )
        # 不再断言 spatial 的通道数:`cls_pool` 池到 1×1 后 view,pooled.shape[1] 与
        # spatial.shape[1] 恒等,重复断言零区分力(位置头 in_channels 取 352 这件事由
        # `tests/test_head_interfaces.py` 的 352≠208 锚点保护)。这里断 spatial 真正
        # 独有的性质:**保留了空间维** —— 若有人把返回值误写成两个 pooled,dim 会退化成 2。
        assert x_cls_spatial.dim() == 4, (
            f"stage4_spatial 必须保留空间维 [B,C,H,W],got {tuple(x_cls_spatial.shape)}"
            " —— 位置头的 CoordConv 需要空间网格"
        )

        cbm = self.cbm_head(x_cls, e_prior)
        z_contrastive = self.supcon_projector(x_cls)

        # ── 三头串接 ────────────────────────────────────────────────────────
        # 依赖顺序:ETE 的 logits 是 T 头的输入之一(T 头内部 detach + softmax,
        # T 的监督不得回流重塑 ETE 表征——ETE 是官方独立计分维度)。
        soft_mask_detached = soft_mask.detach()
        loc_out = self.location_head(x_cls_spatial)
        ete_out = self.extension_head(
            x_cls_spatial, soft_mask_detached, cbm.logits_margin
        )
        nodule_scale_detached = nodule_scale_from_mask(soft_mask)
        t_out = self.clinical_t_head(
            x_cls_spatial, nodule_scale_detached, ete_out["ete_logits"]
        )
        tirads_out = self.tirads_head(
            x_cls_spatial,
            cbm.logits_comp, cbm.logits_echo, cbm.logits_shape,
            cbm.logits_margin, cbm.logits_foci,
        )

        outputs: ModelOutputs = {
            "mask": logits_mask,
            "composition": cbm.logits_comp,
            "echogenicity": cbm.logits_echo,
            "shape": cbm.logits_shape,
            "margin": cbm.logits_margin,
            "foci": cbm.logits_foci,
            "malignancy": cbm.logits_malignancy,
            "concept_vector": cbm.concept_vector,
            "latent_res": cbm.latent_res,
            "z_latent": x_cls,
            "z_contrastive": z_contrastive,
            "physical_vector": v_phys,
            "spin_l1": self.cbm_head.get_l1_penalty(),
            # --- Phase 2 三头接口 ---
            # `soft_mask` 原通路(→ prior_extractor)**有意不 detach**,见 stages.py:27。
            # 这里另发一份 detach 副本给三头,原物理先验通路不受影响。
            "stage4_spatial": x_cls_spatial,
            "soft_mask_detached": soft_mask_detached,
            "nodule_scale_detached": nodule_scale_detached,
            # --- Phase 2.4 三头输出 ---
            "location_side_logits": loc_out["side_logits"],
            "location_level_logits": loc_out["level_logits"],
            "location_probs": loc_out["location_probs"],
            "ete_logits": ete_out["ete_logits"],
            "ete_probs": ete_out["ete_probs"],
            "ete_band_ratio": ete_out["band_ratio"],
            "t_logits": t_out["t_logits"],
            "t_probs": t_out["t_probs"],
            "t_conf": t_out["t_conf"],
            # --- backlog #124:TI-RADS 头输出 ---
            "tirads_logits": tirads_out["tirads_logits"],
            "tirads_probs": tirads_out["tirads_probs"],
            # --- 深监督辅助 logits(plan 任务D,2026-09-20)---
            # 已上采样到与 "mask" 同 shape [B,1,H,W];训练侧由
            # compute_segmentation_loss 按权重消费,推理侧无人读。
            "seg_aux2": aux2_logits,
            "seg_aux3": aux3_logits,
        }

        return outputs

    # ── ROI 范式波3·模型面:第二前向入口(2026-09-25)────────────────────────
    def forward_roi(self, roi_images, roi_meta, seg_features=None):
        """ROI 级前向第二入口 —— 复用**同一 backbone/头实例**,输出键对齐
        `collect_roi_records` 的消费形态(每病灶一条 dict)。

        🔴 零回归铁律:本方法**新增**,不改 `forward` 任何一行;整图路径
        的 outputs 不含 `roi_outputs` 键(分流判据 = 键存在性,
        `validate_step.collect_official_records` 同款约定)。

        Args:
            roi_images: [K,1,S,S] ROI 图像 batch(RoiDataset/roi_collate 产出,
                K = 病灶数;整图训练循环外的独立 ROI batch)。
            roi_meta: per-lesion 元数据 list[dict](长度 K),每条含
                `bbox_xyxy`/`orig_w`/`orig_h`/`spacing_mm`/`roi_mask`(
                [1,H',W'] ROI 域 soft/二值 mask,ETE band 与 d_mm 消费;
                训练侧由 collate 侧装,推理侧由 ROI pipeline 装)/
                `lesion_id`/`patient_id`/`is_main`。
                ⚠️ 这里吃 **list[dict]** 而非 batch tensor dict——collate 后的
                roi_meta tensor 化键(bbox/label)由调用方解包;单一职责,
                本方法不做 collate 逆操作。
            seg_features: None 或分割侧特征(见 `roi_encoder.forward_roi_features`;
                推理 ROI pipeline 传"分割器在同 ROI 上的特征",训练数据面
                单 ROI 路传 None 由 encoder 现算+detach)。

        Returns:
            dict:
              - `"roi_outputs"`: list[dict](长度 K),每条 =
                {lesion_id, patient_id, is_main, malignancy [2],
                location_probs [7], ete_probs [4], ete_logits [4],
                t_conf 标量, tirads_probs [5]} —— 逐键对齐
                `collect_roi_records` 的 roi_outputs 消费契约。
              - `"roi_pooled"` [K,352] / `"roi_spatial"` [K,352,h,w]:
                backbone ROI 特征(下游损失/诊断可复用)。
              - `"roi_concepts"`: 概念头五键 dict(K 维)。
        """
        from src.models.roi_encoder import forward_roi_features

        K = int(roi_images.shape[0])
        feats = forward_roi_features(self, roi_images, seg_features)
        spatial, pooled = feats["spatial"], feats["pooled"]

        # ROI 域 mask:各条 meta 自带 [1,H',W'](H' 不必统一——ETE band/d_mm 都
        # 是 per-ROI 消费,mask 与 ROI 图像同栅格时物理语义最准)。缺失 →
        # 全零(阴性/无 mask 语义,band=0 → ETE 局部特征全 0、d_mm=0,与整图头
        # 阴性语义一致)。**不 stack**(H' 可不同;逐条消费,零广播坑)。
        H = int(roi_images.shape[-1])
        masks = []
        for i in range(K):
            m = roi_meta[i].get("roi_mask")
            if m is None:
                m = torch.zeros(1, H, H, dtype=torch.float32,
                                device=roi_images.device)
            masks.append(m.detach().float())
        # ETE 头吃 [K,1,H,W] 堆叠:缺省补到统一 H(有 mask 的条保持原栅格,
        # 无 mask 的条补零图)。各条 H' 不齐时以**第一条**为准 pad/crop——
        # ETE band 内部 adaptive_avg_pool 到特征分辨率,栅格差异只影响 band
        # 分辨率不影响语义;pad 用零(带外区域,零=无证据)。
        if K:
            target_hw = masks[0].shape[-2:]
            stackable = []
            for m in masks:
                if m.shape[-2:] != target_hw:
                    mm = torch.zeros(1, *target_hw, dtype=m.dtype, device=m.device)
                    hh, ww = min(m.shape[-2], target_hw[0]), min(m.shape[-1], target_hw[1])
                    mm[:, :hh, :ww] = m[:, :hh, :ww]
                    m = mm
                stackable.append(m)
            roi_masks = torch.stack(stackable, dim=0)
        else:
            roi_masks = torch.zeros(
                0, 1, H, H, dtype=torch.float32, device=roi_images.device)

        # d_mm:spacing None → 0(官方无 spacing,物理头走离群点语义,
        # 见 physical_diameter_from_mask 的"不得编造"条)。
        spacings = []
        for i in range(K):
            sp = roi_meta[i].get("spacing_mm")
            spacings.append(0.0 if sp is None else float(sp))
        spacing_t = torch.tensor(spacings, dtype=torch.float32,
                                 device=roi_images.device).unsqueeze(-1)  # [K,1]
        d_mm = physical_diameter_from_mask(roi_masks, spacing_t)          # [K,1]

        # 逐头前向。loc 按病灶逐条(坐标通道 per-ROI;K 条小循环,K=每图病灶数,
        # 实测量级 1-3,批级向量化收益为零、代码复杂度翻倍,不做)。
        loc_probs_rows, side_rows, level_rows = [], [], []
        for i in range(K):
            m = roi_meta[i]
            out = self.roi_location_head(
                spatial[i:i + 1],
                bbox_xyxy=m.get("bbox_xyxy"),
                orig_w=m.get("orig_w") or 0,
                orig_h=m.get("orig_h") or 0,
            )
            loc_probs_rows.append(out["location_probs"][0])
            side_rows.append(out["side_logits"][0])
            level_rows.append(out["level_logits"][0])
        if K:
            location_probs = torch.stack(loc_probs_rows, dim=0)
            location_side_logits = torch.stack(side_rows, dim=0)
            location_level_logits = torch.stack(level_rows, dim=0)
        else:
            location_probs = torch.zeros(0, 7, device=roi_images.device)
            location_side_logits = torch.zeros(0, 3, device=roi_images.device)
            location_level_logits = torch.zeros(0, 3, device=roi_images.device)

        ete_out = self.roi_extension_head(spatial, roi_masks)
        t_out = self.roi_clinical_t_head(spatial, d_mm, ete_out["ete_logits"])
        concepts = self.roi_concept_head(pooled)
        tirads_out = self.roi_tirads_head(spatial, concepts)

        # 良恶性:ROI pooled 过整图 CBM(同一 cbm_head 实例——它是"主干表征
        # →良恶性"的既有通路,ROI 域复用表征、复用头,这是 scoping §C.2
        # "可复用的是主干表征"的直接落点;物理先验 e_prior 给 None 走 CBM
        # 内部零兜底,ROI 域无整图物理量,不编造)。
        cbm = self.cbm_head(pooled, None)
        malignancy = cbm.logits_malignancy                                     # [K,2]

        roi_outputs = []
        for i in range(K):
            m = roi_meta[i]
            roi_outputs.append({
                "lesion_id": m.get("lesion_id"),
                "patient_id": m.get("patient_id"),
                "is_main": int(m.get("is_main", 1)),
                "malignancy": malignancy[i],
                "location_probs": location_probs[i],
                "ete_probs": ete_out["ete_probs"][i],
                "ete_logits": ete_out["ete_logits"][i],
                "t_conf": t_out["t_conf"][i],
                "tirads_probs": tirads_out["tirads_probs"][i],
            })
        return {
            "roi_outputs": roi_outputs,
            "roi_pooled": pooled,
            "roi_spatial": spatial,
            "roi_concepts": concepts,
            "roi_location_side_logits": location_side_logits,
            "roi_location_level_logits": location_level_logits,
            "roi_t_logits": t_out["t_logits"],
            "roi_tirads_logits": tirads_out["tirads_logits"],
            "roi_ete_logits": ete_out["ete_logits"],
            "roi_band_ratio": ete_out["band_ratio"],
            "roi_d_mm": d_mm,
        }
