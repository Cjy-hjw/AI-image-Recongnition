"""官方数据四头训练入口 —— **不动 `train.py`**(legacy 交付路径,已验证)。

与 `train.py` 的分工:`train.py` 跑 legacy 三阶段(分割+概念+良恶性,stages.py 调度);
本脚本跑**官方四头**(位置/ETE/临床T/TI-RADS)的 A/B/C 调度(schedule_official.py),从分割底座
warm-start,官方数据只在云端 —— 本地只能 `--mock-labels --smoke` 验管道数值健康。

## 训练循环复用与损失政策

复用 `train_batch`(loop/train_step.py):它已内建 mixup 后的 `build_official_head_targets`
(四头 targets 组装,只训主病灶 + validity 顶层直传),AMP/scaler/grad-clip/vCLUB/AUCM
对偶步进都在里面,不重造。四头损失由 `MultiTaskLoss` 的固定 λ 外挂算(不进 FAMO)。

**legacy 损失默认保持开启**(`--freeze-legacy` 关):头微调期间继续回传分割+概念+良恶性
损失,守护旧任务不退化(Stage B 解冻 cls Top-2 时尤其重要 —— 让 top-2 同时服务新旧任务,
而非被四头单独拉走导致良恶性 AUC 退化)。传 `--freeze-legacy` 则关 malignancy AUCM
(对偶变量冻结),纯头训练。

## Gate / 护栏

- warm-start:`--resume-from` strict=False,断言 missing_keys ⊆ 四头前缀(多缺一个
  非头键即 fail-loud,杜绝静默半载,Phase B.4)。
- Dice 护栏:warm-start 后在官方 val 上测一次基线,训练中 val Dice 相对基线跌 >1 点
  警告、>3 点 abort(防重演 Stage2 崩塌)。基线与训练 val 同口径(validate_step 逐样本平均)。
- 早停:rules§3(epoch 3 后指标随机 + Dice 熔断)。
- JSONL 日志:V2.1 硬约束,云端落 `/2026aicompetition/workspace/logs`(默认);本地
  `--log-dir logs`。

## 用法

  # 云端真数据正式训练(Stage A 起步)
  python scripts/train_official.py --data-root /2026aicompetition/datasets/training/annotation \
      --xlsx <标注.xlsx> --resume-from checkpoints/best_model_mixtrain_xl_e40.pth \
      --out checkpoints/official_heads_best.pth

  # 本地管道冒烟(无官方数据,合成标签,3 epoch)
  python scripts/train_official.py --mock-labels --smoke --log-dir logs --batch-size 4
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.loss import MultiTaskLoss
from src.models.multi_task_model import MultiTaskThyroidModel
from src.models.losses.components.clinical_t import inverse_frequency_weights
from src.models.losses.components.extension import class_balanced_weights, log_class_prior
from src.training.jsonl_logger import JsonlLogger, OFFICIAL_LOG_DIR
from src.training.reproducibility import configure_reproducibility, seed_worker
from src.training.schedule_official import (
    STAGE_C_END,
    configure_official_stage,
    get_official_stage,
)
from src.training.loop.bn_freeze import set_frozen_modules_eval
from src.training.loop.train_step import train_batch, _roi_labels_from_meta
from src.training.trainer import validate_epoch


# ─────────────────────────────────────────────────────────────────────────────
# official_head_config 组装(从 config.train 的类计数造 λ / 权重张量)
# ─────────────────────────────────────────────────────────────────────────────
def build_official_head_config(config: dict) -> dict:
    """把 config.train 的官方头超参组装成 `MultiTaskLoss(official_head_config=...)`。

    ETE 的 class_weights / log_prior、T/TI-RADS 的 class_weights 是 [C] 张量,由类
    计数构造;随模型 `.to(device)` 迁移(MultiTaskLoss 用 register_buffer 存,故这里
    给 CPU 张量即可)。
    """
    tr = config["train"]
    ete_counts = tr["ete_class_counts"]
    ete_beta = float(tr.get("ete_cb_beta", 0.99))
    ete_la_tau = float(tr.get("ete_la_tau", 0.0))
    ete_qwk_weight = float(tr.get("ete_qwk_weight", 0.0))
    # 🔴 负 τ 必须在入口拒绝(2026-08-20 Gate2 实测发现的口径裂缝):
    # 本函数按 `la_tau > 0` 决定要不要造 log_prior,而 `compute_extension_loss`
    # 的守卫是 `la_tau != 0.0` —— 两者对负值的判定相反,`la_tau=-0.5` 会走进
    # LA 分支却拿到 log_prior=None,报一句与真实病因无关的
    # "la_tau != 0 时必须提供 log_prior"(真实病因是"τ 不该是负数")。
    # 且负 τ 在数学上是**放大**类别不平衡(LA 的符号方向见 extension.py
    # 模块 docstring),与本项目踩过两次的符号反转坑同源(#113/#120),
    # 属于宁可开不了工也不能静默跑歪的那一类。
    assert ete_la_tau >= 0.0, (
        f"ete_la_tau 必须 >= 0，got {ete_la_tau}"
        " —— 负 τ 会放大类别不平衡而非校正(Logit Adjustment 的符号方向见"
        " src/models/losses/components/extension.py 模块 docstring)"
    )
    tirads_counts = tr.get("tirads_class_counts")
    cfg = {
        "loc_loss_weight": float(tr.get("loc_loss_weight", 1.0)),
        "ete_loss_weight": float(tr.get("ete_loss_weight", 1.0)),
        "clin_t_loss_weight": float(tr.get("clin_t_loss_weight", 1.0)),
        "tirads_loss_weight": float(tr.get("tirads_loss_weight", 1.0)),
        "ete_focal_gamma": float(tr.get("ete_focal_gamma", 2.0)),
        "ete_la_tau": ete_la_tau,
        # backlog #128 LA-SoftQWK-OD:0(默认)= 走 CB-Focal 老路径。
        "ete_qwk_weight": ete_qwk_weight,
        "ete_dist_weight": float(tr.get("ete_dist_weight", 0.2)),
        "clin_t_label_smoothing": float(tr.get("clin_t_label_smoothing", 0.0)),
        "tirads_label_smoothing": float(tr.get("tirads_label_smoothing", 0.0)),
        "tirads_ordinal_weight": float(tr.get("tirads_ordinal_weight", 0.0)),
        # β=0 → 不加权(None);>0 → CB 权重。**qwk_weight>0 时 CB-Focal 分支不参与
        # 计算,但权重张量仍按配置构造并传入**——不加特判是有意的:compute_extension_loss
        # 内部按 qwk_weight 短路分支,传入未使用的 class_weights 不产生副作用,
        # 而加特判反而多一条"两处判断条件必须同步"的隐性耦合。
        "ete_class_weights": (
            class_balanced_weights(ete_counts, ete_beta) if ete_beta > 0 else None
        ),
        # log_prior 供 Logit Adjustment 消费:la_tau>0(CB-Focal 分支叠加 LA)或
        # qwk_weight>0(LA-SoftQWK-OD 方案,LA 调整发生在两分支之前)时均需要。
        "ete_log_prior": (
            log_class_prior(ete_counts) if (ete_la_tau > 0 or ete_qwk_weight > 0) else None
        ),
        "clin_t_class_weights": (
            inverse_frequency_weights(tr["clin_t_class_counts"])
            if tr.get("clin_t_use_class_weights", False)
            else None
        ),
        # TI-RADS 真实计数未知(禁本地),config 默认 null + 开关 false → 不加权;
        # 上云拿到真实分布、且判定属中度不平衡(非长尾)后再置 true + 填计数。
        "tirads_class_weights": (
            inverse_frequency_weights(tirads_counts)
            if tr.get("tirads_use_class_weights", False) and tirads_counts
            else None
        ),
        # Stage2 seg-only(2026-09-13):0(默认)= 老路径零回归,权重非 0 时
        # compute_segmentation_loss 内消费(见该函数 docstring)。
        "seg_boundary_weight": float(tr.get("seg_boundary_weight", 0.0)),
        "seg_lovasz_weight": float(tr.get("seg_lovasz_weight", 0.0)),
        # plan 任务C CC-DiceCE(新架构线):默认 0 零回归;权重非 0 时
        # 实例级局部损失生效(见 cc_dice_ce_per_sample docstring)。
        "seg_ccdice_weight": float(tr.get("seg_ccdice_weight", 0.0)),
        # plan 任务D 深监督辅助头(2026-09-20):默认 0 零回归;权重非 0 时
        # aux2/aux3 与 gt 同式 BCE+Dice 生效(见 compute_segmentation_loss
        # docstring 的 deepsup 段)。
        "seg_deepsup_weight": float(tr.get("seg_deepsup_weight", 0.0)),
        # plan 任务L1 悬崖加权(2026-09-23,新架构线):默认 0 = 零回归;权重
        # 非 0 时 train_step 按前代 IoU 缓存查表注入 targets["cliff_weights"]
        # (见 compute_segmentation_loss docstring 的 cliff 段)。
        "seg_cliff_weight": float(tr.get("seg_cliff_weight", 0.0)),
        # 🔴 ROI 范式损失组(2026-09-25,Gate2 A 路 P1-2 修复):默认 0 零回归;
        # 此前 yaml 写了键但本函数不组装 → config→MultiTaskLoss 透传链断裂,
        # yaml 层开关永远到不了损失层(r7 config 静默不读事故同型)。权重非 0
        # 时 compute_roi_head_losses(四头 ROI 损失)生效。
        "roi_head_weight": float(tr.get("roi_head_weight", 0.0)),
        # 债6(2026-09-25,方案 A 简化版):ROI 概念头监督,默认 0 零回归。
        # 同上透传链——漏此行 = yaml 开关到不了损失层。
        "roi_concept_weight": float(tr.get("roi_concept_weight", 0.0)),
    }
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# warm-start:strict=False + missing_keys ⊆ 四头前缀断言(Phase B.4)
# ─────────────────────────────────────────────────────────────────────────────
def load_warm_start(model: nn.Module, ckpt_path: str, device) -> None:
    """加载分割底座 checkpoint。四头键缺失属预期(底座无四头);缺任何**非头**键即
    fail-loud —— 那意味着底座与当前架构不匹配,静默半载会训出垃圾。"""
    # weights_only=False 是**防御性**声明,不是在修已存在的崩溃:torch 2.6 起该参数
    # 默认翻转为 True,会拒绝纯 tensor 之外的对象。A800(torch2.6)实测全部 17 个
    # checkpoint:本项目的分割底座(best_model_mixtrain_xl_e40.pth 等)是纯
    # OrderedDict,默认参数下加载正常;唯一在默认参数下 UnpicklingError 的是
    # authenticity_model.pth(含 numpy 的 feat_mean/feat_std),而它由
    # gatekeeper/authenticity.py:229 自己以 weights_only=False 加载,不走这里。
    # 此处显式关掉,是因为 --resume-from 允许指向 thyroidxl_best.pth 那类
    # {"epoch","stage","state_dict",...} 包装格式,未来格式再变也不会在云上翻车。
    # 加载的是我方自产的可信文件(非外部输入),关掉该校验无安全代价。
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # best_model.pth 是纯 state_dict;若是 {"model": ...} 包装则取出。
    if isinstance(state, dict) and "model" in state and "location_head.0.weight" not in state:
        state = state["model"]
    result = model.load_state_dict(state, strict=False)
    head_pfx = MultiTaskThyroidModel.OFFICIAL_HEAD_PREFIXES
    # 深监督辅助头(plan 任务D,2026-09-20):旧 checkpoint(r5/ep16/CC 快照)
    # 无 aux 键属预期——从旧权重热启动深监督实验是**标准路径**(aux 头随机
    # 初始化 + 主链继承)。列入白名单与四头同待遇;缺任何其他键仍 fail-loud。
    aux_pfx = MultiTaskThyroidModel.SEG_AUX_PREFIXES
    # ROI 头(2026-09-25 训练侧接线,债3):旧底座(分割/四头全量权重)不含
    # ROI 头 46 键属**预期冷启**(新头随机初始化,`--roi-heads` 的设计语义:
    # backbone 热启 + ROI 头冷启)。列入白名单——这兑现了
    # multi_task_model.py:275-279 注释"合法化点在 train_official 的 ROI
    # 白名单(训练侧挂)"的声称(注释在 models 领地不改,由本处代码兑现)。
    # 缺任何其他键(白名单外)仍 fail-loud。
    roi_pfx = MultiTaskThyroidModel.ROI_HEAD_PREFIXES
    bad_missing = [
        k for k in result.missing_keys
        if not k.startswith(head_pfx) and not k.startswith(aux_pfx)
        and not k.startswith(roi_pfx)
    ]
    if bad_missing:
        raise RuntimeError(
            f"warm-start 底座缺失**非四头**键 {bad_missing[:8]}(共 {len(bad_missing)})"
            f" —— checkpoint {ckpt_path} 与当前架构不匹配,拒绝静默半载。"
        )
    if result.unexpected_keys:
        print(f"--> [warm-start] 忽略 {len(result.unexpected_keys)} 个多余键(底座含已弃用字段)")
    # 🔴 Gate2 P1 修复(2026-09-20 审查):只数四头前缀的 missing——旧权重
    # 缺 aux 4 键(任务D 标准路径)会被 len(missing_keys) 混进"四头 N 键随机
    # 初始化"的口径,stdout 事实错误(F1 审查防的"从 stdout 分不出真热启动"
    # 同型雷)。aux 计数下方单独分口径打印。
    n_head_missing = sum(
        1 for k in result.missing_keys if k.startswith(head_pfx))
    # 🔴 ete_only 专属判读(Gate2 审查 F1,2026-09-12):r5 全量热启动时四头
    # missing=0(权重已覆盖),此时打"四头 0 键随机初始化"是**事实错误**且与
    # 纯分割底座(四头 32 键真随机)的日志格式相同——从 stdout 分不出哪个是
    # 真热启动。漏传 --resume-from 时 ETE 小试从随机初始化训,整轮白烧还被
    # 误判为"模式有害"。故按键数分口径打印。
    if n_head_missing == 0:
        print(f"--> [warm-start] 加载 {ckpt_path};四头权重全部命中(missing=0,全量热启动)")
    else:
        print(f"--> [warm-start] 加载 {ckpt_path};四头 {n_head_missing} 键随机初始化"
              f"({'🔴 分割底座,四头为随机' if n_head_missing >= 30 else '部分命中,核对 checkpoint'})")
    # 深监督 aux 键计数(任务D):旧权重缺 4 键(2头×w/b)属预期标准路径,
    # 但必须打进日志——"aux 随机初始化"从 stdout 可见,防"以为热启动了
    # aux 其实是随机"的误读(与四头计数同型但分口径打印)。
    n_aux_missing = sum(1 for k in result.missing_keys if k.startswith(aux_pfx))
    if n_aux_missing:
        print(f"--> [warm-start] 深监督辅助头 {n_aux_missing} 键随机初始化"
              f"(旧 checkpoint 无 aux 键,任务D 标准路径)")
    # ROI 头键计数(债3,分口径同 aux):旧底座缺全部 ROI 键 = 预期冷启,
    # 但同样必须打进日志(stdout 可见,防"以为 ROI 头热启动了其实是随机"的
    # 误读;G1 挡不住"认知与事实错位"这一类,日志是唯一防线)。
    n_roi_missing = sum(1 for k in result.missing_keys if k.startswith(roi_pfx))
    if n_roi_missing:
        print(f"--> [warm-start] ROI 头 {n_roi_missing} 键随机初始化"
              f"(冷启动新头,预期行为;backbone 由本 checkpoint 热启)")



# ─────────────────────────────────────────────────────────────────────────────
# Mock 数据集(--mock-labels):合成图像 + 四头标签,验管道数值健康
# ─────────────────────────────────────────────────────────────────────────────
class MockOfficialDataset(Dataset):
    """纯合成数据集:随机图像 + 随机 mask + 官方新增维度头标签(含边界情形)。

    **不依赖磁盘/公开集** —— 冒烟只验 forward/backward/masked-loss 数值健康,合成即可
    (比"公开集图像"更可移植:CI/陌生容器都能跑,无路径依赖)。刻意覆盖:
      - 全 invalid 样本(所有 valid_*=0,验 masked loss 精确 0、不炸);
      - 峡部(lesion_loc=6,无纵向属性,位置头 clamp 分支);
      - ETE 血管(extrathy_ext=3,最稀有类);
      - TI-RADS 两端类别(I 级=0、V 级=4,验有序项/类权重不偏废边界类)。
    """

    def __init__(self, n: int = 32, img_size: int = 512, seed: int = 0):
        self.n = n
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)
        # 预生成每样本的确定性属性(种子固定 → 可复现)
        self._specs = []
        for i in range(n):
            # 每 8 个里留 1 个全 invalid;每 8 个里 1 个峡部;1 个 ETE 血管;
            # 1 个 TI-RADS I 级;1 个 TI-RADS V 级。
            all_invalid = (i % 8 == 0)
            loc = 6 if (i % 8 == 1) else int(self.rng.randint(0, 7))     # 0..6
            ete = 3 if (i % 8 == 2) else int(self.rng.randint(0, 4))     # 0..3
            t = int(self.rng.randint(0, 2))
            if i % 8 == 3:
                tirads = 0       # I 级
            elif i % 8 == 4:
                tirads = 4       # V 级
            else:
                tirads = int(self.rng.randint(0, 5))                    # 0..4
            self._specs.append((all_invalid, loc, ete, t, tirads, i % 5 != 4))  # is_main: 4/5

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = self.img_size
        all_invalid, loc, ete, t, tirads, is_main = self._specs[idx]
        img = torch.rand(1, s, s, dtype=torch.float32)
        mask = (torch.rand(1, s, s) > 0.7).float()
        v = 0.0 if all_invalid else 1.0
        labels = {
            "composition": 0, "valid_composition": int(v),
            "echogenicity": 0, "valid_echogenicity": int(v),
            "shape": 0, "valid_shape": int(v),
            "margin": 0, "valid_margin": int(v),
            "foci": torch.zeros(4, dtype=torch.float32), "valid_foci": 0,
            "malignancy": int(t), "valid_malignancy": int(v),
            "valid_mask": 1,
            # 官方新增维度头
            "lesion_loc": loc, "valid_lesion_loc": int(v),
            "extrathy_ext": ete, "valid_extrathy_ext": int(v),
            "clin_t_stage": t, "valid_clin_t_stage": int(v),
            "tirads": tirads, "valid_tirads": int(v),
            "is_main": int(is_main),
            "lesion_id": f"mock::{idx}",
            "patient_id": f"mockpid::{idx % 7}",
        }
        return img, mask, labels


# ─────────────────────────────────────────────────────────────────────────────
# 冒烟:官方头梯度健康(独立于训练循环,可确定性断言)
# ─────────────────────────────────────────────────────────────────────────────
def smoke_grad_health(model: nn.Module, criterion: nn.Module, device):
    """手动一次 forward+backward,断言各官方头梯度非零非 NaN;再验全 invalid batch 不炸。

    独立于 `train_batch`(它内部 zero_grad→step→zero,拿不到中间梯度),故这里自造
    最小批直接查 `.grad`。是本地能做的最强管道验证。
    """
    s = model.img_size
    B = 2
    model.train()
    heads = {
        "location_head": model.location_head,
        "extension_head": model.extension_head,
        "clinical_t_head": model.clinical_t_head,
        "tirads_head": model.tirads_head,
    }

    def _run(valid: float):
        model.zero_grad(set_to_none=True)
        img = torch.rand(B, 1, s, s, device=device)
        masks = (torch.rand(B, 1, s, s, device=device) > 0.7).float()
        targets = {
            "mask": masks,
            "composition": torch.zeros(B, dtype=torch.long, device=device),
            "echogenicity": torch.zeros(B, dtype=torch.long, device=device),
            "shape": torch.zeros(B, dtype=torch.long, device=device),
            "margin": torch.zeros(B, dtype=torch.long, device=device),
            "foci": torch.zeros(B, 4, device=device),
            "malignancy": torch.zeros(B, dtype=torch.long, device=device),
            "valid_mask": torch.full((B,), valid, device=device),
            "label_validity": {
                k: torch.full((B,), valid, device=device)
                for k in ("composition", "echogenicity", "shape", "margin", "foci", "malignancy")
            },
            # 官方新增维度头顶层直传 validity
            "lesion_loc": torch.tensor([6, 0], device=device),          # 含峡部
            "valid_lesion_loc": torch.full((B,), valid, device=device),
            "extrathy_ext": torch.tensor([3, 0], device=device),        # 含血管
            "valid_extrathy_ext": torch.full((B,), valid, device=device),
            "clin_t_stage": torch.tensor([1, 0], device=device),
            "valid_clin_t_stage": torch.full((B,), valid, device=device),
            "tirads": torch.tensor([4, 0], device=device),              # 含 V 级
            "valid_tirads": torch.full((B,), valid, device=device),
            # L1 悬崖权重(plan 任务L1):smoke 无 labels/lesion_id 通道,
            # 权重>0 时手动补全 1 权重(等价"查表全 miss"路径)防 fail-loud
            # 拦住冒烟;真训练的注入由 train_step 查表完成。
            **({"cliff_weights": torch.ones(B, device=device)}
               if getattr(criterion, "seg_cliff_weight", 0.0) > 0.0 else {}),
            # ROI 范式(债4,2026-09-25):整图前向无 ROI 键,权重>0 时
            # compute_roi_head_losses 会 fail-loud 拒空算(整图冒烟不产
            # roi_outputs/roi_labels)。冒烟里 ROI 支路由 run_roi_epoch 的
            # MockROI loader 真跑(见 main 的 roi_train_loader),本函数只验
            # **整图头**梯度健康——故此处把 ROI 键以空 list 显式注入
            # (K=0 空批合法路径,损失侧同型早退返回 0)——而非关权重:
            # 关权重会让"配置非 0"这件事在冒烟日志里隐形。
            "roi_labels": [],
        }
        outputs = model(img)
        if getattr(criterion, "roi_head_weight", 0.0) > 0.0:
            # K=0 空批:损失侧合法早退(与 smoke_grad_health 的整图职责一致,
            # 理由见 targets 组装段的 ROI 注释)
            outputs = {**outputs, "roi_outputs": []}
        loss, loss_dict = criterion(outputs, targets, enable_mal_loss=True)
        loss.backward()
        return float(loss.item())

    # ① 有效批:loss 有限 + 每个头至少一个参数有非零有限梯度
    loss_valid = _run(1.0)
    assert np.isfinite(loss_valid), f"有效批 loss 非有限: {loss_valid}"
    for hname, head in heads.items():
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        assert grads, f"{hname} 无任何梯度(未接入计算图?)"
        assert all(torch.isfinite(g).all() for g in grads), f"{hname} 梯度含 NaN/Inf"
        assert any(g.abs().sum().item() > 0 for g in grads), f"{hname} 梯度全零(监督未生效?)"

    # ② 全 invalid 批:masked loss 精确屏蔽四头,loss 仍有限、四头梯度全零、无 NaN
    loss_invalid = _run(0.0)
    assert np.isfinite(loss_invalid), f"全 invalid 批 loss 非有限: {loss_invalid}"
    for hname, head in heads.items():
        for p in head.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{hname} 全 invalid 下梯度含 NaN"
    model.zero_grad(set_to_none=True)
    print(f"--> [smoke] 梯度健康 OK(valid loss={loss_valid:.4f} / invalid loss={loss_invalid:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# 单 epoch 官方训练(复用 train_batch,官方 stage 语义)
# ─────────────────────────────────────────────────────────────────────────────
def run_official_epoch(
    model, loader, optimizer, vclub_optimizer, criterion, scaler, device, config,
    running_losses_history, epoch, freeze_legacy, logger, log_every, head_lr,
):
    """跑一个官方 epoch。返回 avg_loss。JSONL 每 `log_every` 步落一行 train。

    `head_lr` 由调用方传入(调度返回的基准 lr),**不从 `optimizer.param_groups[0]` 取**
    —— 那样依赖"第 0 组恒为头组"这一可变实现细节,组序一改日志就错(Gate2 lane1/lane3)。
    注:Stage B 的 backbone LLRD 组 lr 更小,JSONL 只记基准头 lr。
    """
    model.train()
    criterion.train()
    # 冻结子模块置 eval,防其 BN running stats 更新(BN 冻结遗漏教训)。
    set_frozen_modules_eval(model)

    # train_batch 的 `stage` 仅决定 enable_mal_loss=(stage>1):
    #   freeze_legacy=False → stage=2 → 开 malignancy AUCM(守旧任务);
    #   freeze_legacy=True  → stage=1 → 关(纯头训练)。
    legacy_stage = 1 if freeze_legacy else 2
    slr_beta = config["train"].get("slr_beta", 0.99)
    slr_threshold = config["train"].get("slr_threshold", 2.0)
    label_smoothing = config["train"].get("label_smoothing", 0.0)

    prev_z_latent = prev_concept_vector = None
    running = 0.0
    n = 0
    for batch_idx, (images, masks, labels) in enumerate(loader):
        loss, loss_dict, prev_z_latent, prev_concept_vector = train_batch(
            model, images, masks, labels, optimizer, vclub_optimizer, criterion,
            scaler, device, config, running_losses_history, epoch, legacy_stage,
            update_famo_weights=True, prev_z_latent=prev_z_latent,
            prev_concept_vector=prev_concept_vector, slr_beta=slr_beta,
            slr_threshold=slr_threshold, label_smoothing=label_smoothing,
            spin_l1_weight=0.0,
        )
        running += loss.item()
        n += 1
        if logger is not None and batch_idx % log_every == 0:
            logger.log(epoch=epoch, step=batch_idx, phase="train",
                       loss=float(loss.item()), lr=head_lr,
                       extra=_step_diagnostics(loss_dict, labels))
    return running / max(n, 1)


# ── L1 悬崖加权:前代 IoU 采集(plan 任务L1,2026-09-23)──────────────────────
def _collect_cliff_ious(model, loader, device, criterion=None) -> dict:
    """epoch 尾对**本 epoch 刚训过的 train 池**逐样本采 hard IoU,作下一 epoch
    的悬崖权重信号(L1 核心:前代 IoU = train 侧自采自用,非 val 池查表——
    patient 分组互斥下 val→train 键空间零交集,v1 方案结构性空转已废弃)。

    口径 = per-sample **hard IoU**(pred>0.5 二值化后 inter/(union−inter+eps),
    与 validate_step 的 inter/union 同式但 IoU 而非 dice);GT 全零记 **-1 哨兵**
    → _cliff_weight_of 判 w=1(阴性监督归主项);**pred 全零/GT 非零记真 0.0**
    (2026-09-24 审查 P0 修:完全漏检=最硬悬崖,w=4;旧版 `<=0` 把它和哨兵
    一起吞成 w=1,恰好漏掉最该加压的人口)。

    🔴 **BN/运行态**:`model.eval()` → 采完 `model.train()` 恢复,**try/finally**
    (2026-09-24 审查 P1 修:采集循环任何异常炸出时模型不能永久停在 eval 态;
    下一 epoch 头部 `run_official_epoch` 会再调 `set_frozen_modules_eval`,
    但那只覆盖冻结子模块,不保证 dropout/整体态复原)。eval 期间 BN
    running stats 冻结(采集不改统计)。

    criterion 仅用于读 `seg_cliff_weight`(权重 0 时调用方根本不该调本函数,
    此处仅防御性再判一次):不进 targets、不建损失图,与 validate_epoch 零耦合。

    返回 dict[lesion_id -> iou]。同池自采自用 → 下一 epoch 查表 100% 命中
    (loader shuffle 不改 lesion_id 键空间)。🔴 动态样本同病灶多帧共享同一
    lesion_id,逐帧写入 dict 后值取**最后一帧**——非任意聚合口径,重读采集
    数据请知悉(取 min/均值聚合是未做的改进,当前实测方向=少 boost 非反向错)。
    """
    if criterion is not None and getattr(criterion, "seg_cliff_weight", 0.0) <= 0.0:
        return {}
    was_training = model.training
    model.eval()
    ious: dict = {}
    eps = 1e-8
    try:
        with torch.no_grad():
            for images, masks, labels in loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images, gamma_prior=1.0)
                pred_mask = (torch.sigmoid(outputs["mask"]) > 0.5).float()
                inter = (pred_mask * masks).sum(dim=(2, 3))
                union = pred_mask.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                # IoU = 交 / (并 − 交)(并含两倍重叠,dice 分母是 union、IoU 分母是
                # union−inter;与 validate_step:212-214 的 inter/union 同源两口径)
                iou = inter / (union - inter + eps)
                lids = labels.get("lesion_id")
                if lids is None:
                    continue      # legacy 数据集无键:本函数不该被调(fail 上层)
                for i, lid in enumerate(lids):
                    gt_area = float(masks[i].sum())
                    # str(lid) 规整键(2026-09-24 审查路C P2):collate 对非 str
                    # 的 lid(如 int)产 0-dim Tensor,str() 后是 "tensor(7)" 带壳
                    # 字符串——取 .item() 值(0-dim 标量)再 str,干净且可哈希稳定。
                    if torch.is_tensor(lid):
                        lid = str(lid.item())
                    else:
                        lid = str(lid)
                    ious[lid] = -1.0 if gt_area == 0.0 else float(iou[i])
    finally:
        if was_training:
            model.train()
    return ious


def _cliff_stats_from_ious(ious: dict) -> dict:
    """缓存 dict → JSONL 观测标量(S5 预注册,权重>0 才调用)。

    口径:分母 cliff_n_evaluated = iou_prev 非哨兵(-1)的样本数(有效 GT
    样本,含真 0——完全漏检是最硬悬崖,必须进分母与 boosted 人口,2026-09-24
    审查 P0 修:旧版 `v > 0` 把真 0 连分母都排除,JSONL 对模型最差一档零观测);
    clif_iou_prev_mean 只对被加压(w=4)样本算(诊断"压的是不是真悬崖人口");
    无有效样本 → 全 None(调用方跳过,不填 0,与 _epoch_metrics_extra 的
    None 纪律一致)。
    """
    evaluated = [v for v in ious.values() if v != -1.0]
    boosted = [v for v in evaluated if v < 0.5]
    if not evaluated:
        return {"cliff_n_boosted": 0, "cliff_n_evaluated": 0,
                "cliff_frac_boosted": None, "cliff_iou_prev_mean": None}
    return {
        "cliff_n_boosted": len(boosted),
        "cliff_n_evaluated": len(evaluated),
        "cliff_frac_boosted": round(len(boosted) / len(evaluated), 6),
        "cliff_iou_prev_mean": (round(sum(boosted) / len(boosted), 6)
                                if boosted else None),
    }


# JSONL 诊断字段用到的 valid_* 键 → 输出键名(`n_valid_` 前缀)。
# 顺序即日志字段顺序,便于人眼扫。
# ⚠️ 这些是 **loader 原始值**,不是最终进损失的有效监督数 —— 见 `_step_diagnostics`
#    docstring 里的"两级闸门"说明。`is_main` 单独记,就是为了让二者的差可见。
_VALID_KEYS = (
    "valid_malignancy", "valid_lesion_loc", "valid_extrathy_ext",
    "valid_clin_t_stage", "valid_tirads", "valid_mask", "is_main",
)


def _step_diagnostics(loss_dict, labels) -> dict:
    """把子损失分解 + 各头有效标签数打成 JSONL 的 `extra` 字段。

    🔴 **为什么必须有(2026-08-14,勿删)**:首次真数据训练的日志只有总 loss,
    出现 ~5 / ~0.8 双峰却无从定位是哪个头 —— `train_batch` 返回的 `loss_dict`
    此前被直接丢弃。总 loss 是所有子损失经 SLR/FAMO 动态加权后的标量,**从它
    反推组成是不可能的**,只能靠猜;而本项目已两次栽在"推断被当成事实"上。

    `n_valid_*` 单独记的理由(不能用 loss 是否为 0 替代):masked CE 在该头
    valid 全 0 时返回**精确 0**,与"损失已收敛到 0"在日志里长得一模一样。
    四头是随机初始化的新头,"某头是否真的在受监督"是本轮训练的头号判据
    (列名命中 ≠ 取值解析成功 ≠ 有监督样本),必须直接观测计数而非旁推。

    🔴 **`n_valid_*` 是 loader 原始值,不等于最终进损失的监督数(2026-08-14 实证)**:
    有效监督要过**两级闸门**,都在 `build_official_head_targets`(`train_step.py`)里:
      ① `× is_main` —— 官方口径下四头只评主病灶,次病灶(`病灶N`)标签不进损失;
      ② `× hard.float() × mask_weight` —— mixup/cutmix 触发时,归属不明确的样本屏蔽。
    故 `n_valid_lesion_loc=12` 与 `loss_location=0.0` **可以同时出现且都没错**。
    真实事故:曾据此误判"90% 样本白训、标签解析失败",实际标签解析完全正常,
    是闸门②(MixUp 分支的 `lam ~ Beta(0.2,0.2)` 落在 (0,1),而闸门用了精确相等
    `lam==0.0`/`==1.0`)把四头监督整批清零 —— 已于 2026-08-15 修复,改用与
    DR-CutMix 同一套阈值(`lam<=0.35`/`>=0.65`)。
    🔴 **修复在 `src/training/loop/train_step.py` 的闸门,不在 `mixup.py`**:
    `lam` 还被 CBM 概念损失与 SLR 当软加权系数消费,在 mixup 侧硬化会改掉旧任务的
    数学(此处曾误写成 mixup.py,是第一版被撤回方案的残留,勿据此去 mixup.py 找)。
    **`is_main` 一并记就是为了让这个差可见。**

    容错:任何一项取不到都跳过而非抛异常 —— 诊断字段炸掉整轮训练是本末倒置。
    """
    out: dict = {}
    if loss_dict:
        for k, v in loss_dict.items():
            try:
                out[f"loss_{k}"] = round(float(v.item() if hasattr(v, "item") else v), 6)
            except (TypeError, ValueError, RuntimeError):
                continue  # 非标量项(如权重向量)跳过,不阻断训练
    if isinstance(labels, dict):
        for k in _VALID_KEYS:
            v = labels.get(k)
            if v is None:
                continue
            try:
                out[f"n_{k}"] = int(v.sum().item() if hasattr(v, "sum") else v)
            except (TypeError, ValueError, RuntimeError):
                continue
    return out


# ── 分割护栏阈值(2026-08-16 分割改自训后新增)────────────────────────────────
# 语义:下限门(分割没学起来),与既有的"相对基线跌 N 点"上限门互补。
# 🔴 **与 stage 边界耦合**:下面的 epoch 选点依据是 `schedule_official.STAGE_A_END=10`
# (ep11 起进 Stage B)与 `stages.compute_gamma_prior` 的 γ=0 窗口(ep1-20)。若日后
# 重排 A/B/C 边界或 γ 退火起点,**必须同步改这里** —— 否则护栏会静默退化成
# "warmup 期就判死刑"或"跑完才判",两种失效都不报错。
SEG_WARN_EPOCH, SEG_WARN_DICE = 5, 0.10

# 🔴 **硬 abort 用「预测 mask 的前景占比」判,不用 dice、也不用概率分布形状**
#    (2026-08-16 第三/四轮审查后定稿,中间否掉过两版,过程见下)
#
# 【否掉的第一版】dice 门(ep12/0.05):hard `val_dice` 先 `(pred_prob>0.5)` 二值化,
# 而真实病灶掩膜只占 512² 的 0.1%~3%(静态 M 掩膜实测 20-90px)。分割**确实在学**
# 但前景 logit 尚未越过 0 的阶段可持续很多 epoch,dice 全程恒 0 → 早期用 dice 设硬门
# 必然误杀。本地实测佐证:可训态 40 step 时输出已明显有结构,同期 dice 仍在地板上。
#
# 【否掉的第二版】概率分布 spread(`seg_prob_max − seg_prob_mean`):看似能抓住上一轮
# "处处 0.445"的指纹,但**主会话自查实测证伪,3/8 场景判错**
# (`tmp_main_spread_reverse_hole.py`):
#   ① 有空间结构但整体偏低(max=0.45/mean=0.05,spread=0.396 很大)→ 护栏放行,
#      而 `p>0.5` 仍全零、mask 全零、Sclin 照样 0 —— **假的安全感,最危险**;
#   ② 常量图 + 4 个异常像素(spread=0.545)→ 放行,而整体毫无结构。
# 根因:max 与 mean 都是单点/全局标量,不稳健,几个像素就能骗过。
#
# 【定稿】`seg_over_half_frac` = `(pred_prob > 0.5)` 的像素占比。它**直接就是**预测
# mask 的前景占比,也正是官方 IoU 匹配实际消费的那个量 —— 判据与评分口径同构,
# 不再是间接的分布形状代理。实测 8/8 场景全判对(`tmp_main_overhalf_judge.py`)。
#
# 阈值:下界 1e-4(≈26 像素)—— 最小真实病灶 0.1% ≈ 262 像素,留 10 倍误杀余量。
# 🔴 **只设下界,不设上界(2026-08-16 实测否掉了上界门,勿再加回)**:曾加过
# "占比 >0.5 → abort"防"全图判成前景",实测**会误杀正常收敛的模型** ——
# 随机初始化时输出普遍 >0.5(frac≈1.0),收敛到真值(1% 前景)**必然要穿过 0.5 这条线**:
# 实测 step0 frac=1.000 → step10 0.926 → step20 0.555 → step25 0.219 → step60 0.0099
# (最终等于真值),中间 step10-20 那段会被上界门打死。上界是必经之路不是崩溃信号。
# 真正的"全图判前景"崩溃态会一直卡在高位不下来,那由 ep20 的 dice 兜底门抓
# (frac 恒 1.0 时 IoU 极低 → dice 上不去)。
SEG_FROZEN_EPOCH = 8
SEG_FG_FRAC_LO = 1e-4         # 低于此 = 几乎什么都没预测(含恒全零)

# dice 硬门保留为**最后兜底**,阈值大幅放宽:ep20 是 γ=0 干净监督窗口的末尾
# (`compute_gamma_prior` ep>20 才起步),此时仍连 0.02 都到不了,剩下 15 个 epoch
# 也救不回来。0.02 极低,不构成对"慢但在学"的误杀。
SEG_ABORT_EPOCH, SEG_ABORT_DICE = 20, 0.02
# composite 早停警告阈值。口径含 seg_dice 后由 0.35 下调,见 main() 注释。
COMPOSITE_WARN = 0.30

# ── seg_only 版本化快照(2026-09-18 调教,总纲 §6.4)──────────────────────
# 预注册快照点。🔴 不加快照则训完只有一个权重,仍走"composite(无 AP 无匹配)
# 选模"老路——P0 验尸实锤官方口径下最强权重与 composite 最优不同源。
# 上限锚 SEG_ABORT_EPOCH(=20,γ_prior 单变量窗口),两常量同源防漂移:
# 若提预算须同步改,否则快照点落在 γ>0 区间污染单变量前提。
_SEG_SNAPSHOT_EPOCHS = (4, 8, 12, 16, 20)


def _snapshot_path(out_path: str, epoch: int) -> str:
    """--out → 快照路径:<out 去后缀>_ep<N>.pth,同目录。

    复用 --out 的目录与可写性预检(冒烟/真训共用同一防线);版本化命名与
    `--out 必须版本化` 三硬门同型,防止快照冒充/覆盖 best 文件。
    """
    root, _ext = os.path.splitext(out_path)
    return f"{root}_ep{epoch}.pth"


def _epoch_metrics_extra(metrics: dict, val_dice, val_acc, val_auc, train_loss,
                         cliff_stats: "dict | None" = None) -> dict:
    """epoch 级 val 指标 → JSONL `extra` 字段。

    🔴 **为什么必须写 JSONL(2026-08-15 实证,勿删)**:这些指标此前**只走 stdout**
    (`print`),而 `python` 不加 `-u` 时 stdout 是 8KB 块缓冲 —— 上一轮 35 epoch
    真训要攒到 ep25 才第一次刷新,整个训练前 2/3 完全看不到指标,等于盲跑。
    `JsonlLogger` 逐行 flush,不受管道缓冲影响,是唯一可靠的实时观测面。

    `malignancy_auc_pr` 尤其不能漏:它是目标三的核心项(良恶性 AUC-PR),上一轮
    **完全没打印**,0.888 这个数是靠 `composite×5 − 其余四项`**反推**出来的。

    None 值直接跳过(不填 0)—— 填 0 会把"这项算不出来"伪装成"分数是 0"。
    `JsonlLogger._drop_nonfinite` 另会兜掉 NaN/Inf。

    cliff_stats(plan 任务L1,2026-09-23):权重>0 时调用方传入
    `_cliff_stats_from_ious` 的产物(标量键,与 n_cc_* 惯例一致:权重>0 才
    出现);🔴 开训硬判据之一:epoch≥2 起 cliff_frac_boosted>0,若 ep2-3 恒 0
    = 采集/注入链断,20ep 空转须停机回报(ep1 全 w=1 属预期——首查无缓存,
    但 cliff_n_evaluated 应 >0 供确认覆盖)。
    """
    out: dict = {
        "train_loss": round(float(train_loss), 6),
        "val_dice": round(float(val_dice), 6),
        "val_acc": round(float(val_acc), 6),
    }
    if val_auc is not None:
        out["val_auc_legacy"] = round(float(val_auc), 6)
    for k in (
        "malignancy_auc_pr", "location_weighted_auc", "ete_weighted_kappa",
        "clinical_t_auc", "tirads_weighted_kappa", "composite", "n_lesions",
        # 分割概率分布(诊断,非官方指标)。🔴 **它是分割升温的先行指标**:
        # hard `val_dice` 要等前景 logit 越过 0 才会离开 0,而这三个数会先动 ——
        # 2026-08-15 正是靠 `pred_prob` 全域 <0.5 才定位到"分割全程冻结"。
        # 此前它只在 `batch_idx==0` 打一行 stdout,JSONL 里完全没有。
        "seg_prob_mean", "seg_prob_max", "seg_over_half_frac",
        # L1 cliff 分项 val 均值(2026-09-24 审查 P1 修):val 全 1 注入使
        # val_loss 含 cliff 增量,与旧轮不可比——此键让回读可扣除。
        "val_loss_seg_cliff",
    ):
        v = metrics.get(k)
        if v is None:
            continue
        out[k] = int(v) if k == "n_lesions" else round(float(v), 6)

    # ── backlog #128 的 4 项 ETE 诊断监控 ──────────────────────────────────
    # 语义/阈值见 `metrics_official.ete_monitor_metrics` docstring(出自 DR §5)。
    # 只写日志、不参与 composite、不触发 abort —— ETE 是目标四三项之一,不值得
    # 为它停掉整轮(连带分割/良恶性/位置/T 全废)。
    for k in ("ete_extreme_error_rate", "ete_kappa_gap"):
        v = metrics.get(k)
        if v is not None:
            out[k] = round(float(v), 6)
    # 🔴 **数组必须自行清 NaN**:`JsonlLogger._drop_nonfinite` 只判
    # `isinstance(v, float)`、**不遍历 list**(实测 `[0.2, NaN, 0.4]` 原样写出,
    # `json.dumps` 默认 `allow_nan=True` → 裸 `NaN` → 严格解析器**拒读整行**,
    # 而 V2.1 对"日志读不到"的判罚是「可能被标记为可疑对象」)。这两项恰恰都是数组。
    for k in ("ete_pred_hist", "ete_logit_mean_per_class"):
        v = metrics.get(k)
        if not isinstance(v, (list, tuple)):
            continue
        vals = [float(x) for x in v]
        if not all(math.isfinite(x) for x in vals):
            continue  # 整项丢弃(不填 0)—— 与 _drop_nonfinite 同一取舍:缺字段
                      # 一眼看得出异常,填数字则永远发现不了
        out[k] = [round(x, 6) for x in vals]

    # ── L1 悬崖加权观测(plan 任务L1,权重>0 才由调用方传非 None)────────
    if cliff_stats:
        for k, v in cliff_stats.items():
            if v is None:
                continue          # None 纪律同上:缺字段优于伪 0
            out[k] = v
    return out


def evaluate_seg_guards(epoch: int, val_dice, metrics: dict, epochs: int):
    """分割护栏的**纯判定**:返回 `(abort_reason, abort_message, warn_lines)`。

    🔴 **为什么抽成纯函数(2026-08-16 Gate2 lane3 指出)**:护栏是"防白烧 24 小时"的
    唯一开关,而它原本内联在 `main()` 的 epoch 循环里 —— 需要真跑训练才能触到,
    等于**零测试覆盖**。抽出来后每个分支都能用构造值直接钉死(见
    `tests/test_seg_guards.py`),包括最关键的"上一轮真实失败样本必须被抓住"。

    返回约定(三元组,刻意不复用同一个位置放两种类型 —— 那样调用方必须靠 reason
    是否为 None 来推断第二个值的类型,静态检查也过不了):
      - `reason` 非 None → 调用方须 `_abort(reason, message)`
      - `warn_lines` → 调用方逐条打印,继续训练

    判定顺序**有意如此**(先致命后诊断),各门的阈值论证见模块顶部 SEG_* 常量注释。
    """
    warns: list = []

    # ① NaN:必须最先拦。`float('nan') < 阈值` 恒 False,不在此拦下则后面所有
    #    数值门全部静默失效、一路跑满 —— 正是本轮要消灭的"跑完才发现白跑"。
    if not math.isfinite(val_dice):
        return "nonfinite_dice", (
            f"[分割 abort] epoch {epoch} val_dice={val_dice} 非有限值 —— "
            f"前向或损失数值失控。注意 `NaN < 阈值` 恒为 False,若不在此拦下,"
            f"后面所有护栏会静默失效、白跑满 {epochs} 个 epoch。"), warns

    if epoch >= SEG_WARN_EPOCH and val_dice < SEG_WARN_DICE:
        warns.append(
            f"  [分割 warn] epoch {epoch} val_dice={val_dice:.4f} < {SEG_WARN_DICE} "
            f"—— 分割可能没学起来,盯住下一轮是否上升")

    # ② 主门:预测 mask 的前景占比。判据与官方 IoU 匹配消费的量同构 ——
    #    "预测里到底有没有前景",而不是概率分布的形状(那个代理量已被实测证伪,
    #    见模块顶部 SEG_FG_FRAC_* 注释的【否掉的第二版】)。
    #    诊断量缺失(mock/异常)则跳过 —— 不拿 None 当 0 判死刑。
    fg_frac = metrics.get("seg_over_half_frac")
    if epoch >= SEG_FROZEN_EPOCH and fg_frac is not None:
        frac = float(fg_frac)
        if frac < SEG_FG_FRAC_LO:
            return "seg_no_foreground", (
                f"[分割 abort] epoch {epoch} 预测 mask 前景占比 {frac:.6g} < "
                f"{SEG_FG_FRAC_LO} —— 模型几乎一个结节都不输出"
                f"(2026-08-15 那轮就是全域 pred_prob<0.5、前景占比恒 0,跑满 35 epoch)。"
                f"预测 mask 恒空 → 官方 IoU≥0.5 匹配零命中 → 五项分类/分期全部零样本 "
                f"→ Sclin≈0,继续跑满也是白烧。检查 "
                f"schedule_official.SEG_TRAIN_PREFIXES 是否真的进了 optimizer。"), warns

    # ③ dice 兜底门(ep20 = γ=0 干净监督窗口末尾,阈值 0.02 极低不误杀)
    if epoch >= SEG_ABORT_EPOCH and val_dice < SEG_ABORT_DICE:
        return "seg_not_learning", (
            f"[分割 abort] epoch {epoch} val_dice={val_dice:.4f} < {SEG_ABORT_DICE} "
            f"—— 跑完 γ=0 的干净监督窗口(ep1-20)仍未出分割,剩余 epoch 救不回来。"
            f"注意此门是兜底:前景占比若同时正常(>1e-4),说明是在学但极慢,"
            f"回报 seg_prob_* 三项再判。"), warns

    return None, "", warns


# seg_only 门指标:分类五项降幅门。
# 🔴 2026-09-18 seg_only 调教(总纲 §6.4/P0 翻案):**val_dice≥0.72 硬门已删**——
# 当年 seg_only 判死就死在这把错尺子上(像素 dice 0.6754→0.6861 未过 0.72 门,
# 而官方口径下它是最强的:匹配 0.8469/AP 0.8149 双超 r5)。像素 dice 与官方
# IoU 匹配口径脱节,Lovász 优化的恰是 IoU。选模改为训完对版本化快照各跑
# diagnose_detection 用官方口径筛(train_official 不再单裁上线资格)。
_SEG_GATE_DROP = 0.01        # 分类五项各不降超此值
_SEG_GATE_KEYS = {
    "malignancy_auc_pr": "mal", "tirads_weighted_kappa": "tirads",
    "location_weighted_auc": "loc", "clinical_t_auc": "t",
    "ete_weighted_kappa": "ete",
}


def evaluate_seg_only_gate(val_dice, seg_baseline, metrics):
    """seg_only 六项护栏的**纯判定**(阶段2,2026-09-13;抽成纯函数+测试是
    2026-09-14 Gate2 F7-1 的要求——内联在 main 的 epoch 循环里 = 零测试覆盖,
    复刻 evaluate_seg_guards 抽出前的同型缺陷)。

    🔴 2026-09-18 调教:**val_dice≥0.72 硬门已删**(当年判死用的错尺子,
    P0 验尸实锤:官方口径下 seg_only 匹配/AP 双超 r5;选模改由训后对快照跑
    diagnose_detection 官方口径筛)。val_dice 仍作为入参保留——仅供 JSONL
    留档对照,不再参与判定。

    门 = mal/tirads/loc/t/ete 各不降超 0.01(对照 epoch=0 同源基线)。单变量
    原则下分类五项理论上梯度精确 0(全冻),此门测的是"是否真冻住"——若某项
    跌超 0.01,说明 requires_grad/param_groups 配置有泄漏(BN running stats/
    共享层等)。

    🔴 缺值(含顶层入参 None / dict 内 None)**报"无法判定"而非"已过"**:
    pass 门跳过 None 是把"测不出"当"通过"(abort 门跳过 None 才是保守),
    与 evaluate_seg_guards 的"不拿 None 当 0 判死刑"同一纪律、方向相反,别抄混。
    判定与产物的同源性:门判的是传入的那份指标,调用方传"best 那轮"或
    "末轮"由调用方决定(2026-09-14 修复轮专审 F8:末轮门+best 产物错位仍在,
    最终以 best_epoch 行 JSONL 为准)。

    Returns:
        (passed, lines):passed=None 表示无法判定(缺基线/缺当轮值)。
    """
    lines: list = []
    if seg_baseline is None or metrics is None:
        return None, ["  [SEG-only 门 无法判定] 顶层入参含 None"
                      "(seg_baseline/metrics)"]
    # val_dice 仅留档不判(2026-09-18 删 0.72 硬门);非有限值如实留档。
    _dice_note = (val_dice if (val_dice is not None
                                and math.isfinite(val_dice)) else None)
    # 🔴 NaN/+inf 与 None 同等报"无法判定"(3B 专审 P2-1:nan>0.01 恒 False、
    # baseline-inf=-inf 也不触发 bad,两种"测不出"都被判"已过"——与函数
    # docstring 的保守纪律矛盾)。当前生产链路大概率不可达(sklearn 对 NaN
    # 输入先 raise、metrics_official 的 NaN 出口有 None 守卫),此为最终
    # 所有者的防御性自持,不依赖上游守卫。
    def _finite_or_nan(v):
        return v if (v is not None and math.isfinite(v)) else float("nan")
    _missing_or_nonfinite = [
        short for mk, short in _SEG_GATE_KEYS.items()
        if seg_baseline.get(mk) is None or metrics.get(mk) is None
        or not (math.isfinite(_finite_or_nan(seg_baseline.get(mk)))
                and math.isfinite(_finite_or_nan(metrics.get(mk))))
    ]
    if _missing_or_nonfinite:
        # 逐项列出谁缺,操作者能判断是否可忽略,而不是静默放行。
        return None, [f"  [SEG-only 门 无法判定] 指标缺值/非有限:"
                      f" {_missing_or_nonfinite}"
                      f"(不把'测不出'当'通过')"]
    bad = [f"{short}:{seg_baseline[mk]:.4f}→{metrics[mk]:.4f}"
           for mk, short in _SEG_GATE_KEYS.items()
           if seg_baseline[mk] - metrics[mk] > _SEG_GATE_DROP]
    if bad:
        return False, [f"  [SEG-only 门 未过] 分类项跌超 {_SEG_GATE_DROP}: {bad}"
                       f" —— 疑似冻结泄漏,停机排查(不 abort,留权重供分析)"]
    _dice_str = (f"val_dice={_dice_note:.4f}(仅留档)" if _dice_note is not None
                 else "val_dice 缺失/非有限(仅留档)")
    return True, [f"  [SEG-only 门 已过] 分类五项均未跌超 {_SEG_GATE_DROP},"
                  f"{_dice_str};上线资格以官方口径快照筛为准"
                  f"(diagnose_detection,2026-09-18 删 dice 硬门)"]


def _abort(logger, epoch: int, reason: str, message: str):
    """abort 前先把原因写进 JSONL,再抛异常。

    🔴 **为什么不能只 raise**:异常只走 stderr,而云端 agent 回报时常常只发 JSONL。
    abort 的**原因**是最关键的一条信息(是"分割没学起来"还是"数值失控"?两者处理完全不同),
    丢了它等于让排查从零开始。`JsonlLogger` 逐行 flush,写完立刻可读。

    落盘失败不能掩盖真正的 abort 原因,故 log 本身的异常吞掉(诊断字段炸掉整轮是本末倒置)。
    """
    if logger is not None:
        try:
            logger.log(epoch=epoch, step=0, phase="val",
                       extra={"abort_reason": reason, "abort_message": message})
        except Exception:
            pass
    raise RuntimeError(message)


# ─────────────────────────────────────────────────────────────────────────────
# ROI 范式训练支路(债4 的执行半,2026-09-25 训练侧接线)
# ─────────────────────────────────────────────────────────────────────────────
# 设计(2026-09-26 R1 审查订正,勿改):ROI 训练用**独立 RoiDataset train loader**
# (与整图 train_loader 并行、各自 DataLoader),`train_step.train_batch` 不动(整图
# 支路照旧);epoch 内**先整图 run_official_epoch → 再 run_roi_epoch**。
# 🔴 梯度语义(此前注释写"两次 backward 之间无 zero_grad,梯度累积"——与代码
# 相反,订正):两支路是**串行两套独立 step**——train_batch 在每个整图 batch 内
# 自完成 zero_grad→backward→clip→scaler.step;run_roi_epoch 每个 ROI batch 同样
# 自完整一套。整图 epoch 结束时整图梯度已被消费,不存在跨支路累积;两支路
# 各自 clip(clip_grad_norm_ 都对 optimizer.param_groups 全参数集,对方支路
# 参数 p.grad is None 时天然跳过,范数分母只计本支路活跃梯度——语义安全,
# 但"梯度范数口径不独立"是事实,记录于此)。改这条时序(如真做累积)属
# 改变优化轨迹的行为改动,须单独立项+对拍。
# 损失只算 ROI 四头(整图损失在整图 epoch 已算)。

# ROI 支路 batch 大小(病灶数/batch)。K 量级:每图 1-3 病灶、约 6.6k 样本
# → 数千 ROI 条;224² 单通道前向远轻于整图 512²,batch 32 无显存压力。
_ROI_BATCH_SIZE = 32

# 🔴 E2 review P2-1 修复(2026-09-25):ROI 训练尺寸单源。此前 4 处硬编码
# 224 + 横幅误印 config.data.img_size(512)——训练 224 vs 推理 512 的口径
# 差由此被横幅掩盖。推理侧 build_roi_batch 吃 runner 的 config img_size
# (512);两侧尺寸不同是**已知域差异**(encoder+头跨尺寸未对拍),冻结前
# 若要对齐须改这里+runner 一处并跑 224-vs-512 对拍,勿只改一侧。
_ROI_TRAIN_SIZE = 224


def _meta_rows_from_collate(metas):
    """roi_collate 的 dict-of-lists → list[dict](forward_roi 契约形态)。

    债4 转置桥:collate 把数值键堆 [K] tensor、str/list 键保 list;forward_roi
    逐条 `.get()` 读每病灶的 dict。转置逻辑与
    `roi_dataset.roi_labels_from_meta` 的批形态分支同式(单源复用其转置,
    这里只是显式入口——两处必须同步,改一处查另一处)。
    """
    from src.dataset.roi_dataset import roi_labels_from_meta

    return roi_labels_from_meta(metas)      # 转置(无 gt_ 改名语义也无害:
    #   改名只作用于 5 个 label 键的键名,值原样;forward_roi 不读 gt_* 键)


def _move_roi_masks_to_device(metas, device):
    """R1 review P0-1 修(2026-09-26):meta 里的 roi_mask(CPU,RoiDataset 产)
    → device。forward_roi 的 ETE band/d_mm 消费 roi_masks 时与 roi_images
    (已 to(device))相乘——设备不一致 = RuntimeError,ROI 支路在 CUDA 上 100%
    起不来(本机真 CUDA 已复现:RoiExtensionHead(spatial_cuda, mask_cpu) 必崩)。
    测试 fixture 全 CPU + mock 无 roi_mask 键 → 全仓此前无一处让此路径在 GPU
    上跑过。**只迁 cpu 张量**(已在 device 的 no-op,防重复搬运);None/非张量
    原样跳过(缺键 = forward_roi 全零兜底语义,不是搬运失败)。
    run_roi_epoch 与 validate_roi_epoch 两处同修(共用本助手)——单修训练漏
    验证,val 同一 loader 同崩。"""
    for m in metas:
        rm = m.get("roi_mask")
        if isinstance(rm, torch.Tensor) and rm.device.type != device.type:
            m["roi_mask"] = rm.to(device)


def run_roi_epoch(model, roi_loader, optimizer, criterion, scaler, device,
                  epoch, logger, log_every, head_lr, amp_enabled=True):
    """跑一个 ROI epoch(债4:逐 batch forward_roi → 四头损失 → backward)。

    损失编排:`compute_roi_head_losses(outputs, targets, lambda_roi=...)` 只准
    import 不准改(领地红线);超参从 criterion 读(ete_focal_gamma 等——
    MultiTaskLoss 已持有同键,单一配置源)。targets 只装 `roi_labels`(桥自
    collate 后的 meta dict);outputs 只装 `roi_outputs`——本支路**直连**
    compute_roi_head_losses 而不走 criterion.forward(整图损失组件需要整图
    outputs 键,本支路不产;良恶性不在 ROI 编排内是**契约**(该 docstring),
    ROI 头的 cbm 消费链由整图支路的 malignancy 损失继续监督)。

    AMP:与整图支路同款 autocast+scaler;CPU 时调用方传 amp_enabled=False
    (train_batch 的 autocast(device_type="cuda") 在 CPU 上须关,同款约束)。

    梯度观测:返回 (avg_loss, n_batches);首个 ROI epoch 后的梯度检查在
    main 循环里做(打印 FATAL 但不停,照 cliff abort 门型但只打不停)。
    """
    from src.models.losses.components.roi_heads import compute_roi_head_losses

    model.train()
    set_frozen_modules_eval(model)
    running, n = 0.0, 0
    for batch_idx, (imgs, metas) in enumerate(roi_loader):
        imgs = imgs.to(device)
        # collate 后的 metas 是 dict-of-lists;forward_roi 契约吃 **list[dict]**
        # (multi_task_model.py docstring:roi_meta 逐条 dict,数值键解包归
        # 调用方)——用桥函数转置(债1 的同一条转置路径,meta 逐条重建)。
        _metas = _meta_rows_from_collate(metas)
        # R1 P0-1:roi_mask 上设备(见 _move_roi_masks_to_device docstring)
        _move_roi_masks_to_device(_metas, device)
        roi_labels = _roi_labels_from_meta(_metas)
        targets = {"roi_labels": roi_labels}
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=amp_enabled,
                            dtype=torch.bfloat16):
            outputs = model.forward_roi(imgs, _metas)
            # 债6(2026-09-25):传**完整**返回 dict——四头只读 roi_outputs 键
            # (多传无害),概念监督(concepts_roi)读顶层 roi_concepts 键;
            # 旧窄化 {"roi_outputs": ...} 会在 roi_concept_weight>0 时被
            # concepts_roi fail-loud 拒掉(概念 logits 在顶层,不在逐病灶 dict)。
            loss, detail = compute_roi_head_losses(
                outputs, targets,
                lambda_roi=float(getattr(criterion, "roi_head_weight", 0.0)),
                ete_class_weights=getattr(criterion, "ete_class_weights", None),
                ete_focal_gamma=float(getattr(criterion, "ete_focal_gamma", 2.0)),
                tirads_class_weights=getattr(criterion, "tirads_class_weights", None),
                tirads_label_smoothing=float(getattr(criterion, "tirads_label_smoothing", 0.0)),
                tirads_ordinal_weight=float(getattr(criterion, "tirads_ordinal_weight", 0.0)),
                concept_weight=float(getattr(criterion, "roi_concept_weight", 0.0)),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]],
            max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.item())
        n += 1
        if logger is not None and batch_idx % log_every == 0:
            logger.log(epoch=epoch, step=batch_idx, phase="train",
                       loss=float(loss.item()), lr=head_lr,
                       extra={"roi_branch": True, **_roi_detail_extra(detail)})
    return running / max(n, 1), n


def _roi_detail_extra(detail: dict) -> dict:
    """ROI 损失分项 → JSONL extra(键恒在的子集,round 6;非标量跳过)。"""
    out = {}
    for k, v in (detail or {}).items():
        try:
            out[f"loss_{k}"] = round(float(v.item() if hasattr(v, "item") else v), 6)
        except (TypeError, ValueError, RuntimeError):
            continue
    return out


def validate_roi_epoch(model, roi_val_loader, device):
    """ROI val:逐 batch forward_roi → collect_roi_records → roi_metrics(债4)。

    🔴 **ROI 指标只观测不进选模**(注释即防自作主张):composite/选模逻辑
    照旧只消费整图 val 指标(val_dice/分类五项)——ROI 指标是新头冷启的
    **观测面**(roi_* 键进 epoch 级 JSONL 供回读),掺进 composite 会让
    "整图不退化"的选模语义被未标定的新量污染(与 ete_monitor_metrics 不进
    composite 同一纪律)。

    返回 dict:`roi_loc_auc/roi_ete_kappa/roi_t_auc/roi_tirads_kappa/
    roi_mal_auc_pr/roi_n_records/roi_size`(None 透传——"算不出"不填 0,与
    _epoch_metrics_extra 的 None 纪律一致;roi_size 恒数值=训练域尺寸哨兵,
    224 域指标与 512 推理不可比,见返回处注释)。空 loader → 全 None + n=0。
    """
    from src.training.collect_roi_records import collect_roi_records, roi_metrics

    # 🔴 R1 P1 修(2026-09-26):逐模块记 training 态——只还原顶层布尔会把
    # set_frozen_modules_eval 置 eval 的冻结 BN 翻回 train,running stats 被
    # 本 val 的前向污染(ROI+L1 同开时,紧随其后的 _collect_cliff_ious 在
    # 污染态下采集)。全量快照逐模块还原,与 _collect_cliff_ious 同级防御。
    prev_modes = {id(m): m.training for m in model.modules()}
    model.eval()
    records: list = []
    try:
        with torch.no_grad():
            for imgs, metas in roi_val_loader:
                imgs = imgs.to(device)
                _metas = _meta_rows_from_collate(metas)
                # R1 P0-1 同修:val loader 同一 RoiDataset 产 CPU roi_mask
                _move_roi_masks_to_device(_metas, device)
                roi_labels = _roi_labels_from_meta(_metas)
                outputs = model.forward_roi(imgs, _metas)
                records.extend(
                    collect_roi_records(outputs["roi_outputs"], roi_labels))
    finally:
        for m in model.modules():
            m.training = prev_modes.get(id(m), m.training)
    m = roi_metrics(records)
    # 🔴 R1 P1 修(2026-09-26):口径哨兵——roi_* 六键在 224 训练域算出,
    # 与推理 512 域数值不可比(t_auc 由 d_mm 驱动/ete 由 band 驱动,均随
    # ROI 尺寸漂移)。研究者在 JSONL 侧判"ROI 头是否可用"时若拿它当
    # 512 推理一致性判据会拿到系统性偏移结论。字段钉死口径,判读免歧义。
    return {
        "roi_loc_auc": m.get("location_weighted_auc"),
        "roi_ete_kappa": m.get("ete_weighted_kappa"),
        "roi_t_auc": m.get("clinical_t_auc"),
        "roi_tirads_kappa": m.get("tirads_weighted_kappa"),
        "roi_mal_auc_pr": m.get("malignancy_auc_pr"),
        "roi_n_records": len(records),
        "roi_size": _ROI_TRAIN_SIZE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MockROI(债4 mock 半):手构小 RoiDataset,bbox 给在 64² 图上(20:40),
# 不依赖 MockOfficialDataset 改造(整图 mock 零改动)。无 GT mask 文件 →
# 无 mask_path → roi_mask 缺键 → forward_roi 全零兜底(契约仍真实可验:
# 四头前向真跑、损失真算,mask 语义=阴性)。冒烟就是验真模型前向。
# ─────────────────────────────────────────────────────────────────────────────
def _mock_roi_samples(n: int = 12, img_size: int = 64, seed: int = 0):
    """手构 roi_samples list(expand_roi_samples 产物形态)。"""
    rng = np.random.RandomState(seed)
    s = img_size
    out = []
    for i in range(n):
        # R1 P2 修(2026-09-26):bbox/global_loc 按 i 渐变——此前恒 (20,20,40,40)
        # / (30/s,30/s),loc 头的全局位置通道在 mock 冒烟里恒定,坐标分支
        # (bbox→CoordConv)的真实接线零区分力。渐变保持 64² 内、mask 语义不变
        # (无 GT mask,bbox 只驱动裁剪几何与 loc 头输入)。
        _b = 16 + (i % 8)
        x0 = y0 = _b
        x1 = y1 = _b + 20
        out.append({
            "img_path": f"mock::roi::{i}",
            "is_dynamic": False,
            "frame_index": None,
            "patient_key": f"mockpid::{i % 4}",
            "patient_id": f"mockpid::{i % 4}",
            "lesion_id": f"mockroi::{i}",
            "bbox_xyxy": (x0, y0, x1, y1),
            "orig_w": s, "orig_h": s,
            "spacing_mm": None,
            "global_loc": ((x0 + x1) / 2.0 / s, (y0 + y1) / 2.0 / s),
            "mask_path": None,      # mock 无 GT mask → 键缺失路(forward_roi 兜底)
            "malignancy": int(rng.randint(0, 2)), "valid_malignancy": 1,
            "lesion_loc": int(rng.randint(0, 7)), "valid_lesion_loc": 1,
            "extrathy_ext": int(rng.randint(0, 4)), "valid_extrathy_ext": 1,
            "clin_t_stage": int(rng.randint(0, 2)), "valid_clin_t_stage": 1,
            "tirads": int(rng.randint(0, 5)), "valid_tirads": 1,
            "is_main": 1 if i % 4 != 3 else 0,
        })
    return out


def _write_mock_dcm(path: str, arr: np.ndarray):
    """写最小可读 MONOCHROME2 8bit DICOM(测试侧 _make_dcm 同型,内联防 import)。

    mock RoiDataset 的 `_read_frame` 走 `_read_dcm_gray`(pydicom),PNG/裸字节
    会被当坏 DICOM 拒读——mock 图像必须是真 DICOM。
    """
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    ds = Dataset()
    ds.Rows, ds.Columns = int(arr.shape[0]), int(arr.shape[1])
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.PixelData = arr.astype(np.uint8).tobytes()
    fm = FileMetaDataset()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.MediaStorageSOPClassUID = generate_uid()
    fm.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta = fm
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    try:
        ds.save_as(path, write_like_original=False)          # pydicom 2.x
    except TypeError:                                        # pydicom 3.x
        ds.save_as(path, enforce_file_format=True)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="官方四头 A/B/C 调度训练")
    parser.add_argument("--config", default="configs/thyroid_config.yaml")
    parser.add_argument("--data-root", default=None, help="官方 annotation 根目录")
    parser.add_argument("--xlsx", default=None, help="官方标注 xlsx 路径")
    parser.add_argument(
        "--samples-pkl", default=None,
        help="预抽取产物 <cache-dir>/samples.pkl(scripts/data/preextract_dynamic.py 生成)。"
             "给了就跳过 10-25 分钟的样本重建,动态样本改读 PNG 缓存而非全量解压 .nii.gz")
    parser.add_argument("--epochs", type=int, default=STAGE_C_END, help=f"默认 {STAGE_C_END}(A/B/C 全程)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="头 lr 基准;缺省取 config.train.lr")
    parser.add_argument("--out", default="checkpoints/official_heads_best.pth")
    parser.add_argument("--resume-from", default=None, help="分割底座 checkpoint(warm-start)")
    parser.add_argument("--hflip-prob", type=float, default=0.0, help="水平翻转概率(联动翻位置标签)")
    parser.add_argument("--freeze-legacy", action="store_true",
                        help="关 malignancy AUCM + vCLUB 更新,纯头训练(默认保留守旧任务)")
    parser.add_argument("--ete-only", action="store_true",
                        help="ETE 头小试(D10-15):冻结全部仅解冻 extension_head,"
                             "lr 取 Stage C 低档,配 --resume-from r5 权重热启动训 6-8ep")
    parser.add_argument("--seg-only", action="store_true",
                        help="阶段2 分割微调:冻结四头/cls_backbone/Adaptor,仅训分割"
                             "前向链(decoder@head_lr + seg_backbone@×0.1),配 Boundary/"
                             "Lovász 损失(config.train.seg_boundary_weight/seg_lovasz_weight),"
                             "必须 --resume-from r5 + epoch≤20(γ_prior 边界)+ 版本化 --out")
    parser.add_argument("--union-mask", action="store_true",
                        help="L2 union 监督(plan 任务 L2):同图全部 GT 病灶 mask 并集"
                             "作分割监督。须配 --samples-pkl 为 union 版 preextract 产物"
                             "(loader 对旧 pkl + union 硬拒)")
    parser.add_argument("--roi-heads", action="store_true",
                        help="ROI 范式训练:独立 RoiDataset train/val loader + "
                             "forward_roi 支路(epoch 内先整图后 ROI),ROI 头冷启 "
                             "46 键(warm-start 白名单)。须重建 roi 版 samples.pkl"
                             "(build_official_datasets roi_meta=True;旧 pkl 无 "
                             "roi_lesions 键 → ROI loader 0 样本,入口硬拒)或走 "
                             "直建路径(不传 --samples-pkl)")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log-dir", default=OFFICIAL_LOG_DIR,
                        help=f"JSONL 日志目录;默认云端 {OFFICIAL_LOG_DIR},本地传 logs")
    parser.add_argument("--mock-labels", action="store_true",
                        help="合成数据集(无官方数据),验管道数值健康")
    parser.add_argument("--smoke", action="store_true", help="冒烟:少量样本 + 3 epoch")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config["environment"].get("seed", 42))
    # 官方 split 已在 build_official_datasets 内做患者分组,训练用 shuffle=True 即可,
    # 不走 WeightedRandomSampler,故 sampler generator 弃用(下划线),只留 loader generator。
    _, loader_gen = configure_reproducibility(seed, args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or config["data"]["batch_size"]
    img_size = config["data"]["img_size"]
    lr_base = args.lr if args.lr is not None else config["train"]["lr"]
    epochs = 3 if args.smoke else args.epochs
    # CPU 无 AMP:train_batch 的 autocast(device_type="cuda") 在 CPU 上须关。
    if device.type == "cpu":
        config["train"]["amp"] = False
    print(f"[train_official] device={device} bs={batch_size} img={img_size} "
          f"lr_base={lr_base} epochs={epochs} mock={args.mock_labels} freeze_legacy={args.freeze_legacy}"
          f"{' ETE_ONLY' if args.ete_only else ''}{' SEG_ONLY' if args.seg_only else ''}")
    # 🔴 损失权重无条件打进横幅(Gate2 F7-2,2026-09-14):这两个权重经
    # build_official_head_config 无条件进 criterion,**不挂在 --seg-only 旗标下**
    # ——阶段2 把 config 权重改成非 0 后,任何不带 --seg-only 的重训也会静默
    # 消费它们(r7 同型泄漏),必须横幅可见。
    _hc = config["train"]
    # 🔴 R2 P1-2 修(2026-09-26):roi_concept_weight 必须进横幅——它是 ROI 支路
    # 第二个实验变量,漏印 = "yaml 改错无人见"(r7 config 静默不读事故同型;
    # 且概念监督当前**结构性恒空**,见下方 ROI 门提示,横幅是唯一首屏防线)。
    print(f"[seg 损失配方] boundary_weight={_hc.get('seg_boundary_weight', 0.0)} "
          f"lovasz_weight={_hc.get('seg_lovasz_weight', 0.0)} "
          f"ccdice_weight={_hc.get('seg_ccdice_weight', 0.0)} "
          f"deepsup_weight={_hc.get('seg_deepsup_weight', 0.0)} "
          f"cliff_weight={_hc.get('seg_cliff_weight', 0.0)} "
          f"roi_head_weight={_hc.get('roi_head_weight', 0.0)} "
          f"roi_concept_weight={_hc.get('roi_concept_weight', 0.0)}"
          f"{' (SEG_ONLY 实验变量)' if args.seg_only else ''}")
    # 🔴 ETE 小试的三个启动硬门(Gate2 审查 F1/F2/F4,2026-09-12):
    # ① --ete-only 必须配 --resume-from(r5 全量权重)——漏传则 ETE 头从随机
    #    初始化训,6-8ep 白烧且"其余四项崩"会被误判为模式有害;
    # ② --out 必须显式版本化——默认路径是冒烟产物默认名,与真实产物同名同量级
    #    分不出来(memory: weights-inventory 记过同类误判);
    # ③ ETE 损失配方打进启动横幅——损失是 ete_only 唯一实验变量,云 config 是
    #    可变状态(r7 事故),配方不对横幅/JSONL 当场可见,不用等互斥键事后判。
    if args.ete_only:
        if not args.resume_from:
            parser.error("--ete-only 必须配 --resume-from(r5 全量权重热启动);"
                         "漏传则 ETE 头随机初始化,整轮白烧")
        if args.out == "checkpoints/official_heads_best.pth":
            parser.error("--ete-only 必须显式传版本化 --out(默认路径是冒烟产物名,"
                         "与真实产物分不出来)")
        ete_cfg = config["train"]
        print(f"[ETE-only 配方] cb_beta={ete_cfg.get('ete_cb_beta')} "
              f"focal_gamma={ete_cfg.get('ete_focal_gamma')} "
              f"la_tau={ete_cfg.get('ete_la_tau')} "
              f"qwk_weight={ete_cfg.get('ete_qwk_weight')} "
              f"class_counts={ete_cfg.get('ete_class_counts')} "
              f"(应=0.9/0.0/0.0/0.0/[4459,333,81,5] Q7 新表)")

    # 🔴 seg_only 三硬门(阶段2,照 ete_only 门型,2026-09-13):
    # ① 必须 --resume-from(r5 全量权重)——漏传则分割链以外的四头/分类表征
    #    从随机初始化出发,epoch=0 基线无意义、门"其余不降"必炸;
    # ② 必须版本化 --out(默认路径是冒烟产物名,与真实产物同名同量级分不出;
    #    basename 判,反斜杠/绝对路径/./ 前缀写法不绕过——Gate2 2026-09-14);
    # ③ epoch ≤ SEG_ABORT_EPOCH(=20):γ_prior 在 ep≤20 恒 0(分类→分割非
    #    detach 通路梯度精确 0,单变量成立);ep21 起 γ>0,Stage2 崩塌机制混入
    #    (soft_mask→prior 通路全强度回传)。用常量而非字面量 20——同文件
    #    SEG_ABORT_EPOCH 与 γ 退火起点本就同源,:428 注释明言"重排必须同步改"。
    # 损失配方横幅:Boundary/Lovász 权重是本实验唯一变量,config 是云端可变
    # 状态(r7 事故:增量包漏带 config 整轮跑在已证伪分支),配方必须打进横幅
    # (已上移到无条件区,非 seg_only 轮也可见)。
    if args.seg_only:
        if args.ete_only:
            parser.error("--seg-only 与 --ete-only 互斥")
        if not args.resume_from:
            parser.error("--seg-only 必须配 --resume-from(r5 全量权重热启动)")
        if os.path.basename(args.out) == "official_heads_best.pth":
            parser.error("--seg-only 必须显式传版本化 --out(默认路径是冒烟产物名)")
        if epochs > SEG_ABORT_EPOCH:
            parser.error(f"--seg-only 预算上限 {SEG_ABORT_EPOCH} epoch(γ_prior ep≤"
                         f"{SEG_ABORT_EPOCH} 恒 0 的单变量窗口,常量同源防漂移),"
                         f"got {epochs}")
        seg_cfg = config["train"]
        bw = seg_cfg.get("seg_boundary_weight", 0.0)
        lw = seg_cfg.get("seg_lovasz_weight", 0.0)
        cw = seg_cfg.get("seg_ccdice_weight", 0.0)
        dw = seg_cfg.get("seg_deepsup_weight", 0.0)
        l1w = float(seg_cfg.get("seg_cliff_weight", 0.0))
        # 🔴 量级标定豁免(2026-09-14 云端实证:标定=权重 0 跑 1ep 读分项基线,
        # 与此门"防全 0 空微调"冲突,原版把标定也拦了)。epochs==1 视为标定
        # 放行——空微调白烧至少要 2ep 才成立,1ep 的产物本就是丢弃的探针。
        if (args.smoke is False and epochs > 1
                and bw == 0.0 and lw == 0.0 and cw == 0.0 and dw == 0.0
                and l1w == 0.0):
            # 冒烟/标定允许全 0;真训全 0 = 空微调白烧(r7 同型)。
            parser.error("--seg-only 真训须 config.train.seg_boundary_weight / "
                         "seg_lovasz_weight / seg_ccdice_weight / "
                         "seg_deepsup_weight / seg_cliff_weight 至少一项 > 0"
                         "(全 0 = 空 微调白烧;"
                         "权重 0 的 1ep 量级标定放行)")
        print(f"[SEG-only 配方] boundary_weight={bw} lovasz_weight={lw} "
              f"ccdice_weight={cw} deepsup_weight={dw} cliff_weight={l1w} "
              f"epochs={epochs} (门:mal/tirads/loc/t/ete "
              f"各不降超 0.01,对照 epoch=0 基线;🔴 dice 硬门已删 2026-09-18,"
              f"上线资格以训后快照官方口径筛为准)")

    # ── --roi-heads 四硬门(债4 接线,2026-09-25;照 seg_only 门型)──────────
    # ① 与 --seg-only/--ete-only 互斥:两个 only 模式冻结一切只留单链,
    #    ROI 头 requires_grad=False → forward_roi 支路零梯度 = 白烧;
    # ② 须 --resume-from:ROI 头冷启但 **backbone 必须热启**(随机 backbone
    #    上冷启 46 键新头,整轮白烧)。mock 冒烟豁免(合成数据无真底座可热启,
    #    冒烟只验管道数值健康——豁免理由与 ETE/seg_only 门对 mock 的处理同型);
    # ③ 须版本化 --out(basename 判,反斜杠/绝对路径不绕过);
    # ④ 真训(非 smoke 非 1ep)须 config.train.roi_head_weight > 0:
    #    全 0 = ROI 支路算了损失但 λ=0(精确零梯度)= 空 ROI 微调白烧
    #    (compute_roi_head_losses 权重 0 早退,照 seg_only 全 0 门同型)。
    if args.roi_heads:
        if args.seg_only or args.ete_only:
            parser.error("--roi-heads 与 --seg-only/--ete-only 互斥"
                         "(两 only 模式冻结一切只留单链,ROI 头无梯度 = 白烧)")
        if not args.resume_from and not args.mock_labels:
            parser.error("--roi-heads 必须配 --resume-from(ROI 头冷启但 backbone "
                         "必须热启,否则整轮白烧);mock 冒烟豁免(合成数据无真底座)")
        if os.path.basename(args.out) == "official_heads_best.pth":
            parser.error("--roi-heads 必须显式传版本化 --out(默认路径是冒烟产物名,"
                         "与真实产物分不出来)")
        roi_w = float(config["train"].get("roi_head_weight", 0.0))
        if args.smoke is False and epochs > 1 and roi_w <= 0.0:
            parser.error("--roi-heads 真训须 config.train.roi_head_weight > 0"
                         "(全 0 = 空 ROI 微调白烧:损失早退 λ=0 精确零梯度;"
                         "权重 0 的 1ep 量级标定放行,照 seg_only 门型)")
        # 🔴 R2 P1-2 修(2026-09-26):概念监督健康判据进启动面——
        # roi_concept_weight>0 时,训练 JSONL 的 concept_n_valid_* 恒 0
        # = 概念标签链断(当前**已知结构性恒空**:官方表无独立概念列,
        # 信息只在 TI-RADS详细评分 复合串里,详见 official_loader
        # _parse_concept 注释)。首屏提示判据,防 20ep 白烧无人知。
        concept_w = float(config["train"].get("roi_concept_weight", 0.0))
        if concept_w > 0.0:
            print(f"[ROI-heads 配方] ⚠️ roi_concept_weight={concept_w}>0:"
                  "官方表无独立概念列(成分/回声/形状/边缘 候选名实测不命中),"
                  "概念监督大概率恒空——concept_n_valid_* 恒 0 即坐实,"
                  "20ep 概念头零学习,建议权重归 0 再发轮")
        print(f"[ROI-heads 配方] roi_head_weight={roi_w} "
              f"roi_concept_weight={concept_w} "
              f"roi_size={_ROI_TRAIN_SIZE}(训练;推理=512 已知域差异) "
              f"epochs={epochs}"
              f"(ROI 头冷启46键,债3白名单;epoch 内先整图后 ROI 双支路)")

    # ── --out 可写性预检(fail-fast,必须在任何耗时操作之前)──────────────────
    # 为什么要有:`torch.save` 失败会抛异常直接中断训练,而保存只发生在每轮 val
    # 之后 —— 35 epoch 跑到第 30 轮才发现写不进去,前面全部白跑。
    # 为什么位置这么靠前:再往后就是建数据集 → 建模型 → warm-start 加载 121MB
    # 底座 → 跑一轮 baseline validate,云上合计好几分钟;路径不可写这件事在
    # 第 0 秒就能知道,没有任何理由让它等到那之后。
    # 为什么不能只靠 makedirs:2026-08-13 A800 实测复现过一个 makedirs 完全测
    # 不出的场景 —— `checkpoints/` 目录 owner 是当前用户、可写(makedirs 通过),
    # 但目标文件 official_heads_best.pth 的 owner 是 root、模式 644,torch.save
    # 走 open(w) 覆盖时 Permission denied。**目录可写 ≠ 目标文件可写**。
    # 为什么用 "ab" 而不是 "wb":append 模式不截断,已存在的权重文件零风险;
    # 只有探针自己新建的 0 字节文件才删除(size==0 判据),绝不误删已有权重。
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    try:
        with open(args.out, "ab"):
            pass
        if os.path.getsize(args.out) == 0:
            os.remove(args.out)
    except OSError as e:
        raise SystemExit(
            f"[FATAL] --out 不可写:{args.out}({type(e).__name__}: {e})。"
            f"训练前置检查失败,拒绝空跑 —— 若等到训练结束才发现,整轮算力全废。"
        )
    # 快照路径同一预检(P2 修复,2026-09-18 审查):预检只探 args.out 本身,
    # 快照路径坏(上轮只读残留)要到 ep4 才炸,前面全废。探针语义与 --out 同型
    # ("ab" 不截断、仅 0 字节新建才删);仅 seg_only 真训需要,其余轮次零成本跳过。
    if args.seg_only and args.smoke is False:
        for _se in _SEG_SNAPSHOT_EPOCHS:
            if _se > epochs:
                break  # 快照点在预算外(短训),不预检不落盘
            _sp = _snapshot_path(args.out, _se)
            try:
                with open(_sp, "ab"):
                    pass
                if os.path.getsize(_sp) == 0:
                    os.remove(_sp)
            except OSError as e:
                raise SystemExit(
                    f"[FATAL] seg_only 快照路径不可写:{_sp}"
                    f"({type(e).__name__}: {e})。第 {_se} epoch 才发现 = 前面全废,"
                    f"拒绝开训(修掉上轮只读残留或换 --out)。"
                )

    # ── 数据 ──────────────────────────────────────────────────────────────
    if args.mock_labels:
        n_train = 16 if args.smoke else 64
        n_val = 8 if args.smoke else 24
        train_ds = MockOfficialDataset(n=n_train, img_size=img_size, seed=seed)
        val_ds = MockOfficialDataset(n=n_val, img_size=img_size, seed=seed + 100)
        # ROI 支路 mock(债4):手构 MockROI(64² 图 bbox 按 i 渐变,无 GT mask →
        # roi_mask 缺键 forward_roi 兜底)。冒烟就是验真模型前向,故用真
        # MultiTaskThyroidModel 的 forward_roi 跑(不用 FakeModel)。
        roi_train_samples = _mock_roi_samples(
            n=(12 if args.smoke else 32), img_size=64, seed=seed)
        roi_val_samples = _mock_roi_samples(
            n=(6 if args.smoke else 16), img_size=64, seed=seed + 100)
    else:
        if not args.data_root or not args.xlsx:
            parser.error("非 --mock-labels 模式必须给 --data-root 与 --xlsx")
        from src.dataset.augmentation import get_train_transforms
        from src.dataset.official_loader import build_official_datasets
        # roi_meta=True(债4 数据半):样本携带 roi_lesions,供 ROI loader 展开。
        # 🔴 与 samples_pkl 组合时 pkl 载入不重建 → 无 roi_lesions 键 →
        # ROI loader 0 样本(下方硬拒,不静默空跑)。
        train_ds, val_ds = build_official_datasets(
            args.data_root, args.xlsx, img_size=img_size, seed=seed,
            train_transform=get_train_transforms(), hflip_prob=args.hflip_prob,
            samples_pkl=args.samples_pkl,
            union_mask=args.union_mask,
            roi_meta=args.roi_heads,
        )
        roi_train_samples = None      # 真数据路径:ROI 样本从 val_ds/train_ds 提取
        roi_val_samples = None
    num_workers = 0 if (args.smoke or device.type == "cpu") else config["environment"]["num_workers"]
    # persistent_workers(2026-09-12):num_workers>0 时保持 worker 进程跨 epoch
    # 存活,省每轮的进程池重建(12 workers ≈ 每 epoch 2-5s × 35ep)。
    # 🔴 必须 _persist 守卫:num_workers=0 时传 True 直接 ValueError。
    _persist = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), worker_init_fn=seed_worker, generator=loader_gen,
        persistent_workers=_persist,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), worker_init_fn=seed_worker,
        persistent_workers=_persist,
    )

    # ── ROI 支路 loader(债4;--roi-heads 才建,默认 None 零回归)────────────
    roi_train_loader = None
    roi_val_loader = None
    if args.roi_heads:
        from src.dataset.roi_dataset import RoiDataset, roi_collate, expand_roi_samples
        if args.mock_labels:
            # 🔴 mock RoiDataset 需要真实可读 img_path:_read_frame 走
            # _read_dcm_gray(pydicom)——PNG 会被当 DICOM 读崩。mock 图像内容
            # 无关紧要,直接写**最小可读 DICOM**(复用测试侧 _make_dcm 同型
            # 手构,但测试文件不可 import——此处内联最小版)到 log 目录旁。
            _mock_roi_dir = os.path.join(args.log_dir, "mock_roi_imgs")
            os.makedirs(_mock_roi_dir, exist_ok=True)
            roi_train_samples = _mock_roi_samples(
                n=(12 if args.smoke else 32), img_size=64, seed=seed)
            roi_val_samples = _mock_roi_samples(
                n=(6 if args.smoke else 16), img_size=64, seed=seed + 100)
            for _s in roi_train_samples + roi_val_samples:
                _p = _s["img_path"]
                if not _p.startswith("mock::"):
                    continue
                _fp = os.path.join(_mock_roi_dir, _p.split("::")[-1] + ".dcm")
                if not os.path.exists(_fp):
                    _write_mock_dcm(_fp, np.full((64, 64), 120, np.uint8))
                _s["img_path"] = _fp
            roi_train_ds = RoiDataset(roi_train_samples, roi_size=_ROI_TRAIN_SIZE)
            roi_val_ds = RoiDataset(roi_val_samples, roi_size=_ROI_TRAIN_SIZE)
        else:
            # 真数据:整图 train/val Dataset 的 samples 经 expand_roi_samples
            # 展开(患者分组切分已在 build_official_datasets 内做,ROI 样本
            # 继承同一切分 → 无跨集泄漏)。
            roi_train_ds = RoiDataset(
                expand_roi_samples(train_ds.samples), roi_size=_ROI_TRAIN_SIZE)
            roi_val_ds = RoiDataset(
                expand_roi_samples(val_ds.samples), roi_size=_ROI_TRAIN_SIZE)
            # 🔴 旧 pkl 硬拒(任务书:不是静默空跑):0 样本 = pkl 无
            # roi_lesions 键(半开态)——"号称开了 ROI 实际 0 样本"与
            # --limit 残件同类,须重建 roi 版 samples.pkl 或走直建路径。
            if len(roi_train_ds) == 0:
                parser.error(
                    "--roi-heads 但 ROI train loader 0 样本 —— "
                    f"{'--samples-pkl ' + str(args.samples_pkl) + ' 是旧产物(无 roi_lesions 键)' if args.samples_pkl else '样本构造为空'};"
                    "重建 roi 版 samples.pkl(preextract 重跑)或去 pkl 用直建路径")
        _roi_workers = 0 if (args.smoke or device.type == "cpu") else num_workers
        roi_train_loader = DataLoader(
            roi_train_ds, batch_size=_ROI_BATCH_SIZE, shuffle=True,
            num_workers=_roi_workers, pin_memory=(device.type == "cuda"),
            worker_init_fn=seed_worker, generator=loader_gen,
            collate_fn=roi_collate,
            persistent_workers=_roi_workers > 0,
        )
        roi_val_loader = DataLoader(
            roi_val_ds, batch_size=_ROI_BATCH_SIZE, shuffle=False,
            num_workers=_roi_workers, pin_memory=(device.type == "cuda"),
            worker_init_fn=seed_worker, collate_fn=roi_collate,
            persistent_workers=_roi_workers > 0,
        )
        print(f"--> [ROI loader] train={len(roi_train_ds)} 条 / "
              f"val={len(roi_val_ds)} 条病灶级样本(batch={_ROI_BATCH_SIZE})")

    # ── 模型 + 损失 ───────────────────────────────────────────────────────
    # mock/冒烟不加载 ImageNet 预训练(省时,合成数据无意义);真训按 config.
    pretrained = config["model"]["pretrained"] and not args.mock_labels
    model = MultiTaskThyroidModel(
        backbone_name=config["model"]["backbone"], pretrained=pretrained, config=config,
    ).to(device)
    criterion = MultiTaskLoss(official_head_config=build_official_head_config(config)).to(device)
    vclub_optimizer = torch.optim.Adam(model.vclub.parameters(), lr=1e-4)
    # torch.amp.GradScaler(新 API):`torch.cuda.amp.GradScaler` 在 2.8 已 deprecated
    # (docs/交付与部署/torch2.8迁移方案.md:80 登记的待迁移项;cbm_head.py:37/52 是样板)。
    # 云端就是 torch2.8,新入口直接用新 API,少留一处债。传 device.type 而非硬编码
    # 'cuda':本地 CPU 冒烟时不会刷"CUDA 不可用"的无谓警告。
    # `# type: ignore`:torch 上游没把 GradScaler 放进 torch.amp.__all__(2.5.1/2.8.0
    # 实测均为 False),Pyright 会报 reportPrivateImportUsage —— 但两版**运行时都可用**
    # (已实测构造成功)。压法沿用 cbm_head.py:37/52 的既有样板。
    scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
        device.type, enabled=config["train"].get("amp", True)
    )

    if args.resume_from:
        load_warm_start(model, args.resume_from, device)

    # ── 冒烟:先查梯度健康,再跑短训练循环 ──────────────────────────────────
    if args.smoke:
        smoke_grad_health(model, criterion, device)

    # ── Dice 护栏基线(warm-start 后测一次;mock 无意义则跳过)──────────────
    dice_baseline = None
    seg_baseline = None
    if not args.mock_labels:
        base_metrics: dict = {}
        _, dice_baseline, _, _, _ = validate_epoch(
            model, val_loader, criterion, device, epoch=1, official_metrics_out=base_metrics
        )
        print(f"--> [护栏] warm-start val Dice 基线 = {dice_baseline:.4f}")
        # 🔴 ETE 小试基线留档(Gate2 审查 F3,2026-09-12):门"ETE≥0.24 且其余
        # 四项各不降超 0.01"需要同权重同 split 的逐指标基线,而这正是 warm-start
        # 后第一次 val(r5 权重本身)。此前只取 dice 丢弃其余——门只能拿文档里
        # 不同 epoch 的峰值当基线,口径不可比。收集进 _baseline_pending,
        # 待正式 logger 建好后以 epoch=0/phase=val 写 JSONL。
        _baseline_row = {"note": "warm-start (pre-finetune) r5 metrics"}
        _baseline_row.update({k: v for k, v in base_metrics.items()
                              if isinstance(v, (int, float))})
        _baseline_pending = _baseline_row
        # 🔴 seg_only 护栏的分类五项基线(阶段2,2026-09-13):门=
        # "mal/tirads/loc/t/ete 各不降超 0.01(对照 epoch=0)"
        # 需要逐指标基线,与 ETE 小试同源(epoch=0/phase=val 已写 JSONL,上面)。
        # 此处额外钉运行时断言变量(逐 epoch 判降幅用)。
        if args.seg_only:
            # 🔴 键清单从 _SEG_GATE_KEYS 单源派生(3B 专审 P2-3:两份手工
            # 列表改一处漏一处 → 门静默退化"无法判定");composite 仅作
            # 参考记录(门不消费)。
            seg_baseline = {**{k: base_metrics.get(k) for k in _SEG_GATE_KEYS},
                            "composite": base_metrics.get("composite")}
            print(f"--> [SEG-only 护栏基线] {seg_baseline}")
    else:
        _baseline_pending = None

    logger = JsonlLogger(log_dir=args.log_dir, data_source="official/train_v1")
    if _baseline_pending is not None:
        # phase 走合法枚举 "val"(epoch=0 标识 warm-start 基线),指标全进 extra
        # (规范必填字段一个不少,基线指标是额外诊断键不违规)。
        _extra = {k: v for k, v in _baseline_pending.items()
                  if k not in ("phase", "epoch", "note")}
        # mode 只有 'training'/'inference' 两个合法值(jsonl_logger._MODES),
        # 基线行属训练侧产物故取 'training'(首版误写 'validation' 撞 assert,
        # 冒烟走 mock 路径根本不进这段,本地测不出——云端首跑才炸)。
        logger.log(epoch=0, step=0, phase="val", mode="training",
                   extra={"baseline_note": _baseline_pending["note"], **_extra})
        print(f"--> [护栏] baseline 指标已写 JSONL(epoch=0/phase=val): "
              f"{ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in _extra.items()} }")
    log_every = 1 if args.smoke else 20
    running_losses_history: list = []
    optimizer = None
    prev_stage = None
    best_composite = -1.0
    saw_head_grad = False
    # ROI 支路状态(债4):权重从 config 透传读(criterion 同源);梯度检查
    # 只在首个 ROI epoch 后做一次;前缀动态读模型常量(单源)。
    _roi_weight = float(build_official_head_config(config).get("roi_head_weight", 0.0))
    _roi_grad_checked = False
    _ROI_PFX = MultiTaskThyroidModel.ROI_HEAD_PREFIXES
    # L1 悬崖加权(权重>0 才激活;_last_cliff_stats 进 epoch 级 JSONL extra)
    _cliff_enabled = float(
        build_official_head_config(config).get("seg_cliff_weight", 0.0)) > 0.0
    _last_cliff_stats: dict = {}

    try:
        for epoch in range(1, epochs + 1):
            param_groups, head_lr, stage_name = configure_official_stage(
                model, criterion, epoch, lr_base, config["train"]["weight_decay"],
                freeze_legacy=args.freeze_legacy, ete_only=args.ete_only,
                seg_only=args.seg_only, roi_heads=args.roi_heads,
            )
            cur_stage = get_official_stage(epoch)
            if args.ete_only:
                cur_stage = "ETE"   # 恒定单阶段:全程不重建 optimizer,动量跨 epoch 承接
            if args.seg_only:
                cur_stage = "SEG"   # 同上:恒定单阶段(seg_only 不随 epoch 变参数集)
            # 阶段边界重建 optimizer(参数集合变了);阶段内只更 lr(承接动量)。
            if optimizer is None or cur_stage != prev_stage:
                optimizer = torch.optim.AdamW(param_groups, lr=head_lr)
                print(f"  [optimizer rebuilt] {stage_name} head_lr={head_lr:.2e}")
            else:
                # 参数集合不变,逐组按新 lr 更新(param_groups 顺序与上次一致)。
                for g, ng in zip(optimizer.param_groups, param_groups):
                    g["lr"] = ng["lr"]
            prev_stage = cur_stage

            avg_loss = run_official_epoch(
                model, train_loader, optimizer, vclub_optimizer, criterion, scaler,
                device, config, running_losses_history, epoch, args.freeze_legacy,
                logger, log_every, head_lr,
            )
            saw_head_grad = True  # run_official_epoch 跑通即四头参与了 backward

            # ── ROI 支路(债4,--roi-heads 且权重>0 才跑;先整图后 ROI)──────
            # 梯度语义见 :876 设计注释(2026-09-26 订正):两支路串行两套
            # 独立 step,无跨支路累积。ROI 头梯度观测:首个 ROI epoch 后检查
            # 任一 roi_*_head 参数 .grad 非空(零 = 接线断,打印 FATAL 但不停
            # ——照 cliff abort 门型但只打不停,云上人判)。
            # 🔴 R1 P0-1 已修(2026-09-26):run/validate_roi_epoch 内
            # _move_roi_masks_to_device(此前 roi_mask 恒 CPU,ROI 支路在 CUDA
            # 上 100% 必崩,测试 fixture 全 CPU 测不出)。
            # 🔴 mock 冒烟下 ROI 支路权重=0(config 默认)→ 不跑 ROI epoch,
            # 冒烟只验整图管道 + ROI loader/前向冒烟在 smoke_grad_health 外
            # 不覆盖——专门验前向的入口是 --roi-heads 真训;mock 冒烟要跑
            # ROI 支路须临时改 config 权重(冒烟不走 0 权重硬门)。
            roi_extra = None
            if roi_train_loader is not None and _roi_weight > 0.0:
                roi_avg_loss, _n_roi = run_roi_epoch(
                    model, roi_train_loader, optimizer, criterion, scaler,
                    device, epoch, logger, log_every, head_lr,
                    amp_enabled=bool(config["train"].get("amp", True))
                    and device.type == "cuda")
                if not _roi_grad_checked and epoch >= 1:
                    _roi_grad_checked = True
                    _grads_ok = any(
                        p.grad is not None and p.grad.abs().sum().item() > 0
                        for n, p in model.named_parameters()
                        if n.startswith(_ROI_PFX))
                    if not _grads_ok:
                        print(f"[FATAL] ROI 头梯度全零(epoch {epoch})——"
                              f"forward_roi/损失/optimizer 接线断,后续 epoch "
                              f"全部白烧;不自动停机(云上人判),排查 "
                              f"ROI_HEAD_PREFIXES/param_groups/run_roi_epoch")
                # ROI val(只观测不进选模,见 validate_roi_epoch docstring)
                roi_extra = validate_roi_epoch(model, roi_val_loader, device)
                print(f"  [ROI] ep{epoch} loss={roi_avg_loss:.4f} "
                      + " ".join(f"{k}={v}" for k, v in roi_extra.items()))

            # ── L1 悬崖加权:epoch 尾采前代 IoU 回填缓存(plan 任务L1)──────
            # 顺序在 train 之后、validate 之前:本 epoch 刚训完的权重才是
            # "前代";validate 的 model.eval 由 validate_epoch 自管。首 epoch
            # 也采(resume-from ep16 的 ep1 训练结束后即可采,下一 epoch 用;
            # ep1 训练中仍全 w=1,无损失)。权重 0 时零成本跳过。
            if _cliff_enabled:
                criterion.set_cliff_ious(
                    _collect_cliff_ious(model, train_loader, device))
                _last_cliff_stats = _cliff_stats_from_ious(
                    criterion._cliff_iou_prev)
                # 🔴 开训硬判据的执行物(2026-09-24 审查路C P0 修):任务书声称
                # "ep≥2 起 frac 恒 0 = 链断停机",此前只写 JSONL 无人消费,20ep
                # 空转零告警。frac=0.0(非 None)且分母>0 = 采集/注入链断的
                # 确定性信号(lesion_id 同池自采自用 100% 命中,恒 0 只可能是
                # 链路坏),按任务书停机。ep1 免检:首 epoch 训练中仍全 w=1,
                # 但 ep1 尾已采完,frac 应非 0;None(全哨兵/空缓存)不判死刑。
                if (epoch >= 2 and _last_cliff_stats.get("cliff_frac_boosted") == 0.0
                        and (_last_cliff_stats.get("cliff_n_evaluated") or 0) > 0):
                    _abort(logger, epoch, "cliff_chain_broken",
                           "[cliff abort] epoch {ep} cliff_frac_boosted=0 但分母 "
                           "{n}>0 —— 采集/注入链断(同池自采自用不应恒 0),"
                           "20ep 会空转,停机排查 set_cliff_ious/lesion_id 接线"
                           .format(ep=epoch,
                                   n=_last_cliff_stats.get("cliff_n_evaluated")))

            metrics: dict = {}
            val_loss, val_dice, val_acc, _, val_auc = validate_epoch(
                model, val_loader, criterion, device, epoch=epoch, official_metrics_out=metrics
            )
            composite = metrics.get("composite")
            # 🔴 epoch 级指标必须进 JSONL,不能只 print(2026-08-15 血泪):`print` 走
            # stdout,不加 `python -u` 时是 8KB 块缓冲 —— 上一轮真训攒到 ep25 才第一次
            # 刷新,全程等于瞎跑。JsonlLogger 逐行 flush、不受管道缓冲影响。
            # V2.1 契约核对(`赛事开发规范-赛道七V2.1提取.md:10-65`):必填仅
            # timestamp/epoch/step/phase/mode/data_source,额外字段不禁止。
            _extra = _epoch_metrics_extra(metrics, val_dice, val_acc, val_auc,
                                          avg_loss,
                                          cliff_stats=_last_cliff_stats or None)
            if roi_extra is not None:
                # ROI 指标(None 透传;只观测不进选模,见 validate_roi_epoch)
                _extra.update(roi_extra)
            logger.log(
                epoch=epoch, step=0, phase="val", loss=float(val_loss), lr=head_lr,
                extra=_extra,
            )
            print(
                f"Epoch {epoch:02d}/{epochs} [{stage_name}] train_loss={avg_loss:.4f} "
                f"val_dice={val_dice:.4f} "
                f"seg_p={metrics.get('seg_prob_mean')} "
                f"seg_pmax={metrics.get('seg_prob_max')} "
                f"seg_over_half={metrics.get('seg_over_half_frac')} "
                f"composite={composite} "
                f"mal_auc_pr={metrics.get('malignancy_auc_pr')} "
                f"loc_auc={metrics.get('location_weighted_auc')} "
                f"ete_kappa={metrics.get('ete_weighted_kappa')} t_auc={metrics.get('clinical_t_auc')} "
                f"tirads_kappa={metrics.get('tirads_weighted_kappa')}"
            )

            # ── 分割护栏 ────────────────────────────────────────────────────
            # 🔴 下限护栏(2026-08-16 新增):分割自训后 val_dice 从 ≈0 起步,原来的
            # "相对 warm-start 基线跌 >3 点"护栏在 0.0007 基线上**永远触发不了**,
            # 等于没有护栏。而分割是官方评分闸门(全零 mask → 零匹配 → Sclin≈0),
            # 训不起来必须尽早停,别烧满 35 epoch(云端一轮约 24 小时)。
            # 判定逻辑全部在 `evaluate_seg_guards()`(纯函数,便于测试),阈值论证见
            # 模块顶部 SEG_* 常量注释。
            reason, abort_msg, warn_lines = evaluate_seg_guards(
                epoch, val_dice, metrics, epochs)
            # 🔴 **warn 必须在 abort 之前打印**(2026-08-16 第四轮审查抓到的回归):
            # 抽成纯函数前 warn 是立即 `print` 的,abort 时 stdout 上先有 warn 再崩;
            # 抽出后 warn 收进 list、abort 分支提前 return —— **abort 那一轮的 warn
            # 就丢了**,而那恰恰是最该看到的诊断信息(比如"dice 偏低"与"前景占比为 0"
            # 同时出现,能直接区分是没学起来还是学歪了)。
            for line in warn_lines:
                print(line)
            if reason is not None:
                _abort(logger, epoch, reason, abort_msg)

            # 相对基线护栏(仅真数据 + 有 warm-start 基线时有意义):跌 >3 点 abort、
            # >1 点 warn。分割自训场景下基线≈0、本护栏基本不触发,但 warm-start 回归
            # 可用时它仍是防"重演 Stage2 崩塌"的那道门,保留不动。
            if dice_baseline is not None:
                drop = dice_baseline - val_dice
                if drop > 0.03:
                    _abort(logger, epoch, "dice_collapse",
                           f"[护栏 abort] val Dice {val_dice:.4f} 相对基线 {dice_baseline:.4f} "
                           f"跌 {drop*100:.1f}>3 点 —— 疑似重演 Stage2 崩塌,停机排查。")
                if drop > 0.01:
                    print(f"  [护栏 warn] val Dice 跌 {drop*100:.2f} 点(>1,<3)")

            # 🔴 seg_only 六项护栏(阶段2,2026-09-13):判定在
            # `evaluate_seg_only_gate()` 纯函数(2026-09-14 Gate2 F7-1 抽出,
            # 内联在 epoch 循环里 = 零测试覆盖)。**只在末轮判**且判的是当轮
            # val 指标;⚠️ 落地产物是"composite 最优"那轮的权重(见选模逻辑),
            # 两者可能是不同 epoch——门过≠产物达门,门不过≠产物废,最终以
            # best_epoch 行的 JSONL 指标为准(best 保存时已写 checkpoint 行)。
            # mock 冒烟无 seg_baseline(上面 `not args.mock_labels` 分支跳过)——
            # 冒烟只验管道数值健康,不验护栏本身,跳过此门。
            if args.seg_only and epoch == epochs and not args.mock_labels:
                assert seg_baseline is not None  # not args.mock_labels 已保证赋值
                _gate_pass, _gate_lines = evaluate_seg_only_gate(
                    val_dice, seg_baseline, metrics)
                for _gl in _gate_lines:
                    print(_gl)
                # 🔴 门结果(含"无法判定")无条件进 JSONL(2026-09-14 修复轮
                # 专审 F8):只 print 不落 JSONL,事后无法区分"没判"和"判了
                # 缺值";None 在 json.dumps 里合法(null)。
                logger.log(epoch=epoch, step=0, phase="val",
                           extra={"seg_only_gate_pass": _gate_pass})

            # 早停熔断(rules§3):epoch 3 后 composite 仍偏低 → 警告。
            # ⚠️ 阈值 2026-08-16 由 0.35 下调:composite 口径从"5 项分类均值"变成
            # "含 seg_dice 的 6 项均值",早期 epoch 分割≈0 会把它系统性拉低约 1/6
            # (上一轮 ep1 的 0.363 换算到新口径约 0.303),不调会立刻误报。
            if epoch >= 3 and composite is not None and composite < COMPOSITE_WARN:
                print(f"  [早停 warn] epoch {epoch} composite={composite:.3f} 偏低,关注是否发散")

            # 选模:composite 更优则存(mock 下 composite 可能全 None,存最后一轮兜底)
            if composite is not None and composite > best_composite:
                best_composite = composite
                torch.save(model.state_dict(), args.out)
                print(f"--> 保存 best(composite={composite:.4f})→ {args.out}")
                # checkpoint 字段填实际路径(此前恒 null);V2.1 样例即
                # `"checkpoints/best_model.pth"` 这种相对路径。
                # 🔴 best_epoch + 五项指标必须随行落盘(3B 专审 P1-1):
                # seg_only 门判末轮、产物是本行那轮,注释让操作者"以
                # best_epoch 行 JSONL 为准"——缺这些字段该操作不可执行。
                _best_extra = {"composite_best": round(float(composite), 6),
                               "best_epoch": epoch,
                               "val_dice": round(float(val_dice), 6)}
                _best_extra.update({
                    mk: (round(float(v), 6) if v is not None and math.isfinite(v) else None)
                    for mk, v in ((k, metrics.get(k)) for k in _SEG_GATE_KEYS)})
                logger.log(epoch=epoch, step=0, phase="val", checkpoint=args.out,
                           extra=_best_extra)

            # ── seg_only 版本化快照(ep4/8/12/16/20,2026-09-18 调教)──────────
            # 🔴 当年判死的第二把尺子:只有 best-composite 单文件,选模被 composite
            #    (无 AP 无匹配)垄断。快照让训后能用 diagnose_detection 官方口径
            #    逐个筛(P0 轮 2 实锤:官方口径下最强权重与 composite 最优不同源)。
            # 只在 seg_only 真训落(冒烟/标定不落,防冒烟产物冒充快照);路径=
            # <out 去后缀>_ep<N>.pth,与 --out 同目录,不覆盖 best 文件。
            if (args.seg_only and args.smoke is False
                    and epoch in _SEG_SNAPSHOT_EPOCHS):
                _snap = _snapshot_path(args.out, epoch)
                # 覆盖警告(P2 修复,2026-09-18 审查):同 --out 二跑会静默覆盖首轮
                # 快照且 best_epoch 指针随之指错。不阻断(重跑同 out 本就是重做
                # 同一实验),但必须在 stdout+JSONL 留痕,否则事后无从区分
                # "快照是本轮产物"还是"首轮残留"。
                _snap_reused = os.path.exists(_snap)
                if _snap_reused:
                    print(f"  [快照 warn] {_snap} 已存在,本轮覆盖写(上轮残留;"
                          f"若非有意重跑,先 mv 走旧快照再启动)")
                torch.save(model.state_dict(), _snap)
                logger.log(epoch=epoch, step=0, phase="val", checkpoint=_snap,
                           extra={"snapshot": True,
                                  "snapshot_epoch": epoch,
                                  "snapshot_overwrote_existing": bool(_snap_reused),
                                  "val_dice": round(float(val_dice), 6),
                                  "composite": (round(float(composite), 6)
                                                if composite is not None else None)})
                print(f"--> 保存快照 ep{epoch} → {_snap}")
        # 循环结束:composite 可能全程 None(val 子集单类/无恶性主病灶时各指标均不可算),
        # 此时无条件存最终权重保证有产物。**不能加 `not os.path.exists(args.out)`**——
        # 第二次运行时该文件已存在,兜底会被跳过、训练白跑零产出(Gate2 lane4 查出)。
        if best_composite < 0:
            torch.save(model.state_dict(), args.out)
            print(f"--> 保存 final(全程无有效 composite)→ {args.out}")
    finally:
        logger.close()

    if args.smoke:
        assert saw_head_grad, "冒烟未跑到任何四头 backward"
        print(f"[SMOKE OK] {epochs} epoch 完成,四头管道数值健康,日志 → {logger.path}")


if __name__ == "__main__":
    main()
