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
from src.training.loop.train_step import train_batch
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
    bad_missing = [k for k in result.missing_keys if not k.startswith(head_pfx)]
    if bad_missing:
        raise RuntimeError(
            f"warm-start 底座缺失**非四头**键 {bad_missing[:8]}(共 {len(bad_missing)})"
            f" —— checkpoint {ckpt_path} 与当前架构不匹配,拒绝静默半载。"
        )
    if result.unexpected_keys:
        print(f"--> [warm-start] 忽略 {len(result.unexpected_keys)} 个多余键(底座含已弃用字段)")
    n_head_missing = len(result.missing_keys)
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
        }
        outputs = model(img)
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


def _epoch_metrics_extra(metrics: dict, val_dice, val_acc, val_auc, train_loss) -> dict:
    """epoch 级 val 指标 → JSONL `extra` 字段。

    🔴 **为什么必须写 JSONL(2026-08-15 实证,勿删)**:这些指标此前**只走 stdout**
    (`print`),而 `python` 不加 `-u` 时 stdout 是 8KB 块缓冲 —— 上一轮 35 epoch
    真训要攒到 ep25 才第一次刷新,整个训练前 2/3 完全看不到指标,等于盲跑。
    `JsonlLogger` 逐行 flush,不受管道缓冲影响,是唯一可靠的实时观测面。

    `malignancy_auc_pr` 尤其不能漏:它是目标三的核心项(良恶性 AUC-PR),上一轮
    **完全没打印**,0.888 这个数是靠 `composite×5 − 其余四项`**反推**出来的。

    None 值直接跳过(不填 0)—— 填 0 会把"这项算不出来"伪装成"分数是 0"。
    `JsonlLogger._drop_nonfinite` 另会兜掉 NaN/Inf。
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
          f"{' ETE_ONLY' if args.ete_only else ''}")
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

    # ── 数据 ──────────────────────────────────────────────────────────────
    if args.mock_labels:
        n_train = 16 if args.smoke else 64
        n_val = 8 if args.smoke else 24
        train_ds = MockOfficialDataset(n=n_train, img_size=img_size, seed=seed)
        val_ds = MockOfficialDataset(n=n_val, img_size=img_size, seed=seed + 100)
    else:
        if not args.data_root or not args.xlsx:
            parser.error("非 --mock-labels 模式必须给 --data-root 与 --xlsx")
        from src.dataset.augmentation import get_train_transforms
        from src.dataset.official_loader import build_official_datasets
        train_ds, val_ds = build_official_datasets(
            args.data_root, args.xlsx, img_size=img_size, seed=seed,
            train_transform=get_train_transforms(), hflip_prob=args.hflip_prob,
            samples_pkl=args.samples_pkl,
        )
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

    try:
        for epoch in range(1, epochs + 1):
            param_groups, head_lr, stage_name = configure_official_stage(
                model, criterion, epoch, lr_base, config["train"]["weight_decay"],
                freeze_legacy=args.freeze_legacy, ete_only=args.ete_only,
            )
            cur_stage = get_official_stage(epoch)
            if args.ete_only:
                cur_stage = "ETE"   # 恒定单阶段:全程不重建 optimizer,动量跨 epoch 承接
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
            logger.log(
                epoch=epoch, step=0, phase="val", loss=float(val_loss), lr=head_lr,
                extra=_epoch_metrics_extra(metrics, val_dice, val_acc, val_auc, avg_loss),
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
                logger.log(epoch=epoch, step=0, phase="val", checkpoint=args.out,
                           extra={"composite_best": round(float(composite), 6)})
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
