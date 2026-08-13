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
    tirads_counts = tr.get("tirads_class_counts")
    cfg = {
        "loc_loss_weight": float(tr.get("loc_loss_weight", 1.0)),
        "ete_loss_weight": float(tr.get("ete_loss_weight", 1.0)),
        "clin_t_loss_weight": float(tr.get("clin_t_loss_weight", 1.0)),
        "tirads_loss_weight": float(tr.get("tirads_loss_weight", 1.0)),
        "ete_focal_gamma": float(tr.get("ete_focal_gamma", 2.0)),
        "ete_la_tau": ete_la_tau,
        "clin_t_label_smoothing": float(tr.get("clin_t_label_smoothing", 0.0)),
        "tirads_label_smoothing": float(tr.get("tirads_label_smoothing", 0.0)),
        "tirads_ordinal_weight": float(tr.get("tirads_ordinal_weight", 0.0)),
        # β=0 → 不加权(None);>0 → CB 权重。
        "ete_class_weights": (
            class_balanced_weights(ete_counts, ete_beta) if ete_beta > 0 else None
        ),
        # log_prior 仅 Logit Adjustment(la_tau>0)时需要,否则 None。
        "ete_log_prior": log_class_prior(ete_counts) if ete_la_tau > 0 else None,
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
    print(f"--> [warm-start] 加载 {ckpt_path};四头 {n_head_missing} 键随机初始化(预期)")



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
                       loss=float(loss.item()), lr=head_lr)
    return running / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="官方四头 A/B/C 调度训练")
    parser.add_argument("--config", default="configs/thyroid_config.yaml")
    parser.add_argument("--data-root", default=None, help="官方 annotation 根目录")
    parser.add_argument("--xlsx", default=None, help="官方标注 xlsx 路径")
    parser.add_argument("--epochs", type=int, default=STAGE_C_END, help=f"默认 {STAGE_C_END}(A/B/C 全程)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="头 lr 基准;缺省取 config.train.lr")
    parser.add_argument("--out", default="checkpoints/official_heads_best.pth")
    parser.add_argument("--resume-from", default=None, help="分割底座 checkpoint(warm-start)")
    parser.add_argument("--hflip-prob", type=float, default=0.0, help="水平翻转概率(联动翻位置标签)")
    parser.add_argument("--freeze-legacy", action="store_true",
                        help="关 malignancy AUCM + vCLUB 更新,纯头训练(默认保留守旧任务)")
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
          f"lr_base={lr_base} epochs={epochs} mock={args.mock_labels} freeze_legacy={args.freeze_legacy}")

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
        )
    num_workers = 0 if (args.smoke or device.type == "cpu") else config["environment"]["num_workers"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), worker_init_fn=seed_worker, generator=loader_gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), worker_init_fn=seed_worker,
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

    logger = JsonlLogger(log_dir=args.log_dir, data_source="official/train_v1")
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
                freeze_legacy=args.freeze_legacy,
            )
            cur_stage = get_official_stage(epoch)
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
            logger.log(epoch=epoch, step=0, phase="val", loss=float(val_loss), lr=head_lr)
            print(
                f"Epoch {epoch:02d}/{epochs} [{stage_name}] train_loss={avg_loss:.4f} "
                f"val_dice={val_dice:.4f} composite={composite} "
                f"loc_auc={metrics.get('location_weighted_auc')} "
                f"ete_kappa={metrics.get('ete_weighted_kappa')} t_auc={metrics.get('clinical_t_auc')} "
                f"tirads_kappa={metrics.get('tirads_weighted_kappa')}"
            )

            # Dice 护栏(仅真数据有基线):相对 warm-start 基线跌 >3 点 abort、>1 点 warn。
            if dice_baseline is not None:
                drop = dice_baseline - val_dice
                if drop > 0.03:
                    raise RuntimeError(
                        f"[护栏 abort] val Dice {val_dice:.4f} 相对基线 {dice_baseline:.4f} "
                        f"跌 {drop*100:.1f}>3 点 —— 疑似重演 Stage2 崩塌,停机排查。"
                    )
                if drop > 0.01:
                    print(f"  [护栏 warn] val Dice 跌 {drop*100:.2f} 点(>1,<3)")

            # 早停熔断(rules§3):epoch 3 后 composite 仍随机(≤0.5)且非 None → 警告
            if epoch >= 3 and composite is not None and composite < 0.35:
                print(f"  [早停 warn] epoch {epoch} composite={composite:.3f} 偏低,关注是否发散")

            # 选模:composite 更优则存(mock 下 composite 可能全 None,存最后一轮兜底)
            if composite is not None and composite > best_composite:
                best_composite = composite
                torch.save(model.state_dict(), args.out)
                print(f"--> 保存 best(composite={composite:.4f})→ {args.out}")
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
