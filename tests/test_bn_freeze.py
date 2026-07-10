"""Regression tests for set_frozen_modules_eval (src/training/loop/bn_freeze.py).

2026-07-10 血泪教训：scripts/train_thyroidxl.py 曾遗漏调用此函数——backbone
权重 requires_grad=False，但只要整模型 model.train()，其内部 BatchNorm 仍会用
每个 batch 的真实激活值更新 running_mean/running_var（这只受 module.training
控制，与 requires_grad 无关）。50 epoch 下来把一个训练前就已临界的 BN 通道
（running_var 低至 3.5e-42）冲向数值病态区间，经 backbone 逐层复合放大，
致 checkpoint 报废重训。本测试锁死 set_frozen_modules_eval 的核心契约。

用轻量合成模型隔离测试该函数本身，不拖全量 MultiTaskThyroidModel backbone。
"""
import torch
import torch.nn as nn

from src.training.loop.bn_freeze import set_frozen_modules_eval


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen_conv = nn.Conv2d(3, 8, 3, padding=1)
        self.frozen_bn = nn.BatchNorm2d(8)
        self.trainable_head = nn.Linear(8, 2)

    def forward(self, x):
        x = self.frozen_bn(self.frozen_conv(x))
        x = x.mean(dim=(2, 3))
        return self.trainable_head(x)


def _freeze_backbone(model: _TinyNet) -> None:
    for p in model.frozen_conv.parameters():
        p.requires_grad = False
    for p in model.frozen_bn.parameters():
        p.requires_grad = False


def test_frozen_bn_running_stats_do_not_drift_with_freeze():
    """set_frozen_modules_eval 后，冻结子模块的 BN running_var/running_mean 必须零漂移。"""
    model = _TinyNet()
    _freeze_backbone(model)

    var_before = model.frozen_bn.running_var.clone()
    mean_before = model.frozen_bn.running_mean.clone()

    model.train()
    set_frozen_modules_eval(model)

    x = torch.randn(4, 3, 8, 8)
    with torch.no_grad():
        model(x)

    assert torch.equal(model.frozen_bn.running_var, var_before), \
        "冻结 BN 的 running_var 在 set_frozen_modules_eval 后仍发生漂移"
    assert torch.equal(model.frozen_bn.running_mean, mean_before), \
        "冻结 BN 的 running_mean 在 set_frozen_modules_eval 后仍发生漂移"


def test_bn_running_stats_drift_without_freeze_call():
    """反向验证：不调用 set_frozen_modules_eval 时 running_var 确实会漂移——
    证明上面的测试是在测有效的东西，不是巧合下没漂移（这正是本次事故的复现）。"""
    model = _TinyNet()
    _freeze_backbone(model)

    var_before = model.frozen_bn.running_var.clone()

    model.train()  # 故意不调用 set_frozen_modules_eval

    x = torch.randn(4, 3, 8, 8)
    with torch.no_grad():
        model(x)

    assert not torch.equal(model.frozen_bn.running_var, var_before), \
        "预期未冻结时 running_var 应漂移；若相等说明测试环境本身有问题（假阴性风险）"


def test_trainable_submodule_stays_in_train_mode():
    """未冻结的子模块不应被误设为 eval（防止过度冻结把可训练头也冻住）。"""
    model = _TinyNet()
    _freeze_backbone(model)

    model.train()
    set_frozen_modules_eval(model)

    assert model.trainable_head.training is True
    assert model.frozen_bn.training is False


def test_lora_like_partial_freeze_still_freezes_sibling_bn():
    """模拟 LoRA 场景：模块内混有冻结的 base weight 和可训练的 lora_A，
    但同级的 BatchNorm 是独立子模块、参数全冻结——它必须仍被设为 eval，
    不能因为"父模块里有可训练参数"就连带被放过（此前误判的关键场景）。"""

    class LoRALikeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_weight = nn.Parameter(torch.randn(4, 4), requires_grad=False)
            self.lora_a = nn.Parameter(torch.randn(4, 4), requires_grad=True)
            self.bn = nn.BatchNorm1d(4)
            for p in self.bn.parameters():
                p.requires_grad = False

        def forward(self, x):
            return self.bn(x @ (self.base_weight + self.lora_a * 0.01))

    block = LoRALikeBlock()
    var_before = block.bn.running_var.clone()

    block.train()
    set_frozen_modules_eval(block)

    assert block.bn.training is False, "LoRA 场景下，同级独立 BN 子模块必须仍被冻结"
    assert block.lora_a.requires_grad is True, "可训练的 lora_a 不应被本函数影响"

    x = torch.randn(4, 4)
    with torch.no_grad():
        block(x)

    assert torch.equal(block.bn.running_var, var_before)
