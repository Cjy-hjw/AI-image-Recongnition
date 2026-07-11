"""Regression test for scripts/verify/gates/gate8_adaptor_isolation.py.

验证 Gate 8(TaskAlignmentAdaptor cls→seg 梯度隔离审计)本身的正确性——不依赖
verify_model.py 整体跑通(本地 Windows 读取 configs/thyroid_config.yaml 会触发已知 GBK
编码 bug,见 memory/gate1-windows-gbk-bug.md;本文件直接构造 model/criterion/dummy 输入
调用 gate8_adaptor_isolation(),绕开该文件读取路径,可在本地 pytest 全绿)。

两个方向:
  1. test_gate8_passes_with_correct_detach:
     真实代码(multi_task_model.py `.detach()` 正确调用)下,Gate 必须返回 True,
     且 seg_backbone 梯度精确为 0(不是"很小")。
  2. test_gate8_catches_missing_detach:
     用 monkeypatch 把 torch.Tensor.detach 替换成恒等函数,模拟"误删 .detach()"这个
     具体 bug 场景,Gate 必须返回 False 并在打印内容中报告 seg_backbone 参数的非零
     泄漏——证明 Gate 真的具备检测能力,而不是一个无论对错都返回 True 的伪 Gate。

⚠️ 关键前置条件(实测踩坑记录,勿删):TaskAlignmentAdaptor.gamma 初始化为
nn.Parameter(torch.zeros(1))(零初始化残差门控设计,让 adaptor 训练早期从恒等映射开始
学习)。全新构造的模型 gamma 恒为 0,此时无论 `.detach()` 是否被删除,注入项
`gamma*(f_cls*mask)` 在数值上都恒为 0,两种场景下 seg_backbone 梯度都精确为 0 ——
若 gate8 内部不显式把 gamma 拨到非零值,test_gate8_catches_missing_detach 这个负面
测试会静默通过("Gate 声称检测到位,实际从未被真正触发"),形成假绿,起不到任何回归
守卫作用。gate8_adaptor_isolation() 内部已处理该前置条件(见其源码注释),本文件用
负面测试反向验证该处理确实生效,而不是重新假设它生效。
"""

import torch
import pytest

from src.models.multi_task_model import MultiTaskThyroidModel
from src.models.loss import MultiTaskLoss
from scripts.verify.gates.gate8_adaptor_isolation import gate8_adaptor_isolation


def _build_fixture():
    torch.manual_seed(0)
    model = MultiTaskThyroidModel(backbone_name="efficientnet_b2", pretrained=False)
    criterion = MultiTaskLoss()

    x = torch.randn(2, 1, 512, 512)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)  # 与 verify/runner.py 保持一致的归一化 dummy 输入
    targets = {
        "mask": torch.randint(0, 2, (2, 1, 512, 512)).float(),
        "composition": torch.tensor([0, 3]),
        "echogenicity": torch.tensor([1, 2]),
        "shape": torch.tensor([0, 1]),
        "margin": torch.tensor([2, 0]),
        "foci": torch.randn(2, 4).sigmoid(),
        # 正负样本各 1 个：AUCMLossApprox 在单类 batch 时会走 `*0.0` 退化捷径返回零损失
        # (见 AUCMLossApprox.forward 的 n_pos<0.5 or n_neg<0.5 分支)，那样 loss_mal 就是
        # 恒零常量，backward() 不会产生任何有意义的梯度信号，测不出真正的隔离契约。
        "malignancy": torch.tensor([1, 0]),
    }
    return model, criterion, x, targets


def test_gate8_passes_with_correct_detach(capsys):
    """真实代码：.detach() 正确调用，Gate 必须判定通过。"""
    model, criterion, x, targets = _build_fixture()

    # 模拟前序 Gate（gate3_auc_margin 的 Rule 3.3 / gate6 的 configure_stage_parameters）
    # 会残留在共享 model 实例上的 requires_grad=False 污染状态，验证 gate8 能自愈、
    # 不依赖 Gate 的执行顺序（这正是审计发现的"既有 Gate 会掩盖问题"的具体触发条件）。
    for p in model.seg_backbone.parameters():
        p.requires_grad = False

    result = gate8_adaptor_isolation(model, criterion, x, targets)
    captured = capsys.readouterr()

    assert result is True
    assert "GATE 8" in captured.out

    # gamma 必须被恢复为注入前的原值（全新模型的初始化值 0.0），不留副作用给后续流程。
    assert model.adaptor_stage3.gamma.item() == pytest.approx(0.0)
    assert model.adaptor_stage4.gamma.item() == pytest.approx(0.0)


def test_gate8_catches_missing_detach(capsys):
    """负面测试：monkeypatch 掉 detach 模拟误删 bug，Gate 必须能检测到泄漏并返回 False。"""
    model, criterion, x, targets = _build_fixture()

    orig_detach = torch.Tensor.detach
    torch.Tensor.detach = lambda self: self  # 模拟"误删 .detach()"：detach() 变成恒等函数
    try:
        result = gate8_adaptor_isolation(model, criterion, x, targets)
    finally:
        torch.Tensor.detach = orig_detach  # 必须无条件恢复，避免污染同进程内的其他测试

    captured = capsys.readouterr()

    assert result is False
    assert "GATE 8 Failed" in captured.out
    # 断言失败原因确实是我们要审计的那条契约，而不是别的无关异常
    assert "seg_backbone" in captured.out or "梯度隔离" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
