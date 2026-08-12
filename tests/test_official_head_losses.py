"""阶段 2.4 三头损失编排(`components/official_heads.py` + `MultiTaskLoss` 接线)单测。

覆盖四类风险:
  ① **legacy 零回归** —— targets 无三头键时 total/loss_dict 与接线前逐位一致;
  ② **λ 真的生效** —— λ=0 与 λ=2 的差值恰为 2 倍原始损失(不是"看起来变了");
  ③ **validity 走顶层直传键** —— 只塞 `label_validity` 不塞顶层键会 KeyError,
     而非静默拿垃圾标签当监督;
  ④ **梯度隔离** —— 三头损失不得回流 CBM `margin` 分支与分割解码器
     (ETE/T 头内部 detach 是设计意图,回流即毁 CBM 报告可信度)。
"""
import pytest
import torch

from src.models.losses.components import compute_official_head_losses
from src.models.losses.multitask import MultiTaskLoss

B = 4
NUM_ETE = 4


def _head_outputs(requires_grad: bool = True):
    def t(*shape):
        return torch.randn(*shape, requires_grad=requires_grad)

    return {
        "location_side_logits": t(B, 3),
        "location_level_logits": t(B, 3),
        "ete_logits": t(B, NUM_ETE),
        "t_logits": t(B, 2),
    }


def _head_targets(all_valid: bool = True):
    v = torch.ones(B) if all_valid else torch.zeros(B)
    return {
        "lesion_loc": torch.tensor([0, 4, 6, 1]),
        "valid_lesion_loc": v.clone(),
        "extrathy_ext": torch.tensor([0, 1, 3, 2]),
        "valid_extrathy_ext": v.clone(),
        "clin_t_stage": torch.tensor([0, 1, 1, 0]),
        "valid_clin_t_stage": v.clone(),
    }


def test_skips_entirely_without_head_keys():
    """targets 无三头键 → 返回标量 0.0 与空 detail(legacy train.py 路径零回归)。"""
    total, detail = compute_official_head_losses(_head_outputs(), {"malignancy": torch.zeros(B)})
    assert total == 0.0 and not isinstance(total, torch.Tensor)
    assert detail == {}


def test_partial_keys_only_compute_present_heads():
    """只给位置键 → 只算位置,ETE/T 的 detail 键必须缺席(而非算成 0 混淆监控)。"""
    tg = _head_targets()
    only_loc = {k: v for k, v in tg.items() if "lesion_loc" in k}
    total, detail = compute_official_head_losses(_head_outputs(), only_loc)
    assert isinstance(total, torch.Tensor) and torch.isfinite(total)
    assert "location" in detail and "location_side" in detail
    assert "ete" not in detail and "clin_t" not in detail


def test_lambda_scales_exactly():
    """λ 必须精确线性缩放。λ=0 得 0、λ=2 得 2 倍 —— 钉死"λ 接错位置/被吞"。"""
    out, tg = _head_outputs(requires_grad=False), _head_targets()
    kw = dict(ete_focal_gamma=2.0)
    base, d1 = compute_official_head_losses(out, tg, 1.0, 1.0, 1.0, **kw)
    zero, _ = compute_official_head_losses(out, tg, 0.0, 0.0, 0.0, **kw)
    twice, d2 = compute_official_head_losses(out, tg, 2.0, 2.0, 2.0, **kw)
    assert float(zero) == pytest.approx(0.0, abs=1e-9)
    assert float(twice) == pytest.approx(2.0 * float(base), rel=1e-6)
    # detail 是**未乘 λ 的原始值** —— 两次调用必须相同,否则监控读数随 λ 漂移
    for k in ("location", "ete", "clin_t"):
        assert float(d1[k]) == pytest.approx(float(d2[k]), rel=1e-6)


def test_all_invalid_batch_is_exact_zero_no_nan():
    """整批 validity=0 → 损失精确 0、无 NaN、梯度精确 0(masked 范式的分母 eps)。"""
    out = _head_outputs()
    total, detail = compute_official_head_losses(out, _head_targets(all_valid=False))
    assert torch.isfinite(total) and float(total) == pytest.approx(0.0, abs=1e-9)
    total.backward()
    for k, v in out.items():
        assert v.grad is not None, f"{k} 应有 grad 对象(哪怕全 0)"
        assert torch.isfinite(v.grad).all() and float(v.grad.abs().sum()) == 0.0


def test_validity_must_be_toplevel_key_not_label_validity():
    """🔴 validity 只认顶层直传键。塞进 `label_validity` 不生效 —— 必须 KeyError
    暴露,不能静默把 invalid 样本的垃圾标签当真监督(审查 F2)。"""
    tg = _head_targets()
    del tg["valid_lesion_loc"]
    tg["label_validity"] = {"lesion_loc": torch.ones(B)}
    with pytest.raises(KeyError, match="valid_lesion_loc"):
        compute_official_head_losses(_head_outputs(), tg)


def test_validity_2d_is_rejected():
    """[B,1] validity 会隐式广播成 [B,B] 静默放大 loss(QA_VP §2.2)→ 必须 assert 死。"""
    tg = _head_targets()
    tg["valid_lesion_loc"] = torch.ones(B, 1)
    with pytest.raises(AssertionError, match="一维"):
        compute_official_head_losses(_head_outputs(), tg)


def test_gradient_does_not_reach_margin_or_segmentation():
    """三头损失不得回流 CBM margin / 分割解码器。

    ETE 头吃 `margin`(CBM 边缘概念)与 `soft_mask_detached`,T 头吃 ete_logits 与
    `nodule_scale_detached` —— 全部在头内部 detach。若哪天有人去掉 detach,
    ETE 的监督会重塑"边缘"概念表征、毁掉 CBM 报告可信度,而 loss 照常下降。
    反证:`cls_backbone` **必须**有梯度,否则本测试只是在验证"头没接上"。
    """
    model = _build_small_model()
    x = torch.randn(2, 1, 128, 128)
    outputs = model(x, gamma_prior=0.0)
    tg = {
        "lesion_loc": torch.tensor([0, 4]),
        "valid_lesion_loc": torch.ones(2),
        "extrathy_ext": torch.tensor([0, 3]),
        "valid_extrathy_ext": torch.ones(2),
        "clin_t_stage": torch.tensor([0, 1]),
        "valid_clin_t_stage": torch.ones(2),
    }
    total, _ = compute_official_head_losses(outputs, tg)
    total.backward()

    def grad_sum(prefix):
        return sum(
            float(p.grad.abs().sum())
            for n, p in model.named_parameters()
            if n.startswith(prefix) and p.grad is not None
        )

    assert grad_sum("cbm_head") == 0.0, "三头损失回流了 CBM(margin 未 detach?)"
    assert grad_sum("seg_head") == 0.0, "三头损失回流了分割头(soft_mask 未 detach?)"
    assert grad_sum("dec_stages") == 0.0, "三头损失回流了分割解码器"
    assert grad_sum("cls_backbone") > 0.0, "反证:cls_backbone 必须有梯度(否则头没接上)"


def _build_small_model():
    from src.models.multi_task_model import MultiTaskThyroidModel

    torch.manual_seed(0)
    return MultiTaskThyroidModel(pretrained=False, config={"data": {"img_size": 128}})


def test_multitask_loss_legacy_targets_unchanged():
    """`MultiTaskLoss` 接线后,legacy targets(无三头键)的 loss_dict 键集合不变。"""
    model = _build_small_model()
    crit = MultiTaskLoss()
    outputs = model(torch.randn(2, 1, 128, 128), gamma_prior=0.0)
    targets = {
        "mask": torch.zeros(2, 1, 128, 128),
        "composition": torch.tensor([0, 1]),
        "echogenicity": torch.tensor([0, 1]),
        "shape": torch.tensor([0, 1]),
        "margin": torch.tensor([0, 1]),
        "foci": torch.zeros(2, 4),
        "malignancy": torch.tensor([0, 1]),
    }
    total, ld = crit(outputs, targets)
    assert torch.isfinite(total)
    for k in ("location", "ete", "clin_t"):
        assert k not in ld, f"legacy 路径不该出现三头监控键 {k}"


def test_multitask_loss_adds_head_terms_when_targets_present():
    """targets 带三头键时,total 增量 = λ 加权三头损失,且 loss_dict 出现监控键。"""
    model = _build_small_model()
    crit = MultiTaskLoss(official_head_config={"loc_loss_weight": 1.0,
                                               "ete_loss_weight": 1.0,
                                               "clin_t_loss_weight": 1.0})
    outputs = model(torch.randn(2, 1, 128, 128), gamma_prior=0.0)
    base_targets = {
        "mask": torch.zeros(2, 1, 128, 128),
        "composition": torch.tensor([0, 1]),
        "echogenicity": torch.tensor([0, 1]),
        "shape": torch.tensor([0, 1]),
        "margin": torch.tensor([0, 1]),
        "foci": torch.zeros(2, 4),
        "malignancy": torch.tensor([0, 1]),
    }
    head_targets = {
        "lesion_loc": torch.tensor([0, 4]),
        "valid_lesion_loc": torch.ones(2),
        "extrathy_ext": torch.tensor([0, 3]),
        "valid_extrathy_ext": torch.ones(2),
        "clin_t_stage": torch.tensor([0, 1]),
        "valid_clin_t_stage": torch.ones(2),
    }
    # FAMO 是有状态的动态调度器,两次 forward 会改权重 → 关掉更新才能纯比较增量
    t_legacy, _ = crit(outputs, base_targets, update_famo_weights=False)
    t_full, ld = crit(outputs, {**base_targets, **head_targets},
                      update_famo_weights=False)
    for k in ("location", "location_side", "location_level", "ete", "clin_t"):
        assert k in ld, f"缺三头监控键 {k}"
    expected = float(ld["location"]) + float(ld["ete"]) + float(ld["clin_t"])
    assert float(t_full) - float(t_legacy) == pytest.approx(expected, rel=1e-5)


def test_criterion_buffers_follow_device_and_state_dict():
    """ETE class_weights/log_prior 必须是 buffer(随 .to(device) 迁移 + 进 state_dict)。

    普通属性会留在 CPU,GPU 训练时在损失里报设备不匹配 —— 本地 CPU 永远测不出来。
    """
    cfg = {
        "ete_class_weights": torch.tensor([0.5, 1.0, 1.2, 1.3]),
        "ete_log_prior": torch.log(torch.tensor([0.9, 0.06, 0.03, 0.01])),
        "clin_t_class_weights": torch.tensor([0.86, 1.14]),
    }
    crit = MultiTaskLoss(official_head_config=cfg)
    keys = crit.state_dict().keys()
    for k in ("ete_class_weights", "ete_log_prior", "clin_t_class_weights"):
        assert k in keys, f"{k} 未注册为 buffer,GPU 上会设备不匹配"
    # 未配置时为 None,state_dict 不含该键,且损失侧按"不加权"处理
    bare = MultiTaskLoss()
    assert bare.ete_class_weights is None
    assert "ete_class_weights" not in bare.state_dict()
