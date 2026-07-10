"""Regression tests for src/inference/model_runner.build_model().

2026-07-10 血泪教训：build_model() 曾用 strict=False 静默容忍 checkpoint 与
模型结构不匹配——①未解包 train_thyroidxl.py 的 {"state_dict":...} 包装格式时，
710 个参数全部静默丢失、模型等于随机初始化；②LoRA checkpoint 误喂非 LoRA
模型时，144 个 lora_A/lora_B key 被静默丢弃。现已改为硬校验：
load_state_dict(strict=False) 后 missing/unexpected 非空则 raise RuntimeError。
本测试锁死这个契约，防止回归回静默降级。
"""
import torch

from src.inference.model_runner import build_model
from src.models.multi_task_model import MultiTaskThyroidModel

# img_size 用小尺寸只为让 config 合法；这些测试只做 state_dict 层面操作，不跑 forward。
CONFIG = {
    "model": {"backbone": "efficientnet_b2"},
    "data": {"img_size": 64},
}


def _tiny_model():
    return MultiTaskThyroidModel(backbone_name="efficientnet_b2", pretrained=False, config=CONFIG)


def test_build_model_loads_plain_state_dict(tmp_path):
    """裸 state_dict（train.py 的 torch.save(model.state_dict(), ...) 约定）必须正常加载。"""
    model = _tiny_model()
    ckpt_path = tmp_path / "plain.pth"
    torch.save(model.state_dict(), ckpt_path)

    loaded = build_model(CONFIG, str(ckpt_path), torch.device("cpu"))
    assert loaded is not None


def test_build_model_unwraps_wrapped_checkpoint(tmp_path):
    """train_thyroidxl.py 的 {"epoch","stage","state_dict",...} 包装格式必须被正确解包，
    加载出来的参数必须和原模型逐值相等（此前不解包会导致 710 个参数全部静默丢失）。"""
    model = _tiny_model()
    ckpt_path = tmp_path / "wrapped.pth"
    torch.save({
        "epoch": 5, "stage": "FT-LoRA",
        "state_dict": model.state_dict(),
        "patient_val_auc": 0.8, "dice": 0.5,
    }, ckpt_path)

    loaded = build_model(CONFIG, str(ckpt_path), torch.device("cpu"))
    loaded_sd = loaded.state_dict()
    original_sd = model.state_dict()
    for k in original_sd:
        assert torch.equal(loaded_sd[k], original_sd[k]), \
            f"参数 {k} 未被正确加载（包装格式解包失败会导致此处不相等/仍是随机初始化）"


def test_build_model_raises_on_unexpected_keys(tmp_path):
    """checkpoint 含模型没有的 key（如未合并的 LoRA lora_A/lora_B）必须硬报错，
    不能像旧版 strict=False 那样静默丢弃。"""
    model = _tiny_model()
    sd = model.state_dict()
    sd["some_module.lora_A"] = torch.randn(2, 2)  # 模拟未合并的 LoRA 权重
    ckpt_path = tmp_path / "mismatched.pth"
    torch.save(sd, ckpt_path)

    try:
        build_model(CONFIG, str(ckpt_path), torch.device("cpu"))
        assert False, "checkpoint 含未知 key 时 build_model 必须抛 RuntimeError"
    except RuntimeError as e:
        assert "不匹配" in str(e)


def test_build_model_raises_on_missing_keys(tmp_path):
    """checkpoint 缺模型需要的 key 必须硬报错（防止部分随机初始化悄悄进生产推理）。"""
    model = _tiny_model()
    sd = model.state_dict()
    sd.pop(next(iter(sd)))  # 删掉任意一个真实存在的 key
    ckpt_path = tmp_path / "incomplete.pth"
    torch.save(sd, ckpt_path)

    try:
        build_model(CONFIG, str(ckpt_path), torch.device("cpu"))
        assert False, "checkpoint 缺 key 时 build_model 必须抛 RuntimeError"
    except RuntimeError as e:
        assert "不匹配" in str(e)


def test_build_model_falls_back_to_random_init_when_file_missing(tmp_path):
    """checkpoint 文件不存在时，仍应降级随机初始化 + warning，不报错
    （这个分支不受本次改动影响，向后兼容行为必须保持不变）。"""
    missing_path = tmp_path / "does_not_exist.pth"
    loaded = build_model(CONFIG, str(missing_path), torch.device("cpu"))
    assert loaded is not None
