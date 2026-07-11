"""Regression tests for src/inference/model_runner.build_model().

2026-07-10 血泪教训：build_model() 曾用 strict=False 静默容忍 checkpoint 与
模型结构不匹配——①未解包 train_thyroidxl.py 的 {"state_dict":...} 包装格式时，
710 个参数全部静默丢失、模型等于随机初始化；②LoRA checkpoint 误喂非 LoRA
模型时，144 个 lora_A/lora_B key 被静默丢弃。现已改为硬校验：
load_state_dict(strict=False) 后 missing/unexpected 非空则 raise RuntimeError。
本测试锁死这个契约，防止回归回静默降级。
"""
import torch
import pytest

from src.inference.model_runner import build_model, load_config, preprocess_image, run_forward
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


class TestRunForwardCpuFp16Fallback:
    """2026-07-11 血泪教训（两轮）：
    第一轮——CI docker-build-smoke 在纯 CPU 容器（torch==2.1.0，Dockerfile 锁定
    版本）上跑 video_predict.py 时崩溃——run_forward() 打印了
    "[WARNING] ...Falling back to float32..."，但紧接着仍必然 RuntimeError
    (AutocastCPU only support Bfloat16)。根因：torch==2.1.0 的
    torch.autocast.__enter__ 在 device_type="cpu" 时**不看 enabled 参数**，
    只要 dtype!=torch.bfloat16 就必然 TORCH_CHECK 失败——旧代码 enabled=False
    但 dtype 仍传 float16，降级承诺从未真正生效。
    第二轮——第一轮修复只堵了 precision=="fp16"，却让 precision=="fp32" 的
    dtype 从原本无害的 bfloat16 变成了 float16（误信"enabled=False 时 dtype
    是 no-op"，Gate2 review 也这么判断，但 CI test job 的既有测试
    test_run_forward_fp32_equals_bare_forward 直接实测抓到回归）。最终修复：
    CPU 上 dtype 恒为 bfloat16，不管 precision 取值，靠 enabled 门控生效与否。
    本测试类锁死"CPU 上任意 precision 都不崩溃"这个契约。
    注：torch>=2.5 已放宽此校验，本机（Demo_D env, torch 2.5.1）不会复现旧
    bug 本身，但这些测试在 CI 实际使用的 torch==2.1.0 容器上会真实生效；且无论
    版本，这里断言的"不崩溃 + 返回合法输出"契约本身必须恒成立。
    """

    def _tiny_model(self):
        return MultiTaskThyroidModel(
            backbone_name="efficientnet_b2", pretrained=False, config=CONFIG
        ).to(torch.device("cpu")).eval()

    def test_fp16_precision_on_cpu_does_not_raise(self):
        model = self._tiny_model()
        device = torch.device("cpu")
        img_tensor = preprocess_image(
            torch.zeros(64, 64, dtype=torch.uint8).numpy(), img_size=64, device=device
        )
        outputs = run_forward(model, img_tensor, precision="fp16", device=device)
        assert "malignancy" in outputs
        assert torch.isfinite(outputs["malignancy"]).all()

    def test_fp32_precision_on_cpu_does_not_raise(self):
        """第二轮回归的直接锁死：precision="fp32" 曾在这次修复中被误改成
        dtype=float16 而在 CPU 上崩溃（enabled=False 并不能豁免 CPU 的
        dtype 校验）。"""
        model = self._tiny_model()
        device = torch.device("cpu")
        img_tensor = preprocess_image(
            torch.zeros(64, 64, dtype=torch.uint8).numpy(), img_size=64, device=device
        )
        outputs = run_forward(model, img_tensor, precision="fp32", device=device)
        assert "malignancy" in outputs
        assert torch.isfinite(outputs["malignancy"]).all()

    def test_bf16_precision_on_cpu_still_works(self):
        """确保本次改动没有连带破坏原本就能跑的 bf16-on-cpu 路径。"""
        model = self._tiny_model()
        device = torch.device("cpu")
        img_tensor = preprocess_image(
            torch.zeros(64, 64, dtype=torch.uint8).numpy(), img_size=64, device=device
        )
        outputs = run_forward(model, img_tensor, precision="bf16", device=device)
        assert "malignancy" in outputs
        assert torch.isfinite(outputs["malignancy"]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 设备")
class TestRunForwardGpuPrecisionRegression:
    """Gate2 review（2026-07-11）非阻塞建议补齐：CPU+fp16 修复改了 dtype 表达式
    的 else 分支取值（fp32 时从 bfloat16 变成 float16），虽然 enabled=False 时
    该值不参与任何 cast、纯属无害，但仍需一个 GPU 路径的回归测试锁死"CPU 修复
    没有连带影响 CUDA 上 fp16/bf16/fp32 三种 precision 的实际行为"。"""

    def _tiny_model(self, device):
        return MultiTaskThyroidModel(
            backbone_name="efficientnet_b2", pretrained=False, config=CONFIG
        ).to(device).eval()

    @pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
    def test_gpu_precision_paths_do_not_raise(self, precision):
        device = torch.device("cuda")
        model = self._tiny_model(device)
        img_tensor = preprocess_image(
            torch.zeros(64, 64, dtype=torch.uint8).numpy(), img_size=64, device=device
        )
        outputs = run_forward(model, img_tensor, precision=precision, device=device)
        assert "malignancy" in outputs
        assert torch.isfinite(outputs["malignancy"]).all()


class TestLoadConfig:
    """2026-07-11 修复：predict.py/video_predict.py 此前在 try 块外裸
    open(args.config) → yaml.safe_load → config["environment"]["device"]，
    路径不存在/YAML 格式错误/键缺失都会导致进程裸崩溃、非零 exit code、零输出
    ——与图像/视频加载失败是同一类问题（2026-07-11 早前已修），这次补配置加载。
    本测试锁死 load_config() 本身的契约：合法配置正常返回，三类异常各自可预期
    地抛出（不是静默返回 None 或部分配置），供调用方 try/except Exception 统一
    兜底。"""

    def test_valid_config_loads(self, tmp_path):
        cfg_path = tmp_path / "valid.yaml"
        cfg_path.write_text(
            "environment:\n  device: cuda:0\ndata:\n  img_size: 512\n",
            encoding="utf-8",
        )
        config = load_config(str(cfg_path))
        assert config["environment"]["device"] == "cuda:0"
        assert config["data"]["img_size"] == 512

    def test_missing_file_raises_file_not_found_error(self, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            load_config(str(missing))

    def test_malformed_yaml_raises(self, tmp_path):
        cfg_path = tmp_path / "malformed.yaml"
        # 双引号未闭合的非法 YAML,yaml.safe_load 必抛 YAMLError
        cfg_path.write_text('environment: "device: cuda:0\n', encoding="utf-8")
        with pytest.raises(Exception):
            load_config(str(cfg_path))

    def test_missing_required_key_raises_key_error(self, tmp_path):
        """有效 YAML,但缺 environment.device 这个下游必用的键。"""
        cfg_path = tmp_path / "incomplete.yaml"
        cfg_path.write_text("data:\n  img_size: 512\n", encoding="utf-8")
        with pytest.raises(KeyError):
            load_config(str(cfg_path))
