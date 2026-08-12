"""CLI 入口端到端冒烟测试(predict.py / video_predict.py 当子进程实跑)。

为什么需要:其余 99 个单测全是组件级的,没有一个把交付入口脚本作为整体跑通。
入口的「接线」一旦断裂(import 名错、argparse 参数漂移、产物未落盘、共用
model_runner/postprocess/export 被改坏),组件单测一个都抓不到——只有把脚本当
子进程实跑才能兜住。本测试不验证精度(用随机初始化权重,build_model 在 checkpoint
缺失时降级随机权重 + warning,不报错),只验证「管线连通 + 产物齐全 + JSON 可解析」。

两条路径:
  REJECTED  predict.py + 随机噪声 → gatekeeper 正确拒,落 result.json(含 status/原因)
  PASSED    video_predict.py --skip_gatekeeper → 推理/后处理/报告/JSON/Grad-CAM 全产出

平台 codec 缺失(无法写 mp4)时 PASSED 用例 skip,不让 CI 假失败(对齐 test_video.py)。
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 子进程必须能 import src.* —— 用仓库根做 cwd,并把根注入 PYTHONPATH
_ENV = dict(os.environ, PYTHONPATH=REPO_ROOT)

try:
    from scripts.verify.gen_synthetic_dicom import build_dicom
    _HAS_PYDICOM = True
except ImportError:
    _HAS_PYDICOM = False

try:
    import pylibjpeg  # noqa: F401
    _HAS_PYLIBJPEG = True
except ImportError:
    _HAS_PYLIBJPEG = False


def _run_entry(script, args, timeout=300):
    """在仓库根以子进程跑入口脚本,返回 CompletedProcess。"""
    return subprocess.run(
        [sys.executable, script, *args],
        cwd=REPO_ROOT, env=_ENV,
        capture_output=True, text=True, timeout=timeout,
    )


def _write_noise_png(path, h=400, w=500):
    import cv2
    img = (np.random.rand(h, w) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def _write_noise_mp4(path, n=8, size=128):
    """写一段随机噪声 mp4;codec 不可用时返回 False 让调用方 skip。"""
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 10.0, (size, size), isColor=True)
    if not vw.isOpened():
        vw.release()
        return False
    for _ in range(n):
        vw.write((np.random.rand(size, size, 3) * 255).astype(np.uint8))
    vw.release()
    return os.path.exists(path) and os.path.getsize(path) > 0


# ═════════════════════════════════════════════════════════════════════════════
# REJECTED 路径:predict.py + 噪声 → gatekeeper 拒绝
# ═════════════════════════════════════════════════════════════════════════════

class TestPredictRejectPath:
    def test_noise_image_runs_and_writes_rejection(self, tmp_path):
        img = tmp_path / "noise.png"
        _write_noise_png(img)
        out = tmp_path / "out"
        r = _run_entry("predict.py", [
            "--input_image", str(img),
            "--output_dir", str(out),
            "--precision", "fp32",
        ])
        # 进程必须正常退出(gatekeeper 拒绝是业务结果,不是崩溃)
        assert r.returncode == 0, f"predict.py 非零退出\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        # 本测试只兜「管线连通 + 产物落盘 + JSON 可解析」(见文件头),不把
        # gatekeeper 在随机噪声上的具体判定当硬断言:该判定依赖 GLCM/频域统计的
        # 浮点细节,随平台(numpy 1.x 容器 vs 2.x 本地 + 不同 BLAS)翻转——
        # 本地拒(REJECTED),交付容器可能放行(PASSED)。
        # TODO(真实数据): gatekeeper 判定的平台稳定性待初赛真实样本复核;
        #   若届时仍平台敏感,需在 authenticity.py 加固数值稳定性而非靠手工阈值。
        rj = out / "result.json"
        assert rj.exists(), "predict.py 未落 result.json"
        data = json.loads(rj.read_text(encoding="utf-8"))
        status = data.get("status")
        # R3(2026-08-10)后的硬契约:authenticity 权重缺失在 build_auth_module 降级 None,
        # **绝不再让整例走 ERROR**。故缺权重环境下 predict.py 的最终 status 必须是 PASSED
        # (soft 模式、诊断照出)或 REJECTED(dedup/tamper 硬判据,平台浮点敏感),**不得 ERROR**。
        # 这是正向断言"降级确实生效"(比原来'不是auth-ERROR'更强、且不耦合任何错误文案字面量):
        # 若这里出现 ERROR,要么 R3 降级没生效、要么降级路径在别处崩,两者都是必须抓的回归。
        assert status in ("REJECTED", "PASSED"), (
            f"缺 authenticity 权重时 status={status!r}(期望 PASSED/REJECTED)。"
            f"R3 后该故障应降级 skipped 而非整例 ERROR:\n{data}")


# ═════════════════════════════════════════════════════════════════════════════
# ERROR 路径:输入文件加载失败(不存在/空文件)→ 必须落 result.json,不能零输出
# （2026-07-11 修复的 P0 回归：此前 load_input_image/load_video_frames 加载失败时,
#  两个入口脚本在 try 块外直接 return,只 print 不落盘——output_dir 是空目录,比
#  gatekeeper REJECTED 还差,还绕开了 except Exception 那层安全网）
# ═════════════════════════════════════════════════════════════════════════════

class TestLoadFailureErrorFallback:
    def test_predict_missing_image_writes_error_json(self, tmp_path):
        missing = tmp_path / "does_not_exist.png"
        out = tmp_path / "out"
        r = _run_entry("predict.py", [
            "--input_image", str(missing),
            "--output_dir", str(out),
            "--precision", "fp32",
        ])
        assert r.returncode == 0, \
            f"输入文件不存在时不应崩溃\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        rj = out / "result.json"
        assert rj.exists(), "输入文件不存在时仍必须落 result.json(不能零输出空目录)"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data.get("status") == "ERROR", f"预期 status=ERROR，实际 {data}"
        assert "reason" in data and data["reason"], "reason 不应为空"

    def test_video_predict_missing_video_writes_error_json(self, tmp_path):
        missing = tmp_path / "does_not_exist.mp4"
        out = tmp_path / "vout"
        r = _run_entry("video_predict.py", [
            "--input_video", str(missing),
            "--output_dir", str(out),
            "--precision", "fp32",
        ])
        assert r.returncode == 0, \
            f"输入视频不存在时不应崩溃\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        rj = out / "result.json"
        assert rj.exists(), "输入视频不存在时仍必须落 result.json(不能零输出空目录)"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data.get("status") == "ERROR", f"预期 status=ERROR，实际 {data}"
        assert "reason" in data and data["reason"], "reason 不应为空"

    def test_predict_empty_file_writes_error_json(self, tmp_path):
        """空文件(0 字节,无扩展名)→ cv2.imread 必然解码失败,同样必须落 ERROR。"""
        empty = tmp_path / "empty_no_ext"
        empty.write_bytes(b"")
        out = tmp_path / "out_empty"
        r = _run_entry("predict.py", [
            "--input_image", str(empty),
            "--output_dir", str(out),
            "--precision", "fp32",
        ])
        assert r.returncode == 0, \
            f"空文件不应崩溃\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        rj = out / "result.json"
        assert rj.exists(), "空文件仍必须落 result.json(不能零输出空目录)"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data.get("status") == "ERROR", f"预期 status=ERROR，实际 {data}"
        assert "reason" in data and data["reason"], "reason 不应为空"

    def test_video_predict_empty_file_writes_error_json(self, tmp_path):
        """空文件(0 字节,无扩展名)→ cv2.VideoCapture 打开必然失败,同样必须落 ERROR。
        与 test_predict_empty_file_writes_error_json 对称补齐(2026-07-11 Gate2
        review 非阻塞建议:此前只测了 video 侧 missing,没测 empty,覆盖度不对称)。"""
        empty = tmp_path / "empty_no_ext"
        empty.write_bytes(b"")
        out = tmp_path / "vout_empty"
        r = _run_entry("video_predict.py", [
            "--input_video", str(empty),
            "--output_dir", str(out),
            "--precision", "fp32",
        ])
        assert r.returncode == 0, \
            f"空文件不应崩溃\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        rj = out / "result.json"
        assert rj.exists(), "空文件仍必须落 result.json(不能零输出空目录)"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data.get("status") == "ERROR", f"预期 status=ERROR，实际 {data}"
        assert "reason" in data and data["reason"], "reason 不应为空"


# ═════════════════════════════════════════════════════════════════════════════
# ERROR 路径:--config 加载失败(路径不存在/YAML 格式错误/键缺失)→ 必须落
# result.json,不能零输出(2026-07-11 修复：与上面输入加载失败同一类"try 外裸
# 操作"问题,此前 open(args.config)/yaml.safe_load/config["environment"]["device"]
# 都在 try 块外,见 model_runner.load_config docstring)
# ═════════════════════════════════════════════════════════════════════════════

class TestConfigLoadFailureErrorFallback:
    def test_predict_missing_config_writes_error_json(self, tmp_path):
        img = tmp_path / "noise.png"
        _write_noise_png(img)
        missing_cfg = tmp_path / "does_not_exist.yaml"
        out = tmp_path / "out"
        r = _run_entry("predict.py", [
            "--input_image", str(img),
            "--config", str(missing_cfg),
            "--output_dir", str(out),
            "--precision", "fp32",
        ])
        assert r.returncode == 0, \
            f"--config 路径不存在时不应崩溃\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        rj = out / "result.json"
        assert rj.exists(), "--config 缺失时仍必须落 result.json(不能零输出空目录)"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data.get("status") == "ERROR", f"预期 status=ERROR，实际 {data}"
        assert "reason" in data and data["reason"], "reason 不应为空"

    def test_predict_malformed_config_writes_error_json(self, tmp_path):
        img = tmp_path / "noise.png"
        _write_noise_png(img)
        bad_cfg = tmp_path / "malformed.yaml"
        bad_cfg.write_text('environment: "device: cuda:0\n', encoding="utf-8")
        out = tmp_path / "out2"
        r = _run_entry("predict.py", [
            "--input_image", str(img),
            "--config", str(bad_cfg),
            "--output_dir", str(out),
            "--precision", "fp32",
        ])
        assert r.returncode == 0, \
            f"--config 格式错误时不应崩溃\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        rj = out / "result.json"
        assert rj.exists(), "--config 格式错误时仍必须落 result.json"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data.get("status") == "ERROR", f"预期 status=ERROR，实际 {data}"
        assert "reason" in data and data["reason"], "reason 不应为空"

    def test_video_predict_missing_config_writes_error_json(self, tmp_path):
        vid = tmp_path / "clip.mp4"
        if not _write_noise_mp4(vid):
            pytest.skip("mp4v codec 不可用,跳过本用例")
        missing_cfg = tmp_path / "does_not_exist.yaml"
        out = tmp_path / "vout"
        r = _run_entry("video_predict.py", [
            "--input_video", str(vid),
            "--config", str(missing_cfg),
            "--output_dir", str(out),
            "--precision", "fp32",
        ])
        assert r.returncode == 0, \
            f"--config 路径不存在时不应崩溃\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        rj = out / "result.json"
        assert rj.exists(), "--config 缺失时仍必须落 result.json"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data.get("status") == "ERROR", f"预期 status=ERROR，实际 {data}"
        assert "reason" in data and data["reason"], "reason 不应为空"

    def test_video_predict_malformed_config_writes_error_json(self, tmp_path):
        """与 test_predict_malformed_config_writes_error_json 对称补齐（2026-07-11
        Gate2 review 发现缺口：此前 video_predict.py 侧只测了 config 缺失，没测格式
        错误，虽然 load_config() 本身的 malformed 场景已被
        test_model_runner.py::test_malformed_yaml_raises 单元覆盖，但子进程级冒烟
        应锁死 video_predict.py 真实调用链路，而非只靠间接覆盖）。"""
        vid = tmp_path / "clip.mp4"
        if not _write_noise_mp4(vid):
            pytest.skip("mp4v codec 不可用,跳过本用例")
        bad_cfg = tmp_path / "malformed.yaml"
        bad_cfg.write_text('environment: "device: cuda:0\n', encoding="utf-8")
        out = tmp_path / "vout2"
        r = _run_entry("video_predict.py", [
            "--input_video", str(vid),
            "--config", str(bad_cfg),
            "--output_dir", str(out),
            "--precision", "fp32",
        ])
        assert r.returncode == 0, \
            f"--config 格式错误时不应崩溃\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        rj = out / "result.json"
        assert rj.exists(), "--config 格式错误时仍必须落 result.json"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data.get("status") == "ERROR", f"预期 status=ERROR，实际 {data}"
        assert "reason" in data and data["reason"], "reason 不应为空"


# ═════════════════════════════════════════════════════════════════════════════
# PASSED 路径:video_predict.py --skip_gatekeeper → 全产物
# ═════════════════════════════════════════════════════════════════════════════

class TestVideoPassedPath:
    def test_full_pipeline_produces_all_artifacts(self, tmp_path):
        vid = tmp_path / "clip.mp4"
        if not _write_noise_mp4(vid):
            pytest.skip("mp4v codec 不可用,跳过 PASSED 端到端用例")
        out = tmp_path / "vout"
        r = _run_entry("video_predict.py", [
            "--input_video", str(vid),
            "--output_dir", str(out),
            "--precision", "fp32",
            "--skip_gatekeeper",
            "--enable_gradcam",
            "--max_frames", "4",
        ])
        if r.returncode != 0 and "decode" in (r.stdout + r.stderr).lower():
            pytest.skip(f"平台视频解码不可用: {r.stderr[-200:]}")
        assert r.returncode == 0, f"video_predict.py 非零退出\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"

        # 四件交付产物齐全
        for name in ("result.json", "clinical_report.txt",
                     "mask_predicted.png", "gradcam_cls.png"):
            assert (out / name).exists(), f"缺产物 {name}\nSTDOUT:{r.stdout}"

        # JSON 结构 + 中文可正确还原(评测系统读的就是它)
        data = json.loads((out / "result.json").read_text(encoding="utf-8"))
        assert data["status"] == "PASSED"
        assert data["input_type"] == "video_cine"
        assert 0.0 <= data["malignancy_probability"] <= 1.0
        assert data["gradcam_generated"] == 1
        assert len(data["per_frame_malignancy"]) == data["n_frames_sampled"]
        assert "高度可疑" in data["tirads_level"] or "TR" in data["tirads_level"]

        # 临床报告含中文且非空
        rpt = (out / "clinical_report.txt").read_text(encoding="utf-8")
        assert len(rpt) > 100


# ═════════════════════════════════════════════════════════════════════════════
# ERROR 路径:权重与模型结构不匹配 → build_model 硬报错须优雅降级为 result.json
# （2026-07-10 code review 发现的真实回归：build_model 从 strict=False 静默容忍
#  改为硬 raise 后，若无兜底会让容器裸崩溃、零输出，比预测错误更差）
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildModelErrorFallback:
    def test_video_predict_writes_error_json_on_checkpoint_mismatch(self, tmp_path):
        import torch

        vid = tmp_path / "clip.mp4"
        if not _write_noise_mp4(vid):
            pytest.skip("mp4v codec 不可用,跳过")

        # 构造一个与模型结构不匹配的 checkpoint（模拟未合并的 LoRA checkpoint /
        # 版本漂移场景），video_predict.py 里 build_model() 在 gatekeeper 之前
        # 就会被调用，无需 --skip_gatekeeper 也能确定性触发这条路径。
        bad_ckpt = tmp_path / "mismatched.pth"
        torch.save({"some_bogus_key": torch.randn(2, 2)}, bad_ckpt)

        out = tmp_path / "vout_err"
        r = _run_entry("video_predict.py", [
            "--input_video", str(vid),
            "--output_dir", str(out),
            "--weights", str(bad_ckpt),
            "--precision", "fp32",
            "--skip_gatekeeper",
        ])
        assert r.returncode == 0, \
            f"权重不匹配时进程不应崩溃，必须优雅降级\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"

        rj = out / "result.json"
        assert rj.exists(), "权重不匹配时仍必须落 result.json（不能零输出）"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data["status"] == "ERROR", f"预期 status=ERROR，实际 {data}"
        assert "reason" in data and data["reason"]

    def test_predict_writes_error_json_on_checkpoint_mismatch(self, tmp_path):
        """predict.py 与 video_predict.py 是同款 except Exception 兜底，但调用顺序
        不同（predict.py 先 gatekeeper 后 build_model），此前只测了 video_predict.py
        一侧，predict.py 这条新分支从未被验证过是否可达（2026-07-10 Gate2审查发现）。
        """
        import torch

        img = tmp_path / "noise.png"
        _write_noise_png(img)

        bad_ckpt = tmp_path / "mismatched.pth"
        torch.save({"some_bogus_key": torch.randn(2, 2)}, bad_ckpt)

        out = tmp_path / "out_err"
        r = _run_entry("predict.py", [
            "--input_image", str(img),
            "--output_dir", str(out),
            "--weights", str(bad_ckpt),
            "--precision", "fp32",
        ])
        assert r.returncode == 0, \
            f"权重不匹配时进程不应崩溃，必须优雅降级\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"

        rj = out / "result.json"
        assert rj.exists(), "权重不匹配时仍必须落 result.json（不能零输出）"
        data = json.loads(rj.read_text(encoding="utf-8"))
        # gatekeeper 可能先一票否决（REJECTED，随机噪声图属正常业务结果），也可能
        # 放行后触发 build_model 的权重不匹配 ERROR——两条都验证了「不裸崩溃、
        # 必落 result.json」，只在真走到 ERROR 分支时才断言 reason 非空。
        assert data.get("status") in ("REJECTED", "ERROR"), \
            f"status 非法: {data.get('status')}\nSTDOUT:{r.stdout}"
        if data.get("status") == "ERROR":
            assert "reason" in data and data["reason"]


# ═════════════════════════════════════════════════════════════════════════════
# DICOM 单帧/多帧 cine → video_predict.py --skip_gatekeeper 全产物
# (2026-07-11 P1：docker-build-smoke 输入覆盖扩容第一步，本地先验证同一份
#  scripts.verify.gen_synthetic_dicom.build_dicom 生成的 DICOM 能被真实链路
#  完整跑通——与 test_entrypoint_dispatch.py::_write_dicom 那种只写魔数字节
#  供 bash 路由测试的假文件不同，这里是真正能被 pydicom.dcmread() 解析、
#  video_io._load_dicom_cine 能读出 pixel_array 的合法 DICOM)
# entrypoint.sh 把 .dcm/.dicom 无条件路由到 video_predict.py（单帧也不例外，
# _load_dicom_cine 会把 2D pixel_array 包装成 (1,H,W) 当单帧 cine 处理，见
# 本轮 P1 计划里"已确认不是 bug 的发现"），故这里统一走 video_predict.py。
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_PYDICOM, reason="pydicom 未安装，跳过 DICOM 冒烟测试")
class TestDicomInputPaths:
    def test_single_frame_dicom_passes(self, tmp_path):
        dcm = tmp_path / "single.dcm"
        build_dicom(rows=32, cols=32, n_frames=1).save_as(str(dcm), enforce_file_format=True)
        out = tmp_path / "vout_dcm1"
        r = _run_entry("video_predict.py", [
            "--input_video", str(dcm),
            "--output_dir", str(out),
            "--precision", "fp32",
            "--skip_gatekeeper",
        ])
        assert r.returncode == 0, \
            f"单帧 DICOM 不应崩溃\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        rj = out / "result.json"
        assert rj.exists(), "单帧 DICOM 仍必须落 result.json"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data.get("status") == "PASSED", f"预期 status=PASSED，实际 {data}\nSTDOUT:{r.stdout}"
        for name in ("result.json", "clinical_report.txt", "mask_predicted.png"):
            assert (out / name).exists(), f"缺产物 {name}"

    def test_multi_frame_dicom_cine_passes(self, tmp_path):
        dcm = tmp_path / "cine.dcm"
        build_dicom(rows=32, cols=32, n_frames=4).save_as(str(dcm), enforce_file_format=True)
        out = tmp_path / "vout_dcm4"
        r = _run_entry("video_predict.py", [
            "--input_video", str(dcm),
            "--output_dir", str(out),
            "--precision", "fp32",
            "--skip_gatekeeper",
        ])
        assert r.returncode == 0, \
            f"多帧 DICOM cine 不应崩溃\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        rj = out / "result.json"
        assert rj.exists(), "多帧 DICOM cine 仍必须落 result.json"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data.get("status") == "PASSED", f"预期 status=PASSED，实际 {data}\nSTDOUT:{r.stdout}"
        for name in ("result.json", "clinical_report.txt", "mask_predicted.png"):
            assert (out / name).exists(), f"缺产物 {name}"

    @pytest.mark.skipif(not _HAS_PYLIBJPEG, reason="pylibjpeg 未安装，跳过压缩 DICOM 冒烟测试")
    @pytest.mark.parametrize("n_frames", [1, 4], ids=["single-frame", "multi-frame-cine"])
    def test_compressed_dicom_passes(self, tmp_path, n_frames):
        """2026-07-11 Opus 战略审查发现的致命缺口补测：真实超声 DICOM 绝大多数是
        JPEG/JPEG2000/RLE 压缩，此前 fixture 只测过未压缩 ExplicitVRLittleEndian
        ——若压缩解码插件缺失，`ds.pixel_array` 会抛 NotImplementedError（被
        `_load_dicom_cine` 兜成 ERROR 不崩溃，但等于该动态图样本 100% 丢失预测）。
        `requirements.txt` 已补 pylibjpeg/pylibjpeg-libjpeg，这里验证 JPEG
        baseline 压缩的 DICOM 经真实链路完整跑通产出 PASSED。
        参数化覆盖单帧 + 多帧 cine（2026-07-11 Gate2 review 非阻塞建议补齐：
        此前只测过压缩+单帧组合，压缩+多帧 cine 这个组合未覆盖过）。"""
        dcm = tmp_path / "compressed.dcm"
        build_dicom(rows=32, cols=32, n_frames=n_frames, compressed=True).save_as(
            str(dcm), enforce_file_format=True
        )
        out = tmp_path / "vout_dcm_compressed"
        r = _run_entry("video_predict.py", [
            "--input_video", str(dcm),
            "--output_dir", str(out),
            "--precision", "fp32",
            "--skip_gatekeeper",
        ])
        assert r.returncode == 0, \
            f"压缩 DICOM 不应崩溃\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        rj = out / "result.json"
        assert rj.exists(), "压缩 DICOM 仍必须落 result.json"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data.get("status") == "PASSED", \
            f"预期 status=PASSED(压缩解码插件应正确解码),实际 {data}\nSTDOUT:{r.stdout}"
        for name in ("result.json", "clinical_report.txt", "mask_predicted.png"):
            assert (out / name).exists(), f"缺产物 {name}"


# ═════════════════════════════════════════════════════════════════════════════
# JPG 静态图 → predict.py 全产物(官方确认静态图交付格式就是 JPG,此前只测过 PNG)
# ═════════════════════════════════════════════════════════════════════════════

class TestJpegInputPath:
    def test_jpeg_static_image_reject_path_runs(self, tmp_path):
        """2026-07-11 P1 收尾补测：docs/比赛对齐与风险.md 确认官方静态图交付格式
        是 JPG，但此前所有冒烟/单测全部用 PNG 构造输入，JPG 路径从未被真实验证
        过——cv2.imread 本身对两种格式一视同仁，但零成本的事该补，直接对齐交付
        格式。"""
        img = tmp_path / "noise.jpg"
        arr = (np.random.rand(400, 500) * 255).astype(np.uint8)
        import cv2
        cv2.imwrite(str(img), arr)
        out = tmp_path / "out_jpg"
        r = _run_entry("predict.py", [
            "--input_image", str(img),
            "--output_dir", str(out),
            "--precision", "fp32",
        ])
        assert r.returncode == 0, \
            f"JPG 静态图不应崩溃\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        rj = out / "result.json"
        assert rj.exists(), "JPG 静态图仍必须落 result.json"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data.get("status") in ("REJECTED", "ERROR", "PASSED"), \
            f"status 非法: {data.get('status')}\nSTDOUT:{r.stdout}"


# ═════════════════════════════════════════════════════════════════════════════
# 非空损坏文件 → predict.py 必须 ERROR，不能裸崩溃
# (2026-07-11 P1：与已有的空文件用例区分——空文件是 0 字节，cv2.imread 立即
#  返回 None；这里是非空但内容不构成任何合法图像格式，验证 cv2.imread 在这种
#  "看起来像文件、内容却是垃圾"输入下的失败路径同样被正确兜住)
# ═════════════════════════════════════════════════════════════════════════════

class TestCorruptedNonEmptyFile:
    def test_predict_corrupted_nonempty_file_writes_error_json(self, tmp_path):
        corrupted = tmp_path / "corrupted.png"
        # 非空、有 .png 扩展名，但内容是随机字节，不构成任何合法 PNG 结构
        corrupted.write_bytes(os.urandom(2048))
        out = tmp_path / "out_corrupted"
        r = _run_entry("predict.py", [
            "--input_image", str(corrupted),
            "--output_dir", str(out),
            "--precision", "fp32",
        ])
        assert r.returncode == 0, \
            f"损坏文件不应崩溃\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        rj = out / "result.json"
        assert rj.exists(), "损坏文件仍必须落 result.json（不能零输出）"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data.get("status") == "ERROR", f"预期 status=ERROR，实际 {data}"
        assert "reason" in data and data["reason"], "reason 不应为空"
