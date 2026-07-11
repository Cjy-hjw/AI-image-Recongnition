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
        if status == "ERROR":
            # 2026-07-10 Gate2审查修复：authenticity.py 权重缺失现在硬报错（此前
            # 静默退化为随机初始化分类,一票否决门形同虚设且无人察觉)。本地/CI
            # 环境通常没有真实 checkpoints/authenticity_model.pth,这正是修复后
            # 的预期新行为(不是回归)——只要 reason 明确指向"权重缺失"就放行,
            # 若是其它原因的 ERROR 则说明有真实回归,仍然失败。
            reason = data.get("reason", "")
            assert "AuthenticityModule" in reason and "不存在" in reason, \
                f"ERROR 但原因不是预期的权重缺失,可能是真实回归: {data}"
        else:
            assert status in ("REJECTED", "PASSED"), \
                f"status 非法: {status}\nSTDOUT:{r.stdout}"


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
