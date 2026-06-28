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
        # gatekeeper 应判噪声为非真实图并拒绝
        assert "REJECTED" in r.stdout, f"未走拒绝分支\nSTDOUT:{r.stdout}"
        rj = out / "result.json"
        assert rj.exists(), "拒绝路径未落 result.json"
        data = json.loads(rj.read_text(encoding="utf-8"))
        assert data.get("status") != "PASSED"


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
