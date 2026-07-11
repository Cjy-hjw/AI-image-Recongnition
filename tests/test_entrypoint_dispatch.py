"""entrypoint.sh 交付入口的分发路由 + 防篡改自检 + src 树指纹测试。

entrypoint.sh 是容器唯一入口,负责:① 按输入扩展名分发到 predict.py(静态图)
或 video_predict.py(视频/DICOM cine);② PRODUCTION=1 时对入口脚本/src 全树/
权重做 SHA-256 完整性自检,任一不符即阻断。这些 bash 级逻辑此前无任何测试覆盖。

测试策略(不真跑推理 —— 脚本结尾 `exec python <script>` 否则会加载 torch):
  - 分发路由:PATH 前置一个假 `python` shim(只回显参数),验证 exec 收到的脚本名
  - PRODUCTION 阻断:哈希不符在 exec 之前 exit 1,捕获退出码与错误信息
  - src 树指纹:bash compute_src_tree_sha256 == scripts/seal_container.py 的 Python 实现

bash 不可用(罕见,CI Linux 与本地 Git Bash 均有)时整文件 skip。
"""
import os
import shutil
import stat
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRYPOINT = os.path.join(REPO_ROOT, "scripts", "entrypoint.sh")
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="bash 不可用,跳过 entrypoint 测试")


def _make_python_shim(dir_path):
    """在 dir_path 造一个假 python/python3,回显 `SHIM_ARGS: <args>` 后退出 0。

    entrypoint.sh 结尾 `exec python <script> ...` 会命中它,从而在不加载 torch、
    不真跑推理的前提下捕获被分发到的脚本名与透传参数。
    """
    shim = os.path.join(dir_path, "python")
    with open(shim, "w", encoding="utf-8", newline="\n") as f:
        f.write('#!/bin/sh\necho "SHIM_ARGS: $@"\nexit 0\n')
    os.chmod(shim, os.stat(shim).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    # 某些环境 entrypoint 用 python3;两个都铺上
    shutil.copy2(shim, os.path.join(dir_path, "python3"))
    return shim


def _write_png(path):
    """写一个带合法 PNG 魔数(\\x89PNG\\r\\n\\x1a\\n + IHDR chunk 头)的最小文件。"""
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR")


def _write_mp4(path):
    """写一个带合法 MP4 ftyp box 魔数(isom brand)的最小文件。"""
    with open(path, "wb") as f:
        f.write(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2avc1mp41")


def _write_dicom(path):
    """写一个带合法 DICOM 魔数的最小文件:128 字节任意 preamble + offset 128 处 'DICM'。"""
    with open(path, "wb") as f:
        f.write(b"\x00" * 128 + b"DICM" + b"\x02\x00\x00\x00garbage")


def _write_unrecognized(path):
    """写一段既非图片也非视频/DICOM 魔数的内容(纯文本填充,file(1) 会判为 text/plain)。"""
    with open(path, "wb") as f:
        f.write(b"this is filler content matching no known image/video/DICOM magic bytes at all")


def _run_entrypoint(input_path, extra_args=None, env_extra=None, shim_dir=None):
    env = dict(os.environ)
    if shim_dir:
        env["PATH"] = shim_dir + os.pathsep + env["PATH"]
    if env_extra:
        env.update(env_extra)
    cmd = [BASH, ENTRYPOINT, input_path, *(extra_args or [])]
    return subprocess.run(
        cmd, cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 分发路由:扩展名 → 正确入口脚本(开发模式,绕过哈希自检)
# ═════════════════════════════════════════════════════════════════════════════

class TestDispatch:
    @pytest.mark.parametrize("ext", ["mp4", "avi", "mov", "mkv", "dcm", "dicom",
                                     "MP4", "DCM"])
    def test_dynamic_inputs_route_to_video(self, ext, tmp_path):
        shim = str(tmp_path / "shim")
        os.makedirs(shim)
        _make_python_shim(shim)
        r = _run_entrypoint(f"/tmp/case.{ext}", shim_dir=shim)
        assert r.returncode == 0, f"STDERR:{r.stderr}"
        assert "SHIM_ARGS: video_predict.py --input_video" in r.stdout, \
            f"扩展名 .{ext} 应路由到 video\n{r.stdout}"

    @pytest.mark.parametrize("ext", ["png", "jpg", "jpeg", "bmp", "tif", "PNG"])
    def test_static_inputs_route_to_predict(self, ext, tmp_path):
        shim = str(tmp_path / "shim")
        os.makedirs(shim)
        _make_python_shim(shim)
        r = _run_entrypoint(f"/tmp/case.{ext}", shim_dir=shim)
        assert r.returncode == 0, f"STDERR:{r.stderr}"
        # 看实际分发结果(shim 回显的 exec 参数),而非启动横幅 —— 横幅里
        # "video_predict.py SHA-256" 自检行必然含该字样,不能用全局 not-in 判定
        assert "SHIM_ARGS: predict.py --input_image" in r.stdout, \
            f"扩展名 .{ext} 应路由到 predict\n{r.stdout}"

    def test_extra_flags_passed_through(self, tmp_path):
        shim = str(tmp_path / "shim")
        os.makedirs(shim)
        _make_python_shim(shim)
        r = _run_entrypoint("/tmp/case.png", extra_args=["--precision", "bf16"],
                            shim_dir=shim)
        assert r.returncode == 0
        assert "--precision bf16" in r.stdout, f"透传 flag 丢失\n{r.stdout}"


# ═════════════════════════════════════════════════════════════════════════════
# PRODUCTION=1 防篡改阻断:哈希不符必须在 exec 之前 exit 1
# ═════════════════════════════════════════════════════════════════════════════

class TestProductionIntegrity:
    def test_tampered_predict_hash_blocks(self, tmp_path):
        # EXPECTED_PREDICT_SHA256 仍是占位示例值,实际 predict.py 哈希必然不符 → 阻断
        shim = str(tmp_path / "shim")
        os.makedirs(shim)
        _make_python_shim(shim)
        r = _run_entrypoint("/tmp/case.png", env_extra={"PRODUCTION": "1"},
                            shim_dir=shim)
        assert r.returncode == 1, "PRODUCTION 下哈希不符应 exit 1"
        assert "integrity hash mismatch" in r.stderr.lower() or \
               "mismatch" in r.stderr.lower(), f"未报篡改\nSTDERR:{r.stderr}"
        # 必须在 exec python 之前阻断 —— shim 不应被调用
        assert "SHIM_ARGS" not in r.stdout, "阻断失败:已执行到 python"

    def test_dev_mode_bypasses_integrity(self, tmp_path):
        # 不设 PRODUCTION → 开发模式,跳过哈希校验直接分发
        shim = str(tmp_path / "shim")
        os.makedirs(shim)
        _make_python_shim(shim)
        r = _run_entrypoint("/tmp/case.png", shim_dir=shim)
        assert r.returncode == 0
        assert "DEVELOPMENT MODE" in r.stdout
        assert "SHIM_ARGS" in r.stdout, "开发模式应正常分发到 python"

    def test_src_tree_mismatch_blocks_after_entry_hashes_pass(self, tmp_path):
        """覆盖 src 树阻断分支(reviewer E2):predict/video 哈希都对、仅 src 树不符,
        必须在第二道 exit 拦下,不放行到 exec。否则 src 防篡改链等于没有。"""
        sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
        import seal_container
        # 把 entrypoint.sh + _src_tree_hash.sh 拷到临时区,注入真实 predict/video
        # 哈希(令前两道通过),但 EXPECTED_SRC_TREE 故意留错值 → 触发 src 树阻断
        ep = tmp_path / "entrypoint.sh"
        ep.write_text(
            (open(ENTRYPOINT, encoding="utf-8")).read()
            .replace("EXPECTED_PREDICT_SHA256=\"4d7c50a1d62c9adb5a9513e72c4dedafc1cb2d4c5236c9a6957ec7dfd5a9ca22\"",
                     f'EXPECTED_PREDICT_SHA256="{seal_container._sha256_file(os.path.join(REPO_ROOT, "predict.py"))}"')
            .replace("EXPECTED_VIDEO_PREDICT_SHA256=\"0000000000000000000000000000000000000000000000000000000000000000\"",
                     f'EXPECTED_VIDEO_PREDICT_SHA256="{seal_container._sha256_file(os.path.join(REPO_ROOT, "video_predict.py"))}"'),
            encoding="utf-8", newline="\n")
        shutil.copy2(os.path.join(REPO_ROOT, "scripts", "_src_tree_hash.sh"),
                     tmp_path / "_src_tree_hash.sh")
        shim = str(tmp_path / "shim")
        os.makedirs(shim)
        _make_python_shim(shim)
        env = dict(os.environ, PRODUCTION="1",
                   PATH=shim + os.pathsep + os.environ["PATH"])
        r = subprocess.run([BASH, str(ep), "/tmp/case.png"],
                           cwd=REPO_ROOT, env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 1, f"src 树不符应 exit 1\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        assert "src/ tree integrity" in r.stderr.lower() or \
               "src/ tree" in r.stderr, f"未报 src 树篡改\nSTDERR:{r.stderr}"
        assert "SHIM_ARGS" not in r.stdout, "阻断失败:已执行到 python"


# ═════════════════════════════════════════════════════════════════════════════
# src 树聚合指纹:bash(entrypoint)与 Python(seal_container)字节级一致
# ═════════════════════════════════════════════════════════════════════════════

class TestSrcTreeHashParity:
    def test_bash_matches_python(self):
        # bash 端:source entrypoint.sh 实际使用的同一份 _src_tree_hash.sh,
        # 调真正的 compute_src_tree_sha256(而非内联手抄,杜绝 drift 假绿:
        # 一旦该函数被改坏,本测试必须能抓到)
        hash_sh = os.path.join(REPO_ROOT, "scripts", "_src_tree_hash.sh")
        r = subprocess.run(
            [BASH, "-c", f'cd "{REPO_ROOT}"; source "{hash_sh}"; '
                         f'compute_src_tree_sha256'],
            capture_output=True, text=True, timeout=120,
        )
        bash_sha = r.stdout.strip()
        assert len(bash_sha) == 64, f"bash 指纹异常: {r.stdout}\n{r.stderr}"

        # Python 端:直接调 seal_container
        sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
        import seal_container
        py_sha, files = seal_container.src_tree_sha256(REPO_ROOT)
        assert len(files) > 0, "src 树未找到 .py 文件"
        assert bash_sha == py_sha, \
            f"src 树指纹不一致!\n  bash:  {bash_sha}\n  python:{py_sha}"


# ═════════════════════════════════════════════════════════════════════════════
# 内容嗅探兜底:扩展名与实际 magic bytes 不符时重新路由;完全无法识别时硬失败
# (官方"动态图"交付格式未知风险的加固,见 scripts/entrypoint.sh 第5节注释)
# ═════════════════════════════════════════════════════════════════════════════

class TestContentSniffFallback:
    def test_ext_says_image_content_is_video_reroutes_to_video(self, tmp_path):
        # 扩展名 .jpg(静态图桶)但内容其实是 MP4 ftyp box -> 必须以内容为准重路由到 video
        shim = str(tmp_path / "shim")
        os.makedirs(shim)
        _make_python_shim(shim)
        case = tmp_path / "case.jpg"
        _write_mp4(case)
        r = _run_entrypoint(str(case), shim_dir=shim)
        assert r.returncode == 0, f"STDERR:{r.stderr}"
        assert "SHIM_ARGS: video_predict.py --input_video" in r.stdout, \
            f"内容嗅探为视频时应重路由到 video_predict.py\n{r.stdout}"
        assert "WARNING" in r.stdout and "mismatch" in r.stdout.lower(), \
            f"重路由必须打印不一致警告,不能静默\n{r.stdout}"

    def test_ext_says_video_content_is_image_reroutes_to_predict(self, tmp_path):
        # 扩展名 .mp4(动态桶)但内容其实是 PNG -> 必须以内容为准重路由到 predict
        shim = str(tmp_path / "shim")
        os.makedirs(shim)
        _make_python_shim(shim)
        case = tmp_path / "case.mp4"
        _write_png(case)
        r = _run_entrypoint(str(case), shim_dir=shim)
        assert r.returncode == 0, f"STDERR:{r.stderr}"
        assert "SHIM_ARGS: predict.py --input_image" in r.stdout, \
            f"内容嗅探为图片时应重路由到 predict.py\n{r.stdout}"
        assert "WARNING" in r.stdout and "mismatch" in r.stdout.lower(), \
            f"重路由必须打印不一致警告,不能静默\n{r.stdout}"

    def test_matching_ext_and_content_no_spurious_warning(self, tmp_path):
        # 扩展名与内容一致时不应误报 mismatch 警告(嗅探只做交叉校验,不该添噪声)
        shim = str(tmp_path / "shim")
        os.makedirs(shim)
        _make_python_shim(shim)
        case = tmp_path / "case.png"
        _write_png(case)
        r = _run_entrypoint(str(case), shim_dir=shim)
        assert r.returncode == 0, f"STDERR:{r.stderr}"
        assert "SHIM_ARGS: predict.py --input_image" in r.stdout
        # 只判定"扩展名/内容不一致重路由"这条特定警告未被误触发;不能用裸 "WARNING"
        # 断言 —— 启动横幅本身就有与本功能无关的 "[WARNING] Trained weights not found"。
        assert "mismatches content sniff" not in r.stdout, \
            f"扩展名与内容一致不应误报 mismatch 重路由警告\n{r.stdout}"

    def test_dicom_content_without_whitelisted_ext_routes_to_video(self, tmp_path):
        # 扩展名 .raw(不在动态白名单内,按扩展名会被误判成静态图)但内容有合法 DICOM
        # 魔数(offset 128 "DICM")-> 必须靠内容嗅探识别出 DICOM 并重路由到 video_predict.py。
        # 这正是 DICOM 128 字节 preamble 特殊性(不能只靠扩展名/file(1) magic db 猜)的回归验证。
        shim = str(tmp_path / "shim")
        os.makedirs(shim)
        _make_python_shim(shim)
        case = tmp_path / "case.raw"
        _write_dicom(case)
        r = _run_entrypoint(str(case), shim_dir=shim)
        assert r.returncode == 0, f"STDERR:{r.stderr}"
        assert "SHIM_ARGS: video_predict.py --input_video" in r.stdout, \
            f"DICOM 魔数应重路由到 video_predict.py,不受扩展名白名单限制\n{r.stdout}"
        assert "WARNING" in r.stdout, f"扩展名与内容(DICOM)不一致应报警告\n{r.stdout}"

    def test_unrecognized_content_hard_fails_with_error_json(self, tmp_path):
        # 文件真实存在且可读,但内容既非 DICOM 也非 image/*、video/* -> 不能瞎猜路由,
        # 必须 exit 1 + 不 exec 任何 python 入口 + 落 status=ERROR 的 result.json。
        shim = str(tmp_path / "shim")
        os.makedirs(shim)
        _make_python_shim(shim)
        case = tmp_path / "case.jpg"
        _write_unrecognized(case)
        out_dir = tmp_path / "out"
        r = _run_entrypoint(str(case), extra_args=["--output_dir", str(out_dir)],
                            shim_dir=shim)
        assert r.returncode == 1, f"无法识别格式应 exit 1\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
        assert "SHIM_ARGS" not in r.stdout, "无法识别格式不该分发到任何 python 入口"
        assert "CRITICAL ERROR" in r.stderr, f"未报无法识别格式错误\nSTDERR:{r.stderr}"

        result_json = out_dir / "result.json"
        assert result_json.exists(), f"无法识别格式必须落 ERROR result.json 到 --output_dir\n{r.stderr}"
        import json
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        assert payload["status"] == "ERROR", f"result.json status 应为 ERROR: {payload}"
        assert "reason" in payload and payload["reason"], f"result.json 缺 reason: {payload}"

    def test_nonexistent_file_skips_sniff_falls_back_to_extension(self, tmp_path):
        # 文件不存在(评测/开发中常见占位场景,见既有 TestDispatch 用例)-> 无法嗅探,
        # 必须退化为纯扩展名路由,不能因为读不到文件就误判成"无法识别"从而硬失败。
        shim = str(tmp_path / "shim")
        os.makedirs(shim)
        _make_python_shim(shim)
        missing = tmp_path / "does_not_exist.mp4"
        r = _run_entrypoint(str(missing), shim_dir=shim)
        assert r.returncode == 0, f"文件不存在不该硬失败,应退化为扩展名路由\nSTDERR:{r.stderr}"
        assert "SHIM_ARGS: video_predict.py --input_video" in r.stdout
        assert "mismatches content sniff" not in r.stdout, "无嗅探依据时不应误报 mismatch 重路由警告"

