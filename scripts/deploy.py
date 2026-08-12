import os
import sys
import paramiko

# === 服务器连接凭证:优先环境变量,其次 账号密码/deploy.env(该目录 .gitignore
# 已排除、严禁入库,见目录内 ssh.md)。两者都缺失才 fail-fast——不再要求每次
# 手动 export,但仍不写死进本文件本身。 ===
_CRED_FILE = os.path.join(os.path.dirname(__file__), "..", "账号密码", "deploy.env")


def _load_env_file_fallback(keys):
    """把 _CRED_FILE 里 KEY=VALUE 行灌进 os.environ,仅填充当前未设置的 key。"""
    if not os.path.isfile(_CRED_FILE):
        return
    # utf-8-sig:凭证文件常由 Windows 编辑器(记事本 / PowerShell Set-Content)生成,
    # 带 UTF-8 BOM 时纯 "utf-8" 读法会把首行 key 读成 "﻿DEPLOY_HOST",与 keys
    # 集合比对失败 → 该 key **静默**不加载,最终报"缺少 DEPLOY_HOST"而文件里明明
    # 写着,极难排查。utf-8-sig 在无 BOM 时行为与 utf-8 完全一致,无副作用。
    with open(_CRED_FILE, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k in keys and not os.environ.get(k):
                os.environ[k] = v.strip()


_load_env_file_fallback({"DEPLOY_HOST", "DEPLOY_PORT", "DEPLOY_USER", "DEPLOY_PASSWORD"})

host = os.environ.get("DEPLOY_HOST")
port = os.environ.get("DEPLOY_PORT")
user = os.environ.get("DEPLOY_USER")
password = os.environ.get("DEPLOY_PASSWORD")
remote_base = os.environ.get("DEPLOY_REMOTE_BASE", "/home/cjy/AI_image_Recognition")

_missing = [
    name for name, val in (
        ("DEPLOY_HOST", host), ("DEPLOY_PORT", port),
        ("DEPLOY_USER", user), ("DEPLOY_PASSWORD", password),
    ) if not val
]
if _missing:
    sys.exit(
        "[deploy] 缺少环境变量: " + ", ".join(_missing) + "\n"
        "  用法: DEPLOY_HOST=<ip> DEPLOY_PORT=<port> DEPLOY_USER=<user> "
        "DEPLOY_PASSWORD=<pwd> python scripts/deploy.py"
    )
port = int(port)

ignore_dirs = {
    ".git",
    ".cache",
    "__pycache__",
    ".omp",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    "runs",
    "data",
    "账号密码",  # 凭证目录,整目录级排除——不能只靠 .md 后缀过滤(deploy.env 就漏过一次)
}
ignore_files = {"deploy.py", "bash.cmd", "deploy_bundle.tar", "upload.py"}

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port=port, username=user, password=password)
sftp = ssh.open_sftp()


# Remote mkdir -p helper
def remote_mkdir_p(remote_path):
    dirs = remote_path.split("/")
    current = ""
    for d in dirs:
        if not d:
            continue
        current += "/" + d
        try:
            sftp.mkdir(current)
        except IOError:
            pass


local_base = os.getcwd()
print("Starting deployment to server...")

for root, dirs, files in os.walk(local_base):
    rel_path = os.path.relpath(root, local_base)
    # Filter ignored directories, but keep scripts/data/ (Python modules, not image data)
    dirs[:] = [
        d for d in dirs
        if d not in ignore_dirs or (d == "data" and rel_path == "scripts")
    ]
    if rel_path == ".":
        remote_dir = remote_base
    else:
        remote_dir = os.path.join(remote_base, rel_path).replace("\\", "/")

    remote_mkdir_p(remote_dir)

    for f in files:
        if (
            f in ignore_files
            or f.endswith(".pyc")
            or f.endswith(".tmp")
            or f.endswith(".md")
            # 本地临时/调试产物(审查证据脚本、探针日志)。`.gitignore` 已有 `/tmp_*`,
            # 但 deploy 不读 gitignore —— 本轮 Gate2 的 16 个 tmp_ 脚本因此被整批传上
            # 服务器。它们对服务器毫无用处,还会污染目录、干扰后续 agent 判断哪些是
            # 正式代码。
            or f.startswith("tmp_")
        ):
            continue
        local_file = os.path.join(root, f)
        remote_file = os.path.join(remote_dir, f).replace("\\", "/")

        try:
            remote_stat = sftp.stat(remote_file)
            local_stat = os.stat(local_file)
            # 增量跳过判据(R5 修复 + Gate2 收紧):**仅 size 相同不足以判定内容一致**——
            # 同尺寸内容变更(改配置值、改字符串、换等长注释)会被静默跳过,服务器跑陈旧
            # 代码且零告警。
            #
            # 判据为「size 相同 **且** mtime 近似相等(|Δ|<2s)」才跳过。关键是下方 put 后
            # 用 `sftp.utime` 把**远端 mtime 盖成本地 mtime**——于是比较的是"上次部署时的
            # 本地 mtime"与"当前本地 mtime"这两个**同源 epoch 值**,与服务器墙钟无关:
            #   · 文件没改 → 两者相等 → 跳过;
            #   · 文件改了(编辑必更新 mtime,即便 size 不变)→ Δ 变大 → 重传。
            # 旧判据 `local.mtime <= remote.mtime+2` 用的是服务器上传时的墙钟,服务器时钟
            # **超前**时 local 永远"不新于"remote → 同尺寸真改动被静默漏传(Gate2 finding)。
            # 现判据双向对称、不受时钟偏差影响。2s 容差吸收 SFTP 秒级截断(utime 存整秒)。
            if (remote_stat.st_size == local_stat.st_size
                    and remote_stat.st_mtime is not None
                    and abs(local_stat.st_mtime - remote_stat.st_mtime) < 2):
                continue
        except IOError:
            pass

        print(f"Uploading: {rel_path if rel_path != '.' else ''}/{f}")
        sftp.put(local_file, remote_file)
        # 关键:把远端 mtime 盖成本地 mtime,让下次比较基于同源 epoch(见上方判据注释)。
        # paramiko 默认 put 不保留 mtime(远端取服务器上传时刻),不显式 utime 则时钟偏差
        # 会让判据失效。失败不致命(退化为按 size 判)——只记警告不中断整批部署。
        try:
            local_stat = os.stat(local_file)
            sftp.utime(remote_file, (local_stat.st_atime, local_stat.st_mtime))
        except (IOError, OSError) as e:
            print(f"  [WARNING] utime 同步失败(不影响本次上传,仅下次可能多传一次): {e}")

sftp.close()
ssh.close()
print("SUCCESS: Deployment to server completed successfully!")
