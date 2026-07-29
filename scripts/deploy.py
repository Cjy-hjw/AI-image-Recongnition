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
    with open(_CRED_FILE, "r", encoding="utf-8") as f:
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
        ):
            continue
        local_file = os.path.join(root, f)
        remote_file = os.path.join(remote_dir, f).replace("\\", "/")

        try:
            remote_stat = sftp.stat(remote_file)
            local_stat = os.stat(local_file)
            if remote_stat.st_size == local_stat.st_size:
                continue
        except IOError:
            pass

        print(f"Uploading: {rel_path if rel_path != '.' else ''}/{f}")
        sftp.put(local_file, remote_file)

sftp.close()
ssh.close()
print("SUCCESS: Deployment to server completed successfully!")
