import os
import paramiko

host = "182.91.14.5"
port = 22038
user = "root"
password = "LIUpeng!@2026"
remote_base = "/root/AI_image_Recognition"

ignore_dirs = {".git", ".cache", "__pycache__", ".omp", ".ipynb_checkpoints", "runs", "data"}
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
    # Filter ignored directories in-place
    dirs[:] = [d for d in dirs if d not in ignore_dirs]
    
    rel_path = os.path.relpath(root, local_base)
    if rel_path == ".":
        remote_dir = remote_base
    else:
        remote_dir = os.path.join(remote_base, rel_path).replace("\\", "/")
        
    remote_mkdir_p(remote_dir)
    
    for f in files:
        if f in ignore_files or f.endswith(".pyc") or f.endswith(".tmp"):
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
