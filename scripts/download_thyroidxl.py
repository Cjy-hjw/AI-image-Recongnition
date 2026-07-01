"""
ThyroidXL数据集下载脚本
数据源: https://huggingface.co/datasets/hunglc007/ThyroidXL
镜像源: https://hf-mirror.com/datasets/hunglc007/ThyroidXL

使用说明:
1. 安装依赖: pip install huggingface_hub
2. 设置HF_ENDPOINT环境变量使用镜像:
   export HF_ENDPOINT=https://hf-mirror.com  # Linux/Mac
   set HF_ENDPOINT=https://hf-mirror.com     # Windows CMD
   $env:HF_ENDPOINT="https://hf-mirror.com"  # Windows PowerShell
3. 运行: python scripts/download_thyroidxl.py

注意: ThyroidXL是Gated Dataset，需要先在HuggingFace上申请访问权限
"""

import os
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download, login

def compute_file_hash(filepath, algorithm='sha256'):
    """计算文件哈希值"""
    hash_obj = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def download_thyroidxl(
    output_dir='data/public_datasets/ThyroidXL',
    token=None,
    use_mirror=True
):
    """
    下载ThyroidXL数据集

    Args:
        output_dir: 输出目录
        token: HuggingFace访问令牌（Gated Dataset必需）
        use_mirror: 是否使用HF-Mirror镜像
    """
    # 设置镜像
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print(f"使用镜像源: {os.environ['HF_ENDPOINT']}")

    # 登录（Gated Dataset需要）
    if token:
        login(token=token)
        print("✓ HuggingFace登录成功")
    else:
        print("警告: 未提供token，可能无法访问Gated Dataset")
        print("请访问 https://huggingface.co/settings/tokens 获取访问令牌")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n开始下载ThyroidXL到: {output_path.absolute()}")
    print("=" * 60)

    try:
        # 下载数据集
        dataset_path = snapshot_download(
            repo_id="hunglc007/ThyroidXL",
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"\n✓ 下载完成: {dataset_path}")

        # 验证下载的文件
        print("\n文件清单:")
        files = list(output_path.rglob('*'))
        total_size = 0

        manifest = {}
        for f in files:
            if f.is_file():
                size = f.stat().st_size
                total_size += size
                file_hash = compute_file_hash(f)
                manifest[str(f.relative_to(output_path))] = {
                    'size': size,
                    'sha256': file_hash
                }
                print(f"  {f.name}: {size / 1024 / 1024:.2f} MB")

        print(f"\n总大小: {total_size / 1024 / 1024:.2f} MB")

        # 保存manifest
        manifest_file = output_path / 'download_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"\n✓ Manifest已保存: {manifest_file}")

        # 检查关键文件
        expected_files = ['README.md', 'data']  # 根据实际情况调整
        missing = []
        for fname in expected_files:
            if not (output_path / fname).exists():
                missing.append(fname)

        if missing:
            print(f"\n⚠ 缺失文件: {', '.join(missing)}")
        else:
            print("\n✓ 所有预期文件已就绪")

        return True

    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n可能的原因:")
        print("1. 未申请ThyroidXL访问权限（Gated Dataset）")
        print("2. HuggingFace token无效或未提供")
        print("3. 网络连接问题")
        print("\n解决方案:")
        print("1. 访问 https://huggingface.co/datasets/hunglc007/ThyroidXL 申请访问")
        print("2. 获取token: https://huggingface.co/settings/tokens")
        print("3. 使用代理或HF-Mirror镜像")
        return False

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='下载ThyroidXL数据集')
    parser.add_argument('--output', type=str, default='data/public_datasets/ThyroidXL',
                        help='输出目录')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace访问令牌（Gated Dataset必需）')
    parser.add_argument('--no-mirror', action='store_true',
                        help='不使用HF-Mirror镜像')

    args = parser.parse_args()

    # 尝试从环境变量读取token
    token = args.token or os.environ.get('HF_TOKEN')

    success = download_thyroidxl(
        output_dir=args.output,
        token=token,
        use_mirror=not args.no_mirror
    )

    if success:
        print("\n" + "=" * 60)
        print("✓ ThyroidXL数据集下载完成")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ 下载失败，请检查上述错误信息")
        print("=" * 60)
        exit(1)
