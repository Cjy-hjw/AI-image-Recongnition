"""
预训练权重离线打包脚本
用于下载Phase 4所需的4个预训练模型权重

权重清单（调研报告 Table 2）:
1. ConvNeXt-Nano (timm) - 62MB - Apache-2.0
2. EfficientNet-B2 (timm) - 36MB - Apache-2.0
3. SSCD-ResNet50 (Facebook Research) - 100MB - MIT
4. MedSAM ViT-B (HF) - 350MB - Apache-2.0

总计: ~548MB
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

def compute_hash(filepath: Path, algorithm='md5') -> str:
    """计算文件哈希"""
    hash_obj = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def download_timm_weights(model_name: str, output_dir: Path) -> Optional[Path]:
    """
    下载timm预训练权重

    Args:
        model_name: timm模型名称（如 'convnext_nano.in12k_ft_in1k'）
        output_dir: 输出目录
    """
    try:
        import timm
        print(f"\n下载 {model_name}...")

        # timm自动下载到~/.cache/torch/hub/checkpoints/
        # 这里触发下载并获取路径
        model = timm.create_model(model_name, pretrained=True)

        # 获取权重文件路径（timm默认缓存位置）
        cache_dir = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'

        # 查找对应权重文件
        weight_files = list(cache_dir.glob(f'*{model_name.split(".")[0]}*.pth'))
        if not weight_files:
            weight_files = list(cache_dir.glob(f'*{model_name.split("_")[0]}*.pth'))

        if weight_files:
            src = weight_files[0]
            dst = output_dir / src.name

            # 复制到项目目录
            if not dst.exists():
                import shutil
                shutil.copy2(src, dst)
                print(f"[OK] 已复制: {dst.name} ({dst.stat().st_size / 1024 / 1024:.2f} MB)")
            else:
                print(f"[OK] 已存在: {dst.name}")

            return dst
        else:
            print(f"[WARN] 未找到缓存文件，使用state_dict直接保存")
            dst = output_dir / f'{model_name}.pth'
            import torch
            torch.save(model.state_dict(), dst)
            print(f"[OK] 已保存: {dst.name} ({dst.stat().st_size / 1024 / 1024:.2f} MB)")
            return dst

    except Exception as e:
        print(f"[FAIL] 下载失败: {e}")
        return None

def download_sscd_weights(output_dir: Path) -> Optional[Path]:
    """下载SSCD权重（Facebook Research GitHub Release）"""
    try:
        import urllib.request

        url = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
        dst = output_dir / "sscd_disc_mixup.torchscript.pt"

        if dst.exists():
            print(f"\n[OK] SSCD权重已存在: {dst.name}")
            return dst

        print(f"\n下载 SSCD-ResNet50...")
        print(f"URL: {url}")

        # 使用urllib下载（会自动使用环境变量中的代理）
        urllib.request.urlretrieve(url, dst)

        print(f"[OK] 已下载: {dst.name} ({dst.stat().st_size / 1024 / 1024:.2f} MB)")
        return dst

    except Exception as e:
        print(f"[FAIL] 下载失败: {e}")
        print("备用方案: 手动从 https://github.com/facebookresearch/sscd-copy-detection 下载")
        return None

def download_medsam_weights(output_dir: Path) -> Optional[Path]:
    """下载MedSAM权重"""
    try:
        from huggingface_hub import hf_hub_download

        print(f"\n下载 MedSAM ViT-B...")

        # 方法1: 从官方HF repo下载
        dst = output_dir / "medsam_vit_b.pth"

        if dst.exists():
            print(f"[OK] MedSAM权重已存在: {dst.name}")
            return dst

        # MedSAM官方仓库
        filepath = hf_hub_download(
            repo_id="wanglab/medsam",
            filename="medsam_vit_b.pth",
            cache_dir=output_dir.parent / '.hf_cache',
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )

        print(f"[OK] 已下载: {dst.name} ({dst.stat().st_size / 1024 / 1024:.2f} MB)")
        return Path(filepath)

    except Exception as e:
        print(f"[FAIL] 下载失败: {e}")
        print("备用URL: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return None

def main():
    # 输出目录
    output_dir = Path(__file__).parent.parent / '.cache_weights'
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("预训练权重离线打包")
    print("=" * 70)
    print(f"输出目录: {output_dir.absolute()}")

    # 检查代理设置
    if not os.environ.get('http_proxy') and not os.environ.get('HTTP_PROXY'):
        print("\n[WARN] 未检测到代理设置")
        print("建议先运行:")
        print('  $env:http_proxy="http://127.0.0.1:1203"')
        print('  $env:https_proxy="http://127.0.0.1:1203"')

        response = input("\n继续下载? (y/n): ")
        if response.lower() != 'y':
            print("已取消")
            return

    # 权重清单
    manifest: Dict[str, dict] = {}

    # 1. ConvNeXt-Nano
    print("\n" + "=" * 70)
    print("1/4 ConvNeXt-Nano")
    print("=" * 70)
    weight = download_timm_weights('convnext_nano.in12k_ft_in1k', output_dir)
    if weight:
        manifest['convnext_nano'] = {
            'path': weight.name,
            'size_mb': weight.stat().st_size / 1024 / 1024,
            'md5': compute_hash(weight, 'md5'),
            'sha256': compute_hash(weight, 'sha256'),
            'license': 'Apache-2.0',
            'source': 'timm'
        }

    # 2. EfficientNet-B2
    print("\n" + "=" * 70)
    print("2/4 EfficientNet-B2")
    print("=" * 70)
    weight = download_timm_weights('efficientnet_b2.ra_in1k', output_dir)
    if weight:
        manifest['efficientnet_b2'] = {
            'path': weight.name,
            'size_mb': weight.stat().st_size / 1024 / 1024,
            'md5': compute_hash(weight, 'md5'),
            'sha256': compute_hash(weight, 'sha256'),
            'license': 'Apache-2.0',
            'source': 'timm'
        }

    # 3. SSCD-ResNet50
    print("\n" + "=" * 70)
    print("3/4 SSCD-ResNet50")
    print("=" * 70)
    weight = download_sscd_weights(output_dir)
    if weight:
        manifest['sscd_resnet50'] = {
            'path': weight.name,
            'size_mb': weight.stat().st_size / 1024 / 1024,
            'md5': compute_hash(weight, 'md5'),
            'sha256': compute_hash(weight, 'sha256'),
            'license': 'MIT',
            'source': 'Facebook Research'
        }

    # 4. MedSAM ViT-B
    print("\n" + "=" * 70)
    print("4/4 MedSAM ViT-B")
    print("=" * 70)
    weight = download_medsam_weights(output_dir)
    if weight:
        manifest['medsam_vit_b'] = {
            'path': weight.name,
            'size_mb': weight.stat().st_size / 1024 / 1024,
            'md5': compute_hash(weight, 'md5'),
            'sha256': compute_hash(weight, 'sha256'),
            'license': 'Apache-2.0',
            'source': 'wanglab/medsam'
        }

    # 保存manifest
    manifest_file = output_dir / 'weights_manifest.json'
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("下载完成")
    print("=" * 70)
    print(f"[OK] 成功下载: {len(manifest)}/4 个权重文件")

    total_size = sum(m['size_mb'] for m in manifest.values())
    print(f"[OK] 总大小: {total_size:.2f} MB")
    print(f"[OK] Manifest已保存: {manifest_file}")

    # 打印清单
    print("\n权重清单:")
    for name, info in manifest.items():
        print(f"  {name}:")
        print(f"    文件: {info['path']}")
        print(f"    大小: {info['size_mb']:.2f} MB")
        print(f"    MD5: {info['md5']}")
        print(f"    许可: {info['license']}")

    if len(manifest) < 4:
        print("\n[WARN] 部分权重下载失败，请检查网络或手动下载")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
