# pylibs 全新安装任务书(2026-08-31 版本 2)

> pylibs 目录已被用户手动删除。本任务书是**唯一安装依据**,不要读 requirements.txt
> (里面含 torch 行,--target 安装会把镜像自带的 torch 重下成 CPU 版,已踩过坑)。
> 包清单已把推理 + 训练两侧合并、torch 系全部剔除、全部小依赖显式列入,照抄即可。

## 唯一安装命令(一条装完全部)

```bash
pip install --target=/2026aicompetition/workspace/pylibs \
  numpy==2.4.4 \
  safetensors \
  timm \
  pydicom==3.0.2 \
  pylibjpeg==2.1.0 \
  pylibjpeg-libjpeg==2.4.0 \
  pylibjpeg-openjpeg==2.5.0 \
  opencv-python-headless==4.14.0 \
  nibabel \
  Pillow \
  pyyaml \
  scipy \
  scikit-learn \
  scikit-image \
  albumentations \
  fastapi \
  uvicorn \
  requests \
  lazy_loader \
  huggingface_hub \
  pandas \
  openpyxl \
  paramiko \
  -i https://mirrors.aliyun.com/pypi/simple/
```

说明:
- **torch / torchvision / torchaudio 绝对不装**(镜像自带 2.8.0+cu128)。
- 🔴 **timm 的官方依赖声明里就含 torch/torchvision,--target 模式下 pip 看不见镜像
  自带的 torch,会把 torch 一起下载进 pylibs——这是必然发生的,不是装错**。处理:
  让 pip 装完(勿中断,中断留残缺目录),然后按验证第 2 步删掉 torch 系目录即可,
  import 会自动落回镜像自带版。
- pylibjpeg 三件套版本**精确锁定**——其他版本绑 numpy1 ABI,numpy2 下 import 即炸。
- opencv 锁 4.14.0(5.0 炸 gatekeeper)。
- 装完若 pip 提示装了别的 torch 相关包,一并删掉(见下)。

## 验证(5 条全过才算完成)

```bash
# 1. torch 未被污染:必须输出 2.8.0+cu128 True
PYTHONPATH=/2026aicompetition/workspace/pylibs python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2. torch 残留检查:输出必须为空
ls /2026aicompetition/workspace/pylibs | grep -i "^torch"

# 3. 全量 import:必须输出 ALL OK 2.4.4
PYTHONPATH=/2026aicompetition/workspace/pylibs python -c "import cv2,pydicom,nibabel,skimage,PIL,pylibjpeg,lazy_loader,timm,yaml,scipy,fastapi,pandas,openpyxl,albumentations; import numpy; print('ALL OK', numpy.__version__)"

# 4. 功能级 DICOM/NIfTI API
PYTHONPATH=/2026aicompetition/workspace/pylibs python -c "import pydicom, nibabel; from skimage.feature import graycomatrix; print('API OK')"

# 5. 推理入口可加载:必须输出 SRC OK
cd /2026aicompetition/workspace/cloud_infer && PYTHONPATH=/2026aicompetition/workspace/pylibs python -c "from src.inference.model_runner import build_model, load_config; print('SRC OK')"
```

faiss 不装(有 NumPy 降级路径,不影响)。

**把 5 条的输出原样贴回,不要只报结论。**
