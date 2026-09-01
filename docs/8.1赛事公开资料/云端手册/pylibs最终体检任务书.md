# pylibs 最终体检任务书(2026-08-31)

> 背景:pylibs 已全部装完,个别包版本曾被装错(openjpeg/libjpeg),已按任务书重装。
> 本任务书做**最终体检**,逐条跑、逐条贴回,不做任何修改动作(除非某条 FAIL 且命令里给了修法)。

## 第 1 步:版本核对表

跑:

```bash
ls /2026aicompetition/workspace/pylibs | sort
```

对照下表(只列**必须精确匹配**的;其余包任意新版即可):

| 目录名(截断版) | 必须版本 |
|---|---|
| numpy | 2.4.6 或 2.3.2(2.x 即可,**禁止 1.x**) |
| pylibjpeg | 2.1.0 |
| pylibjpeg_libjpeg / libjpeg 相关 | 2.4.0 |
| pylibjpeg_openjpeg / openjpeg 相关 | 2.5.0 |
| cv2(opencv) | 4.x(**禁止 5.x**) |
| pydicom | 3.0.x |

同时确认 pylibs 下**没有任何** torch / torchvision / torchaudio 开头的目录(有就删,镜像自带)。

## 第 2 步:关键扩展逐个 import

```bash
PYTHONPATH=/2026aicompetition/workspace/pylibs python -c "import _openjpeg; print('_openjpeg OK')"
PYTHONPATH=/2026aicompetition/workspace/pylibs python -c "import _libjpeg; print('_libjpeg OK')"
PYTHONPATH=/2026aicompetition/workspace/pylibs python -c "import pydicom; print('pydicom OK', pydicom.__version__)"
```

任何一条出现 `numpy.core.multiarray failed to import` 或 `compiled using NumPy 1.x`
→ 说明该包是 numpy1 编译版,**版本装错了**,不是 numpy 的问题,**禁止降级 numpy**,
按第 1 步的版本表 force-reinstall 对应包(--no-deps)。

## 第 3 步:全量 import

```bash
PYTHONPATH=/2026aicompetition/workspace/pylibs python -c "
import importlib
mods = ['cv2','pydicom','nibabel','skimage','PIL','pylibjpeg','lazy_loader',
        'torch','torchvision','numpy','yaml','tqdm','scipy','faiss','timm',
        'fastapi','uvicorn','requests','pandas','openpyxl','albumentations','safetensors']
for m in mods:
    try:
        mod = importlib.import_module(m)
        print('OK  ', m, getattr(mod,'__version__',''))
    except Exception as e:
        print('FAIL', m, '|', type(e).__name__, e)
"
```

判据:全部 OK(torch 行必须 2.8.0+cu128)。

## 第 4 步:功能级验证

```bash
cd /2026aicompetition/workspace/cloud_infer && \
PYTHONPATH=/2026aicompetition/workspace/pylibs python -c "
import torch
from src.inference.model_runner import build_model, load_config
c = load_config('configs/thyroid_config.yaml')
m = build_model(c, 'checkpoints/official_heads_seg_best.pth', torch.device('cpu'))
print('MODEL BUILD OK')
"
```

判据:打印 `Loaded model weights from ...` + `MODEL BUILD OK`。
这一条通过 = 权重、代码、依赖三者全部就位,与测评运行路径一致。

## 第 5 步:输出汇报

把第 1-4 步**每条的完整输出原样贴回**,不要只报"都通过了"。
有任何 FAIL 不要自己发挥修法,原样贴回等指示。
