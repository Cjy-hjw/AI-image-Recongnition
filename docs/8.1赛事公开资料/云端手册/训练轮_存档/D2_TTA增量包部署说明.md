# D2 TTA 增量包部署说明(cloud_tta_incr.zip)

> 改动性质:**纯推理侧优化**(静态图 hflip TTA,只平均分割概率,分类头不融合)。
> 不涉及训练、不涉及权重、不涉及 config——`official_heads_seg_best.pth` 原样不动。
>
> 包内仅 2 文件(本地 commit `ed2d2f8`,全量测试 1084 passed+1 skipped):
> - `src/inference/case_pipeline.py`(TTA 主体 + 开关 `AICOMP_STATIC_TTA`,默认开)
> - `src/inference/postprocess.py`(`decode_predictions` 新增 `mask_prob_override` 参数)

## 部署步骤

1. **上传** `cloud_tta_incr.zip` 到云平台。

2. **解压到 cloud_infer 目录**(注意:是 `cloud_infer/`,测评容器读这棵树;
   不是 workspace 根):
   ```bash
   cd /2026aicompetition/workspace/cloud_infer
   unzip -o /path/to/cloud_tta_incr.zip   # 🔴 -o 必须带,防"替换同名目录"抹掉兄弟文件
   ```

3. **验证(三条,全过才算部署成功)**:

   ```bash
   cd /2026aicompetition/workspace/cloud_infer

   # ① TTA 开关存在且默认开(期望输出 True)
   PYTHONPATH=. python -c "from src.inference.case_pipeline import _static_tta_enabled; print(_static_tta_enabled())"

   # ② 新参数存在(期望无报错,输出含 mask_prob_override)
   grep -n "mask_prob_override" src/inference/postprocess.py | head -3

   # ③ 推理链路可跑通(期望无异常,LocationProbs 非均匀)
   PYTHONPATH=. python -c "
from src.inference.model_runner import load_config, build_model
import torch
config = load_config('configs/thyroid_config.yaml')
model = build_model(config, 'checkpoints/official_heads_seg_best.pth', torch.device('cpu'))
print('模型加载 OK')"

   # ④ 可选:服务探活
   # curl -s http://localhost:8000/health
   ```

4. **跑分前核对**:TTA 默认开,**无需设任何环境变量**。想临时关闭对拍用
   `AICOMP_STATIC_TTA=0`。

## 判读

- ① 输出 `True` = TTA 已生效(默认开);输出 `False` 说明有人设过 env,查
  容器环境变量分栏。
- ③ 报 `RuntimeError: 权重与模型结构不匹配` = 解压错了树/丢了文件,重新按
  步骤 2 解压。
- 云端若跑官方测评:期望 Dice/AP 双升或持平(预期 +0~0.35 分);若 Dice/AP
  任一明显下跌,回滚方式 = 用上一版 cloud_infer 备份覆盖这两个文件,或上传时
  先 `cp` 备份:
  ```bash
  cp src/inference/case_pipeline.py src/inference/case_pipeline.py.bak_tta
  cp src/inference/postprocess.py src/inference/postprocess.py.bak_tta
  ```
