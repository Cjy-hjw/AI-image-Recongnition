# Repository Guidelines

## 1. 项目概述 (Project Overview)
本项目为 **“2026全国医保影像AI识图大赛” 赛道七（基于超声的甲状腺癌智能检测与影像规范性审核）** 的全链路深度学习解决方案。系统需通过严格的真实性与规范性审核（门槛防御体系），并在此基础上实现甲状腺结节的病灶识别、ACR TI-RADS 标准的多维度分级评估、良恶性辅助诊断，以及具备高可解释性的关键点位自动标注与结构化文本推理输出。

---

## 2. 架构与数据流 (Architecture & Data Flow)
系统分为 **门槛审核** 与 **多任务诊断** 两个核心阶段：

### 2.1 门槛审核阶段 (Gatekeeper Pipeline)
输入图像必须通过门槛项校验（FPR < 1%, Precision > 98%）后才能进入后续的诊断流程：
1. **真实性鉴别**: 
   - 提取散斑抑制各向异性扩散（SRAD）后的残余高频声学纹理。
   - 使用误差级别分析（ELA）和高通滤波器捕捉生成模型（GAN/Diffusion）的伪影。
   - 利用灰度共生矩阵（GLCM）提取纹理方差，通过轻量级 `EfficientNet-B0` 区分物理仿体与真人超声图像。
2. **去重与篡改检测**:
   - **去重 (两级检索)**: 第一级基于感知哈希（pHash/dHash）计算海明距离快速初筛；第二级使用 DINOv2 提取深层嵌入向量，存入 FAISS GPU 向量库进行高精度精筛。
   - **拼接与复制移动检测**: 使用 FAST 关键点 + RANSAC 对齐 + LBP 纹理描述 + SVM 分类器作为基线，检测复制移动和拼接边缘伪影。进阶方案使用离散小波变换（DWT）分离低频与高频信息进行频域匹配。

### 2.2 多任务诊断阶段 (Multi-task Diagnosis MTL)
通过统一的多任务学习（MTL）网络，共享 Encoder 并行处理多个下游任务：
- **主干网络 (Backbone)**: 选用 2D `EfficientNet-B2`（或 `ConvNeXt-Tiny`）作为 U-Net 编码器以适配 8GB 显存，引入门控任务注意力（Gated Task Attention, GTA）将局部空间边界特征与全局语义诊断特征进行通道和空间层面的解耦过滤。**严禁使用任何 3D 网络**，超声是纯 2D 模态。
- **并行任务头**:
  1. **病灶检测与分割**: 预测结节的像素级掩码（Segment Mask）与 Bound Box。
  2. **ACR TI-RADS 评估头**:
     - *成分 (Composition)*、*回声 (Echogenicity)*、*形状 (Shape)*、*边缘 (Margin)*：形式化为多类别分类任务，使用 **Softmax 激活 + 分类交叉熵损失（Categorical Cross-Entropy）**。
     - *强回声灶 (Echogenic Foci)*：非互斥的多标签分类任务，使用 **Sigmoid 激活 + 二元交叉熵损失（Binary Cross-Entropy）**。
     - 对高危特征（如“长大于宽”、微钙化点）赋予更高的假阴性惩罚权重。
  3. **概念瓶颈模型 (CBM)**: 潜空间残差概念瓶颈模型（Latent-Residual CBM），融合直接可解释性的线性直连路径与捕捉复杂交互的潜特征非线性路径，并结合可微的 ACR TI-RADS 积分 MSE 正则化项，确保推理决策流严格符合临床医学共识。
  4. **可解释性输出 (XAI)**: 结合 Grad-CAM 与特征图在图上自动标注病灶边缘及钙化关键点位，并可生成高度对齐的规范结构化文本临床报告。

```
Raw Image -> [SRAD + ELA / GLCM + EfficientNet] -> 真实性校验
            -> [pHash + DINOv2 + FAISS] -> 去重校验
            -> [FAST+RANSAC+LBP+SVM / DWT] -> 篡改校验
                      |
                      v (通过门槛项)
           [Swin-v2 / ConvNeXt / MFS-Unet 2D 骨干网络]
            /         |         \
           v          v          v
     [结节分割/检测] [TI-RADS概念向量] [Grad-CAM关键点]
                      |
                      v (CBM决策层)
               [良恶性诊断与TR分级]
                      |
                      v
               [多模态文本报告]
```

---

## 3. 关键目录结构 (Key Directories)
```
.
├── configs/               # YAML 训练超参配置文件
├── reference/             # 行业报告与技术方案设计指南
├── src/
│   ├── dataset/           # 数据加载器与超声弹性形变、散斑模拟数据增强管线
│   ├── gatekeeper/        # 真伪鉴别、去重、拼接检测、复制-移动检测模块
│   ├── models/            # 2D 多任务网络骨干、CBM 瓶颈层与各类分类/分割头
│   └── utils/             # ELA 滤波、pHash、FAISS 检索、Grad-CAM 等辅助工具
├── tests/                 # 基于 pytest 的单元与集成测试脚本
├── train.py               # 训练与多折交叉验证 CLI 启动入口
├── evaluate.py            # Hold-out 验证集评估脚本
├── predict.py             # 沙箱环境单卡/批量推理入口脚本（离线部署）
├── pyrightconfig.json     # Pyright 静态类型分析配置
└── requirements.txt       # 环境依赖配置清单
```

---

## 4. 重要文件 (Important Files)
- `train.py`: 训练、监督标签校正（SLR）优化器与 weights & biases 实验追踪集成入口。
- `evaluate.py`: 用于在锁定的 20% Hold-out 验证集上验证 SOTA 指标。
- `predict.py` / `inference.py`: 沙箱提交入口，必须符合无网络、标准输入输出（JSON / 标注重构图）的赛题要求。
- `pyrightconfig.json`: 用于配置 Python 解释器及 Anaconda 虚拟环境路径，提供 LSP 静态分析支持。
- `Dockerfile`: 构建沙箱运行镜像的标准容器配置文件。

---

## 5. 运行时与工具链偏好 (Runtime & Tooling Preferences)
- **Python 运行环境**: Python 3.10 / 3.11。
- **指定 Anaconda 虚拟环境路径**: `D:\program\Python\anaconda\envs\Demo_D`（解释器指向 `D:\program\Python\anaconda\envs\Demo_D\python.exe`）。
- **包管理工具**: `pip`（沙箱环境部署需支持基于离线依赖包目录的无网离线安装）。
- **核心依赖库**:
  - `torch`, `torchvision`, `torchaudio` (CUDA 12.1+ 兼容版本)
  - `monai` (医疗影像变换与分割评估)
  - `timm` (2D 预训练模型加载)
  - `faiss-gpu` (去重嵌入检索)
  - `opencv-python`, `scikit-image`, `scikit-learn` (传统图处理、GLCM、RANSAC、LBP + SVM 篡改分类)
  - `PyWavelets` (DWT 变换)
  - `wandb` (实验与指标追踪)
  - `mamba-ssm`, `causal-conv1d` (可选，Vision Mamba CUDA 算子)
- **硬件约束**: 单张 **NVIDIA A800 (80GB)** GPU。显存充裕，支持大 Batch Size（16/32）与高分辨率（$512\times512$/$1024\times1024$）2D 训练。**严禁使用 3D 网络**。
- **沙箱约束**: 物理隔离，无外网。必须保证所有模型推理流程在完全离线状态下运行，依赖包需提前离线下载。

---

## 6. 开发管理命令 (Development Commands)

### 6.1 环境配置与依赖离线化
```bash
# 1. 激活指定环境并安装 CUDA 编译的 PyTorch 框架
conda activate Demo_D
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 2. 安装项目依赖项
pip install monai timm wandb faiss-gpu opencv-python scikit-image scikit-learn PyWavelets

# 3. 离线依赖打包（供离线沙箱使用）
pip freeze > requirements.txt
pip download -r requirements.txt -d ./offline_packages
# 沙箱环境无网离线安装：
pip install --no-index --find-links=./offline_packages -r requirements.txt
```

### 6.2 代码质量规范、格式化与静态类型分析
```bash
# 运行 Pyright 静态类型检查
pyright

# 运行 Ruff 静态代码检查与格式化
ruff check .
ruff format .
```

### 6.3 运行训练与评估
```bash
# 启动多任务多折交叉训练
python train.py --config configs/thyroid_config.yaml --device cuda:0

# 评估最终模型性能
python evaluate.py --model_path checkpoints/best_model.pth --data_path data/val
```

### 6.4 远程服务器（A800）部署与背景运行
由于服务器具备大算力，建议在服务器上直接通过国内镜像进行依赖补全和后台静默训练：

1. **校园网登录认证（连通外网）**：
   ```bash
   curl -G "http://10.0.1.5/drcom/login" \
     --data-urlencode "DDDDD=23032304023" \
     --data-urlencode "upass=zd79718812@.." \
     --data-urlencode "0MKKey=123456" \
     --data-urlencode "R1=0" --data-urlencode "R2=" --data-urlencode "R3=0" --data-urlencode "R6=0" \
     --data-urlencode "para=00" --data-urlencode "v4ip=10.12.0.4" --data-urlencode "terminal_type=1" \
     --data-urlencode "lang=zh-cn" --data-urlencode "jsVersion=4.2" --data-urlencode "v=" \
     -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)" \
     -H "Referer: http://10.0.1.5/"
   ```

2. **创建独立 Python 虚拟环境**：
   ```bash
   # 克隆已有 CUDA 支持的 d2l_env 环境，避免重新下载 2.4 GB 框架包
   /root/miniconda3/bin/conda create --name thyroid_env --clone d2l_env -y
   ```

3. **在线国内镜像安装其余库**：
   ```bash
   /root/miniconda3/envs/thyroid_env/bin/pip install opencv-python-headless monai \
     albumentations scikit-image scikit-learn faiss-cpu pyyaml \
     -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. **后台静默挂载多任务模型训练**：
   ```bash
   # 注入 Hugging Face 国内镜像站，利用 8 线程并行 dataloader 启动 120 轮高清训练
   cd /root/AI_image_Recognition
   export HF_ENDPOINT=https://hf-mirror.com
   (nohup /root/miniconda3/envs/thyroid_env/bin/python -u train.py --config configs/thyroid_config.yaml > train.log 2>&1 < /dev/null &) &
   ```

---

## 7. 代码规范与通用模式 (Code Conventions & Common Patterns)
- **纯 2D 计算限制**: 所有数据输入和模型算子必须为 2D。严禁声明或引入 `nn.Conv3d`、`nn.MaxPool3d` 等任何 3D 组件。
- **无数据泄漏验证 (Patient-Grouped)**: 划分交叉验证集和测试集时，必须按 **患者 ID (Patient ID)** 进行分组（Grouped），确保同一患者的不同切片图像不会同时出现在训练集与验证集中。
- **命名规范**:
  - 类名采用大驼峰（PascalCase），例如 `ConceptBottleneckModel`、`SRADFilter`。
  - 函数与变量名采用蛇形命名法（snake_case），例如 `extract_glcm_features`、`compute_srad_diffusion`。
  - 权重保存命名格式统一为 `best_model.pth` 或 `checkpoint_epoch_{N}.pth`。
- **错误与异常处理**:
  - 处理图像前，必须验证输入 Tensor 维度为 `[Batch, Channel, Height, Width]` 且 Channel 数符合要求。
  - 显式捕获并记录 CUDA Out-Of-Memory (OOM) 异常，并在必要时清空缓存（`torch.cuda.empty_cache()`）。
- **监督标签校正 (SLR)**: 针对超声数据的主观标注偏差，引入 SLR 机制。在训练迭代中动态跟踪样本损失，若样本损失长期过高（疑似标签噪声），对其损失权重进行平滑衰减，避免模型过拟合于错误的人工标注。

---

## 8. 测试与质量保证 (Testing & QA)

### 8.1 测试策略与框架
- **单元测试**: 使用 `pytest` 框架对数据预处理逻辑（如 SRAD 扩散、ELA、DWT 降噪）、感知哈希计算、以及 CBM 逻辑关系计算等非深度学习数学公式模块进行覆盖。
- **静态类型安全**: 提交前必须通过 `pyright` 静态类型分析检查，零 Errors 与 critical warnings。
- **资源限制**: 运行模型推理时，显存峰值开销限制在 80% A800 (约 64GB) 以下，单帧图像平均推理耗时必须 $< 2.0$ 秒。

### 8.2 内部评测指标基线 (SOTA Baselines)
开发与集成测试必须满足以下基线性能标准：
- **门槛审核模块**: False Positive Rate (FPR) < 1%, Precision > 98%，拼接及篡改分类 F1-score > 91.5%，Accuracy $\approx$ 98.49%。
- **结节分割**: Dice 系数 > 82%，Intersection over Union (IoU) > 76.2%。
- **良恶性分类**: AUC > 0.90。
- **TI-RADS 分级评估**: 成分、回声、形状、边缘、强回声灶五个维度的平均加权 F1-score > 0.82。
- **关键点标注**: 与专家标注边界框的 IoU > 0.70。
- **定量可解释报告**: 医师逆向认知测试准确率 > 87.1%。

### 8.3 数据验证集划分
- 采用 **患者分组的分层 5 折交叉验证（Patient-Grouped Stratified 5-Fold CV）** 作为日常开发调优基准。
- 提取数据集中 **固定的 20% 作为 hold-out 验证集** 进行完全隔离保护，仅在发布新版本模型或提交沙箱前进行最终的指标核验。
