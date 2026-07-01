# ThyroidXL数据集获取指南

## 数据集信息

- **来源**: HuggingFace `hunglc007/ThyroidXL`
- **类型**: Gated Dataset（需要手动申请访问权限）
- **规模**: 4,093例患者 / 11,635张B-mode超声图像
- **标注**: BBox + 像素Mask + ACR TI-RADS分级 + 病理金标准（FNAC）
- **许可**: CC BY 4.0（允许学术竞赛使用）

## 获取步骤

### 1. 注册HuggingFace账户并申请访问

1. 访问 https://huggingface.co/join 注册账户（需代理）
2. 访问数据集页面: https://huggingface.co/datasets/hunglc007/ThyroidXL
3. 点击 "Request Access" 按钮
4. 填写申请表单（说明用于学术竞赛研究）
5. 等待审批（通常1-3个工作日）

### 2. 获取访问令牌（Token）

1. 登录HuggingFace后访问: https://huggingface.co/settings/tokens
2. 点击 "New token" 创建新令牌
3. Token类型选择 "Read"（只需下载权限）
4. 复制生成的token（格式: `hf_xxxxxxxxxxxxx`）

### 3. 配置本地Token

**方法1: 环境变量（临时）**
```powershell
$env:HF_TOKEN="hf_xxxxxxxxxxxxx"  # 替换为你的token
```

**方法2: huggingface-cli登录（持久）**
```powershell
# 先设置代理
$env:http_proxy="http://127.0.0.1:1203"
$env:https_proxy="http://127.0.0.1:1203"

# 登录（会提示输入token）
& 'D:\program\Python\anaconda\envs\Demo_D\python.exe' -m huggingface_hub login
```

### 4. 下载数据集

**使用项目脚本下载**:
```powershell
# 设置代理（必需）
$env:http_proxy="http://127.0.0.1:1203"
$env:https_proxy="http://127.0.0.1:1203"
$env:NODE_OPTIONS="--dns-result-order=ipv4first"

# 运行下载脚本
& 'D:\program\Python\anaconda\envs\Demo_D\python.exe' scripts/download_thyroidxl.py --output data/public_datasets/ThyroidXL
```

**手动Python下载**:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="hunglc007/ThyroidXL",
    repo_type="dataset",
    local_dir="data/public_datasets/ThyroidXL",
    resume_download=True
)
```

## 预期下载内容

根据调研报告，数据集应包含：
- 图像文件（PNG格式）
- 标注文件（BBox坐标、Mask掩膜）
- 元数据CSV（患者信息、TI-RADS级别、病理标签）
- README文档

预计总大小: ~1.2 GB

## 数据验证

下载完成后需验证：
1. 文件完整性（SHA256校验）
2. 样本数量（11,635张图像）
3. 良恶性分布（64.6% 良性 vs 35.4% 恶性）
4. 标注schema完整性（所有字段存在）

## 备用方案

如果HuggingFace访问困难：
1. 联系论文通讯作者直接获取（hungslc007@gmail.com）
2. 使用HF-Mirror镜像（设置 `HF_ENDPOINT=https://hf-mirror.com`）
3. 通过学术合作渠道（深圳大学AI学院）

## 故障排除

**问题1: 403 Forbidden**
- 原因: 未获得访问权限或token无效
- 解决: 检查申请状态，确认token有效

**问题2: 连接超时**
- 原因: 未设置代理
- 解决: 参照上方代理配置

**问题3: 下载中断**
- 原因: 网络不稳定
- 解决: 脚本支持断点续传，重新运行即可

## 下一步

下载完成后：
1. 运行 `python scripts/verify_thyroidxl.py` 验证数据完整性
2. 生成数据统计报告（样本分布、标注覆盖率）
3. 继续 Task 1.2: 获取ThyUS2Path数据集
