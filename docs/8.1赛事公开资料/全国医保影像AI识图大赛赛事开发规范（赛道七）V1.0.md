# 全国医保影像AI识图大赛赛事开发规范（赛道七）V1.0



- **编制单位**：全国医保影像AI识图大赛技术保障组  
- **编制日期**：2026年7月  

## 一、 训练日志输出规范



- **格式要求**：必须采用 **JSON Lines (JSONL)** 格式，每行一个有效的 JSON 对象。  
- **文件路径**：必须保存在 `/2026aicompetition/workspace/logs` 目录下（如 `/2026aicompetition/workspace/logs/training.log`）。  
- **审查风险警示**：如平台未能读取到指定日志文件，可能会被标记为可疑对象，一切后果由参赛队伍自负。  
- **完整日志字段表**：  

| **字段名**        | **类型** | **必填** | **说明**                          | **示例**                          |
| ----------------- | -------- | -------- | --------------------------------- | --------------------------------- |
| `timestamp`       | string   | 是       | ISO 8601 格式的时间戳             | `"2025-04-28T10:30:00.123Z"`      |
| `epoch`           | integer  | 是       | 当前训练轮次（从 1 开始）         | `5`                               |
| `step`            | integer  | 是       | 当前迭代步数                      | `1000`                            |
| `phase`           | string   | 是       | 运行阶段：`train`, `val`, `test`  | `"train"`                         |
| `mode`            | string   | 是       | 模型状态：`training`, `inference` | `"training"`                      |
| `data_source`     | string   | 是       | 数据来源标识                      | `"official/train_v1"`             |
| `loss`            | float    | 否       | 损失函数值（训练/验证阶段）       | `0.234`                           |
| `lr`              | float    | 否       | 当前学习率                        | `0.001`                           |
| `checkpoint`      | string   | 否       | 保存的模型文件路径                | `"checkpoints/model_epoch5.pth"`  |
| `pretrained_from` | string   | 否       | 预训练权重路径或来源标识          | `"official/torchvision/resnet50"` |

## 二、 推理服务 API 开发要求



参赛队伍需严格按照以下两个 HTTP 交互接口开发推理服务：  

### 1. 交互一：测评内容接收（`/call`）



- **接口路径与端口**：`POST [http://127.0.0.1:8000/call](http://127.0.0.1:8000/call)`

    

- **请求头**：`Content-Type: application/json`（无认证拦截）  

- **响应时限**：必须采取**异步响应**机制。接收到请求后需在 **5 秒内** 返回 HTTP 状态码 `200`（超时未响应将判定为请求异常）。  

- **入参关键字段**：  

  - `request_id`：请求唯一标识。  
  - `team_id` / `track_code`：队伍与赛道标识。  
  - `input.evaluation_id`：评测任务 ID。  
  - `input.dataset_path`：测试数据集路径（服务需支持批量数据文件读取）。  

### 2. 交互二：回传推理结果（回调）



- **回调地址**：`POST {训推平台服务地址}/api/competition/inference/callback/`

    

- **输出路径规范**：`/2026aicompetition/workspace/answer/{evaluation_id}/{结果集}`

    

  - `/2026aicompetition/workspace/answer` 已由平台创建。  
  - `{evaluation_id}` 目录**必须由参赛者代码在推理时动态创建**。  

- **关键关联逻辑**：选手必须保存 `/call` 接收到的 `request_id` 和 `evaluation_id`，并在回调时回传，否则评测无法闭环，校验将不通过。  

- **回调 Body**：  

  JSON

  ```
  {
      "request_id": "2f4f3f2a-6320-44c2-bd49-de29bd64dc09",
      "evaluationId": 454899866564564,
      "predPath": "canswer/454899866564564"
  }
  ```

### 3. 运行环境与启动机制



- **前台长进程**：启动推理服务时，命令必须是一个**长期运行的前台进程**（如 `vllm` 启动脚本），否则服务会被关闭。  

## 三、 测评容器与健康检查要求



- **容器类型**：测评阶段必须创建容器类型为“测评容器”的实例（普通训练容器无法发起测评）。  
- **健康检查接口**：服务必须在 `8000` 端口提供 `GET /health` 接口，并返回 HTTP `200`。  
- **平台探针规则**：  
  - **启动探针**：每 5 秒检查一次，最长等待 1800 秒（30分钟）。服务启动成功后等待 60 秒进入就绪状态。  
  - **探活探针**：每 30 秒检查一次，若连续失败 3 次，平台将自动重启测评容器。  
- **环境与镜像保存**：  
  - **脚本保存**：建议将代码/脚本保存在挂载目录 `/2026aicompetition/workspace` 下。  
  - **镜像导出**：若在普通容器中安装了额外环境，需通过平台“保存镜像”功能导出为自定义镜像，创建“测评容器”时选择该自定义镜像。  

## 四、 金标准（标注）数据格式规范



- **文件结构**：
  - 一级目录为患者号（`PatientId`）。  
  - 原始图像为 DICOM 格式，Mask 掩码为 `.nii.gz` 格式（如 `M_1_1_mask.nii.gz`、`病灶1_1_mask.nii.gz`）。  
- **临床元数据表格（`.xlsx`）核心字段**：  
  - `PatientId` / `AccessionNumber` / `SeriesUid` / `DetailDescription`：患者及序列脱敏信息。  
  - `RoiName`：标注名称（**`M` 表示主病灶**，非主病灶为 `病灶1`、`病灶2` 等）。  
  - `RoiVolume` / `RoiVoxelCount` / `RoiHUMaxValue` / `RoiHUMinValue` / `RoiHUAverageValue` / `RoiHUStandardDeviation`：ROI 几何与灰度统计。  
  - `STUDY->CLINICAL->备注`：重点审核、图像质量问题跳过、序列缺失跳过、构建失败跳过、报告缺失跳过、阴性数据跳过、无。  
  - `ROI->CLINICAL->良恶性`：良、恶、无。  
  - `ROI->CLINICAL->TI-RADS`：I类 ~ V类。  
  - `cancer_label`：阳性、阴性。  
  - `patho_histology`：恶性肿瘤、非肿瘤性、交界性肿瘤、良性肿瘤、无。  
  - `extrathy_ext`（甲状腺外侵犯）：甲状腺包膜、毗邻软组织、血管、无、无病灶（阴性）、NA/UNK。  
  - `lesion_loc`（主病灶位置）：左上叶、左下叶、左中叶、右上叶、右下叶、右中叶、峡部、无（阴性）、NA/UNK。  
  - `clin_t_stage`（临床显著性分级 T）：有、无。  

## 五、 选手预测输出格式规范



### 1. 目录存放结构



预测输出采用两层文件夹存放：`{StudyUid}/{SeriesUid}`。  

Plaintext

```
/2026aicompetition/workspace/answer/{evaluation_id}/
└── {StudyUid}/
    ├── prediction.json                         # 必须叫 prediction.json
    └── {SeriesUid}/
        ├── THY_US_POS_001_primary.nii.gz        # 分割 Mask 文件
        └── THY_US_POS_001_nodule_2.nii.gz      
```

### 2. 结节掩码（Mask）规范



- **格式与值域**：必须为 NIfTI 格式（`.nii` 或 `.nii.gz`）。像素编码：`1` 表示结节，`0` 表示背景。  
- **尺寸一致性**：Height × Width 必须与对应原始输入超声图像严格一致。  
- **静态数据**：每个预测结节输出 2D 二值掩码。阴性病例输出空文件夹。  
- **动态数据**：
  - 每个结节输出 **3D 二值掩码**，Z 轴对应全视频帧序列（必须预测并输出所有帧）。  
  - **评估规则**：评测时仅抽取标注帧计算指标，未标注帧不计分。  
  - **动态 AP 计算**：采用病灶级汇总——每个预测病灶仅以其唯一的 `Detection.confidence` 进入全测试集排序一次，避免同一病灶因多帧而被重复计数。  

### 3. 主病灶匹配与评估规则



- **输出要求**：模型需对**所有检出的结节**输出完整的属性预测。  
- **评估逻辑**：评测脚本会自动根据空间重叠度寻找与金标准主病灶（`RoiName` 为 `M`）匹配的预测结节。  
- **关键特有字段评估**：**甲状腺外侵犯**、**主病灶位置**、**T临床显著性分级** 仅针对匹配成功的主病灶进行评估计分。  

### 4. `prediction.json` 完整 JSON 字典规范



  

| **字段路径**                 | **类型** | **详细说明 / 值域**                                          | **必填** |
| ---------------------------- | -------- | ------------------------------------------------------------ | -------- |
| `StudyUID`                   | string   | 检查唯一标识                                                 | 是       |
| `ProcessingTime_ms`          | int      | 推理总耗时（毫秒）                                           | 是       |
| `Prediction.MaskFiles`       | string[] | 所有预测结节 Mask 文件名列表                                 | 是       |
| `Prediction.Nodules[]`       | array    | 结节预测详情（阴性时为空数组 `[]`）                          | 是       |
| └─ `NoduleID`                | string   | 结节唯一标识（如 `"N1"`）                                    | 是       |
| └─ `MaskFile`                | string   | 对应的 Mask 文件名                                           | 是       |
| └─ `Detection.confidence`    | float    | `[0, 1]` 病灶级检测置信度（动态数据代表整个序列置信度）      | 是       |
| └─ `Malignancy.predicted`    | bool     | `true`（恶性） / `false`（良性）                             | 是       |
| └─ `Malignancy.confidence`   | float    | `[0, 1]` 恶性置信度                                          | 是       |
| └─ `TI_RADS_Class`           | int      | `1`, `2`, `3`, `4`, `5`                                      | 是       |
| └─ `ExtrathyroidalExtension` | int      | `1`: 无, `2`: 包膜, `3`: 软组织, `4`: 血管                   | 是       |
| └─ `Location`                | int      | `1` ~ `7`（仅无结节时可为 `0`）                              | 是       |
| └─ `LocationProbs`           | float[7] | 7 个位置类别的概率分布，顺序固定且和为 1                     | 是       |
| └─ `T_ClinSig.T`             | int      | `0`（无临床显著性） / `1`（有临床显著性）                    | 是       |
| └─ `T_ClinSig.T_conf`        | float    | `[0, 1]` T 临床显著性置信度                                  | 是       |
| `Interpretation`             | object   | 包含 `Conclusion`（文本总结）和 `KeyFindings`（关键发现列表） | 推荐     |