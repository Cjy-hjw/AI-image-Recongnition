# src/utils/AGENTS.md — 工具函数与可视化

## Package Structure

| File | Responsibility |
|:---|:---|
| `report_generator.py` | `calculate_tirads_points()` — ACR TI-RADS 计分; `generate_report()` — 临床诊断报告生成 |
| `grad_cam.py` | Grad-CAM 热力图生成与叠加(双主干: cls/seg 分支) |

---

## 1. TI-RADS 计分 (`report_generator.py`)

### `calculate_tirads_points(composition, echogenicity, shape, margin, foci)`

严格遵循 ACR TI-RADS 白皮书(2017)评分表:

| 属性 | 分值 |
|:---|:---|
| **Composition** | Cystic/Spongiform(0), Mixed(1), Solid(2) |
| **Echogenicity** | Anechoic(0), Iso/Hyperechoic(1), Hypoechoic(2), Very Hypoechoic(3) |
| **Shape** | Wider-than-tall(0), Taller-than-wide(3) |
| **Margin** | Smooth/Ill-defined(0), Lobulated(2), Extrathyroidal(3) |
| **Foci** | None(0), Macrocalcification(1), Peripheral(2), Punctate(3) |

### `generate_report(image_id, patient_id, breakdown, malignancy_prob)` → `str`

生成可打印的临床报告文本,包含患者信息、各维度预测、TI-RADS 分级及恶性概率。

---

## 2. Grad-CAM (`grad_cam.py`)

双主干显式分支选择:
- `branch="cls"` → `cls_backbone`(EfficientNet-B2),解释恶性分类决策
- `branch="seg"` → `seg_backbone`(ConvNeXt-Nano),解释分割决策

`_resolve_gradcam_target_layer(model, branch)` 取末个 blocks 或末个 Conv2d。
