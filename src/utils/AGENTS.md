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

- **硬校验(2026-07-10 Gate2审查修复,此前该文件零测试覆盖)**：
  1. `calculate_tirads_points` 对 comp/echo/shape/margin/foci 各 idx 越界一律 `raise ValueError`（不再 `.get(idx,("未知",0))` 静默按0分处理）。实测案例:margin_idx=5(非法,应为2分档)此前被静默记0分,总分从真值3(TR3,~4.8%风险,建议穿刺)算成1(TR2,~1.5%风险,无需处置)——静默跨越了临床处置决策边界。`FOCI_MAP.get(i)` 此前唯独缺默认值,foci_probs 若>4维会在越界处直接 `TypeError` 崩溃,现同样归入越界硬校验。
  2. `generate_report` 对 `malignancy_prob=None/nan/inf` 一律 `raise ValueError`。此前 `nan >= 0.5` 在 Python 里恒为 `False`,会**无任何报错**地打印"良性倾向"——本该判"无效/不确定"的情形被静默判成良性,是癌症筛查场景的假阴性风险。
  3. 两处硬校验均依赖 `predict.py`/`video_predict.py` 的外层 `except Exception` 转 `ERROR` result.json 兜底(为此同步把 `predict.py` 的 gatekeeper 审计也纳入了 try 包裹范围)。
  4. 测试见 `tests/test_report_generator.py`(13个用例,覆盖评分表边界/越界idx/NaN/None/inf)。评分表本身(COMPOSITION/ECHOGENICITY/SHAPE/MARGIN/FOCI 映射 + TR1-5 分级区间)经核对与 ACR TI-RADS 白皮书(2017)完全一致,不是bug来源。

---

## 2. Grad-CAM (`grad_cam.py`)

双主干显式分支选择:
- `branch="cls"` → `cls_backbone`(EfficientNet-B2),解释恶性分类决策
- `branch="seg"` → `seg_backbone`(ConvNeXt-Nano),解释分割决策

`_resolve_gradcam_target_layer(model, branch)` 取末个 blocks 或末个 Conv2d。
