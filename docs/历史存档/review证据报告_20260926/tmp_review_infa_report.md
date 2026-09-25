# R-A 审查报告 · ROI 数据流与坐标几何(ROI 新架构轮)

审查员:R-A(对抗式)· 日期:2026-09-25
范围:`src/inference/roi_pipeline.py`(新)/ `v21_output.py` ROI 段 / `case_pipeline.py` 调用点 /
`runner.py` 接线 / `nodule_split.crop_nodule` 几何 / `roi_dataset.crop_roi`(训练侧镜像基准)/
`multi_task_model.forward_roi` + `roi_heads` + `roi_encoder`(坐标链消费端)/ `metrics_official.aggregate_by_roi_instance`(聚合口径基准)
实跑环境:`D:/program/Python/anaconda/envs/Demo_D/python.exe -X utf8`
基线:`pytest tests/test_roi_inference.py` → **41 passed**;`test_v21_output.py` 44 / `test_tiling_inference.py`+`test_static_tta.py`+`test_nodule_split.py` 121 / `test_diagnose_detection.py`+`test_detection_audit.py` 74 全绿。

---

## 🔴 给主会话的紧急提示(非 finding,但必须先读)

1. **本次审查进行中,主会话修改了被审代码**:`src/inference/v21_output.py` 的
   `_build_entries_and_save_masks` 在我审查期间被重命名为 `_build_entries_and_save_slices`
   (连同 tests/scripts/AGENTS.md/docs 6 个文件)。我已复跑基线确认当前状态全绿
   (`test_roi_inference` 41 / `test_v21_output` 44 / `test_diagnose_detection`+`test_detection_audit` 74),
   **但 R-B(lane B)已写出的报告 §基线("85 passed / 106 passed")是重命名前的旧态**;
   其 finding 的行号/引用需按新名复核一遍再采信。我未改任何仓库文件(仅新增 24 个
   `tmp_review_infra_probe*.py` 探针,按下述纪律保留)。
2. **`tmp_review_infb_report.md` 的 FINDING 5(ROI 裁剪几何两侧不一致)与我方
   F1 同题**(它给 P1 单列、未察觉 resize 是数学恒等、也未给"两侧都不 pad 才是自洽解"的选项)。
   两报告须在主会话合并裁决,勿分别采纳导致相反处方(它建议"改成 pad-to-square" —— 那是错方向,
   详见 F1)。

## 结论摘要

| # | 严重度 | 一句话 |
|---|---|---|
| F1 | **P0** | 训练侧 ROI = bbox 紧切片(无上下文),推理侧 ROI = bbox + 25% 边距 → 同一结节的两套 ROI 几何/内容系统性不同,35 epoch 训的 ROI 头在推理时吃**未经标定的输入分布**;正确修法是**去掉推理侧 pad**(不是把训练侧也 pad 方)。 |
| F2 | P1 | `AICOMP_ROI_HEADS=1` 对**全进程**生效而诊断侧 `run_static_inference` 不注入 `orig_img_np`;诊断链 `case_result` 会带 `roi_heads=True` 但实跑整图链(打点与实跑路径不符,且 `--mechanism-audit` 的开训硬判据消费的就是这个 case_result)。 |
| F3 | P2 | `roi_meta['pad_hw']=(0,0)` 在推理侧恒为假值(真实 pad 由 crop_nodule 内部决定、既有 4 像素也有 5 像素,不可由 bbox 反推);键集对齐了但**语义未对齐**,任何未来按 pad_hw 还原坐标的消费方会静默错。 |
| F4 | P2 | `_build_entries_and_save_slices` 的 context 用 `.get()` 取值 → 半键集 context 静默产出 `Location=1/ETE=1/T=0` 占位编码(合法 JSON、指标按错值计)。当前两生产调用方都给全键,是潜在而非现实缺陷。 |
| F5 | P2 | 每帧每实例都打一条 `[ROI-dynamic] 单帧实例` stdout(`n==1` 时),动态病例数百帧 → 数百行日志噪声(不影响 stdout 正确性,但污染训练/诊断日志)。 |
| F6 | P2 | `case_pipeline` 注释称 ROI 开时会返回"原图域 pred_mask",实测 tiling 分支的 `pred_mask` 是 **512² tile 域**(`mask_prob_override` 才是原图域);注释与代码不一致(至少会炸,但注释误导后继改动)。 |

**PASS 清单(试过什么反例才排除)**见文末。

---

## F1 — 【P0】训练/推理 ROI 裁剪几何不一致:一边 bbox 紧切片、一边 25% 上下文边距

**【file:line】** `src/inference/roi_pipeline.py:120-133`(`crop_roi_for_instance` → `nodule_split.crop_nodule`)
· 基准 `src/dataset/roi_dataset.py:264-277`(`RoiDataset.crop_roi`,训练侧唯一生产者)
· 消费端 `src/inference/model_runner.py:113-124`(`preprocess_image` 强方 resize)
· 契约自述 `src/inference/roi_pipeline.py:57-62`

**【机制】** 两侧是**两种不同的补法**,不是"一个 pad 一个不 pad":

| | 训练侧 `RoiDataset.crop_roi` + `resize_img_to` | 推理侧 `crop_roi_for_instance` + `preprocess_image` |
|---|---|---|
| 第一步 | `frame[y0:y1, x0:x1]`(**bbox 紧切片,无上下文边距**) | 先 `crop_nodule(..., pad_ratio=0.25)`(**外扩 25% 上下文**) |
| 第二步 | 短边补 0 到**正方形**(`side=max(h,w)`) | 不补方,保持 `(H',W')`(细长) |
| 第三步 | `cv2.resize`(双线性)到 `roi_size` | `cv2.resize`(双线性)到 `img_size` |

🔴 **关键判读(与 R-B FINDING 5 的分歧点)**:第二步的两条路**在像素上是同一个操作** ——
"pad 到方形再 resize 成方形" ≡ "直接 resize 成方形"(cv2 双线性是分离的按轴重采样,
pad 区与重采样节点重合的实测误差仅 8/255×1 像素=1.6%,边界瞬变且**推理侧无 pad 区**)。
所以 **第二步不是 bug**,不能拿它当处方。真正的差异是**第一步**:

- 训练 ROI 只含 `mask_bbox` 紧框内的像素 → **结节外腺体全部在 ROI 外**;
- 推理 ROI 含 25% 边距 → 同一病灶的 ROI 多出成圈的外部组织。

**【反例 / 实跑证据 · 探针 22、24】**
```
$ python tmp_review_infra_probe22.py
训练 ROI shape (9, 9) → (224, 224)  pad_hw (0, 5)
推理 ROI shape (12, 6) → (224, 224) actual (1, 1, 7, 13)
patch 逐像素相同? False     像素差最大 = 210  平均 = 69.672
（方形 bbox 3,4,9,10 也是 False —— 训练侧 6×6 原样 vs 推理侧 10×10 带边距）
$ python tmp_review_infra_probe24.py
结节 20×20 + 紧贴右缘的腺体(120):
  训练 ROI (20,20):  覆盖列 [20,40) → 右缘 40:46 腺体**完全在 ROI 外**;ROI 内 120 值像素数 = 0
  推理 ROI (30,30):  覆盖列 [15,45) → 含腺体 5 列;          ROI 内 120 值像素数 = 100
```

**【后果(为何是 P0 而非 R-B 的 P1)】** 本轮已合成架构、模型面、训练端三路审查修好的
**同一棵树** 里,唯一没做的是"训练输入 == 推理输入"。后果不是崩溃而是**静默劣化**:
ROI 头(尤其 ETE band / TI-RADS margin——判据本体就是"结节与周围组织的交界")在 35 epoch 里
学的是"紧框 ROI 的纹理统计",推理时收到"带 25% 边距 ROI",两者的结节占比、band_ratio 分布、
邻域上下文全部漂移;`local-→-线上`的传递比例会因此系统性偏小,而**判线时无法归因**(要么归因"ROI 没用",
要么归因"训练不够",都不会指向这里)。对"上云标定 λ → 20ep → 判线才换权重"的当前计划,
这是在标定之前就必须先钉死的输入契约。

**【建议处方(不要采 R-B 的"改训练侧 pad-to-square")】** 去掉推理侧上下文边距,使两侧同为
"bbox 紧切片 → 补方 → resize":
```python
# crop_roi_for_instance:不再走 crop_nodule 的 25% pad
x0, y0, x1, y1 = inst_bbox_xyxy
patch = np.asarray(frame)[y0:y1, x0:x1]
# 补方(pad 右/下,与训练侧 crop_roi 同式)→ 交给 preprocess_image resize
```
⚠️ 三条附带要求:
1. **两侧同时改 = 恒等**,单侧改才是修复(R-B 已正确指出这一伪证风险,但选错了方向);
2. ROI_PAD_RATIO 常量与 `roi_pipeline` docstring「复用在生产被逐结节分类消费过的 crop_nodule」
   的论证随之失效(`crop_nodule` 其实**零生产消费方**,全仓只有该函数与测试引用)——须同步改文档/AGENTS;
3. 改完必须补一条**跨侧几何对拍**回归测试(同一 bbox,`RoiDataset.crop_roi` 与
   `crop_roi_for_instance` 产出的 patch 逐像素相等)——这是本轮缺的那条镜像锚,
   `test_roi_inference.py` 的 `test_crop_roundtrip_pixelwise_non_square` 只验"patch == frame[框]"
   的自一致性,**对训练侧零约束**(零区分力)。

---

## F2 — 【P1】`AICOMP_ROI_HEADS` 是全局 env,但诊断镜像链不注入 `orig_img_np` → 打点与实跑路径不符

**【file:line】** `src/service/runner.py:410-411`(env 开 → `result["orig_img_np"] = img_np`)
· `src/inference/case_pipeline.py:229-241, 336-340`(`roi_heads=True` 只依赖 **env**,不依赖是否真注入)
· `scripts/diagnose_detection.py:179-196`(`run_static_inference`,**只注入 orig_hw**)

**【机制】** env 是进程级:`AICOMP_ROI_HEADS=1` 同时让 runner 与诊断脚本里的
`diagnose_static_case` 走 `_roi_on=True` 分支 → `case_result["roi_heads"] = True`。
但诊断脚本的镜像只注入 `orig_hw`(不注入 `orig_img_np`,因为它**不落盘**、走
`run_static_inference_from_result` 的整图共享链)。于是:

- 诊断侧 `case_result` 打点 `roi_heads=True`,而实跑路径是**整图共享**;
- `--mechanism-audit` 的开训硬判据(总纲 §6.4:ep≥2 且 `cliff_frac_boosted==0` 即 abort)消费的
  正是这个 `case_result` 链,任何"这条 case 走的是哪条路"的离线统计都会读错;
- 更危险的是**未来**:若 diagnostic 链后来复用同一 `case_result` 直落 `_write_static`(省一次前向
  是自然优化),就会在"env 开 + 无 `orig_img_np`"下**硬 raise `V21OutputError`**,而原因离根因很远。

**【反例 / 实跑证据 · 探针 16】**
```
$ python tmp_review_infra_probe16.py       # env 关 vs env=1 各跑一次诊断链
env 关 : (None, [()], [0.48199779])
env 开 : (True, [()], [0.48199779])
→ 开关开时诊断侧多出 roi_heads 标记但实例/置信度逐位相同: True
```

**【建议】** 二选一:(a) 诊断脚本也在 env 开时注入 `orig_img_np`(成本为零,`img_np` 在手);
或 (b) `roi_heads` 键的判据从"env 开"改成"`orig_img_np` 存在"(即"这条路**真跑了** ROI")。
推荐 (b) —— 打点的语义应是"实跑路径",不是"开关状态"。

---

## F3 — 【P2】`roi_meta['pad_hw']=(0,0)` 是**恒假值**,"键集对齐"被误当作"语义对齐"

**【file:line】** `src/inference/roi_pipeline.py:149-151, 169` · 基准 `src/dataset/roi_dataset.py:24, 289`

**【机制】** docstring 的理由是"`crop_nodule` 不 pad 到方形故无补边量"。事实是 `crop_nodule`
**确实补了边**(按轴 pad_ratio=0.25,`pad_x=f(bbox 尺寸)`),只是补在横纵两轴而非补到方。
实测同一函数返回 **4 像素**(pad_ratio 0.25 落在 bbox 宽 15 上,round(3.75)=4)与 **5 像素**
(bbox 高 20,round(5)=5)——都**无法**从 `bbox_xyxy` 反推,因为框已被钳位裁剪,pad 量依赖
裁剪前尺寸。故任何按 `pad_hw` 还原"ROI 像素 ↔ 原图坐标"的消费方(训练侧该键的既定用途)
在推理侧会**静默偏 4-5 像素**,不报错。

当前实测无消费方(`grep pad_hw`:生产代码只有生产者与透传,`roi_heads.py:91` 的 docstring 声称
"坐标还原由消费端做"但代码里没有任何消费端)→ 潜在缺陷,不是现实故障。

**【实跑证据 · 探针 2/23】**
```
$ python tmp_review_infra_probe2.py
   meta.bbox_xyxy = (4, 6, 12, 11)   # E2 P1-1 修复后的实例半开框 ✓
   actual 裁剪框  = (2, 5, 14, 12)   # 实际生效框,含 pad(左右各 2、上下各 1)
   pad_hw = (0, 0)                   # ← 与 actual 矛盾
$ python tmp_review_infra_probe23.py
细长 bbox w6 h20 → patch (30, 10)  actual (8, 0, 18, 30)   # pad_x=1? 实测 2;pad_y=5
```

**【建议】** 删掉 `pad_hw` 键(键集"对齐"若无消费方就是负债,且它给的是假值),或填**真实 pad 量**
并通过 `crop_roi_for_instance` 一并返回。二选一都必须同步 `roi_dataset` 侧注释说明两侧键义不同。

---

## F4 — 【P2】context `.get()` 静默占位:半键集 context 会产出 `Location=1 / ETE=1 / T=0`

**【file:line】** `src/inference/v21_output.py:389-398`(`context.get("ete_code", pj.ETE_NONE)` 等)
· 兜底 `src/inference/prediction_json.py:264-271`(location_probs=None → 均匀占位 → argmax=0 → Location=1)

**【机制】** `malignancy_confidence` / `tirads_class` 用 `[]`(缺即 KeyError,**安全**),但
`ete_code` / `location_probs` / `location` / `t_clin_sig` / `t_confidence` 用 `.get()` 带默认值:
缺这些键的 context 会写出"合法但全是占位"的编码(ETE=1=无侵犯 / Location=1=左上叶 / T=0),
`validate_prediction_json` 全过。生产两个调用方都给全 9 键(实测键集逐键相同,探针 13),
所以是潜在缺陷;风险在于"新增第三个调用方"或"ROI 路某键改名"时会**静默按占位计分**。

**【反例 / 实跑证据 · 探针 15】**
```
$ python tmp_review_infra_probe15.py     # context = {"malignancy_confidence":0.9,"tirads_class":4}
{'Location': 1, 'ExtrathyroidalExtension': 1, 'T_ClinSig': {'T': 0, 'T_conf': 0.0}, ...}
→ 未报错、JSON 合法
※ 全空 context 会 KeyError('malignancy_confidence')(探针 14)——即"缺得够多才炸"
```

**【建议】** 把 `_build_entries_and_save_slices` 的 context 取值改成**双侧同契约**:要么全用 `[]`
(fail-loud,推荐),要么在进入循环前 `assert set(context) == REQUIRED_KEYS` 一次校验。
`.get()` 的向后兼容理由(CLI 不产四头概率)在当前代码体系下不成立——CLI 走 `build_probability_export`,
不经本函数(该理由在 `_tirads_class_from_breakdown` 的 docstring 里已被 Gate2 纠正过一次,同型第二次)。

---

## F5 — 【P2】动态 ROI 逐帧打点:`n_frames_this_inst == 1` 时每帧每实例打一行日志

**【file:line】** `src/inference/v21_output.py:687-693`(`_run_roi_dynamic` 的 `print`)

**【机制】** E2 review P1-2 意图是"给单帧实例打点,离线可统计人口"。但静态路径也走
`_run_roi_static`(单帧、不打点),而动态路径是**逐帧**调用本函数:若某个实例只有 1 帧
(`frame_indices` 长度 1),打点会触发;更实际的问题是 `n_frames_this_inst == 1` 的判定在
**每帧**都成立不了,但**每帧**都会重新计算并可能打印(取决于实例帧数)。

先确认实际行为:**30 个单帧实例 = 30 行**,若生产里存在大量"1 帧实例"(分割碎裂导致的短实例,
`_split_3d` 的时间闭运算已缓解但非零),单病例日志可达数百行;而 runner 的 `logger` 与
nohup 日志会把它带进云端训练日志(总纲已两次栽在"日志污染")。

**【实跑证据 · 探针 19/20】**(打印本身正常,窗口/实例分组正确 — 见 PASS 清单):
```
$ python tmp_review_infra_probe19.py
窗口数 = 3  各窗 (ids, frames): N1 [0..15] / N1 [16..31] / N1 [32..39]
→ 每窗单实例 & 帧号递增: True
```

**【建议】** 改成**病例级聚合打点**(函数末尾统计"单帧实例数/总实例数"打一行),
或降到 `logger.debug`。当前形态不影响数值正确性。

---

## F6 — 【P2】`case_pipeline` 注释与代码不一致:tiling 分支的 `pred_mask` 不是原图域

**【file:line】** `src/inference/case_pipeline.py:231-236`(注释:"返回的 pred_mask 是原图域")
· `src/inference/tiling.py:183-195`(`mask_prob_override = prob_full` 才是 [H,W] 原图域;
`decode_predictions` 对 `pred_mask` 仍 `sigmoid(outputs["mask"])`,shape = 512² tile 域)

**【机制】** `decode_predictions(masks_prob_override=...)` 只把 **override** 用于
`pred_mask/mask_binary`(`postprocess.py:239-249`),而 tiling 传的是 `prob_fused`(原图域)——
但 `case_pipeline` 的注释写"tiling 分支返回的 pred_mask 是原图域、img_tensor 是重建的中心 tile 张量"。
实测 tiling 分支的 `result["preds"]["pred_mask"].shape == (532,727)`(原图域)是**因为 override 生效**,
不是 tile 域;而 `RoiLocationHead` 吃的是 `spacing/bbox` 而不是 mask,所以当前无功能影响。
风险是后继改动拿"原图域 pred_mask"这条注释去推导 tiling+ROI 的兼容性(而两者已被互斥 raise 挡住,
但注释里的理由写错了:真实理由是"两条未验证路径叠加不可归因 + tiling 早退后本层没有原图",
而非"pred_mask 已是原图域")。

**【建议】** 订正注释:去掉"pred_mask 是原图域"的断言(它是 `mask_prob_override` 的语义,
decode 内部对 `pred_mask` 的契约仍是"与 outputs['mask'] 同形")。

---

## PASS 清单(试过什么反例才排除)

1. **静态 ROI 逐结节独立(不串味)** — 探针 10/11:FakeRoi 按 patch 内容分支,
   两实例 bbox → 原图亮/暗块,N1/N2 输出 `loc 7/mal 0.9975/tir 5` vs `loc 1/mal 0.0025/tir 1`,
   **逐结节不同且与实例序对齐** → 排除"四头值挂错结节 / 全实例共享"。
2. **动态 chunk(16)不跨实例、时序不乱** — 探针 19/20:
   40 帧单实例 → 3 窗 `[0..15]/[16..31]/[32..39]`;两实例各 20 帧 → 4 窗、每窗**只含单实例 id**。
   → 排除"跨实例混窗(会让 A 的均值含 B 的帧)"。
3. **帧号语义:用 `frame_indices`(原始帧号)而非下标** — 探针 19 窗口帧号递增且与
   `iter_instance_frames_dynamic` 的 docstring 一致;`_run_roi_dynamic` 缺帧时 `continue`
   而 `n_frames_this_inst` 只在 yield 时 +1 → 排除"取错帧静默用邻近帧"。
4. **ROI 裁剪坐标可逆** — 探针 2:`patch == frame[y0:y1, x0:x1]` 逐像素相等(非方阵 13×17);
   `crop_roi_for_instance` 与手工 `crop_nodule(bbox_xywh)` 输出**逐位相同** → 排除 off-by-one/xywh 混淆。
5. **3D 实例 bbox 取前四项忽略 z** — 探针 2:`_roi_instance_bbox((4,6,8,5,2,7)) == (4,6,12,11)`
   → 排除 z 混入 x/y。
6. **动态实例 bbox = 全帧空间并集,单帧 mask 必落在框内** — 探针 2 的 D 段:
   `(2..3,1..2)` 与 `(8..9,12..13)` 两帧 → 并集框 `(1,2,14,10)`,两帧 patch 同形 → 排除
   "逐帧各算 bbox 导致同 lesion 多份矛盾几何"。
7. **`global_loc` 计算与训练侧同式** — 探针 2/22:`((x0+x1)/2)/W` 实测与手算逐位一致;
   与 `roi_dataset.expand_roi_samples:183-184` 同式(同样用实例半开框)→ 排除 E2 P1-1 回归。
8. **mal 逐帧 softmax 后取概率均值 vs 训练侧同口径** — 探针 5:三帧 logits 两侧
   `mal_prob` 差 1.5e-8;`tirads_probs`/`t_conf` 逐位一致;训练侧 `aggregate_by_roi_instance`
   的 `mal_prob` 源 = `collect_roi_records:165-167` 的 **logits softmax**,而推理侧
   `_aggregate_roi_frame_rows:717-723` 是**逐帧 softmax 后均值** → 两者都在概率域,
   排除"一侧 logits 均值再 softmax"(会系统性压扁置信度)的错配。
9. **`_qwk_round` 编码换算复刻无偏差(期望取整 lo=1)** — 探针 9/17:
   `_roi_nodule_context` 与 `_shared_nodule_context` 对同一分布给同码;
   `t_clin_sig = int(t_conf>0.5)` 与 `argmax(t_probs)` 在 4 组边界 `(0.5,0.5)/(0.4,0.6)/(0.6,0.4)/(0.9,0.1)`
   上**逐组相同**(ROI 输出只有 t_conf 标量,两类时数学等价)→ 排除"T 少 +1 / argmax 口径错"。
10. **context 键集与整图共享 context 完全一致** — 探针 13:9 键零缺零多
    → 排除 `build_nodule_entry` 的 `.get()` 静默取默认(F4 的"当前形态"验证)。
11. **ROI 输出值域/有限性** — 探针 9(真 tiny 模型 + `_softmax_mal_logits`):
    mal 0.6535 / t_conf 0.5267,三个概率向量全在 [0,1] 且无 NaN
    → 排除"裸 logit 直通"(§J.5 已修的 P0-1 未回归)。
12. **`build_roi_batch` 逐张预处理与生产单帧路径逐位一致** — 探针 12:
    `torch.equal(t[i:i+1], resize(img)/255)` 对非方 patch(20×20、5×30)两张均 True
    → 排除"第二条 resize 实现/镜像漂移"。
13. **ROI 批量前向不互相污染(BN eval)** — 探针 12:批 vs 单张逐位相同
    → 排除"两实例的 patch 在批内互相影响(BatchNorm train 态残留)"。
14. **零实例豁免** — 探针 11 阴性路径:`test_zero_instances_no_forward` 覆盖
    (41 passed 内含),零实例不调 `build_roi_batch`(该函数对空列表 raise)。
15. **bbox 越界防御** — 探针 6:出界 bbox → `crop_nodule` 抛 `ValueError`
    (整例冒泡,不静默裁空) → 排除"静默裁到空 patch 喂模型"。
16. **1×1 实例** — 探针 6:`(4,6,5,7)` → patch `(1,1)`,不崩 → 排除"极小结节裁剪退化"。
17. **ROI 与整图四头编码口径可并存** — 探针 17:整图 `argmax+1` 与 ROI `_qwk_round/argmax+1`
    各自作用于自己的分布,同一输入下数值不同属预期(随机头),不构成编码冲突。
18. **`_roi_forward_chunk` 的 validate 生效** — `test_nan_malignancy_raises` /
    `test_inf_location_raises` / `test_wrong_length_raises_valueerror` 3 项实跑绿。

## 未覆盖(明确交回主会话裁决)

- **ROI 头 warm-start 的实际配比**:探针 17 显示同一输入下整图/ROI 四头输出差异明显
  (mal 0.5197 vs 0.3993、ete argmax 1 vs 3)。这**不是 bug**(ROI 头是新结构 + 新训),
  但要回答"开了 ROI 后线上会涨还是跌",必须先决定 F1 的输入契约,否则任何量级判断都无意义。
- **`Detection.confidence` 与 ROI 四头来自两个处理器**(检测项仍走整图分割软图):
  我已在 `write_case_v21` / `_build_entries_and_save_slices` 的 docstring 补了说明
  (作为"审查期间主会话已改的同一文件"的一部分),但**检测项在 ROI 轮不会变**这一事实
  会影响"ROI 支路整体上线"的表述,归主会话。

## 产物纪律

新增探针(均保留,未删;未改任何其他仓库文件):
`tmp_review_infra_probe1.py`(诊断链 env 耦合)、`probe2`(几何可逆)、`probe3/4`(训练侧聚合与 instance_idx 源)、
`probe5`(聚合口径对拍)、`probe6`(边界)、`probe7/10/11`(逐实例独立)、`probe8/22/23/24`(裁剪几何两侧对拍)、
`probe9/17`(真模型值域与准)、`probe12`(预处理/批一致性)、`probe13/14/15`(context 键集与占位)、
`probe16`(env 盲区)、`probe18`(case_result 键写入点)、`probe19/20`(动态分窗)、`probe21`(pad_hw 消费方 grep)。
