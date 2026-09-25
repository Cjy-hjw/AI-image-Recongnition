# R-B 审查报告 · 官方输出契约与健壮性(ROI 新架构轮)

审查员:R-B(对抗式)· 日期:2026-09-25
范围:`roi_pipeline.py`(新)/ `v21_output.py`(+339)/ `postprocess.py`(+133)/ `case_pipeline.py` / `runner.py` / `nodule_split.py` / `tiling.py`
方法:逐字段核官方契约 + 实跑反例。**不采信注释声称**,每条 finding 附实跑命令与输出。
实跑环境:`D:/program/Python/anaconda/envs/Demo_D/python.exe -X utf8`(torch 2.5.1+cu121 / py3.11)。
基线测试:`pytest tests/test_roi_inference.py tests/test_v21_output.py -q` → **85 passed**;
`pytest tests/test_prediction_json.py tests/test_service_runner.py tests/test_model_runner.py -q` → **106 passed**;
`pytest tests/test_nifti_io.py tests/test_case_discovery.py tests/test_diagnose_detection.py -q` → **108 passed**。
(即:下述 finding 全部**在测试全绿的前提下**成立。)

---

## FINDING 1 — 🔴 P0-阻塞:ROI 路径下 ETE 头与 T 头的物理判据输入**恒为空**(band_ratio≡0 / d_mm≡0)

**【severity】P0(功能级:两项目标四得分项的特征判据被静默抹平)**
**【file:line】**
- `src/inference/roi_pipeline.py:136-182`(`roi_meta_for_instance` 全键集,**无 `roi_mask`**)
- `src/models/multi_task_model.py:498-532`(`m = roi_meta[i].get("roi_mask")` → None → 全零 mask → `d_mm = physical_diameter_from_mask(零mask, 0.0) = 0`)
- `src/models/multi_task_model.py:557`(ETE 头吃这张全零 mask)
- 对照:训练侧生产者 `src/dataset/roi_dataset.py:304-331`(`meta["roi_mask"] = ...`)

**【机制】**
`forward_roi` 的契约文档写「推理侧由 ROI pipeline 装 `roi_mask`」(multi_task_model.py:465-468),
但 `roi_pipeline.roi_meta_for_instance()` 返回的键集**没有 `roi_mask`**;我全仓 grep
`src/inference/` + `src/service/` 对 `roi_mask` 命中数 = **0**(唯一生产者是训练侧 `RoiDataset.__getitem__`)。
于是模型侧走 `m is None → torch.zeros(1,H,H)` 兜底:
- `RoiExtensionHead.forward(roi_spatial, roi_masks)` → `boundary_band_from_mask(全零图)` → band 恒空 → `local = 0/(0+eps) = 0`、`band_ratio = 0`;
- `d_mm = physical_diameter_from_mask(roi_masks, spacing_t=0.0)`;`spacing_mm` 恒 `None` → spacing 张量全 0 → **返回 `torch.zeros_like`,连 mask 面积都不参与**;
- `RoiClinicalTHead` 的 scale 通道 `[d_mm, log(clamp(d_mm,1e-4))]` = `[0, -9.21]` **对所有病灶恒等**。

**反例(实跑)** `tmp_review_infb_dynmask.py` + `tmp_review_infb_realmodel_roi2.py`(真 `MultiTaskThyroidModel`,后端 `RetinaNet/convnext_nano` 结构,仅权重随机):
```
[probe] K=3 roi_mask 缺键/全零 = [True, True, True]      # 三条 meta 全部缺键
band_ratio [0.0, 0.0, 0.000977]                          # 前两条(缺键/显式零 mask)恒 0
d_mm [0.0, 0.0, 0.0]                                     # 第三条给了 GT-like mask 仍为 0(spacing=0 短路)
```
`_run_roi_dynamic` 端到端(`tmp_review_infb_dynmask.py`)实跑确认三条 meta 全部 `roi_mask` 缺键。
即便补上 mask,`d_mm` 仍恒 0 —— 两个独立缺陷叠加,后者被 docstring 明确"有意"(mask_features.py:157-162 「spacing=None → 恒返回 0」),前者不是。

**【影响】**
ETE(目标四 40% 内的一项,加权 Kappa)与 T(目标四内一项,AUC 吃 `t_conf` 排序)在 ROI 路径下
的判据输入退化为常量。这不是"数值差一点",而是**该路径本轮达不到承诺的收益**;
若上云标定按"ROI 四头真实输出"预期收益,会得到近零增量而归因困难(且 `_shared_nodule_context` 的旧路已用整图 soft_mask,故**现状整图路径反而有真 mask**,ROI 开了可能反而退化)。
注:`_roi_nodule_context` 消费的四头值本身合法(见 PASS 4),所以**不会报错、prediction.json 完全合规**,只有指标低。

**【建议】** 三选一必须先裁决(我不断言哪个是正解,但必须显式化):
1. `roi_meta_for_instance` 装 ROI 域 mask:**必须与图像同裁剪管线**(`crop_nodule` 的 pad 后框),再 resize 到 roi_size,即 `_crop_mask_like_image` 的等价物 —— `tmp_review_infb_instantiation.py` 实测:图像 patch 150×150 ↔ 按实例 bbox 裁的 mask 100×100、按裁剪框裁的才是 150×150;形状不齐时 `forward_roi` 的 stack 循环会零填充错位(实得 `boundary_band_from_mask(零图)=0`),**不报错**。
2. 或把 `spacing_mm` 与 mask 都显式置空并在文档/任务书把 ETE/T 两项从"ROI 收益"里**移除**(诚实降级)。
3. 无论哪条,`run_roi_forward` 应加一条**可区分性断言**:`band_ratio` 全 0 且 K>0 时 fail-loud(当前静默)。

---

## FINDING 2 — 🔴 P0:ROI 前向在 bf16 autocast 下必崩(`BFloat16` 无 numpy 转换);整图路径不崩

**【severity】P0(开 `AICOMP_ROI_HEADS` + `precision=bf16` → 该例 `ERROR` 占位 = 该例零输出)**
**【file:line】** `src/inference/v21_output.py:197-204`(`_to_np`,**无 `.float()`**)· 同一实现 `src/inference/postprocess.py:83-88` · 消费点 `v21_output.py:225-229`(`_roi_nodule_context`)/`:611`(`_softmax_mal_logits`)/`:720-730`(`_aggregate_roi_frame_rows`)
**【机制】** `v.detach().cpu().numpy()` 对 `bfloat16` 张量直接抛 `TypeError: Got unsupported ScalarType BFloat16`。
`precision="bf16"` 走 `run_forward` 的 `torch.autocast(dtype=torch.bfloat16)`(`model_runner.py:151`),
`with autocast` 块内 **未注册 autocast 的算子**(如 `torch.full`/`torch.as_tensor`/部分索引拷贝)按 autocast 语义**原样输出 bf16 张量**;
`_to_np` 没有 `.float()` 兜底 → TypeError。
**【反例(实跑)** `tmp_review_infb_abspath.py`:
```
v21_output._to_np(bf16) FAIL TypeError: Got unsupported ScalarType BFloat16
v21_output._to_np(fp16) OK
postprocess._to_np(bf16) FAIL TypeError            # 两处独立实现,同一缺陷
bf16 ROI dynamic FAIL: TypeError Got unsupported ScalarType BFloat16   # 端到端 _run_roi_dynamic
```
**【为什么整图路径不崩(这是差异,不是共性缺陷)】** `src/inference/model_runner.py:168-172`
```python
if precision in ["fp16","bf16"]:
    outputs["malignancy"] = torch.clamp(outputs["malignancy"].float(), ...)   # ← 唯一的 .float()
```
`precision=bf16` → 该分支同样执行 → 整图 `malignancy` 已被强制转 fp32;其余四头的 probs 由各头内部
`torch.softmax(logits.float(), dim=-1)` 产出 fp32(`roi_heads.py:167/279/372`,
`heads/location.py:185-186`)。
→ **结论:ROI 路径 `_to_np` 的 bf16 崩溃未被任何上游 `.float()` 覆盖**(该差异是本 finding 成立的关键,已单独实跑核对)。
**【触发条件】**(a) `precision` 显式为 bf16,(b) 或模型任一 ROI 张量落在未注册 autocast 的算子上而保持 bf16
(**本次无法在本地判定 (b) 的实测发生概率 —— 环境 torch 2.5.1(fake 模型不 autocast),交付镜像 torch 2.8.0,
我**没有**在 torch 2.8 autocast 语义下实跑,不做概率断言**;故本条按"条件触发"报,并附带下面的零成本硬化建议)。
**【建议(零成本、双实例)】** 两处 `_to_np` 各加 `.float()`:`v = v.detach().float().cpu().numpy()`。
与 fp16 路径逐位等价(fp16→fp32 无损),且消除"裸 logits 未过 clamp"的整图/ROI 路径不一致。

---

## FINDING 3 — 🔴 P0:动态 ROI 路径只对 `frame_indices ⊆ 抽样帧` 的实例输出四头 → 违反"所有检出结节都须给全字段",其余实例整例 ERROR

**【severity】P0(整例 ERROR = 该例零输出;或按另一路径 = 逐结节缺字段)**
**【file:line】** `src/inference/v21_output.py:661-696`(逐 `instances` 调 `_run_roi_dynamic`)· `src/inference/roi_pipeline.py:208-244`(`iter_instance_frames_dynamic`,`frame is None → continue`)· `v21_output.py:681-686`(零命中 raise)
**【机制】** 动态实例来自 `split_nodules(binary_3d)`(全帧 3D mask,`:859`),
其 `frame_indices` = **该 3D 实例出现过预测的所有帧**;而 ROI 消费的 `frame_by_index` 只含**抽样帧**
(`:883-884`,`{f["frame_index"]: f["frame_np"] for f in per_frame}`)。
`AICOMP_MAX_FRAMES` 一旦回退抽样(全帧为默认 `0`,但 env 可回退),实例帧 ⊄ 抽样帧即触发:
`iter_instance_frames_dynamic` 静默 `continue` → `n_frames_this_inst == 0` → `V21OutputError`
→ `runner._process_case` 的 `except Exception` → `_write_case_error` 写 **ERROR 占位(空 Nodules)**。
**【反例(实跑)** `tests/test_roi_inference.py::test_dynamic_zero_available_frames_raises` 自身即该反例
(官方口径:该病例仍应有 1 个预测结节)。`_split_3d` 的闭运算只在**时间轴**做(不影响空间连通),
但**连通帧号会跨抽样**;`v21_output.py:854-855` 的 `gap` 公式本身承认"AICOMP_MAX_FRAMES 回退抽样时本公式仍必要"。
**【官方依据】** 「模型应对**所有检出的结节**均输出完整的属性字段…评估脚本会自动根据空间匹配结果进行筛选」
(评审规则「三新维度仅对主病灶评估」条)—— 不输出 = 该结节四头一分不得;而当前实现是**整例**无输出,更重。
**【并存问题】** 与 FINDING 1 同源:`instances` 里出现"零可用帧"的实例,直接证明"该结节在抽样帧上有预测"这一前提不成立。
**【建议】** `iter_instance_frames_dynamic` 缺帧时**退化为最近/首帧的四头值**并打点统计(而不是 make-or-break 整例),
或者在 `_run_roi_dynamic` 前按官方口径"所有检出结节"逐实例给兜底 context(与 `_shared_nodule_context` 同款整图值),
二者都必须保留 `n_frames_this_inst==0` 的**可观测打点**(当前 print 只覆盖 n==1)。

---

## FINDING 4 — 🟠 P1:`spacing_mm=None` 使 T 头物理尺度对**所有**病灶恒 `d_mm=0`,且 `log(0)→-9.21` 落在与"无结节"同一量级

**【severity】P1(单列,不崩;但 T 一项在 ROI 路径上退化为"无物理量可判")**
**【file:line】** `src/models/mask_features.py:174-200`(`spacing_mm is None → torch.zeros_like`;`math.pi`)
**【机制/实跑】** `tmp_review_infb_realmodel_roi2.py`:`roi_d_mm [0.0, 0.0, 0.0]`,即使第三条带了前景 mask 仍为 0(走 `spacing_mm` 归一化路径得到 `spacing_mm=0.0`,乘法结果恒 0)。docstring 已把它标为有意语义,但**FINDING 1 的 `roi_mask` 缺口让"空 mask → 0"与"有 mask 但无 spacing → 0"两种状态在数值上不可区分**,头无法区分"无结节"与"不知尺度"。
**【建议】** 显式记录(任务书/文档)ROI 路径 T 头 scale 通道为常量;或给 `RoiClinicalTHead` 增加"spacing 缺失"标志位。

---

## FINDING 5 — 🟠 P1:ROI 裁剪几何与训练数据面口径不一致(且非最坏情形):裁剪不 pad-to-square,而 H/W 长宽差异被 `preprocess_image` 强压成 512×512

**【severity】P1(不报错、产物合法;只影响 ROI 头输入分布 → 准确率)**
**【file:line】** `src/inference/roi_pipeline.py:120-133`(`crop_roi_for_instance` → `crop_nodule`)· `src/inference/roi_pipeline.py:57-62` 契约注(已自述该差异)· `src/inference/model_runner.py:113-124`(`preprocess_image` → `cv2.resize(img_np,(img_size,img_size))` 强方)
**【机制】** 训练侧 `RoiDataset.crop_roi`(`roi_dataset.py:264-277`)是 **pad-to-square**(`side=max(h,w)`,右/下补 0),`roi_collate` 侧 resize 到 `roi_size`;推理侧 `crop_nodule` 是**按轴独立 pad**(不撑方)。
`preprocess_image` 把 `(H',W')` 直接 resize 成方阵:
- 细长结节(如 `tmp_review_infb_holdout.py` 的 30×46)各向异性比例 = 46/30 ≈ 1.53;
- 若两侧都不 pad,训练/推理各向异性比例相消,是**良性的**;
- 但 `roi_collate` 反推:训练 patch 已是方形 → 方阵 resize 是**恒等**(零各向异性),而推理侧带 1.53 → **ROI 域宽高比 / 腺体-结节面积比 / d_px 全部与训练分布不一致**。
**【实跑证据】** `tmp_review_infb_holdout.py`:
```
原始实例框(xyxy) = (100,80,130,100) -> loc 头 bbox_xyxy
crop_nodule 实际裁剪框 = (92,75,138,105)   # pad 后
patch shape (30, 46)   crop box size (46, 30)
```
**【建议】** 二选一显式裁决并写进训练/推理同一契约:`crop_roi_for_instance` 改成 pad-to-square 与训练侧对齐,或训练侧改按轴 pad。**不要只改一侧** —— 两侧同时改是恒等,反而掩盖问题(与 F4/M3 同款伪证风险)。

---

## FINDING 6 — 🟠 P1:`d_mm` 的物理值在训练面与推理面分别为 0 与正数 → ROI 头学到的是纯噪声特征轴

**【severity】P1(训练有效性;不是崩溃)**
**【file:line】** 训练 `src/training/loop/*`(+ `train_official.py run_roi_epoch`)经 `roi_labels_from_meta` 组装 `roi_meta`,**同样**只透传 `spacing_mm`(恒 None)。
**【机制】** 训练侧 `spacing_mm` 恒 None(官方影像无 spacing)→ 训练时 `d_mm` 恒 0;推理侧若按 FINDING 1 的建议补上 mask 而 spacing 仍为 None,则**同样恒 0**(一致,无害)。但只要任何一侧把 `spacing_mm` 填成非 None(如未来补 DICOM PixelSpacing),`physical_diameter_from_mask` 的**单点短路**就会让 `d_mm=0` 与 `d_mm>0` 两侧分布劈裂。
**【实跑/依据】** `model_runner.build_model`:`missing=44` 含 `roi_*` 时 raise 已实测(FINDING 8);`spacing_mm` 在 `roi_pipeline.py:153-154` 被显式写死为 `None` 透传。
**【建议】** 把 `spacing_mm` 的"有/无"作为**显式的、两侧同源**的开关写入契约文档;若要启用,必须训练侧同时提供(否则 T 头 scale 通道在推理侧出现训练从未见过的值域)。

---

## FINDING 7 — 🟡 P2:`_write_dynamic` 的 `_dynamic_det_conf` 在"部分帧缺概率图"时**静默少平均**,而 `soft_by_frame` 恰是抽样帧的字典

**【severity】P2(数值偏置,不报错)**
**【file:line】** `src/inference/v21_output.py:284-296`(`soft is None → continue`,注释自述"生产构造下不可达")· `v21_output.py:832`(`soft_by_frame = {int(fi): sf for fi, sf in zip(frame_indices, soft_frames)}`)
**【机制】** 该 `continue` 的前提"实例帧 ⊆ 抽样帧"与 **FINDING 3 的同一前提**;二者会同时被抽样回退打破。一旦打破,`_dynamic_det_conf` 的分母(实例出现帧数)与实际参与平均的帧数不等,`Detection.confidence` 系统性偏高,而 AP 排序口径随之变化 —— 而且**它不会像 FINDING 3 那样 raise**(`n_frames_this_inst>0` 但部分帧缺图时不触发零帧守卫)。
**【实跑】** `tests/test_roi_inference.py::test_dynamic_zero_available_frames_raises` 构造的 `per_frame 帧号 (0,2,4)` + 实例 `frame_indices [1,3]` 正是"部分缺图"可达性反例的直接证据(该测试只覆盖了"全缺"分支)。
**【建议】** 与 FINDING 3 一并处置:缺帧时应 raise 或显式回退并打点,不要 `continue`。

---

## FINDING 8 — 🟡 P2:ROI 输出的 `tirads_probs` / `ete_probs` / `location_probs` **未做"和为 1"校验**;ROI 路径的 `_qwk_round` 对未归一分布会产出越界码的风险链

**【severity】P2(当前不触发,原因见下;防御面缺口)**
**【file:line】** `src/inference/postprocess.py:154-162`(`_check_len` 只查 shape + finite)· `v21_output.py:232-233`(`_qwk_round(tirads_probs)` / `_qwk_round(ete_probs)`)· 对照整图路径 `v21_output.py:479-480` + `prediction_json.build_nodule_entry` 的 `_finite_probs`(也**只**查长度 + finite,不查和)
**【机制】** `_qwk_round` = `int(round(Σ j·p_j)) + lo`;若上游分布和显著 >1 或 <0,`E[j]` 可越出 `[0,len-1]`,产出 `tirads_class=6` 或 `0`。
**【实测未触发的理由(试过的反例)**】`tmp_review_infb_locprobs.py`:200 seed × 3 scale(1/10/40)穷举 `compose_location_probs` 输出,`max|Σ−1| = 1.79e-7 < 1e-6`;`tmp_review_infb_jsoncontract.py` 三档 logits scale(1/8/40)下 `locprobs_sum ∈ {1.000000063, 0.999999944, 1.000000089}`,长度 7/4/5 全部正确;ROI 头内部均为 `torch.softmax(...float())`。故**当前不可达**。
**【建议】** 若未来 `validate_roi_probs` 被复用/被绕过(如有人直接构 dict),这层缺"和为 1"的校验;`build_nodule_entry` 的 `_finite_probs` 同样。可加"和∈[1-1e-3,1+1e-3]"(不严 1e-6,避免 fp16 往返误杀)。

---

## FINDING 9 — 🟡 P2(信息):Predictions 目录白名单与动态全帧回填 —— **本次 ROI 改动未引入任何额外落盘**,但 ROI 路径的 print 打点只到 stdout

**【severity】P2(实测无风险,记录以闭合该项)**
**【机制/实跑】** 
- `roi_pipeline` / `_run_roi_static` / `_run_roi_dynamic` 全仓唯一落盘点是 `_build_entries_and_save_masks` → `save_binary_mask`(`.nii.gz`)+ `_atomic_write_json`(`prediction.json`);ROI 改动**只加了 stdout print**(`v21_output.py:692-693`)。
- `tmp_review_infb_txtlog.py` 实跑:目录内容 `[]`(stdout 打点不落盘)。
- 动态 3D 全帧回填未被动过:`tmp_review_infb_*` 无涉及,但仍实跑核对了 `scatter_frames_to_full_length` 契约(`nifti_io.py:192-244`,帧号越界/长度不齐/尺寸不一致三条均 raise)与 `save_binary_mask` 的 `(H,W)→(W,H,1)` 单帧 Z 维(`:88-99`)。
- ⚠️ 但 **FINDING 3 的 ERROR 路径会让整例输出变成"空 Nodules"占位**(`_write_case_placeholder`,`runner.py:555+`)—— 这不是"多余文件",是"该例全部指标不计分"。已在 F3 计入严重度。

---

## FINDING 10 — 🟡 P2:旧 ckpt(无 ROI 键)在推理侧 **strict 拒**(已实测,与训练侧口径一致);但 `_ensure_built` 的异常只落日志,契约上属"伪装健康"

**【severity】P2(行为已实测确认;运维可观测性缺口)**
**【file:line】** `src/inference/model_runner.py:70-90`(`aux_only_missing` 白名单,`roi_*` **不在**白名单)· `src/service/runner.py:174-200`(`_ensure_built` 惰性构建,异常如何暴露未在本文件内处理)
**【实跑证据】**
```
python tmp_review_infb_load.py
RuntimeError: [FATAL] 权重与模型结构不匹配，拒绝静默部分加载：checkpoints\official_heads_best.pth
  missing keys (44): ['roi_location_head...', 'roi_extension_head...']   unexpected keys (0)
```
**结论:strict 白名单口径 = 训练侧 `ROI_HEAD_PREFIXES` 不在 `OFFICIAL_HEAD_PREFIXES` 的注释一致(F8 PASS)**。
**【建议(仅记录)**】部署文档需明确:ROI 轮上线必须**代码+权重成对**;旧权重 + 新代码(带 ROI 头)在 `build_model` 即拒;
`_ensure_built` 的 raise 若被上层吞成"`/health` 仍 200",与 #123 同类风险(已在 backlog)。

---

# PASS 清单(试过什么反例才排除)

| # | 核验点 | 反例/方法 | 结果 |
|---|---|---|---|
| P1 | **StudyUID = PatientId、MaskFiles↔Nodules[].MaskFile 一一对应** | `prediction_json.build_prediction_json:381-393` 逐行读:二者同源 `mask_files` 单列表;`validate_prediction_json` 双侧 sorted 相等 + 重复即 raise;`_validate_relative_mask_path` 拦绝对路径/`..`(`:196-231`);`v21_output:354-355` 用正斜杠拼相对路径 | 契约满足,无 finding |
| P2 | **置信度值域 [0,1] 且非 NaN/Inf** | `tmp_review_infb_jsoncontract.py` 三档 logits scale(1/8/40):`mal_conf/t_conf` 全落 [0,1];`_clamp01` + `validate_prediction_json:508-515` 双闸;`_softmax_mal_logits` 已把裸 logits 转概率(E2 P0-1 修复经我复核成立) | 满足 |
| P3 | **int 字段类型(TI_RADS_Class/ExtrathyroidalExtension/Location/T/ProcessingTime_ms)** | `build_nodule_entry` 逐个 `int(...)` + 值域 raise;`_qwk_round` 返回 `int`;`validate_prediction_json` 验 `isinstance(ProcessingTime_ms,int)` + 三类越界 | 满足 |
| P4 | **`_roi_nodule_context` 与 `_shared_nodule_context` 同键同编码** | `tmp_review_infb_jsoncontract.py`:key set 完全相等(shared-only=∅ / roi-only=∅);Location `argmax+1`、T `int(t_conf>0.5)` 不+1 与 `_qwk_round(lo=1)` 逐行比对 | 满足 |
| P5 | **LocationProbs 索引序 7 类 + 和为 1** | `tmp_review_infb_locprobs.py` 200 seed×3 scale 穷举:`max|Σ−1|=1.79e-7 < 官方容差 1e-6`;顺序由 `heads/location.py:121-135` import 期断言 + `label_schema.LOCATION_LABELS` 单一权威 | 满足 |
| P6 | **动态 3D z 轴 = 原视频全帧数** | `scatter_frames_to_full_length`(`nifti_io.py:192-244`)三条 raise 逐条读 + `_write_dynamic:798-815` 缺 `n_total_frames` fail-loud;ROI 改动**未触碰**该函数与调用点(全仓 grep `scatter_frames_to_full_length` 仅 4 处,均非本 diff) | 未被 ROI 改动破坏 |
| P7 | **故障降级三态(崩 / 静默垫底 / 优雅回退)** | 逐分支读 + 实跑:`run_roi_forward` 无 `forward_roi` → `RoiPipelineError`(`roi_pipeline.py:275-282`,已测 `test_model_without_forward_roi_raises_no_fallback`);ROI 前向异常 → 整例冒泡(已测 `test_dynamic_roi_error_bubbles_whole_case`);`_write_static` 缺 `orig_img_np` → `V21OutputError`(`tmp_review_infb_paths.py` 实测 raise);`_to_np` bf16 → 整例 ERROR(即 FINDING 2) | 三态**均有明确归属**,唯一缺陷是 F2 的 bf16 |
| P8 | **旧权重 strict 口径与训练侧一致** | `tmp_review_infb_load.py` 实测 `missing=44`(全 `roi_*`)→ raise;`ROI_HEAD_PREFIXES` 不在 `OFFICIAL_HEAD_PREFIXES`;`aux_only_missing` 白名单**不含** `roi_*` | 一致(F10) |
| P9 | **ROI 目录白名单** | `tmp_review_infb_txtlog.py`:stdout 打点不落盘;落盘点全仓唯一 | 无多余文件(F9) |
| P10 | **fp16 数值路径** | `tmp_review_infb_roictx.py`:fp16 张量经 `_to_np` OK,四头编码正常 | 满足 |

---

# 附:本次产出的探针(全部保留在工作区,勿删)
- `tmp_review_infb_sumprobs.py` · `tmp_review_infb_roictx.py` · `tmp_review_infb_bf16.py`
- `tmp_review_infb_load.py` · `tmp_review_infb_paths.py` · `tmp_review_infb_dynmask.py`
- `tmp_review_infb_realmodel_roi.py` · `tmp_review_infb_realmodel_roi2.py` · `tmp_review_infb_holdout.py`
- `tmp_review_infb_abspath.py` · `tmp_review_infb_jsoncontract.py` · `tmp_review_infb_instantiation.py`
- `tmp_review_infb_disk2d.py` · `tmp_review_infb_txtlog.py` · `tmp_review_infb_locprobs.py`
