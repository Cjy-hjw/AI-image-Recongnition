# ROI 新架构训练端 review · R1 路(训练循环编排)

**范围**:`scripts/train_official.py` 的 ROI 训练编排区(`run_roi_epoch` / `_roi_detail_extra` /
`validate_roi_epoch` / `_mock_roi_samples` / `main` 的 ROI 构建区):910–1060、1360–1437、1534–1610。
**基线**:`pytest tests/test_roi_train_wiring.py -q` → **24 passed**(实跑确认)。
**领地声明**:主树未做任何功能改动,唯一一次改动是"就地探针 + 立即还原",`git diff` 中无 `R1-PROBE` 残留
(`grep` 零命中);探针脚本保留:`tmp_review_r1_probe.py` / `tmp_review_r1_probe2.py` / `tmp_review_r1_smoke.log`。

---

## P0 — ROI 支路在 CUDA 上必然崩:roi_mask 不上设备

**位置**:`scripts/train_official.py:934`(`imgs = imgs.to(device)`)+
`src/models/multi_task_model.py:499-519`(`masks` 组装)+ `src/dataset/roi_dataset.py:319-320`
(`roi_mask` 产出)。

**机制**:`RoiDataset.__getitem__` 用 `torch.tensor(numpy)` 产 `roi_mask`,**恒在 CPU**。
`run_roi_epoch` 只把 `imgs` 搬到 device,`roi_mask` 经 `roi_collate`(保 list)、
`_meta_rows_from_collate`(逐条透传)、`forward_roi` 一路**原样搬运**:

```python
masks.append(m.detach().float())          # m 仍 CPU
roi_masks = torch.stack(stackable, dim=0) # roi_images.device(多任务模型:519 行)
```

`torch.stack(cpu_tensors, device=cuda_tensor.device)` 在 device 不一致时抛
`RuntimeError: Expected all tensors to be on the same device`。

**可复现证据**(`tmp_review_r1_probe.py` A2,实跑):
```
A2. 把 model 参数搬到 'meta' 设备(模拟 model 在 cuda)、roi_mask 留在 CPU:
forward_roi(meta device) FAILED: RuntimeError: Tensor on device cpu is not on
the expected device meta!
```
小确认:以本项目 `_ROI_TRAIN_SIZE=224` 标定 `torch.stack([cpu_tensor], device="cuda")`
→ `RuntimeError: Expected all tensors to be on the same device, but found at least two
devices, cuda:0 and cpu!`(栈上 3×3 张量本体在 CPU → 必然)

**影响面**:ROI 头**全部 46 键**的编码器通路都吃 `roi_masks`(ETE band + T 头 d_mm),
即 ROI 支路在 CUDA 上 100% 起不来。`_ROI_TRAIN_SIZE=224` 是 E2 review 引入的,**说明训练侧
从未在 CUDA 上真跑过一次**;mock 冒烟在本机走 CUDA,但 `roi_head_weight=0` 使
`run_roi_epoch` 不执行(train_official:1545 `_roi_weight > 0.0`),**整条 ROI 支路在 CUDA 上零覆盖**
——本机 cuda 可用(冒烟横幅 `device=cuda`)却始终没碰到这条路径。

**为什么现有测试全绿**:`test_roi_train_wiring.py::test_run_roi_epoch_smoke_real_model` 用
`torch.device("cpu")` + `amp_enabled=False`;`_mock_batch` 的 metas 里**根本没有 `roi_mask` 键**
→ 走 `m is None` 全零兜底分支,`roi_masks` 由 `roi_images.device` 新建 → 恰好同设备。
`test_roi_dataset.py` 只断言 `roi_mask` 值域/几何,从不让它在 GPU 上前向。**测试 fixture 天然绕开了
唯一会崩的分支。**

**建议修法**(最小、单点):`run_roi_epoch` 内 imgs 迁移后同步迁移 mask—

```python
imgs = imgs.to(device)
for _m in _metas:                       # _metas 由 _meta_rows_from_collate 产出
    _rm = _m.get("roi_mask")
    if _rm is not None:
        _m["roi_mask"] = _rm.to(device)
```
或治本:模型面 `masks.append(m.detach().float().to(roi_images.device))`
(注意 `multi_task_model.py` 是 **models 领地**,本次红线不可改;跨领地改动须单独出单)。
无论选哪条,**须补一条 CUDA 冒烟闸门**:`--roi-heads` 真训前先跑
`run_roi_epoch(model.cuda(), one_batch_with_mask_on_cpu, ...)`,否则类 bug 只会在云端第一轮
GPU 上暴露(整轮白烧)。

---

## P0(次) — 动态 ROI 样本在"非均匀采样视频"上系统性丢样本 → 检测/分割被反降

**位置**:`src/dataset/roi_dataset.py:198-209`(交集帧过滤)、`scripts/train_official.py:1388-1392`
(真数据 ROI loader 由 `expand_roi_samples(train_ds.samples)` 展开);
触发链在 `src/dataset/official_loader.py` 的动态样本构建(`frame_index = fidx`,fidx 来自
`_nonzero_mask_frames`,`load_nifti_frames` 无 `cache_img` 时按 `max_frames=32` **均匀抽样**;
官方序列长度实测 3~750–800 帧)。

**机制**:整图动态样本的 `frame_index` **不是病灶全部标注帧**,而是
`non-zero mask frames ∩ uniform_sample_indices(n_total, 32)`。
`expand_roi_samples` 只保留 `s["frame_index"] ∈ les["frame_indices"]` 的病灶
(roi_dataset:200-201)→ **未被抽中的标注帧不产 ROI 样本**。

**可复现证据**(`tmp_review_r1_probe.py` C + 独立枚举):
```
n_total=100 -> indices[0:6]=[0, 3, 6, 10, 13, 16] ... last=99   # 帧 1/2/4/5 永不进入
n_total=200 -> indices[0:6]=[0, 6, 13, 19, 26, 32] ... last=199 # 帧 1..5 永不进入
n_total=750 -> indices[0:6]=[0, 24, 48, 72, 97, 121] ... last=749
```
枚举 `n_total∈[6,799]`:**n_total ≥ 172 时,采样集与 `{3,4,5}` 交集恒为空**(172–799 全落入
一条连续坏区间)。即"标注帧集中在序列前段、且视频不短"的全部动态病灶 **ROI 样本数 = 0**。

**影响面与判读**:
- ROI 支路在**病灶最清晰的前段**(临床录制的起始段)系统性缺失监督 → ROI 头学到的分布与
  推理时 `roi_pipeline`(同一 uniform 采样,却可能因"该帧不在标注集"而拿不到候选)两面都不对齐。
- ⚠️ **与 L2 的交互是实质风险,不只是"少几个样本"**:L2 union 监督下 `cache_msk` 是**全图全部病灶并集**
  (official_loader L2 注释 + `_static_union_mask`/`_dynamic_union_mask_frame`),
  所以"帧 f 只标了病灶 B"这个事实**在整图 GT 里被抹平**——模型会在帧 f 学到"帧 f 的并集区域",
  而 ROI 头在帧 f 只收到 A 的监督(若 A 在帧 f 无标注则交集为空、A 的 ROI 样本被丢)。这是一个
  整图/ROI **监督语义不一致**的口径裂缝,会在同一 lesion_id 上产生互相矛盾的目标。
- ⚠️ 严重性判读**有不确定性**(R1 诚实标注):339 个动态 mask、3.44 帧/病灶,若绝大多数序列短
  (≤40 帧)则采样为全帧、本项为**零影响**;坏区间只在 n_total≥172 出现。但**这不改变"必须加哨兵"**
  的结论:当前全仓**没有任何一处**统计 `n_dropped / n_kept`,静默丢监督 = 不可观测(rules §3 + r7 同型雷)。

**建议修法**:
1. 零成本先加**哨兵**(不阻塞开训):在 `expand_roi_samples` 返回处统计
   `dropped = Σ(len(les.frame_indices) - 被保留帧数)` 与 `n_roi_samples`,
   `_safe_print` 一行 + 落 JSONL(`roi_n_dropped_frames`),云端 1ep 即可判"这条路是否有量级"。
2. 若量级确实存在,正确修法是**改成"按 mask 非零帧产样本"而不是"按整图样本帧产样本"**:
   即在 `expand_roi_samples` 里对静态/动态统一用 `les["frame_indices"]` 展开动态病灶
   (现成的 `frame_indices` 已在数据面下发,roi_dataset:200 注释称"此分支是防御性兜底"——
   实际它承担了全部过滤职责,注释与事实不符)。此改动会改变样本数,**必须与整图 sidecar
   口径同批对拍**后再上线(属于训练侧 A/B,不在本路权限内)。

---

## P1 — 与设计注释相反:所谓"两支路梯度累积到同一 optimizer"并不存在

**位置**:`scripts/train_official.py:878-881`(设计注释)、`:941`(`run_roi_epoch` 逐 batch
`optimizer.zero_grad()`)、`:1527-1531`(整图 `run_official_epoch`)、
`src/training/loop/train_step.py:286/334/342`。

**事实**:`train_step.train_batch` 每 batch **内部完成** `optimizer.zero_grad()`
→ `backward` → `clip` → `scaler.step(optimizer)`。所以:

1. 整图 epoch 结束时,整图梯度**早已被消费**(且被 clip 到 1.0 后再 step);
2. ROI epoch 第一个 batch 的 `optimizer.zero_grad()` 把**任何残留梯度清零**(含 ROI 头若从整图
   支路拿到的梯度);
3. ROI 支路每 batch 自己 zero_grad → step,**不存在跨支路累积**。

**可复现证据**(`tmp_review_r1_probe2.py` D/E,最小 SGD 复现,实跑):
```
整图 epoch 结束:p_whole.grad = 3.0(train_batch 已 step,梯度已消费)
ROI 第一个 batch 的 optimizer.zero_grad() 后 p_whole.grad = None (train_batch step 后通常为 None)
  => 不存在'累积'——两条支路各自 step,互不叠加(实现是串行 SGD,不是累积)。
```

**为什么这是发现而非纯文档问题**:注释位于模块"设计(主线拍板,勿改)"段,是后人改这块代码的
**唯一契约来源**。它声称的不变量(`两次 backward 之间无 zero_grad`)与代码相反,会直接误导两类改动:
①"既然累积,把 ROI 的 clip 去掉/合并到整图的 clip";②"既然累积,ROI loss 要不要除 batch 数"。
另外它掩盖了**真实语义**:ROI 支路与整图支路是**串行两套 step**,梯度裁剪各算各的
(整图 `train_step:340` 用 `optimizer.param_groups` 全参数集,与 `run_roi_epoch:961-963` 同式)。

**建议修法**:把注释改写为事实(或按设计意图真做累积:整图侧不再 step、ROI 侧统一 step——
但那是会改变优化轨迹的行为改动,须单独立项 + 对拍,不在本轮)。**在本轮,只改注释**。

---

## P1 — ROI 支路训练与验证**尺寸/口径不一致**且无哨兵:训练 224 单 ROI,验证 224 结果与推理 512 对齐

**位置**:`:886-893`(`_ROI_TRAIN_SIZE=224`,注释自认"推理=512 已知域差异")、
`:1389-1392`(val 用 `_ROI_TRAIN_SIZE`)、`:986-1024`(`validate_roi_epoch` 直接在
ROI 域算四项指标)。

**事实**:
- 训练:224² ROI,**mask 由 `resize_mask_to` 从原分辨率 mask 缩到 224**(`roi_dataset:312`)。
  → `d_mm = physical_diameter_from_mask(roi_masks, spacing=0)`(全 None)在 224 → 与 512 推理
  **数值不同**;ETE band 的内外圈宽度按 224 特征分辨率算 → band 物理含义随尺寸漂移。
- 推理:`build_roi_batch` 吃 runner 的 `img_size=512`,`roi_pipeline` 装 `roi_mask` 也是 512 域。
- 而 `validate_roi_epoch` 用**与训练完全相同的 224 域**算 `roi_ete_kappa` / `roi_t_auc`,
  并且**没有给模型喂任何尺寸/域标记**。

**影响面**:`roi_*` 六键是"只观测不进选模"的观测面,**不污染 composite** → 不会自动选到错权重。
真正的问题是**观测面自身失真**:`roi_t_auc`(由 `d_mm` 驱动)与 `roi_ete_kappa`(由 band 驱动)在
224 域上的值**不能作为交付前 512 推理的一致性判据**,而当前 JSONL 没有任何字段提示这一点。
研发者若按 roi_* 曲线决定"ROI 头是否可用",会拿到系统性偏移的结论。

**建议修法**(按代价升序):
1. 零成本:`validate_roi_epoch` 的返回/JSONL 里加 `roi_size=224` 与
   `roi_domain_note="train-domain val (224); 推理 512 须单独对拍"`,把口径钉在产物里。
2. 中成本:val 用 `_ROI_TRAIN_SIZE` 训练、**另跑一次 512 域的 ROI val**(同一 val 人口),
   两列并列进 JSONL → 直接量出域差;这是"冻结前若要对齐须跑 224-vs-512 对拍"
   (注释自称的待办)的低成本执行物。
3. 不建议现在改训练尺寸(会牵动 `_ROI_BATCH_SIZE` 显存与全链数字)。

---

## P1 — `validate_roi_epoch` 的 `try/finally` 恢复态不完整(与同文件 L1 修复同型)

**位置**:`scripts/train_official.py:1001-1015`。

```python
was_training = model.training
model.eval()
try: ... 
finally:
    if was_training: model.train()
```

**问题**:只还原 `model.training` 这一个**顶层布尔**。若调用前模型处于
"整图 `train()` + `set_frozen_modules_eval(model)` 已把冻结子模块置 eval"的混合态,
`effort` 后统一 `model.train()` 会把**冻结的 backbone BN 全部翻回 train 模式**。
`run_roi_epoch`(:930-931)与 `run_official_epoch`(:402-405)各自都会重调
`set_frozen_modules_eval`,故**下一 epoch 头部会自愈** → 不会静默损害训练。

**但仍有两处实际暴露面**:
1. `_collect_cliff_ious`(L1,调用点 `train_official.py:1572-1574`,**位于 ROI val 之后**)。
   ⚠️ 位置修正(主会话复核后自纠):`:1544-1565` 的 ROI 支路(含 `validate_roi_epoch`)
   **整体包在 `if roi_train_loader is not None and _roi_weight > 0.0:` 之内**,所以
   "快照态被 ROI val 打散"只发生在 **ROI 真开(权重>0)** 的轮次——L1-only 轮次里
   `set_frozen_modules_eval` 仍新鲜,该声称成立。ROI+L1 同开时:ROI val 的 finally
   把冻结 BN 翻 train → `_collect_cliff_ious` 的 `model.eval()` 之后、其
   `finally: model.train()`(同 :1015 同型)之前,冻结 BN **处于 train 模式**,
   running stats 被采集数据污染,而它自己的 docstring(:448-452)声称"eval 期间 BN
   running stats 冻结"。`_cliff_enabled` 与 `roi_heads` 无互斥门,同开是正常用法。
2. 同型的 `validate_epoch` 在 `src/training/trainer.py`,本路未审;若同样是裸 `train()`,
   则本问题在整图链同样存在(**建议 R2/R3 或主会话顺手核一遍**)。

**可复现证据(静态 + 可构造)**:`set_frozen_modules_eval`(`bn_freeze.py:6-9`)是"按当前
`requires_grad` 判冻结模块"的**一次快照**;`model.train()` 会递归把所有子模块置 train(含冻结者)。
`train_official.py:230-247` 的 ROI val 键集测试只断言 `model.training`(单布尔),**测不出 BN 态**。

**建议修法**:与 `_collect_cliff_ious` 同款不足以,应记全量:

```python
was_training = model.training
prev_modes = {id(m): m.training for m in model.modules()}
model.eval()
try: ...
finally:
    for m in model.modules():
        m.training = prev_modes.get(id(m), m.training)   # 逐模块还原
```
(或调用方在 roi val 之后显式补一次 `set_frozen_modules_eval(model)`,零成本且最小。)

---

## P2 — ROI 支路的梯度和**若 ROI 头参数落在 optimizer 里,会被整图 `clip_grad_norm_` 一起裁剪**

**位置**:`train_step.py:340-341` vs `train_official.py:961-963`。

两支路都用**同一个 `optimizer` 的全参数集**做 `clip_grad_norm_(max_norm=1.0)`。
当 `roi_heads=True` 时 `configure_official_stage` 把 ROI 头并入 `always_train`
(schedule_official:310-311)与满 `lr` 组(:329),所以 ROI 头**在** `optimizer.param_groups` 里
——已由 `test_roi_heads_true_trainable_and_in_optimizer` 钉死(实跑 24 passed)。

后果:整图支路每 batch 的 `clip_grad_norm_` 会把当时 `p.grad is None` 的 ROI 头跳过(安全),
但 ROI 头随后在 ROI 支路里自己的 `clip` 也只算 `optimizer.param_groups`(= 全参数),
**把整图参数场(mask None)一并纳入范数分母**——语义上没问题(None 不计入 norm),
但两个支路互不知道对方的存在,**梯度范数口径不独立**。属"可接受但需记录"的设计事实,
当前 `run_roi_epoch:961-963` 的注释未提这一点。

**建议**:把这句写进 `run_roi_epoch` docstring(一句话),并在 JSONL 记一次
`roi_grad_norm`(与整图 `_step_diagnostics` 并列)以便回读时能判裁剪是否吃掉了 ROI 头梯度。
非阻塞。

---

## P2 — 死键:非标量 detail 被吞 + 分支名与注释不符

**位置**:`:968-971`(`logger.log(..., extra={"roi_branch": True, **_roi_detail_extra(detail)})`)
+ `concepts_roi.py:62-65` 的声称。

**事实**:`_roi_detail_extra`(:975-983)对每个 detail 值 `float(v.item() if hasattr(v,"item") else v)`,
异常 `continue`。概念计数键的**值是 python `float`**(`concepts_roi.py:59` "计数键恒在恒数值
(不是 0 张量,是 float)")→ `float(0.0)` 正常,**不会被吞**。但:
- 若某天有人把计数改成张量,`float(tensor)` 会走 `.item()` 分支仍然通过——掩盖;
- 注释"非标量跳过"描述的是一个**当前不存在的**跳过路径(所有 detail 都是 0 维张量或 float),
  属"注释描述了不存在的分支",与 `concepts_roi.py:62-65` 的声称(自动进 JSONL)一致但**无测试**
  覆盖计数键进 JSONL。

**建议**:加一条断言式测试(detail 含一个 `torch.zeros(3)` → `_roi_detail_extra` 跳过且不抛),
把"非标量跳过"从注释升级为契约;顺手在 `concepts_roi.py` 的声称处指向该测试。

---

## P2 — mock ROI 的 `global_loc` 硬编码为常数,loc 头在冒烟里得不到信号

**位置**:`:1050`(`"global_loc": (30 / s, 30 / s)` 对所有 mock 样本)。

`RoiLocationHead` 的全局位置通道因此恒定 → mock 冒烟下的 loc 头只有 bbox(=20:40 恒定)输入,
**坐标分支的真实接线(bbox/orig_w/orig_h → CoordConv)在冒烟里无法被验证**。
mock 是"验真模型前向"的唯一入口(`:1319-1321` 注释),这个常量化削弱了它的区分力。

**建议**:mock 样本按 `i` 生成渐变 bbox(如 `(10+i, 10+i, 30+i, 30+i)` 夹到 64²),
零成本、保持"无 mask"语义不变,给 loc 头至少一个变动维度。

---

## P2 — `normalize` 前的 `roi_mask` 与图像**缩放口径不同**,但代码未显式声明

**位置**:`src/dataset/roi_dataset.py:272`(`resize_img_to`) vs `:312`(`resize_mask_to`)。

图像双线性、mask 最近邻,这是**正确**的(mask 必须保持 {0,1},:296-299 注释已说明)。
本路**验证过** `test_roi_mask_crop_geometry_matches_image` 逐像素对拍(:256)确认几何一致,
故判 PASS。留下此条仅因它属 D 类"未在 ROI 训练编排处声明"的跨模块契约(排查 224/512
域差时容易误判 mask 有额外插值误差)。**无需修**。

---

## 判 PASS 的部分(试过什么反例才排除)

| 攻击面 | 试过的反例 | 结论 |
|---|---|---|
| 患者分组泄漏(`expand_roi_samples` 继承同一切分) | 逐行读 `expand_roi_samples`(:168-218):**只从入参 `samples` 逐条展开**,无任何重新划分/顺序依赖;`patient_grouped_split`(pathology_dataset)在 :1239 一次性切分、带 `train_pids & val_pids` 断言;roi_train 取 `train_ds.samples`、roi_val 取 `val_ds.samples`,`launch` 处再无第三个来源 | **PASS**(切分确实继承,无泄漏) |
| `_roi_labels_from_meta` 键名映射 | 27 项桥测试 + 损失/尺子双端消费对拍(`test_bridge_consumed_by_loss_and_records`);实跑 24 passed | PASS |
| 梯度 FATAL 检查的 `n` 遮蔽 | `f"..."` 不含 `n`,**无悬空引用**;`p.grad is not None` 兜住"未 backward 过"的 ROI 头;`startswith(_ROI_PFX)` 的 `_ROI_PFX` 是 str 元组,`startswith` 需要元组→合法 | PASS(仅"只打不停"是**有意的**设计,非 bug) |
| 整图 `zero_grad` 漏/多 | 读 `train_step.py:286` 确认在 batch 内、且 `:347 scaler.update()` 后无残留;ROI 每 batch `zero_grad` 在 `:941`,`backward` 之前 | PASS(但注释不实,见 P1) |
| BN/eval 状态:下一 epoch 是否自愈 | `run_official_epoch:402-405` 与 `run_roi_epoch:930-931` 都重调 `set_frozen_modules_eval`;`run_roi_epoch` 在 train 前 set→**自愈成立** | PASS(跨 epoch);val 内部窗口见 P1 |
| AMP dtype / scaler 行为 | 实跑:autocast(bf16) + `GradScaler(enabled=True)` → `scale/unscale` 成对、`w.grad` 被正确还原、参数正确更新;**不是 no-op 但也无副作用**;`GradScaler("cpu", enabled=False)` 在 CPU 测里正确短路;bf16 不溢出故 inf-check 恒不触发 | PASS |
| `--roi-heads` 硬门 | 六态测试(互斥/版本化 out/缺 resume/权重 0 真训)+ 实跑确认 24 passed | PASS |
| `roi_val_loader` 顺序 | 实读 `:1409-1414`:**显式 `shuffle=False` + `collate_fn=roi_collate`**,非依赖 train_ds 决定 | PASS |
| `pin_memory` 与 `expandable_segments` | 静态标注整数键经 pin_memory 会 cp 回 CPU(PyTorch 语义,无法在本机复跑云 env var);`forward_roi` 内**不把 int64 转 device**(`:575 int(m.get("is_main"))`,`.get()` 出 python int)→ 实际消费路径不受影响 | PASS(对 ROI 支路无影响);整图 `labels` 索引走 `.to(device)` 是既有路径,不属本路 |

---

## 总判

**ROI 训练编排的"接线骨架"是对的**(键契约、患者切分、调度/白名单、AMP 配置、zero_grad 时序均排除),
但**不允许上云**:存在 **1 个必崩级 P0(`roi_mask` 不上设备,ROI 支路在 CUDA 上 100% 起不来)**
和 **1 个有量级的 P0-次(动态 ROI 样本在非均匀采样视频上系统性丢监督,且与 L2 并集 GT 构成监督口径裂缝)**,
另有 3 个 P1(注释不实/域差无哨兵/eval 态恢复不完整)。建议顺序:
① 修 `roi_mask` 设备迁移 + **补一条 CUDA 上的 ROI 冒烟闸门**(本路最大的系统性漏洞是
"全仓无任何一处让 ROI 支路在 GPU 上跑过");② 加 `n_dropped` 哨兵把 P0-次变成可观测数字;
③ 修注释、加 `roi_size` 口径字段、补 `set_frozen_modules_eval` 恢复;④ 再走 E2 前置清单的其余债。
