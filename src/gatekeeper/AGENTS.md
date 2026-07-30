# src/gatekeeper/AGENTS.md - Ultrasound Image Gatekeeping Specifications

This protocol defines the 3-layer gatekeeper pipeline for detecting fake, duplicated, or tampered ultrasound images.

---

## 0. SYNTHESIS ENGINE (`synthesis.py`)

在线合成引擎,为 gatekeeper 基础项(真实性/规范性)提供可训练/可测试数据。全部函数支持 `rng_or_seed` 参数确保确定性可复现。

### 导出函数

| 函数 | 功能 |
|:---|:---|
| `make_phantom_like(img, rng_or_seed)` | 模拟体模:平滑匀化 + 高 GLCM homogeneity |
| `make_synthetic_like(img, rng_or_seed)` | 模拟 GAN/扩散合成:频域周期伪影 + 激进 JPEG 重压 |
| `make_near_duplicate(img, rng_or_seed)` | 近重复:重采样/裁剪/水印/亮度偏移 |
| `make_copy_move(img, rng_or_seed, patch_frac)` | → `(tampered, mask, bbox)` |
| `make_splice(img, donor, rng_or_seed, patch_frac)` | 硬粘贴(留尖锐缝,供几何检测器)→ `(spliced, mask, bbox)` |
| `make_splice_poisson(img, donor, rng_or_seed, patch_frac)` | 泊松/羽化拼接(`cv2.seamlessClone`,无硬缝,供 CAT-Net DCT/BAG 取证;seamlessClone 不可用退化高斯羽化;小图区间守卫)→ `(spliced, mask, bbox)` |
| `inject_nakagami_speckle(img, rng_or_seed, m_range)` | 一阶 Nakagami 乘性散斑(env=√Gamma(m,1/m) 归一均值1),m_range 默认 (0.6,1.4) 覆盖欠/充分发育(m=1 即 Rayleigh) |
| `domain_randomize(img, rng_or_seed)` | 域随机化:Nakagami 散斑 → 高斯模糊 k∈{3,5,7} → JPEG 重压 Q∈[15,75],缩小合成样本与官方真实 JPG 域差 |

### 类别约定

| Label | 含义 |
|:---|:---|
| 0 | 真实图像 |
| 1 | 体模(phantom) |
| 2 | AI 合成(GAN/扩散) |

---

## 1. AUTHENTICITY CLASSIFIER (`authenticity.py`)

- **Feature extraction**: 13 维真实特征:GLCM(4)+ 频域(2)+ ELA(2)+ DND 双纳卡甘米散斑 m_global/m1/m2(3)+ TAI 结构张量各向异性 global/local(2)。前 8 维区分 GAN/合成,后 5 维(DND+TAI)区分物理体模(PVA/琼脂)的声学物理本质。改维度须同步 `authenticity.py`、`authenticity_trainer.FEAT_DIM`、test 断言。
- **训练数据**: 从真实训练图提取 label=0(真实)、`make_phantom_like` 生成 label=1(体模)、`make_synthetic_like` 生成 label=2(合成),特征标准化后 40 epoch 训练。
- **Model**: Shallow MLP 三分类器(`AuthenticityClassifier`)。
- **Weights**: Trained during `train.py` startup, saved to `checkpoints/authenticity_model.pth`. 推理端因 ckpt 含 numpy `feat_mean/feat_std`,torch≥2.6 必须 `torch.load(..., weights_only=False)`;该路径只允许加载本项目自产可信权重。正式训练出固定权重后须同步 `AuthenticityModule.EXPECTED_CKPT_SHA256`,否则 `weights_only=False` 缺少完整性约束会重新暴露 pickle 任意代码执行风险。
- **验证**: 训练结束在验证集报告三类准确率。
- **Red-line**: authenticity MLP 权重变更须重新在物理体模数据集上全量训练；不可仅调参或微调已有权重。
- **权重缺失 = 硬报错（2026-07-10 Gate2审查修复）**：`load_weights(path)` 此前对不存在的路径直接返回（无 else 分支），模型停留在随机初始化态、零日志零标志位，`predict()` 仍返回合法 0/1/2——真实性检测器悄悄变成随机分类器，一票否决门失效或随机误杀且无人察觉。现改为 `raise RuntimeError`，交给 `predict.py`/`video_predict.py` 的外层 `except Exception` 转 `ERROR` result.json（为此同步把 `predict.py` 的 gatekeeper 审计调用纳入了 try 包裹范围，此前只包了 build_model 往后）。**production 容器必须打包好 `authenticity_model.pth`**，缺失即视为部署事故，不再静默降级。此红线**未被下方 2026-07-12 软化改动触及**——软化只针对"分类结果是否驱动 REJECT"，权重缺失/损坏仍硬报错。
- **分类结果降级为软 flag（2026-07-12,统一软flag制度）**：`src/inference/gatekeeper_runner.py::run_gatekeeper_audit` 不再对 `is_phantom>0`（phantom/synthetic 判定）触发 `REJECTED`，仅打印 `[Gatekeeper 软 flag]` 审计日志、`is_phantom`/`is_gan` 字段仍保留在（若因去重/篡改触发的）`REJECTED` result.json 里供审计；**2026-07-12 code-review(Angle B)补强**：若本次审计因去重/篡改被拒绝且 phantom/synthetic 同时命中，人类可读的 `reason` 文本会额外附加 `"(soft-flag: Physical Phantom suspected)"`/`"(soft-flag: GAN/Synthetic suspected)"` 标注（早期版本这条并发信号只在结构化字段里，`reason` 纯文本完全看不到，人工审计/日志 grep 场景会漏读）——`reason` 的第一个分句仍是真正驱动 REJECTED 的原因（duplicate/tampered），soft-flag 标注只是追加说明，不影响拒绝语义。**根因**：`authenticity_trainer` 的 phantom/synthetic 训练标签 100% 来自 `synthesis.py` 程序化合成代理（`make_phantom_like`/`make_synthetic_like`），从未见过真实体模/真实 AI 合成图；`extract_nakagami_features` docstring 自陈判别方向"是合成训练域的特性,非真实物理结论,真训时须用官方真假样本重标定方向"；argmax 无任何置信度门槛；全仓零真实图像 FPR 实测（对比 tampering 有 359 张真实图 0%/27.3% 的明确数字）。**恢复硬拒绝的唯一闸门**：官方真实干净 holdout 上预注册、带置信区间的 FPR 验收达标（不是"加个置信度阈值"就能恢复——置信 margin 只能当软侧附加护栏，不能当恢复硬拒绝的凭据）。详见 `docs/P2红线修复实施计划.md` B1、`docs/P2启动定论_Opus复核.md`。

> **历史说明**: 旧版用 `torch.randn` 随机噪声 + 随机标签做假训练(完全无用占位),已于本次重写为真实特征训练流程。

---

## 2. DEDUPLICATION PIPELINE (`deduplication.py`)

Two-stage duplicate detection:

| Stage | Method | Threshold |
|:---|:---|:---|
| **Primary** | 64-bit perceptual hash (pHash) + Hamming distance | Distance ≤ 12 = duplicate |
| **Secondary** | 可插拔深度描述符 + Cosine Similarity | 按后端取阈值(生产 config:SSCD 0.86 / ResNet18 0.995)|

- **后端可插拔(SSCD 优先,降级 ResNet18)**:`FeatureExtractor` 探测 SSCD 离线 TorchScript 权重(`checkpoints/sscd_disc_mixup.torchscript.pt` 或 `~/.cache/torch/hub/checkpoints/`,来源 facebookresearch/sscd-copy-detection,Apache-2.0/MIT,95MB,512 维 GeM,对 ≤80% 局部裁切/水印/色调偏移近似不变)。权重缺失则**降级** torchvision ResNet18,行为与历史一致——保证离线无权重环境下测试/推理零破坏。两后端均输出 **512 维 L2 归一**向量;`feat_dim` 由 init 跑 dummy 前向**实测探测**(reviewer 必修2,防误传 1024 维 disc_large 静默错算),FAISS `IndexFlatIP(feat_dim)` 与 NumPy 双路兼容。**ResNet18 降级后端自身权重缺失时**(`~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` 不存在)**改为硬 `raise RuntimeError`**(2026-07-11 修复),不再静默走 `pretrained=True` 联网下载——此前这条二次降级路径会在离线交付容器场景(答疑 Q26「不能联网」)下让 torchvision 尝试联网、进程挂死;现在由 `predict.py`/`video_predict.py` 外层 `except Exception` 统一兜成 `status=ERROR` 的 `result.json`,不再有静默挂起的可能。
- **按后端阈值映射(守红线)**:`Deduplicator` 的 `cosine_threshold` 接受三态——`None`→取后端推荐(SSCD 0.86 / ResNet18 0.95);`dict`→按 `backend_name` 取值(生产 config `{sscd:0.86, resnet18:0.995}`,缺键回退推荐);标量→两后端同值(向后兼容)。**单一标量无法同时服务两后端**:SSCD 治近重复漏检需 0.86,ResNet18 守一票否决红线「宁可漏检不可误杀真实影像」需 0.995。`gatekeeper_runner` 透传时缺省传 `None`(非 0.95,否则兜底值顶替 None 锁死映射逻辑)。
- **权重交付**:SSCD TorchScript 需赛前手动下载放入 `checkpoints/`;比赛交付为商用离线容器,SSCD 协议合规。

---

## 3. TAMPERING DETECTION (`tampering.py` + `tampering_catnet.py`)

| Attack Type | Detection Method | 当前能力边界 |
|:---|:---|:---|
| **Copy-Move** | Dense block-matching to find duplicate pixel regions | **降级模式下软 flag,不硬拒绝**:超声天然自相似(斑点/腺体纹理)使真实自相似块与复制块无法用统计/NCC/位移特征分开,触碰「宁可漏检不可误杀」一票否决红线。实测数据(359张真实图 holdout、27.3%误报率、5组对拍实验)见下方"硬拒绝路 vs 软 flag 路"完整说明 |
| **Splicing(硬粘贴)** | 轴对齐长直缝几何判据:Sobel 强阶跃边 → HoughLinesP 找长直线(≥0.14·min(H,W))→ 按 ±10° 分水平/垂直缝 → **要求一条水平缝与一条垂直缝端点在 14px 内角点正交相交**(硬粘贴矩形特征) | 经两轮失败定案:像素/区域方差方案在含结节的干净图 20/20 误报(结节本身即方差异常区),触碰一票否决红线故弃用。现方案干净图 **0/80 零误报**、真实组织拼接召回 **~84%**,放弃纯噪声 donor 等极端对抗样本部分召回以换零误报 |
| **Splicing(羽化/泊松融合)** | `tampering_catnet.py` — CAT-Net v2 风格 DCT/BAG 取证:`block_artifact_energy` 估计 8×8 块伪影栅格(BAG)主导相位,滑窗找局部相位错位区(来自另一 JPEG 的错位粘贴会破坏全图一致栅格,即使 RGB 边界被融合) | 几何缝检测器抓不到「无长直阶跃」的羽化拼接;BAG 取证补此盲区。**红线守护**:① 全图 `grid_strength<阈值`(无 JPEG 栅格)→ 弃权返回未篡改,不对未压缩真实图取证;② `grid_strength` 分母加绝对能量地板,防真实平滑背景+结节图因 median 塌缩被放大成误报;③ 局部相位换算回全图坐标系 `(p+y)%block`,消除窗口原点偏移(否则非 block 整倍 stride 会让干净 JPEG 全图误报)。`min_flag_blocks` 聚类门压随机误报 |

**可插拔 + 降级**(沿用 dedup 范式):`CatNetTamperingDetector` 探测 `checkpoints/catnet_v2.torchscript.pt` 深度 HRNet 权重——存在则 `backend_name="catnet"` 加载 TorchScript;缺失/torch 不可用则安静降级 `backend_name="dct-bag"`(纯 NumPy/cv2 BAG 取证)。

**⚠️ 硬拒绝路 vs 软 flag 路(2026-07-12 统一软flag制度,红线守护)**:`detect()` 不再对全部检测器取无差别并集,而是按可信度分两路:
> **⚠️ 2026-07-29 顶层封顶**:本节所述"硬拒绝路"现被 `gatekeeper_runner.py` 的 `gatekeeper.enforcement_mode`(默认、交付固定 `soft`)统一封顶——`is_tampered=True` **不再等价于 REJECT**,soft 模式下只置审计字段 `would_reject_in_hard_mode=True`、诊断照常输出;仅显式配置 `hard` 时才真正 REJECTED。理由:官方赛前动员会(2026-07-28)明确赛道七豁免全部医保安全考核,gatekeeper 零计分,硬拒绝成纯负期望(拒对零收益、误拒白丢该病例全部临床得分)。**本节以下关于各检测器可信度分级的判定与 FPR 验收闸门全部照旧有效**,只是最终执行层多了一道默认软化。详见 `src/inference/AGENTS.md` §2。
- **硬拒绝路(`is_tampered=True`,hard 模式下 → REJECT)**: 深度 CAT-Net **自身**前向命中(若权重在且本次调用成功) ∪ BAG 相位不一致(仅当 `bag_hard_reject=True` 显式开启时;**默认 `False`**,见下)。CAT-Net 有自身学习式验证;BAG 有「无 JPEG 栅格则弃权」门作红线保护,但阈值仅经 8 个合成种子小样本验证(远小于 copy_move/splice 的 359 张真实图量级),故**默认不作硬拒绝**,待官方真实干净 JPG holdout 预注册 FPR 验收达标后才可经 `configs/thyroid_config.yaml` 的 `gatekeeper.bag_hard_reject: true` 开启。
- **软 flag 路(记录 `last_soft_flags`,不拒绝)**: copy_move / splice 几何块级判据 **永远**软 flag,`bag_hard_reject=False` 时 BAG 命中同样软 flag。根因(copy_move/splice):经典块匹配在超声模态无法兼顾零误报与有效召回——全量 holdout(359张真实图)实测,若硬拒绝则 copy_move 误报 **27.3%**;5 组对拍实验(statistical thr 扫描非单调 / NCC 像素级 / NCC+位移投票)均证明误报与召回同涨同跌,压到零误报时召回仅 5–8%(形同虚设)。修复后硬拒绝误报 **0/359 = 0%**,软 flag 记录 31.8%(仅审计,不影响放行)。
  - **⚠️ 双开关互锁(2026-07-29 明确)**:软/硬由**两把独立的锁**共同决定,只切一把不生效——`enforcement_mode` 决定"硬判据命中后是否 REJECTED",`bag_hard_reject` 决定"BAG 命中算不算硬判据"。组合矩阵:`enforcement_mode=soft`(交付)⇒ **任何命中都不拒绝**;`enforcement_mode=hard` + `bag_hard_reject=false` ⇒ 只有 CAT-Net 自身命中拒绝,BAG 仍只进 soft_flags;`hard` + `true` ⇒ CAT-Net 与 BAG 都拒绝。**copy_move/splice 无对应开关,无论两把锁怎么切都只进 soft_flags**(误报 27.3%,恒不驱动 would_reject_in_hard_mode)。切换时务必同时确认另一处——`hard` 单独切换不会让 BAG 参与拒绝。
  - **2026-07-29 软flag落 audit**:`last_soft_flags` 现被 `gatekeeper_runner.collect_soft_flags()` 读取、归一化后写进 audit 的 `soft_flags` 键(计入 `flagged` 但**恒不进** `would_reject_in_hard_mode`)——此前生产零读取点,仅被这三路命中的图 `flagged=False`、result.json 完全沉默,信号只出现在 stdout。
- **与 CatNet 状态解耦(2026-07-12 修正,取代旧"升级路径")**:此前版本("2026-07-08 修正"/"2026-07-10 Gate2审查修复")用 `catnet_ok_this_call`(本次 CatNet 前向是否真成功)驱动 `block_level_hard`,CatNet 权重落地且本次调用成功后会把 copy_move/splice **自动**升级为硬拒绝。**这个耦合本身被判定为设计缺陷已删除**:"CatNet 这次没崩溃"(甚至只是判定为 clean)和"copy_move/splice 判据本身是否可信"是两件独立的事,不能用前者驱动后者的信任等级——即使 CatNet 权重落地且每次调用都成功,copy_move/splice 依然停留在 27.3% 真实图误报率,不会因为 CatNet 能跑就变得可信。现 `detect()` 内 `block_level_hard`/`catnet_ok_this_call` 变量已删除,copy_move/splice **无条件**软 flag。恢复其硬拒绝的唯一闸门是它们**各自**在真实域 holdout 上的 FPR 验收达标,而非"CatNet 能否跑通"。回归测试:`tests/test_gatekeeper.py::TestCatNetDetector::test_geometric_never_hard_rejects_even_when_catnet_ok`。
- **BAG 硬拒绝开关**:`CatNetTamperingDetector(bag_hard_reject=False)` 默认值,经 `src/inference/gatekeeper_runner.py::build_tamper_detector` 从 `config["gatekeeper"].get("bag_hard_reject", False)` 透传。回归测试:`tests/test_gatekeeper.py::TestBagHardRejectToggle`(默认软 flag + 开关恢复硬拒绝两个方向均覆盖,monkeypatch 强制 `bag_phase_inconsistency` 命中,不依赖合成样本自然触发)。
- **硬拒绝来源标签(`last_hard_reject_source`,2026-07-12 code-review 补强)**:`detect()` 硬拒绝时把来源("catnet"|"bag_phase")写入 `self.last_hard_reject_source`(未拒绝时为 `None`,每次 `detect()` 调用开头重置)。根因:统一软flag制度后硬拒绝路只剩 CatNet 自身命中与 BAG(开关开启时)两种,但 `gatekeeper_runner.py` 的 REJECTED `reason` 文案此前坍缩成笼统的 "Tampering detected (hard-reject path)",审计时无法区分具体命中源;现 `gatekeeper_runner.py` 用 `getattr(tamper_detector, "last_hard_reject_source", None)` 拼接更精确的 reason("Tampering detected (CAT-Net deep model)" / "(BAG phase inconsistency)"),旧版/stub 对象无此属性时向后兼容退化为 "Tampering detected (source unspecified)"(2026-07-29 随 enforcement_mode 软化去掉了文案里的 "hard-reject" 字样——soft 模式下并不发生拒绝,旧措辞会误导审计)。回归测试:`tests/test_gatekeeper.py::TestCatNetDetector::test_catnet_own_hit_sets_hard_reject_source`/`test_no_hard_reject_leaves_source_none`、`tests/test_video.py::TestGatekeeperSoftFlagPolicy::test_tamper_reason_labels_catnet_source`/`test_tamper_reason_labels_bag_phase_source`/`test_tamper_reason_falls_back_to_generic_text_without_source_attr`。
- **官方数据后待验证**: 2026-08-01 后可试 ROI 限定检测(只在结节区块匹配,背景自相似区不检)——子agent 评审认为这是 V2 最值得验证的方向。同时是 copy_move/splice/BAG 恢复各自硬拒绝资格的预注册 FPR 验收窗口。

> **Gate2 跟进项(待真实数据验证)**:并集对误报率单调不减(P(任一命中)≥max各路),红线安全依赖每路各自保守。BAG 单路对干净单次 JPEG 由「全图弃权门 + 局部窗弃权 + `min_flag_blocks` 聚类」三重把守(全图同一栅格 → 局部相位恒等全图相位 → 零命中);但**重采样/缩放后的 JPEG** 会破坏 8×8 栅格对齐使局部相位散乱,属并集 FP 理论敞口(通常先被全图低 `grid_strength` 弃权门拦下)。**2026-08-01 官方真实 JPG 数据到位后须实测**:BAG 单路真图误报率 + copy-move/splice 两路几何检测器红线表现。

---

## 4. GATEKEEPER SCOPE

- **Production inference only**: Gatekeeper runs in `predict.py` (单图) and `video_predict.py` (动态图) for deployment safety.
- **Offline evaluation bypass**: evaluate.py 路径跳过 gatekeeper（holdout 为已知干净真值，详见 src/inference/AGENTS.md Usage Contract）。
