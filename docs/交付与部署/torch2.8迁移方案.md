# torch 2.8 / CUDA 12.8 迁移方案(官方 5090 环境)

> **状态(2026-08-11 更新,方向已变)**:
> - ❌ **§4 Dockerfile 自建镜像 / sshd 那套不执行**——用户已定**直接用官方基础镜像**(平台内网 registry 4 选 1,见下),权重走平台上传,不做本地→云的镜像迁移。该章仅存档。
> - ✅ **§2 requirements + §3 代码 API 兼容仍然要做**——不管镜像谁提供,我们的代码都要在 torch2.8+numpy2 下不崩。预演 venv 在**本地 Windows** `C:\Users\cjy\envs\torch28`,**其唯一职责 = 模拟云端目标环境**(官方镜像 torch2.8.0+cuda12.8),提前抓出上云才会炸的崩溃/静默行为变更 —— 模型只提交一次、禁联网,上云才发现即整场零分。
> - ⚠️ **A800 与本地 torch28 职责严格分工,勿混淆**:**A800 只负责训练 + Linux/真 CUDA 契约验证,不模拟云环境**(它是 Ubuntu16.04/glibc2.23,物理装不了 torch2.7+,曾建到一半已止损清理,**别再试**);**模拟云环境只在本地 torch28 venv 做**。二者互补:本地有目标版本但没 CUDA、A800 有真 CUDA 但版本上不去,故 fp16/CUDA 相关契约走 A800(`probe_heads_cuda.py`),版本兼容性走本地。静态扫描结论见 §3.1,实测验证见 §2。 `/home/cjy/envs/torch28` 做兼容性预演(不动 `thyroid_env`),静态扫描结论见 §3.1,实测验证见 §2。
> - ⚠️ **不可拆开单独改 requirements**:版本锁是围绕 base 配平的联动约束,只改一半必崩 CI。

## 0. 官方环境实况(2026-08-12 容器内 `nvidia-smi` 实测定稿,推翻 2026-08-11 误判)

**资源**:赛事资源类型-GPU5090*1 → **14 vCPU / 64G 内存 / RTX 5090(显存 32G)**。
🔴 **显存 = 32G**(云 agent 官方容器内 `nvidia-smi` 实测 2026-08-12;用户实测确认——2026-08-11 曾凭口述误判为"约 16G/当前分配 14G",控制台"14"实为 **vCPU 数**而非显存档位,该误判已作废)。答疑 Q33 标称"32G 显存 / 16核 / 96G 内存 / 1T 硬盘"此时与实测显存恰好一致。规划一律按 **32G**。详见 `docs/项目现状总纲.md` §3。

**可选基础镜像(框架=Pytorch)**:
| # | tag | 取舍 |
|---|---|---|
| **1** | `10.1.63.4/public/caih/pytorch:2.8.0-cuda12.8-cudnn9-runtime-20260715` | ✅ **我方选此** |
| 2 | `...pytorch:2.11.0-cuda12.8-cudnn9-runtime-20260715` | ✗ 版本过新,破坏性变更多、踩坑资料少 |
| 3 | `...pytorch:2.9.1-cuda12.8-cudnn9-runtime-20260715` | ✗ 同上 |
| 4 | `...pytorch:2.7.0-cuda12.8-cudnn9-runtime-20260715` | ✗ 比 2.8 旧却无实质差别 |

全部 cuda12.8-cudnn9-**runtime**(无 nvcc,不能现场编译 → 依赖必须有 manylinux wheel)。5090 = Blackwell sm_120,强制 CUDA≥12.8。

## 0.1 决定性事实(2026-08-11 A800 实测,推翻旧假设)

**A800 现役 thyroid_env = `torch 2.6.0+cu124 + numpy 2.2.6`,所有训练/混训/评估跑通。**
→ **"numpy<2 是 torch C-ABI 硬约束"只对 requirements 声称的旧交付 base(torch2.1)成立**;真实环境早已 numpy2+torch2.6 共存无恙。迁到 torch2.8+numpy2 的"最大崩溃点"真实环境已验证过,风险低于初判。

## 1. 目标环境

- **base 镜像**:官方 #1 `pytorch:2.8.0-cuda12.8-cudnn9-runtime-20260715`(见 §0)。
- **卡**:5090(Blackwell sm_120),CUDA 12.8。
- **官方约束**:**显存 32G/容器**(2026-08-12 实测定稿)/ 14 vCPU / 64G 内存。(私有镜像需容器内自启 sshd —— **我方不走私有镜像,此条不适用**。)

## 2. requirements.txt 改动清单(逐行)

> ✅ **已本地实测验证(2026-08-11 首验 / 2026-08-12 复跑 / 2026-08-13 三跑)**:独立 venv `C:\Users\cjy\envs\torch28`(**py3.12.13** + **torch 2.8.0+cpu** + **numpy 2.4.4** + timm 1.0.28 + cv2 4.14 + **pandas 2.3.3 + openpyxl 3.1.5**(2026-08-13 补装))跑全套 pytest = **788 passed / 6 skipped / 0 failed**(2026-08-13;历史:首验 597 → 08-12 **620** → 08-13 补装前 775/7 → 补装后 788/6)。torch2.8 + numpy2 + timm1.x 三者**零崩溃**,迁移风险已证实很低。
> 🔴 **补装 pandas/openpyxl 的原因(2026-08-13,别再让它退回去)**:此前该 venv 按 `requirements.txt` 装,而那是**推理交付容器**的清单(不含 pandas 是正确设计,推理不读 xlsx),于是 `tests/test_official_loader.py` **整模块**被 `pytest.importorskip("pandas")` 跳过 —— **"云环境预演"对 `official_loader`(三头标签的唯一来源)覆盖率为 0**,一直在空转最关键的模块。补装后 collected 从 782 升到 **794,与 `Demo_D` 主环境完全一致**。
> ⚠️ **仍不可跨环境比 passed 数**:两环境 collected 现已同为 794,但 skip 数不同(torch28 = 6 / Demo_D = 1,依赖集差异);判回归须**同环境**前后对拍,勿拿两环境数字相减。`Demo_D` 主环境当前口径 = **794 collected / 793 passed + 1 skipped**(本行旧写的 737 已过时)。
> ⚠️ 曾误记为"py3.10.18 + venv 建在 A800",两处均已实测纠正(A800 glibc 2.23 装不了 torch2.7+,那个 venv 从未建成)。

| 行 | 现状 | 改为 | 理由 |
|---|---|---|---|
| torch | `>=2.1.0` | `>=2.8.0`(或按 base 精确 pin `==2.8.0`) | 对齐 base;**已实测 2.8 全绿** |
| torchvision | `>=0.16.0` | `>=0.23.0` | 联动;已实测 |
| **numpy** | `>=1.24,<2` | **`>=2.0`**(放开封顶) | torch2.8 按 numpy2 ABI 编译;**已实测 numpy 2.4.4 全绿** |
| **opencv-python-headless** | `>=4.8.0`(**无上限**) | 🔴 **`>=4.8.0,<5`(必须封顶)** | **本次实测捞到的真雷,见下方 §2.1** |
| pylibjpeg 三件套 | `<2.2.0`/`<2.3.0` 封顶 | 放开上限 | 原封顶是为躲 numpy>=2,现拥抱 numpy2 反而要新版;已实测 2.1.0/2.4.0/2.5.0 可用 |
| timm | `>=0.9.0` | `>=1.0` | **已实测 1.0.28 下 backbone 加载正常**(此前标为"待实测的低置信项",现已排除) |
| Pillow | `>=9.5.0,<11` | 放宽(实测 12.2.0 可用) | 已实测 |
| 其余(pydicom/nibabel/scipy/sklearn/skimage/albumentations/faiss/fastapi/uvicorn) | — | 不动 | 已实测(scipy 1.18 / skimage 0.26 / faiss 1.15)全绿 |

### 2.1 🔴 OpenCV 5.0 破坏 `detect_splicing`(本次预演捞到的真雷)

**现象**:torch2.8 环境首跑全套 pytest = 7 failed,全部在 `tests/test_gatekeeper.py`(Splicing/CatNet/BAG),报 `TypeError: cannot unpack non-iterable numpy.int32 object`。

**双环境对拍锁根因**(numpy 两边同为 2.4.4,唯一变量是 cv2):

| 环境 | cv2 | 结果 |
|---|---|---|
| 旧(torch2.5) | **4.13.0** | 54 passed |
| 新(torch2.8) | **5.0.0** | **7 failed** |
| 新(torch2.8) 降 cv2 后 | **4.14.0** | **54 passed** ✅ |

**根因**:`src/gatekeeper/` 的 `detect_splicing` 用 `cv2.HoughLinesP`,OpenCV 5.0 改了返回值形状,内部解包语句直接抛 TypeError。**不是 numpy、不是 torch 的问题。**

**危害**:`requirements.txt` 原写 `opencv-python-headless>=4.8.0` **无上限** → 上云构建镜像时 `pip install` 会装到 cv2 5.0 → `detect_splicing` 抛异常。gatekeeper 虽零计分,但**须确认该异常是否被上层 except 兜住**;若未兜住会拖垮整个病例 = 该例零分。

**修法**:`opencv-python-headless>=4.8.0,<5`。

### 2.2 系统性教训:所有依赖都该封顶 major

本次是运气好(本地预演抓到);`scipy`/`skimage`/`albumentations`/`timm` 等**同样是 `>=` 无上限**,任何一个在未来发布破坏性大版本,构建镜像时都会静默装上。**官方禁联网 + 只能提交一次**,构建那一刻装错版本 = 无法补救。
→ **建议交付前用 `pip freeze > requirements.lock.txt` 把全部版本 pin 死**,镜像用 lock 文件构建,而不仅是加上限。

## 3. 代码改动(torch API 兼容)

**崩溃级:无**(A800 torch2.6 已跑通,2.6→2.8 无破坏性 API)。

**行为/warning 级(可选改,消 FutureWarning)**:
- `train.py:246` `torch.cuda.amp.GradScaler(...)` → `torch.amp.GradScaler('cuda', ...)`(2.8 仅 deprecated 不崩;`cbm_head.py:37/52` 已是正确样板)。**行号已从 173 校正为 246**(2026-08-13 实核)。⚠️ 上游未把 `GradScaler` 放进 `torch.amp.__all__`(2.5.1/2.6.0/2.8.0 三版实测 `'GradScaler' in __all__` 均为 **False**),Pyright 会报 `reportPrivateImportUsage`,但三版**运行时均实测可构造** —— 迁移时按 `cbm_head.py` 样板加 `# type: ignore` 即可,别因为 IDE 报红就回退。
  - ✅ **`scripts/train_official.py` 已迁(2026-08-13)**,写法 `torch.amp.GradScaler(device.type, enabled=...)` —— 传 `device.type` 而非硬编码 `'cuda'`,CPU 环境不刷"CUDA 不可用"警告(torch28 CPU 环境实测:传 `'cuda'` 会 warn 并 disable)。`train.py` 是 legacy 交付路径,**本次刻意不动**,仍待迁。
- 多处 `from torch.cuda.amp import autocast` → `torch.amp.autocast('cuda', ...)`。
- `torch.load` 所有调用点:**当前均安全**(auth 存 numpy 的已显式 `weights_only=False`,其余纯 tensor 白名单覆盖)。建议显式化 + 注释,非必须。
  - 🔬 **2026-08-13 A800(torch 2.6.0,即 `weights_only` 默认翻转的首个版本)逐文件实测,把上面的推断升级为证据**:`checkpoints/` 下 17 个 `.pth` 用**默认参数**加载 —— **16 个 OK**(分割底座 `best_model*.pth`/`thyroidxl_merged*.pth`/`official_heads_best.pth` 均为纯 `OrderedDict`;`thyroidxl_best*.pth`/`thyus2path_lora_best.pth` 的 `{"epoch","stage","state_dict",...}` 包装、`malignancy_mil_*.pth`、`training_state.pth` 的 `{"epoch","criterion","famo"}` 也都通过 —— **python 基本类型在 weights_only 白名单内**,包装 dict 本身不是问题);**唯一失败 = `authenticity_model.pth`**(`UnpicklingError`,因含 numpy 的 `feat_mean`/`feat_std`),而它由 `gatekeeper/authenticity.py:229` 自己以 `weights_only=False` 加载,**生产路径不受影响**。
  - `scripts/train_official.py:113` 已显式 `weights_only=False` —— 属**防御性**(允许 `--resume-from` 指向包装格式、未来格式再变也不翻车),**不是在修已存在的崩溃**。
- `deduplication.py:103` `models.resnet18(pretrained=True)` → `weights=` 写法(仅 deprecated)。

### 3.1 静态兼容性扫描结论（并入自 `torch28兼容性扫描.md`，源文件已删）

> 扫描日期 2026-08-11，方法为逐项 grep + 实读代码确认。以下为**增量**结论；与 §2/§3 重复的（requirements 逐行、GradScaler/autocast/resnet18 别名）不在此重复。

**`torch.load` 调用点全量盘点（纯 tensor 加载点未显式传 `weights_only`）**：
| file:line | 内容 | 风险 |
|---|---|---|
| `train.py:180` | `load_state_dict(torch.load(args.resume))` | resume 为自产 `best_model.pth`（纯 state_dict），torch 2.6 起默认 `weights_only=True` 可安全加载；但未显式声明，行为依赖 fed 文件内容 |
| `train.py:192` | `torch.load(state_path)` | `training_state.pth` 全纯 tensor+原生数字，安全但未显式声明 |
| `scripts/pretrain_malignancy_mil.py:193` | `torch.load(args.resume_from)` | Phase1 自产 checkpoint 全纯 tensor+数字，安全但未显式声明 |
| `scripts/data/duipai_probe_vs_head.py:41` | `torch.load(CKPT)` | 训练自产，全 tensor+数字，安全但未显式声明 |

→ 建议统一显式 `weights_only=True`（纯 tensor 内容即安全），消除默认值隐患。**交付主链已全部显式化**：`src/inference/model_runner.py:50` / `scripts/merge_lora_checkpoint.py:59` 均 `weights_only=True`，`authenticity.py:229` 显式 `weights_only=False`（含 numpy 标准化统计 `feat_mean`/`feat_std`），`deduplication.py:91` `weights_only=True`——交付容器无默认值隐患。

**numpy 2 移除别名全仓零命中（已 grep 排除，无需改）**：`np.float_`/`np.complex_`/`np.object_`/`np.bool`/`np.round_`/`np.in1d`/`np.product`/`np.NaN`/`np.Inf`/`np.NINF` 均无裸使用；`np.array(..., copy=False)` 零命中；`.ptp()`/`.itemset()`/`.asscalar()` 零命中。所有 dtype 均为 `np.float32`/`np.uint8` 等合法类型。

**Python 3.11 移除项零命中**：`inspect.getargspec`/`asyncio.coroutine`/`distutils`/`imp.load` 全仓 grep 零命中，代码无 3.10-only 依赖。

**TorchScript 边界**：`src/gatekeeper/deduplication.py:77`/`tampering_catnet.py:169` 用 `torch.jit.load`（SSCD/CAT-Net `.torchscript.pt`）——**torch 2.9 起才正式 deprecated，2.8 完整支持**，交付锁 2.8 即可，2.9+ 需迁 `torch.export`。

**NEP 50 标量提升热区（已定位，升级后跑 `tests/test_gatekeeper.py` 对拍）**：`synthesis.py:97` `np.clip(...).astype(np.uint8)` 末尾兜底安全；`deduplication.py:202,220` 显式 astype 安全；`tampering_catnet.py:240` `.../ 255.0` NEP50 升 float64 低风险。审计为零计分防御项，漂移不致命但会改软 flag 边界。

**未用新 API**：`torch.compile`/`torch.func`/`vmap`/`functorch` 全仓零命中；`torch.set_float32_matmul_precision("high")`（`train.py:49`）2.8 仍支持；AMP 训练热路径均用 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` 通用 API（bf16 无 GradScaler，规避 fp16 溢出），2.8 语义不变。

## 4. Dockerfile 改动 —— ❌ 本章作废(存档)

> **2026-08-11:方向已定为「直接用官方基础镜像 + 权重走平台上传」,不自建/不上传私有镜像。**
> 故本章的换 base / 装 sshd / 权重 COPY 均不执行。保留仅为万一改走私有镜像时的参考。

<details><summary>展开原方案(不执行)</summary>

1. `FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` → `FROM <官方>/pytorch:2.8.0-cuda12.8-cudnn9-runtime`(注意官方镜像仓库地址 `10.1.63.4/public/caih/...`)。
2. **sshd**(官方要求私有镜像自启):`apt-get install -y --no-install-recommends openssh-server` + 配 `sshd_config`(PermitRootLogin/PasswordAuthentication)+ `EXPOSE 22` + 启动 wrapper(`service ssh start && exec python -m src.service.app`,不能塞进有哈希校验的 entrypoint.sh)。
3. **权重 COPY(#111)**:`COPY checkpoints/*.pth ...`。
4. py 版本:base 从 py3.10 → py3.11/3.12,确认无 3.10-only 语法。

</details>

## 5. CI 同步(关键,别漏) —— 🔴 **未完成的有效待办**

- `.github/workflows/ci.yml` 的 test job **现仍是 `container: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`**(2026-08-12 主会话实读 `ci.yml:39` 核实),**必须同步换** torch2.8 base,否则 CI 测的不是交付环境(违反铁律2)。**本条是本文档唯一未落地的实质待办,勿因 §4 作废就以为整篇存档。**
- **docker-build-smoke / entrypoint-dispatch-smoke**:换 base 后重验。
- 🔴 **权重 COPY(#111)是必做项,不是"若加了"**:交付 backlog #111 明确要求 Dockerfile 补权重 COPY(见 `docs/项目现状总纲.md` §7),换 base 后冒烟步骤须相应调整。此处与 §4 存档中的 sshd 不同 —— **sshd 已随自建镜像方案一起作废,权重 COPY 没有**。
- push 后 `gh run list`(走代理)核 headSha + success。

## 6. 验证顺序(换云时执行)

> ⚠️ **A800 做不了这套预演(2026-08-11 实测锁死根因)**:A800 是 **Ubuntu 16.04 / glibc 2.23**,而 torch 2.7+ 的官方 wheel 全部要求 `manylinux_2_28`(glibc ≥ 2.28)。pip 在平台标签过滤阶段就丢弃全部 2.7+ 版本(表现为"可用版本停在 2.6.0",易误判为网络问题)。**升 glibc = 换系统且无 root,此路不通**。
> → 验证只能在 ① 官方平台起容器,或 ② 本地机器 docker 拉同版本 `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` 里做。

1. 在上述可行环境里装齐依赖 → 确认**依赖能装**(numpy2 + pylibjpeg 新版 + faiss 有 wheel)。
2. 我们的权重在 torch2.8 能 `torch.load` 加载(重点 `thyroidxl_merged_v2.pth` / `authenticity_model.pth`)。**静态扫描已确认交付主链全部显式传了 `weights_only`,理论安全**(见 §3.1),仍需实测坐实。
3. `verify_model.py` + 一次真实推理跑通、数值正常。
4. **显存复核按 32G**(2026-08-12 实测定稿;本地 A800 80G 不 OOM ≠ 平台不 OOM)。
5. 断网复跑(模拟 Q26 联网禁令),确认无联网尝试。
6. timm 1.x 下 backbone 权重 tag(`convnext_nano.in12k_ft_in1k` / `efficientnet_b2.ra_in1k`)是否仍存在——扫描报告标为唯一需实测的低置信项。

## 7. 前提状态(2026-08-11 更新)

- ✅ **权重可上传**(用户确认官方平台支持)→ 不必打进镜像,#111 的"权重进镜像"不再是唯一解。
- ✅ **不自建镜像**:直接用官方 base(§0),故"镜像大小上限 / 私有镜像 sshd / 本地能否拉 `10.1.63.4` 内网 registry"三条**全部不再阻塞**(内网 registry 本地本来就拉不到,只能平台内用——而我们正是在平台内用)。
- ⏳ **仍待官方**:容器与权重的**交付时序**(权重什么时候、怎么进容器),这决定 #123 启动硬门怎么写。已在用户提交赛事组的问题清单里。
- ⏳ **待实测**:容器内真实显存(`nvidia-smi`)、timm 1.x backbone 权重 tag 可用性。
