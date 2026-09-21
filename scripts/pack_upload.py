"""打包上云包。

用法:
    python scripts/pack_upload.py            # 三个包全打
    python scripts/pack_upload.py infer      # 推理包(代码+权重+手册)
    python scripts/pack_upload.py train      # 训练底座权重

产出:
    C:\\Users\\cjy\\Desktop\\cloud_infer.zip           推理包:代码+权重+手册(约 42MB)
    C:\\Users\\cjy\\Desktop\\cloud_train_weights.zip   训练底座(约 104MB,为下次训练准备)

## 代码包为什么是 41 个 .py 而不是 src/ 全树 88 个

内容 = `python -m src.service.app` 的**静态 import 闭包**(`scripts/verify_closure.py`
实测 41)。`src/dataset`(11) / `src/training`(15) / `src/models/losses`(15) 等推理
路径**永不触达**。

🔴 判据用静态闭包不用运行时探针:运行时只覆盖本次实际走到的分支,而推理服务有大量
条件路径(gatekeeper 降级 / 静动分发 / 坏帧兜底),某次没触发 ≠ 部署时不触发。静态
闭包按 import 传递闭包算,只多留不少留。(实测:运行时 30 个 vs 静态 41 个,少的 11 个
正是没走到的降级分支。)

新增 import 后须跑 `verify_closure.py` 更新 `_CLOSURE`,漏一个 = 云端 ImportError。

## 三个包各装什么(按代码实测的加载点,非凭印象)

**推理代码包** —— 41 个 .py + `configs/` + `requirements.txt` + `README.md`(操作
手册)+ `AGENTS.md`(目录地图)。

**推理权重包**:

| 文件 | 必需? | 缺了会怎样 |
|---|---|---|
| 主模型权重 | 🔴 必需 | **云端训练产物,不在包内**,由 `AICOMP_WEIGHTS` 指定 |
| `resnet18-f37072fd.pth` | 🔴 必需 | 去重后端兜底权重。SSCD 主后端权重我方没有 → 必然降级 resnet18;此时缓存也缺 → 尝试联网下载 → **云端禁联网必然 raise** → `build_service_runner` 崩 → 服务起不来 = 整场零分 |
| `authenticity_model.pth` | 可选 | 只把真实性审计标 skipped、诊断照常,且该项零计分。4.7K 顺手带上 |

**训练底座包**(为下次训练准备,与推理无关):

| 文件 | 用途 |
|---|---|
| `best_model_mixtrain_xl_e40.pth` | 分割 warm-start 底座(公开集 Dice 97.71%/87.15%)。实测 793 张量 / 144 个 seg 键 / **四头 0 个** —— 纯分割底座,不能当推理主权重 |
| `*_pretrained.safetensors` ×2 | ImageNet 骨干初始化 |

⚠️ `authenticity_model.pth` 要放**主权重同目录**(`gatekeeper_runner.py:110` 取
`dirname(weights_path)`),不是固定的 `checkpoints/`。
"""
import hashlib
import os
import shutil
import sys
import zipfile

REPO = r"C:\Users\cjy\Desktop\AI image Recongnition"
OUT_DIR = r"C:\Users\cjy\Desktop"

# ── 推理代码闭包(scripts/verify_closure.py 实测,新增 import 后须重算)──────
_CLOSURE = """
src/gatekeeper/__init__.py
src/gatekeeper/authenticity.py
src/gatekeeper/deduplication.py
src/gatekeeper/tampering.py
src/gatekeeper/tampering_catnet.py
src/inference/__init__.py
src/inference/calibration.py
src/inference/case_pipeline.py
src/inference/export.py
src/inference/gatekeeper_runner.py
src/inference/io.py
src/inference/model_runner.py
src/inference/nifti_io.py
src/inference/nodule_split.py
src/inference/postprocess.py
src/inference/prediction_json.py
src/inference/tiling.py
src/inference/v21_output.py
src/inference/video_aggregate.py
src/models/__init__.py
src/models/blocks.py
src/models/cbm_head.py
src/models/heads/__init__.py
src/models/heads/clinical_t.py
src/models/heads/extension.py
src/models/heads/location.py
src/models/heads/tirads.py
src/models/label_schema.py
src/models/lora_utils.py
src/models/mask_features.py
src/models/multi_task_model.py
src/models/physical_prior.py
src/models/pretrained_loader.py
src/service/__init__.py
src/service/app.py
src/service/case_discovery.py
src/service/config.py
src/service/runner.py
src/service/tasks.py
src/utils/__init__.py
src/utils/grad_cam.py
src/utils/report_generator.py
""".split()

_CODE_EXTRA = [
    ("configs/thyroid_config.yaml", "configs/thyroid_config.yaml"),
    ("requirements.txt", "requirements.txt"),
    ("docs/8.1赛事公开资料/云端手册/操作说明.md", "README.md"),
    ("docs/8.1赛事公开资料/云端手册/速查表.md", "AGENTS.md"),
]

# ── 推理权重:(本地绝对路径, 包内相对路径) ─────────────────────────────────
_WEIGHTS = [
    # 🔴 必需:去重后端的兜底权重。SSCD 主后端权重我方没有 → Deduplicator 必然
    # 降级到 resnet18(deduplication.py:80-86);此时若本地缓存也没有,代码会尝试
    # `models.resnet18(pretrained=True)` 联网下载,而**云端禁联网(答疑Q26)必然
    # 失败 → RuntimeError**(deduplication.py:102-115)。该对象在
    # `build_service_runner`(runner.py:169-170)构建期创建 → **服务起不来 =
    # 整场零分**。这是权重包里唯一非可选项。
    (os.path.expanduser(r"~\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth"),
     "torch_hub_checkpoints/resnet18-f37072fd.pth"),
    # 可选:缺失只是把真实性审计标 skipped、诊断照常(gatekeeper_runner.py:102-119),
    # 且真实性在赛道七零计分。才 4.7K,带上省一条启动警告。
    (os.path.join(REPO, "checkpoints_a800", "authenticity_model.pth"),
     "checkpoints/authenticity_model.pth"),
]

# ── 训练底座权重(为下次训练准备,推理用不上)────────────────────────────────
# 🔴 `best_model_mixtrain_xl_e40.pth` 是**分割 warm-start 底座**(公开集 Dice
# 97.71%/87.15%),实测 793 张量 / 144 个 seg 键 / **四头 0 个** —— 它不含官方四头,
# **不能当推理主权重**(装了四头就是随机初始化)。推理主权重只能是云端训练产出的那份。
#
# ImageNet 骨干(convnext_nano / efficientnet_b2 的 .safetensors)**不装**:
# 云端已有(上一轮训练包传过),重传只是白等上传时间。

# 🔴 当前交付权重的分辨率(2026-09-11 B路审查 F5-1 防线):r5 = 512。
# 出包前用它对 config 的 img_size 做硬校验,不配套即拒绝出包。
# 640 实验胜出并交付 640 权重时,这里与云端 AICOMP_WEIGHTS 同步改 640。
_DELIVERED_IMG_SIZE = 512

_TRAIN_WEIGHTS = [
    (os.path.join(REPO, "checkpoints_a800", "best_model_mixtrain_xl_e40.pth"),
     "checkpoints/best_model_mixtrain_xl_e40.pth"),
]


def _emit(out_dir, items, label):
    """items: [(abs_src, rel_dst)];产出目录 + zip + SHA256SUMS.txt。"""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    copied, missing = [], []
    for src, rel in items:
        if not os.path.isfile(src):
            missing.append((src, rel))
            continue
        dst = os.path.join(out_dir, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    if missing:
        for s, r in missing:
            print(f"  🔴 缺文件 {r}  <- {s}")
        raise SystemExit(f"{label}: {len(missing)} 个文件缺失,拒绝出包")

    lines = []
    for rel in sorted(copied):
        with open(os.path.join(out_dir, rel.replace("/", os.sep)), "rb") as fh:
            lines.append(f"{hashlib.sha256(fh.read()).hexdigest()}  {rel}")
    with open(os.path.join(out_dir, "SHA256SUMS.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    zp = out_dir + ".zip"
    if os.path.exists(zp):
        os.remove(zp)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out_dir):
            for fn in files:
                ap = os.path.join(root, fn)
                z.write(ap, os.path.relpath(ap, out_dir).replace("\\", "/"))

    print(f"\n=== {label} ===")
    print(f"  {zp}   ({os.path.getsize(zp)/1024/1024:.1f} MB, {len(copied)+1} 文件)")
    return copied


def build_code():
    items = [(os.path.join(REPO, r.replace("/", os.sep)), r) for r in _CLOSURE]
    items += [(os.path.join(REPO, s.replace("/", os.sep)), d) for s, d in _CODE_EXTRA]
    items += _WEIGHTS          # 权重与代码同包:少一次上传 + 少一次人工审核窗口
    # 🔴 出包前硬校验 config 分辨率与推理权重的配套性(2026-09-11 B路审查 F5-1):
    # config 的 img_size 烘焙进 checkpoint(prior_extractor.x_coords buffer 形状),
    # 512 权重 + 640 config(或反向)= build_model 的 load_state_dict 当场
    # RuntimeError → 测评容器起不来 = 整场零分。云端 r5 交付权重是 512;
    # 640 实验期间本地 config 是 640,**绝不能**带着它出推理包(实验权重
    # 未定胜负前,交付面始终是 r5+512)。要出 640 包须先改此处常量。
    import yaml
    with open(os.path.join(REPO, "configs", "thyroid_config.yaml"),
              "r", encoding="utf-8") as fh:
        _cfg_img_size = int(yaml.safe_load(fh)["data"]["img_size"])
    if _cfg_img_size != _DELIVERED_IMG_SIZE:
        raise SystemExit(
            f"🔴 拒绝出包:configs/thyroid_config.yaml 的 img_size={_cfg_img_size},"
            f"但交付权重是 {_DELIVERED_IMG_SIZE}(r5)。640 实验的 config 不能进"
            f"推理包——云端 r5 权重遇到 640 config 会 build_model 崩(整场零分)。"
            f"640 实验上线交付时,先改本文件的 _DELIVERED_IMG_SIZE 并确认权重已换。")
    copied = _emit(os.path.join(OUT_DIR, "cloud_infer"), items, "推理包(代码+权重)")
    n_py = sum(1 for r in copied if r.endswith(".py"))
    print(f"  .py = {n_py}")
    print("\n  云端:")
    print("    unzip -o cloud_infer.zip                   # 🔴 必须 -o,别走「替换同名目录」")
    print("    sha256sum -c SHA256SUMS.txt | grep -v OK   # 应无输出")
    print(f'    find src -name "*.py" | wc -l              # 期望 {n_py}')
    print('    python -c "import src.service.app; print(\'OK\')"')
    print("\n  🔴 解压后须手工搬一个文件(它要落在家目录,压缩包到不了那儿):")
    print("    mkdir -p ~/.cache/torch/hub/checkpoints")
    print("    cp torch_hub_checkpoints/resnet18-f37072fd.pth ~/.cache/torch/hub/checkpoints/")


def build_train_weights():
    _emit(os.path.join(OUT_DIR, "cloud_train_weights"), _TRAIN_WEIGHTS, "训练底座包")
    print("\n  用途:**下次训练**的 warm-start 底座,推理用不上。")
    print("    checkpoints/* → 项目根的 checkpoints/")
    print("  🔴 `best_model_mixtrain_xl_e40.pth` 不含官方四头(实测 four_heads=0),")
    print("     **不能当推理主权重** —— 装了四头就是随机初始化,输出随机数。")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "infer", "code", "weights"):
        build_code()
    if which in ("all", "train"):
        build_train_weights()
