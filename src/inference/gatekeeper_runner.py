"""Gatekeeper audit pipeline — production inference only (NOT evaluate)."""

import os
import cv2

from src.gatekeeper.authenticity import AuthenticityModule
from src.gatekeeper.deduplication import Deduplicator
from src.gatekeeper.tampering_catnet import CatNetTamperingDetector


def gatekeeper_cfg(config) -> dict:
    """取 `config["gatekeeper"]`,非 dict(缺键 / None / 标量 / 列表)一律返回 {}。

    唯一入口——本模块**所有**读 gatekeeper 段的地方都必须走这里。
    2026-07-29 /code-review 实测复现:此前只有 `resolve_enforcement_mode` 自己做了
    isinstance 兜底,`build_seeded_deduplicator`/`build_tamper_detector` 仍直接
    `config["gatekeeper"]["..."]`,于是 `gatekeeper: hard`(YAML 少一层缩进)、
    `gatekeeper: null`、整段缺失三种写法都会在模式已降级为 soft **之后**抛
    KeyError/TypeError,被入口 except 兜成 `status="ERROR"` —— **整例零诊断**,
    恰好是 enforcement_mode 软化要避免的后果。降级必须贯穿整条读取路径,
    只在解析模式那一处兜底等于没兜。
    """
    gk = config.get("gatekeeper")
    return gk if isinstance(gk, dict) else {}


def build_seeded_deduplicator(config, device):
    """构建并用训练集种子初始化 Deduplicator。

    抽出此函数是为了让视频/cine 推理**只播种一次**、跨帧复用——否则逐帧调用
    `run_gatekeeper_audit` 会对每帧重复读取并 ResNet 编码最多 200 张训练图,
    O(帧数×200) 的冗余严重拖慢延迟(赛道有推理时延自动评分)。
    """
    # 两个阈值的缺省值都显式对齐 **configs/thyroid_config.yaml 的交付值**——畸形
    # 配置下宁可按交付行为继续跑,也不要因 KeyError 让整例零诊断(见 gatekeeper_cfg)。
    #   phash_threshold 缺省 1     = 交付值。
    #   cosine_threshold 缺省全映射 = 交付值 {sscd:0.86, resnet18:0.995},**而非 None**。
    #     传 None 会让 Deduplicator 取类内推荐值,resnet18 后端是 0.95 —— 比交付值
    #     宽松,等于畸形配置下静默把去重判据从 0.995 放松到 0.95(误报变多,踩
    #     「宁可漏检不可误杀真实影像」红线)。原则:降级要降到**交付行为**,不是
    #     降到库默认行为。config 可给标量(两后端同值)或映射,Deduplicator 内部按
    #     backend_name 取值、缺键回退后端推荐值,故这里给全映射即可。
    gk = gatekeeper_cfg(config)
    train_path = (config.get("data") or {}).get("train_path")

    # 值级畸形兜底(2026-07-30 /code-review Angle C):`gatekeeper_cfg()` 只兜了
    # 「gatekeeper 段是不是 dict」,兜不住段内**单个值**写歪——运维手改 yaml 把
    # `cosine_threshold: "high"`(字符串)、`phash_threshold: "x"` 这类非数值写进
    # 交付配置时,`float("high")` 会穿透到 Deduplicator/int() 里抛 ValueError,被
    # 入口兜成整例 ERROR = 零诊断,正是软化要避免的后果。阈值是「宁可跑严一点也
    # 不能零输出」的审计参数,值非法时**降级到交付默认**而不是抛异常。
    raw_cos = gk.get("cosine_threshold", {"sscd": 0.86, "resnet18": 0.995})
    if isinstance(raw_cos, dict):
        cos_thr = raw_cos
    else:
        try:
            cos_thr = float(raw_cos)
        except (TypeError, ValueError):
            print(f"--> [WARNING] cosine_threshold={raw_cos!r} 非法,降级交付默认映射",
                  flush=True)
            cos_thr = {"sscd": 0.86, "resnet18": 0.995}
    raw_ph = gk.get("phash_threshold", 1)
    try:
        phash_thr = int(raw_ph)
    except (TypeError, ValueError):
        print(f"--> [WARNING] phash_threshold={raw_ph!r} 非法,降级交付默认 1", flush=True)
        phash_thr = 1

    deduplicator = Deduplicator(
        device=device,
        phash_threshold=phash_thr,
        cosine_threshold=cos_thr,
        sscd_weights=gk.get("sscd_weights"),
    )
    if train_path and os.path.exists(train_path):
        # sorted():os.listdir 顺序取决于文件系统/inode,跨容器 rebuild、跨平台
        # (Win 开发机 vs Linux 容器)会变——不排序则被选入去重种子库的前 200 张
        # 可能不同,导致同一张边界重复图 is_duplicate 判定跨环境不一致(踩「宁可漏检
        # 不可误杀」红线)。排序后种子集合确定,判定可复现。
        train_samples = sorted(os.listdir(train_path))
        limit = min(len(train_samples), 200)
        for i in range(limit):
            p = os.path.join(train_path, train_samples[i], "image.png")
            if os.path.exists(p):
                train_img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if train_img is not None:
                    deduplicator.add_image(train_img)
    return deduplicator


def build_auth_module(weights_path):
    """构建并加载 AuthenticityModule，供视频逐帧复用（只加载一次权重）。

    单图 predict.py 无需调用——run_gatekeeper_audit 内部按需懒建；
    视频 video_predict.py 在帧循环前调用此函数，传入 auth_module 参数。
    """
    auth_module = AuthenticityModule()
    auth_weights_path = os.path.join(
        os.path.dirname(weights_path), "authenticity_model.pth"
    )
    auth_module.load_weights(auth_weights_path)
    return auth_module


def build_tamper_detector(config):
    """构建 CatNetTamperingDetector，供视频逐帧复用（避免每帧重探 weights 文件）。

    单图 predict.py 无需调用——run_gatekeeper_audit 内部按需懒建；
    视频 video_predict.py 在帧循环前调用此函数，传入 tamper_detector 参数。
    """
    gk = gatekeeper_cfg(config)
    return CatNetTamperingDetector(
        catnet_weights=gk.get("catnet_weights"),
        bag_hard_reject=gk.get("bag_hard_reject", False),
    )


def resolve_enforcement_mode(config) -> str:
    """读取 `gatekeeper.enforcement_mode`,归一化为 'soft' / 'hard'(缺省 'soft')。

    2026-07-28 官方赛前动员会:赛道七豁免全部医保安全考核,总分 = 临床诊断得分
    Sclin × 100,gatekeeper **零计分贡献**;且官方明确"真实病例被误判为伪造,诊断
    结果仍参与目标三四五计分"——前提是必须先输出诊断。故硬拒绝在本赛道已成
    **纯负期望**:拒对了零收益,误拒一次即白丢该病例全部临床得分(唯一分数来源)。
    默认 soft:审计照跑、结果照记,但不再中断诊断输出。

    非法值(拼写错误等)一律降级 soft 并打警告——失败方向必须偏向"多输出一份诊断"
    而非"多拒一张真实图"(统一软flag制度,见 src/gatekeeper/AGENTS.md §1)。整段写歪
    (YAML 少一层缩进写成 `gatekeeper: hard`、写成 None/列表)由 `gatekeeper_cfg()`
    统一兜成 {},本模块所有读该段的地方共用同一降级语义。
    """
    raw = gatekeeper_cfg(config).get("enforcement_mode", "soft")
    mode = str(raw).strip().lower()
    if mode not in ("soft", "hard"):
        print(f"--> [WARNING] 未知 gatekeeper.enforcement_mode={raw!r},降级为 'soft'")
        return "soft"
    return mode


def collect_soft_flags(tamper_detector) -> list:
    """把 `CatNetTamperingDetector.last_soft_flags` 归一化成可 JSON 序列化的 list。

    2026-07-29 /code-review 发现:`last_soft_flags` 在生产代码里**零读取点**——
    `run_gatekeeper_audit` 只解包 `is_tampered`,于是仅被 copy_move/splice/BAG 命中
    (三者均为软判据,恒不进 is_tampered)的图,`audit["flagged"]` 为 **False**、
    result.json 完全沉默,软 flag 只出现在 stdout。而 copy_move 在 359 张真实图上
    误报 27.3%,信号量不小却全丢,与本模块 docstring 自陈的「审计结论必须能落进
    result.json」直接矛盾(那次修复只覆盖了 phantom/synthetic)。

    `last_soft_flags` 每次 `detect()` 开头重置(tampering_catnet.py:286),故视频
    逐帧复用同一 detector **无跨帧污染**。旧 detector 对象无该属性 → getattr 兜 []。

    **防御式归一化(2026-07-30 /code-review Angle A 补强)**:软 flag 是纯审计字段,
    它的畸形**绝不能让整例审计崩溃**(崩了会被入口兜成 ERROR = 零诊断,比丢一条
    审计记录严重得多)。故对三层畸形都兜底:
      ① 非二元组元素        → {"type": str(item), "bbox": None};
      ② bbox 不是可迭代序列  → bbox=None;
      ③ bbox 元素无法转 int  → 整条 flag 丢弃(bbox 已不可信,留着只会误导复核)。
    长度≠4 的 bbox 目前三个检测器都不会产出(None 或 4-list),暂原样透传——
    归一化的目标是"不崩 + 可序列化",不是替上游校验几何语义。
    """
    flags = getattr(tamper_detector, "last_soft_flags", None) or []
    out = []
    for item in flags:
        try:
            kind, bbox = item
        except (TypeError, ValueError):
            out.append({"type": str(item), "bbox": None})
            continue
        if bbox is None:
            out.append({"type": str(kind), "bbox": None})
            continue
        try:
            norm_bbox = [int(v) for v in bbox]
        except (TypeError, ValueError):
            continue  # bbox 元素畸形(含 None/str/NaN):丢整条,不炸整例
        out.append({"type": str(kind), "bbox": norm_bbox})
    return out


def run_gatekeeper_audit(img_np, config, device, weights_path, output_dir,
                         deduplicator=None, auth_module=None,
                         tamper_detector=None):
    """Run full gatekeeper audit (authenticity, dedup, tampering).

    Args:
        deduplicator:    可选，预建 Deduplicator（视频复用，省去重复播种）。
        auth_module:     可选，预建 AuthenticityModule（视频复用，省去重复加载权重）。
        tamper_detector: 可选，预建 CatNetTamperingDetector（视频复用，省去重复文件探测）。
        三者均为 None 时内部即时构建——单图路径行为不变，向后兼容。

    Returns:
        (passed: bool, audit: dict)

        `audit` **恒为 dict**(旧契约在通过时返回 None,已改)——审计结论必须能落进
        result.json,否则 phantom/synthetic 软 flag 只打印在 stdout、评测方拿不到。
        字段:
          flagged                 任一审计项命中(含仅软 flag 的 phantom/synthetic/
                                  copy_move/splice/BAG-相位)
          would_reject_in_hard_mode  **仅**去重/篡改硬判据命中 → hard 模式下会 REJECTED。
                                  软 flag 恒不计入(见 soft_flags)
          enforcement_mode        本次生效的模式
          reason                  人类可读命中说明(无命中为 "")
          is_phantom/is_gan/is_duplicate/is_tampered  逐项审计位
          soft_flags              [{"type": "copy_move"|"splice"|"bag_phase",
                                    "bbox": [x1,y1,x2,y2]|None}, ...]

        仅当 `enforcement_mode == "hard"` 且 would_reject 为真时返回
        `(False, {..., "status": "REJECTED"})`,可直接交 `save_rejection_json`。
    """
    print("Running Gatekeeper Audit...")
    mode = resolve_enforcement_mode(config)
    if auth_module is None:
        auth_module = build_auth_module(weights_path)
    # 权重缺失/损坏仍在 AuthenticityModule.load_weights 内硬报错(未改动,红线保留)。
    is_phantom, auth_probs = auth_module.predict(img_np)

    if deduplicator is None:
        deduplicator = build_seeded_deduplicator(config, device)

    is_duplicate, dup_reason = deduplicator.is_duplicate(img_np)
    # CAT-Net v2 取证检测器:detect() = 深度CAT-Net ∪ BAG相位(默认软) ∪ 几何(恒软)。
    # 缺权重时降级 dct-bag(纯 NumPy)。
    if tamper_detector is None:
        tamper_detector = build_tamper_detector(config)
    is_tampered, _tam_mask, _tam_bbox = tamper_detector.detect(img_np)
    # 软 flag(copy_move/splice/BAG-相位)恒不进 is_tampered,必须单独取出落进 audit,
    # 否则只出现在 stdout、评测方与人工复核都拿不到(见 collect_soft_flags)。
    soft_flags = collect_soft_flags(tamper_detector)

    # authenticity 软 flag(2026-07-12,统一软flag制度):authenticity_trainer 的
    # phantom/synthetic 训练标签 100% 来自 synthesis.py 程序化合成代理,从未见过
    # 真实体模/真实AI合成图;代码自陈 Nakagami/TAI 判别方向"是合成训练域的特性,
    # 非真实物理结论"(authenticity.py 的 extract_nakagami_features docstring);
    # 且 argmax 无任何置信度门槛。全仓零真实图像 FPR 实测(对比 tampering 有 359
    # 张真实图 0%/27.3% 的明确数字)。故不再作一票否决,仅记录审计,不驱动 REJECT。
    # 恢复硬拒绝的唯一闸门:官方真实干净 holdout 上预注册、带置信区间的 FPR 验收
    # 达标(见 docs/P2红线修复实施计划.md B1、docs/P2启动定论_Opus复核.md §五.5)。
    if is_phantom == 1:
        print("--> [Gatekeeper 软 flag] authenticity: Physical Phantom 疑似(不拒绝,仅记录)", flush=True)
    elif is_phantom == 2:
        print("--> [Gatekeeper 软 flag] authenticity: GAN/Synthetic 疑似(不拒绝,仅记录)", flush=True)

    # 去重/篡改命中 = "hard 模式下会拒绝"。soft 模式下只作 flag,不中断诊断。
    would_reject = bool(is_duplicate or is_tampered)

    rejections = []
    if is_duplicate:
        rejections.append(f"Duplicate image: {dup_reason}")
    if is_tampered:
        # 2026-07-12 code-review(Angle A)发现:此前坍缩成笼统的
        # "hard-reject path"丢失了具体命中源(CatNet深度模型 vs BAG,
        # 二者可信度/验证量级截然不同)。改用 CatNetTamperingDetector 新增的
        # last_hard_reject_source 拼更精确的文案;tamper_detector 为向后兼容
        # 保留的旧对象(无该属性)时退化为原通用文案。
        source = getattr(tamper_detector, "last_hard_reject_source", None)
        if source == "catnet":
            rejections.append("Tampering detected (CAT-Net deep model)")
        elif source == "bag_phase":
            rejections.append("Tampering detected (BAG phase inconsistency)")
        else:
            rejections.append("Tampering detected (source unspecified)")
    # phantom/synthetic 已降级为软flag、不独立驱动 REJECTED,但仍把软flag信息并入
    # reason——否则(2026-07-12 code-review Angle B 发现)is_phantom/is_gan 字段虽仍
    # 保留在 JSON 里,但人类可读的 reason 文本会静默丢失这条并发信号。
    if is_phantom == 1:
        rejections.append("(soft-flag: Physical Phantom suspected)")
    elif is_phantom == 2:
        rejections.append("(soft-flag: GAN/Synthetic suspected)")
    # 篡改软 flag 同理并入 reason(与 phantom 同一处理),人类可读文本不能漏这条信号。
    for f in soft_flags:
        rejections.append(f"(soft-flag: {f['type']} suspected)")

    audit = {
        "flagged": bool(rejections),
        "would_reject_in_hard_mode": would_reject,
        "enforcement_mode": mode,
        "reason": ", ".join(rejections),
        # is_phantom/is_gan 为审计字段(信息记录,不驱动 REJECT)
        "is_phantom": int(is_phantom == 1),
        "is_gan": int(is_phantom == 2),
        "is_duplicate": int(is_duplicate),
        "is_tampered": int(is_tampered),
        # copy_move/splice/BAG-相位软命中详情。**只计入 flagged,绝不进
        # would_reject_in_hard_mode**——三者恒为软判据(copy_move 在 359 张真实图上
        # 误报 27.3%),升级为拒绝依据会直接踩「宁可漏检不可误杀真实影像」红线。
        # 恢复其硬拒绝权限的唯一闸门是各自真实域 holdout FPR 验收达标。
        "soft_flags": soft_flags,
    }

    if would_reject and mode == "hard":
        print(f"--> [REJECTED] Gatekeeper audit failed: {audit['reason']}")
        return False, {**audit, "status": "REJECTED"}

    if would_reject:
        # soft 模式:命中照记、诊断照出。丢一次诊断 = 丢该病例全部临床得分,
        # 而审计本身在赛道七不产生任何得分(见 resolve_enforcement_mode)。
        print(f"--> [Gatekeeper 软 flag] {audit['reason']}(不拒绝,继续诊断)", flush=True)
        return True, audit

    if audit["flagged"]:
        return True, audit

    print(
        "--> [PASSED] Gatekeeper audit successfully verified image authenticity and normalization."
    )
    return True, audit
