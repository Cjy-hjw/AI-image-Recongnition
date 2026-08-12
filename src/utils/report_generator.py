import math

# TI-RADS clinical attributes definitions & point configurations
COMPOSITION_MAP = {
    0: ("囊性或几乎完全囊性 (Cystic / Almost entirely cystic)", 0),
    1: ("海绵状 (Spongiform)", 0),
    2: ("囊实性混合 (Mixed cystic and solid)", 1),
    3: ("实性 (Solid)", 2)
}

ECHOGENICITY_MAP = {
    0: ("无回声 (Anechoic)", 0),
    1: ("高回声或等回声 (Hyperechoic / Isoechoic)", 1),
    2: ("低回声 (Hypoechoic)", 2),
    3: ("极低回声 (Very hypoechoic)", 3)
}

SHAPE_MAP = {
    0: ("宽大于长 (Wider-than-tall)", 0),
    1: ("长大于宽 (Taller-than-wide)", 3)
}

MARGIN_MAP = {
    0: ("边缘光滑 (Smooth)", 0),
    1: ("边缘模糊 (Ill-defined)", 0),
    2: ("边缘不规则或分叶状 (Lobulated / Irregular)", 2),
    3: ("甲状腺外侵犯 (Extrathyroidal extension)", 3)
}

FOCI_MAP = {
    0: ("粗大彗星尾伪影 (None or Large comet-tail artifacts)", 0),
    1: ("粗大钙化 (Macrocalcifications)", 1),
    2: ("边缘/环状钙化 (Peripheral calcifications)", 2),
    3: ("点状强回声 (Punctate echogenic foci / Microcalcification)", 3)
}

def _lookup(map_, idx, label):
    """按 idx 查 TI-RADS 属性表，越界即硬报错（2026-07-10 Gate2审查发现）。

    此前用 `map_.get(idx, ("未知", 0))` 静默把越界 idx 当 0 分处理——上游概念头
    维度契约被破坏（如 index 从预期4类变5类）时不报错、不告警，只在报告正文里
    悄悄出现"未知"二字。实测案例：margin_idx=5（非法，本应是2分档）被静默记0分，
    总分从真值3(TR3,~4.8%风险,建议穿刺)算成1(TR2,~1.5%风险,无需处置)——静默跨越
    了真实临床处置决策边界。越界必须视为上游contract violation硬拦截，不能带着
    错误分数继续跑完整条报告生成链路。
    """
    if idx not in map_:
        raise ValueError(
            f"TI-RADS {label}_idx={idx!r} 超出映射表范围 {sorted(map_.keys())}——"
            "上游概念头输出维度契约被破坏，拒绝静默按0分处理"
        )
    return map_[idx]


def calculate_tirads_points(comp_idx, echo_idx, shape_idx, margin_idx, foci_probs, foci_threshold=0.5):
    """
    Computes total ACR TI-RADS points based on predicted indices.
    """
    points = 0

    comp_desc, comp_pts = _lookup(COMPOSITION_MAP, comp_idx, "comp")
    echo_desc, echo_pts = _lookup(ECHOGENICITY_MAP, echo_idx, "echo")
    shape_desc, shape_pts = _lookup(SHAPE_MAP, shape_idx, "shape")
    margin_desc, margin_pts = _lookup(MARGIN_MAP, margin_idx, "margin")

    points += (comp_pts + echo_pts + shape_pts + margin_pts)

    foci_desc_list = []
    # Foci is multi-label, can select multiple positive classes
    for i, prob in enumerate(foci_probs):
        if prob >= foci_threshold:
            desc, pts = _lookup(FOCI_MAP, i, "foci")
            points += pts
            foci_desc_list.append(f"{desc} (+{pts}分)")
            
    if not foci_desc_list:
        foci_desc_list.append("无明显异常强回声灶 (+0分)")
        
    # Classify Risk Level
    if points == 0:
        tr_level = "TR1 (良性 / Benign)"
        risk = "恶性风险 < 0.3%"
        action = "无需随访或细针穿刺 (FNA)"
    elif points <= 2:
        tr_level = "TR2 (无恶性可疑 / Not suspicious)"
        risk = "恶性风险 ~1.5%"
        action = "无需随访或细针穿刺 (FNA)"
    elif points == 3:
        tr_level = "TR3 (轻度可疑 / Mildly suspicious)"
        risk = "恶性风险 ~4.8%"
        action = "结节 >= 2.5cm 建议随访; >= 1.5cm 建议细针穿刺 (FNA)"
    elif points <= 6:
        tr_level = "TR4 (中度可疑 / Moderately suspicious)"
        risk = "恶性风险 ~9.1%"
        action = "结节 >= 1.5cm 建议随访; >= 1.0cm 建议细针穿刺 (FNA)"
    else:
        tr_level = "TR5 (高度可疑 / Highly suspicious)"
        risk = "恶性风险 >= 35%"
        action = "结节 >= 1.0cm 建议随访; >= 0.5cm 建议细针穿刺 (FNA)"
        
    breakdown = {
        "composition": (comp_desc, comp_pts),
        "echogenicity": (echo_desc, echo_pts),
        "shape": (shape_desc, shape_pts),
        "margin": (margin_desc, margin_pts),
        "foci": foci_desc_list,
        "total_points": points,
        "tr_level": tr_level,
        "risk": risk,
        "action": action
    }
    return breakdown

def generate_report(image_id, patient_id, breakdown, malignancy_prob):
    """
    Generates a structured text explanation report.
    """
    if malignancy_prob is None or not math.isfinite(malignancy_prob):
        # 2026-07-10 Gate2审查发现：此前无校验，NaN 场景下 Python `nan >= 0.5`
        # 恒为 False，会**无任何报错**地打印"良性倾向"——本该是"无效/不确定"的
        # 结果被静默判成良性，是癌症筛查场景下的假阴性风险。None 场景直接
        # `TypeError` 崩溃（虽是崩溃但同样不可接受，此处一并拦截给出明确原因）。
        # 硬拦截交给上层 predict.py/video_predict.py 的 except Exception 转
        # ERROR result.json，不允许带着无效概率继续生成一份"看起来正常"的报告。
        raise ValueError(
            f"generate_report 收到非法 malignancy_prob={malignancy_prob!r}——"
            "上游推理必须产出有限概率值，拒绝据此生成确定性诊断结论"
        )
    report = []
    report.append("="*60)
    report.append("  甲状腺结节超声智能辅助诊断及规范化审核报告")
    report.append("="*60)
    report.append(f"图像编号: {image_id}")
    report.append(f"患者编号: {patient_id}")
    report.append(f"智能分类恶性概率: {malignancy_prob*100:.2f}%")
    report.append(f"预测结论: {'恶性可疑 (Suspicious for Malignancy)' if malignancy_prob >= 0.5 else '良性倾向 (Benign Nodule)'}")
    report.append("-"*60)
    report.append("  ACR TI-RADS 声学特征定量评估:")
    # CBM 五概念头(成分/回声/形状/边缘/钙化)在当前基线 thyroidxl_merged_v2 中从未
    # 收到语义监督——train_thyroidxl.py 只训 malignancy + 分割,概念头仅作为恶性预测
    # 的隐藏中间表示被间接训练,其 argmax 索引与 TI-RADS 语义无监督对应关系(证据见
    # docs/历史存档/比赛对齐与风险.md「接线」条)。故下列五项为架构预演输出、语义未经验证,不可
    # 作独立诊断依据;仅恶性概率与分割 mask 可信。8-01 官方概念标注重训建立语义监督后
    # 删除本段与下面两行 append。
    report.append("  [注] 声学五分项(成分/回声/形状/边缘/钙化)概念头未经语义监督,为架构")
    report.append("       预演输出、语义未经验证,仅供参考;恶性概率与分割 mask 方为可信结论。")
    
    comp_desc, comp_pts = breakdown["composition"]
    report.append(f"  1. 成分: {comp_desc} (+{comp_pts}分)")
    
    echo_desc, echo_pts = breakdown["echogenicity"]
    report.append(f"  2. 回声: {echo_desc} (+{echo_pts}分)")
    
    shape_desc, shape_pts = breakdown["shape"]
    report.append(f"  3. 形状: {shape_desc} (+{shape_pts}分)")
    
    margin_desc, margin_pts = breakdown["margin"]
    report.append(f"  4. 边缘: {margin_desc} (+{margin_pts}分)")
    
    report.append("  5. 强回声灶:")
    for foci_desc in breakdown["foci"]:
        report.append(f"     - {foci_desc}")
        
    report.append("-"*60)
    report.append(f"  总积分: {breakdown['total_points']} 分")
    report.append(f"  风险评级: {breakdown['tr_level']}")
    report.append(f"  临床统计学恶性概率: {breakdown['risk']}")
    report.append(f"  临床处置建议: {breakdown['action']}")
    
    # Intrinsic explainability validation
    cbm_decision = "恶性可疑" if breakdown["total_points"] >= 4 else "良性倾向"
    ai_decision = "恶性可疑" if malignancy_prob >= 0.5 else "良性倾向"
    consistency = "完全一致" if cbm_decision == ai_decision else "存在诊断微调差异"
    
    report.append(f"  逻辑可追溯性校验 (CBM对齐): {consistency}")
    report.append("="*60)
    
    return "\n".join(report)
