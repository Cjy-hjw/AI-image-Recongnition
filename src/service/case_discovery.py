"""V2.1 HTTP 服务的「病例发现」纯函数:把一个 dataset_path 目录解开成 case 列表。

## 背景:为什么需要这一层

V2.1 交付形态是常驻 HTTP 服务(`src/service/app.py` 的 `/call`)。平台下发一个
**目录**(`dataset_path`,内含批量测评病例),服务须遍历该目录逐病例推理,每病例
落 `{PatientId}/prediction.json`(见 `docs/训练数据定稿.md` §1.2)。runner 契约是
`runner(dataset_path, output_dir) -> None`,其中第一步就是「这个目录里有哪些病例、
每个病例的输入文件是什么、按静/动怎么分派」—— 即本模块。

**本模块只做「发现 + 元信息提取」,不接推理**(推理核心 `case_pipeline` 由主线接线,
A3/A2 后续做)。故本模块**不依赖 torch / 模型**,可独立单测,且 import 链不拉起
任何重依赖 —— 唯一可选依赖是 pydicom(读 DICOM 头取 PatientID/SeriesInstanceUID),
缺失时降级走路径推断,见下。

## 静/动识别与元信息来源

静/动区分沿用全项目统一口径(定稿 §1.3):**扩展名分派** ——
`.dcm`(含 `.dicom`)→ static,`.nii`/`.nii.gz` → dynamic。复用
`src.inference.nifti_io.is_nifti_path` 判 NIfTI(它正确处理 `.nii.gz` 双段后缀,
`os.path.splitext` 只切 `.gz` 会漏),不在此重写扩展名逻辑。

元信息(PatientId / subdir_uid)按输入类型取不同来源:

- **静态 `.dcm`**:优先读 DICOM 头 —— `PatientID`(0010,0020)与
  `SeriesInstanceUID`(0020,000E)。这是唯一**可靠**来源(DICOM 头自含),且与
  `src/inference/io.py::load_input_image` 的 PHI 匿名化读取走同一套 pydicom.dcmread
  路径(但本模块**只读元数据、不读像素、不做匿名化写盘** —— 发现层与推理层解耦,
  匿名化是推理时的副作用)。DICOM 头读不到(无 pydicom / 文件非标准 DICOM /
  字段缺失)→ 降级走路径推断,见 `_fallback_patient_id_from_path` 的 @云TODO。

- **动态 `.nii.gz`/`.nii`**:NIfTI 无 DICOM 头,PatientId/AccessionNumber **只能
  从路径推断**。训练数据定稿 §1.11 提示动图形如 `{PatientId}/{PatientId}/{PatientId}.nii.gz`
  (一患者一动态序列,定稿 §4.2 项1 实测确认)—— 据此**参数化**取某级父目录名作
  PatientId。但**确切目录层级本地拿不到官方真实数据无法锁死**(定稿 §4.2 是云端
  agent 在训练集 `annotation/` 下实测,推理输入侧目录结构是 `verification/original/`
  下的 3 层嵌套哈希,定稿 §4.2 项2 仅写「靠扩展名区分静动」、未给确切层级),故此处
  用**可配置的「向上第 N 级父目录」参数**,默认按训练数据口径取 2 级父目录,并显式
  标 @云TODO:云端首跑校正。

  `subdir_uid`(V2.1 动态输出子目录名 = AccessionNumber,定稿 §1.2)来源官方**未明确**
  (定稿 §1.2 注 + §4.1 项1:动态 `.nii.gz` 无 DICOM 头,推理时 AccessionNumber 从何
  获取是「待问官方」项)→ 标 @待官方,先用占位(取 PatientId 或文件 stem)并注释,
  云端拿到真值后在此处替换。

## 防御性设计的纪律

官方真实目录结构本地拿不到(数据禁本地,定稿 §0 证据来源 E5 是云端 agent 真读),
**所有「从路径推断」的逻辑都做成小函数 + 清晰 @云TODO 注释**,宁可结构清晰易改,
别写死一套猜测糊上去 —— 云端首跑发现结构对不上时,改一个函数即可,不用全仓 grep。
这是本模块最重要的纪律:**把不确定的地方标出来,而不是假装确定**。

## 返回的 case dict 形态

    {
      "input_path":  <str, 该病例输入文件的绝对路径>,
      "kind":        "static" | "dynamic",   # .dcm→static, .nii.gz/.nii→dynamic
      "patient_id":  <str>,                  # 静态=DICOM PatientID,动态=路径推断
      "subdir_uid":  <str>,                  # 静态=SeriesInstanceUID,动态=AccessionNumber(待官方)
    }

调用方(后续 A3/A2 的 runner 接线)据此分派:
  - static → `load_input_image` + `diagnose_static_case`;
  - dynamic → `load_nifti_frames` + `diagnose_dynamic_case`;
  - 落盘子目录名:`{patient_id}/`(顶层) + `{subdir_uid}/`(静态 SeriesUid / 动态
    AccessionNumber,见定稿 §1.2 输出结构)。
"""

import os

from src.inference.nifti_io import is_nifti_path

# 静态图扩展名(与 `src/inference/io.py::load_input_image` 的 _is_dicom 判定一致:
# 该函数用 `path.lower().endswith((".dcm", ".dicom"))`)。此处复用同口径,避免两处
# 扩展名列表分叉致静/动分派边界错位(如 `.DICOM` 大小写漏判)。
_DICOM_EXTS = (".dcm", ".dicom")

# 应跳过的非影像文件扩展名(黑名单)。仅当文件既非 DICOM 也非 NIfTI 时才用它做
# 二次确认/注释;主判定是「不是影像」即跳过,见 _classify_file。
# @云TODO:官方 dataset_path 里除 .dcm/.nii.gz 外是否还会混入 .json/.txt/.md/
# .xlsx(标注表)/子目录 README 等,云端首跑观察后如有其它需跳过的类型在此补。
_NON_IMAGE_EXTS = (".json", ".txt", ".md", ".xlsx", ".csv", ".xml", ".html")


def discover_cases(dataset_path):
    """递归遍历 dataset_path,识别所有病例文件,返回 case dict 列表。

    Args:
        dataset_path: 平台 `/call` 下发的 **目录**路径(定稿 §1.2:一个目录内含
            批量测评病例)。允许不存在或为空 —— 返回 [] 而非抛异常,把「目录无效」
            的处置权交回 runner(runner 须落 `_error.json` 回应平台,见
            `tasks._write_task_error`;发现层不越俎代庖落盘)。

    Returns:
        list[dict]:每项形如模块 docstring 描述的 case dict。**顺序**按 os.walk
        的目录遍历序(确定性,便于复现与对拍),同目录内按文件名排序。

    Raises:
        不抛异常 —— IO 错误(Dataset 头读失败、文件不可读)一律记为「跳过该文件」
        并通过降级路径推断元信息。runner 拿到空列表时自行决定是否落 ERROR。
        **理由**:发现层是「尽量多发现」语义,单文件异常不该拖垮整批;且平台只有
        一次提交机会,异常崩在发现阶段 = 整批零输出,比「跳过坏文件、其余正常」更糟。
    """
    if not dataset_path or not os.path.isdir(dataset_path):
        # 目录不存在/不是目录 → 返回 [](runner 负责落 ERROR,本层不越权)。
        return []

    cases = []
    # os.walk(topdown=True) 保证先访问目录再访问其下文件,顺序确定性。
    for root, _dirs, files in os.walk(dataset_path, topdown=True):
        # 同目录内文件名排序,保证跨平台(Linux 容器 vs Windows 本地测试)顺序一致,
        # 便于对拍与回归(顺序非业务语义,但测试断言 `==` 需要稳定序)。
        for name in sorted(files):
            full = os.path.join(root, name)
            case = _build_case(full)
            if case is not None:
                cases.append(case)
    return cases


def _build_case(path):
    """单文件 → case dict;非影像文件返回 None(跳过)。

    分派(定稿 §1.3 扩展名口径):
      - `.dcm`/`.dicom`  → static  → `_build_static_case`
      - `.nii`/`.nii.gz` → dynamic → `_build_dynamic_case`
      - 其余             → None(跳过非影像)
    """
    lower = path.lower()
    if lower.endswith(_DICOM_EXTS):
        return _build_static_case(path)
    if is_nifti_path(path):
        return _build_dynamic_case(path)
    # 非影像文件(.json/.txt/.md/.xlsx 等标注/说明文件)→ 跳过。
    # 不抛异常:dataset_path 可能混入标注 xlsx、README、子目录说明等,属正常。
    return None


def _build_static_case(path):
    """静态 `.dcm` 病例:优先读 DICOM 头取 PatientID/SeriesInstanceUID。

    与 `src/inference/io.py::load_input_image` 的 DICOM 读取走同一套
    `pydicom.dcmread`,但本模块**只读元数据、不读 pixel_array、不做 PHI 匿名化写盘**
    —— 避免在发现阶段就把整张像素解码一遍(浪费 IO),也避免匿名化写盘副作用
    混进发现层(那是推理层 `load_input_image` 的职责)。

    降级路径:DICOM 头读不到(PatientID/SeriesInstanceUID 缺失或 pydicom 不可用)
    → 走 `_fallback_patient_id_from_path` 从路径推断,并标 @云TODO。
    """
    patient_id = None
    series_uid = None

    try:
        import pydicom  # 局部 import:pydicom 是推理容器必装依赖,但发现层不应在
        # import 时就硬依赖它(测试可能 monkeypatch 掉、或 CI 基镜像缺)。缺失时
        # 走降级路径,不崩。
    except ImportError:
        pydicom = None

    if pydicom is not None:
        try:
            # stop_before_pixels=True:只读元数据头,不解码像素(发现层不需要像素,
            # 且部分非标准 DICOM 可能像素编码损坏但头合法 —— 不读像素就不会被它拖垮)。
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            # ds.get(...) 在字段缺失时返回 None(不抛 KeyError),且兼容 pydicom 2.x/3.x。
            # 用属性名而非 (0010,0020) 元组:属性名更可读,且 pydicom 自动映射标准 tag。
            # 0010,0020=PatientID / 0020,000E=SeriesInstanceUID(DICOM 标准 tag)。
            patient_id = _clean_dicom_value(ds.get("PatientID"))
            series_uid = _clean_dicom_value(ds.get("SeriesInstanceUID"))
        except Exception:
            # pydicom.dcmread 失败(文件非标准 DICOM / 损坏 / 非 DICOM 内容碰巧 .dcm 后缀)
            # → 静默降级走路径推断。不抛:发现层是「尽量多发现」语义。
            patient_id = None
            series_uid = None

    # 降级:DICOM 头没取到 → 从路径推断。
    if not patient_id:
        # @云TODO(云端首跑校正):官方推理输入目录结构是 verification/original/{哈希}/
        # {DICOM UID}/{文件}(定稿 §4.2 项2),与训练集 annotation/{pid}/{series}/*.dcm
        # 不同层。此处取「父目录名」作 PatientId 占位 —— 真值须云端首跑时对照
        # DICOM 头(若有)或目录命名规律校正。**优先信 DICOM 头**,只在头缺失时才用此兜底。
        patient_id = _fallback_patient_id_from_path(path)
    if not series_uid:
        # @云TODO(同上):SeriesInstanceUID 占位取「文件所在目录名」(训练集口径:
        # 子目录名 = SeriesInstanceUID,定稿 §1.11)。推理侧若不同,云端校正。
        series_uid = _fallback_subdir_uid_from_path(path)

    return {
        "input_path": os.path.abspath(path),
        "kind": "static",
        "patient_id": patient_id,
        "subdir_uid": series_uid,
    }


def _build_dynamic_case(path):
    """动态 `.nii.gz`/`.nii` 病例:无 DICOM 头,PatientId/AccessionNumber 只能从路径推断。

    训练数据定稿 §1.11:动图目录 `{PatientId}/{PatientId}/{PatientId}.nii.gz` →
    取**向上第 2 级父目录名**作 PatientId(文件本身 1 级 + 中间一层 PatientId 目录 1 级)。

    但**推理输入侧目录结构本地拿不到官方真实数据无法锁死**(定稿 §4.2 项2 仅写
    「靠扩展名区分静动」、未给确切层级),故层级用参数 `_DYNAMIC_PATIENT_DEPTH`
    (默认 2),云端首跑发现对不上时改这一个常量即可,不用动逻辑。

    `subdir_uid`(V2.1 动态输出子目录名 = AccessionNumber,定稿 §1.2)来源官方
    **未明确**(定稿 §1.2 注 + §4.1:动态 .nii.gz 无 DICOM 头,AccessionNumber 从何
    获取待官方)→ @待官方,先用 PatientId 占位(动态一患者一序列,定稿 §4.2 项1,
    PatientId 在动态语境下与 AccessionNumber 的输出子目录定位语义等价),云端拿
    到真值后在此替换。
    """
    patient_id = _infer_ancestor_name(path, _DYNAMIC_PATIENT_DEPTH)
    # @待官方:AccessionNumber 来源未明确(见模块 docstring)。动态一患者一序列
    # (定稿 §4.2 项1)下,用 PatientId 作 subdir_uid 占位不会产生子目录冲突。
    # 官方真值到位后替换此处。
    subdir_uid = patient_id if patient_id else _fallback_subdir_uid_from_path(path)

    return {
        "input_path": os.path.abspath(path),
        "kind": "dynamic",
        "patient_id": patient_id,
        "subdir_uid": subdir_uid,
    }


# ── 路径推断小函数(集中一处,便于云端校正) ────────────────────────────────
# 所有「从路径推断 PatientId/UID」的逻辑都收在这两个函数里,带 @云TODO。
# 云端首跑发现官方目录结构对不上时,只改这两个函数 + 上面的 _DYNAMIC_PATIENT_DEPTH
# 常量,不用动 discover_cases 主流程 —— 这是「防御性 + 参数化」的落点。


def _fallback_patient_id_from_path(path):
    """DICOM 头缺失时,从路径推断 PatientId。

    训练集口径(定稿 §1.11):静态图目录 `{PatientId}/{SeriesInstanceUID}/*.dcm` →
    取**向上第 2 级父目录名**(文件本身 1 级 + SeriesUid 目录 1 级 = 2 级)作 PatientId。

    @云TODO(云端首跑校正):推理输入侧目录是 verification/original/{哈希}/{DICOM UID}/
    {文件}(定稿 §4.2 项2,3 层嵌套哈希),与训练集层级不同 —— 真值须云端首跑时对照
    DICOM 头(优先)或目录命名规律校正。**静态优先信 DICOM 头**,此函数仅兜底。
    """
    return _infer_ancestor_name(path, _STATIC_PATIENT_DEPTH)


def _fallback_subdir_uid_from_path(path):
    """SeriesInstanceUID / AccessionNumber 缺失时,从路径推断。

    静态:训练集子目录名 = SeriesInstanceUID(定稿 §1.11)→ 取「父目录名」(1 级)。
    动态:AccessionNumber 来源官方未明确(@待官方,见 _build_dynamic_case)。

    @云TODO(云端首跑校正):同上,推理侧目录层级不同,真值须校正。
    """
    return _infer_ancestor_name(path, 1)


# ── 推断层级常量(云端首跑后在此校正) ──────────────────────────────────────
# 训练集口径(定稿 §1.11):静态 {pid}/{series}/*.dcm → pid 在第 2 级父;
# 动态 {pid}/{pid}/{pid}.nii.gz → pid 在第 2 级父。
# 推理侧口径(定稿 §4.2 项2):verification/original/{哈希}/{DICOM UID}/{文件} →
# 层级不同,@云TODO 云端首跑后改这两个常量。
_STATIC_PATIENT_DEPTH = 2
_DYNAMIC_PATIENT_DEPTH = 2


def _infer_ancestor_name(path, depth):
    """取 path 向上第 depth 级父目录名;越界(路径浅于 depth)返回文件 stem 兜底。

    Args:
        path: 文件路径(绝对或相对)。
        depth: 向上几级父目录(1=直接父目录,2=祖父目录,...)。

    返回的目录名**去空**(空串/纯空白 → None),调用方据此走更上一级兜底或留 None。
    越界时返回文件 stem(不带扩展名的文件名)作占位 —— 比返回 None 更可能在极端
    浅层路径下给出可用标识(如 `xxx.nii.gz` 直接在 dataset_path 根下时,depth=2
    越界,取 stem `xxx` 作 PatientId 占位,标 @云TODO)。
    """
    # os.path.normpath 统一分隔符后 split,避免尾部 sep 产生空段(如
    # `a/b/` → ['a','b',''] 的空尾段干扰 depth 计算)。
    # 越界判定:dirname 链到达根(dirname(parent)==parent,Windows 下 dirname('pytest-0')
    # == 'pytest-0' 不变)或 depth 耗尽后 name 为空/为根名,都属「路径浅于 depth」→
    # 兜底取文件 stem(去 .nii.gz / .dcm 双/单段后缀)。
    parent = os.path.normpath(path)
    for _ in range(depth):
        new_parent = os.path.dirname(parent)
        if new_parent == parent:  # 已到根,无法再向上
            parent = ""
            break
        parent = new_parent
    name = os.path.basename(parent) if parent else ""
    name = name.strip() if name else ""
    if not name or name == parent:  # 越界 或 到达根后 name==parent(Windows 卷根)
        # 兜底取文件 stem(去 .nii.gz / .dcm 双/单段后缀)。
        # @云TODO:这种浅层路径在官方数据里是否会出现未知 —— 云端首跑观察。
        stem = os.path.basename(path)
        # 去双段后缀 .nii.gz 先,再去单段后缀。
        for ext in (".nii.gz", ".nii", ".dcm", ".dicom"):
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
                break
        return stem.strip() or None
    return name


def _clean_dicom_value(v):
    """DICOM 字段值 → 去空白字符串;None/空 → None。

    pydicom 返回的值可能是 MultiValue(多值 list-like)、PersonName 或 bytes,
    统一 str() 后去首尾空白;空串/纯空白视为缺失 → None(调用方据此走降级)。

    DICOM 的 PatientID 极少是多值,但 SeriesInstanceUID 在某些拼接场景可能被存成
    MultiValue —— str() 后取首个等价于 MultiValue.__str__ 的表现,够用(发现层
    只要一个可用标识,不求 UID 语义严格唯一)。
    """
    if v is None:
        return None
    s = str(v).strip()
    return s or None
