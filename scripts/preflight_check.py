"""上云训练前自检 —— 30 秒内把所有关于官方数据的假设一次验完。

为什么要有这个脚本(必读,决定了它该怎么用):
2026-08-14 第一次云端真训直接 KeyError 崩在读 xlsx 那一步,根因是我方把「图像性质」
这个**从未验证过的推测列名**当成必需列(证据链里原文写的是"推断为…需 Excel 验证",
被误当成已确认)。那一崩浪费了一整轮上传审核 + GPU 排队。

比崩溃更坏的是**静默**:临床列名对不上时不报错、只是该头全程 `valid_*=0` 零监督,
训练照常跑完 35 epoch,最后拿到一个"合法但什么也没学到"的模型。

所以本脚本的设计原则是:**只读、不训练、不写任何文件**,把所有"崩"和"静默"的假设
在真正开训前全部暴露成人能看懂的文字。跑完再决定要不要启动训练。

用法:
    python preflight_check.py --data-root <官方annotation目录> --xlsx <标注xlsx>

    # 路径未知时先不传,它会自己找并告诉你找到了什么
    python preflight_check.py
"""
import argparse
import os
import sys
import glob

# 🔴 **必须在任何 print 之前做**:强制 stdout/stderr 走 UTF-8。
# 本脚本的输出几乎全是中文(报告文字 + 74 个中文列名 + 中文取值),而云端官方容器
# 常见 `LANG=C`/`POSIX` locale → `sys.stdout.encoding` 为 ascii;Windows 中文版
# 控制台则是 gbk。两者下打印中文/emoji 都会 `UnicodeEncodeError` **直接崩**,
# 而且崩点恰好在有 FAIL 要报告的时候(全绿路径反而可能侥幸跑完)——即"最需要看
# 结果时什么都看不到"。实测:`PYTHONIOENCODING=ascii python -c "print('\U0001f534')"`
# → UnicodeEncodeError。errors="replace" 兜底:极端环境下退化成 `?` 也好过崩掉。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")   # py3.7+
    except Exception:
        pass

# 本脚本位于 <包根>/scripts/ 下,需把**包根**加进 sys.path 才能 `import src.*`
# (与 train_official.py 同款写法:dirname(dirname(abspath(__file__))))。
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG_ROOT)

DEFAULT_ROOT = "/2026aicompetition/datasets/training/annotation"

ok_count = 0
fail_count = 0
warn_count = 0


def report(level: str, title: str, detail: str = "") -> None:
    global ok_count, fail_count, warn_count
    mark = {"ok": "[ OK ]", "fail": "[FAIL]", "warn": "[WARN]", "info": "[INFO]"}[level]
    print(f"{mark} {title}")
    if detail:
        for line in str(detail).splitlines():
            print(f"       {line}")
    if level == "ok":
        ok_count += 1
    elif level == "fail":
        fail_count += 1
    elif level == "warn":
        warn_count += 1


def section(name: str) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")


# 逻辑字段 → 该列在官方表里的语义(用于人工挑列时判断"像不像")。
_FIELD_MEANING = {
    "patient_id":   "患者ID(32位hex),同时是磁盘一级目录名",
    "roi_name":     "病灶名:主病灶=`M`,次病灶=`病灶1`/`病灶2`…",
    "series_uid":   "序列UID(1.2.826.0.1...),静态图=磁盘二级子目录名",
    "malignancy":   "良恶性,取值形如 `良`/`恶`/`无`",
    "lesion_loc":   "主病灶位置,取值形如 `2左中叶`/`7峡部`(数字+中文)",
    "extrathy_ext": "甲状腺外侵犯,取值形如 `1无`/`2甲状腺包膜`/`3毗邻软组织`/`4血管`",
    "clin_t_stage": "临床T分期,取值形如 `0无`/`1有`",
    "tirads":       "TI-RADS 总分类,取值形如 `V类`/`IV类`(罗马数字+类)",
    "slice_index":  "动态图标注帧号(整数,从1起)",
}


def _suggest_alias_fix(O, columns, missed, df=None) -> None:
    """未命中时,从真实列名里挑最相近的几个,打印成可直接粘贴的修复代码。

    存在的意义:让云端**当场自己改完继续跑**,而不是回传一轮等开发者改完重新上传
    (那一轮往返要过人工审核,成本以小时计)。挑选用「归一化后的字符重合度」排序,
    再让人肉眼确认语义——**脚本只负责缩小范围,最终由人拍板**,因为选错列会导致
    用错标签训练,那比零监督更糟。

    🔴 **相似度高 ≠ 是对的列,必须看取值再拍板**(2026-08-14 审查实证的真实陷阱):
    真表里 `TI-RADS详细评分` 列的取值是 `"成分:2分,回声:2分,形态:0分,..."` 这种
    **复合串**,它对 `成分`/`回声` 的字符相似度很高、会被排到首选;一旦选中,
    `_parse_concept`(只取数字前缀)对它必然全部解析失败 → 该头零监督;更坏的情形是
    某列语义相近但编码不同,解析"成功"却全是错标签——**那比零监督更糟,且自检全绿、
    训练跑完才暴露**。所以下面对每个候选都打印真实取值样例,选之前先看值对不对。
    """
    import difflib

    print("\n" + "=" * 70)
    print("🔧 未命中字段的修复建议(可直接自己动手,不必回传等我)")
    print("=" * 70)

    cols = [str(c) for c in columns]
    norm_cols = {c: O._normalize_col(c) for c in cols}

    for logical in missed:
        aliases = O._FIELD_ALIASES.get(logical, ())
        print(f"\n  ▸ {logical}  ——  {_FIELD_MEANING.get(logical, '')}")
        print(f"    现有候选名(全都没命中):{list(aliases)}")

        # 用归一化后的串做相似度排序,把最像的真实列名推上来
        targets = [O._normalize_col(a) for a in aliases]
        scored = []
        for c in cols:
            best = max((difflib.SequenceMatcher(None, norm_cols[c], t).ratio()
                        for t in targets), default=0.0)
            scored.append((best, c))
        scored.sort(reverse=True)
        top = [c for s, c in scored[:5] if s > 0.3]

        if top:
            print(f"    最可能是这几列之一(按相似度排序,**先看取值再决定**):")
            for i, c in enumerate(top, 1):
                print(f"      {i}. {c!r}")
                # 🔴 打印真实取值 —— 相似度只看列名字符,看不出语义。
                #    `TI-RADS详细评分` 的值形如 "成分:2分,回声:2分,..." 会冒充 `成分` 列。
                if df is not None:
                    try:
                        vc = df[c].value_counts(dropna=True)
                        miss = int(df[c].isna().sum())
                        sample = ", ".join(f"{v!r}×{n}" for v, n in list(vc.items())[:5])
                        print(f"         取值: {sample}")
                        print(f"         非空 {len(df) - miss}/{len(df)},唯一值 {len(vc)} 种")
                    except Exception:
                        pass
            print(f"    → 确认后,在 src/dataset/official_loader.py 的 _FIELD_ALIASES 里,")
            print(f"      把 \"{logical}\" 这一行的候选名**开头**加上你选中的那个列名,例如:")
            print(f"          \"{logical}\":  ({top[0]!r}, ...原有候选名保持不动...),")
            print(f"    ⚠️ 取值格式与 {logical} 的预期不符就**别选**(宁可零监督也别喂错标签),")
            print(f"       把全部真实列名和取值发回给开发者判断。")
        else:
            print(f"    ⚠️ 真实列名里没有任何一个与现有候选相似 —— 该字段在这份表里"
                  f"可能确实不存在。")
            if logical in ("malignancy", "lesion_loc", "extrathy_ext",
                           "clin_t_stage", "tirads"):
                print(f"      它是监督列,缺了对应的头就零监督。请把上面 [4] 打印的"
                      f"**全部真实列名**发回给开发者判断。")

    print("\n  改完后重跑本脚本确认全绿,再启动训练。")
    print("  ⚠️ 只改 _FIELD_ALIASES 里的**候选列名字符串**,别动其它逻辑。")


def main() -> int:
    ap = argparse.ArgumentParser(description="官方数据训练前自检(只读)")
    ap.add_argument("--data-root", default=None, help=f"默认 {DEFAULT_ROOT}")
    ap.add_argument("--xlsx", default=None, help="标注 xlsx;不传则自动搜")
    ap.add_argument("--sheet", default="ROI级别")
    ap.add_argument("--log-dir", default=None,
                    help="JSONL 日志目录,默认 V2.1 强制路径 /2026aicompetition/workspace/logs;"
                         "试训用 logs。本脚本只探测可写性,写完立刻删除探针文件。")
    ap.add_argument("--resume-from", default=None,
                    help="warm-start 底座 checkpoint;传了就验它能否加载、键集是否匹配"
                         "(否则要等训练建完模型才崩)")
    args = ap.parse_args()

    # ── [1] 运行环境 ────────────────────────────────────────────────────
    section("[1] 运行环境")
    print(f"       Python  {sys.version.split()[0]}")
    print(f"       cwd     {os.getcwd()}")

    for mod in ("torch", "numpy", "pandas", "openpyxl", "timm", "cv2", "nibabel", "pydicom"):
        try:
            m = __import__(mod)
            report("ok", f"{mod:10s} {getattr(m, '__version__', '(no __version__)')}")
        except ImportError as e:
            lvl = "fail" if mod in ("torch", "numpy", "pandas", "openpyxl") else "warn"
            report(lvl, f"{mod:10s} 未安装", str(e))

    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            report("ok", f"GPU {p.name}",
                   f"显存 {p.total_memory / 2**30:.1f} GB / CUDA {torch.version.cuda}")
        else:
            report("fail", "CUDA 不可用", "训练会退到 CPU,速度不可接受")
    except Exception as e:
        report("fail", "GPU 探测失败", str(e))

    # numpy 版本是已知雷区:requirements.txt 的 numpy<2 是旧推理容器约束,
    # 若被降级到 1.x,与 torch2.8 的组合我方从未验证过。
    try:
        import numpy
        if numpy.__version__.startswith("1."):
            report("warn", f"numpy {numpy.__version__} 是 1.x",
                   "我方只在 numpy 2.4.4 + torch2.8 上验证过。若这是被 "
                   "`pip install -r requirements.txt` 降级导致的,请报告。")
    except Exception:
        pass

    # ── [2] 离线权重(缺失=静默用随机主干训完,最阴的失败模式)──────────────
    section("[2] 离线预训练权重")
    for name, why in (
        ("checkpoints/convnext_nano_pretrained.safetensors", "分割主干;缺失会联网重试~70s后崩"),
        ("checkpoints/efficientnet_b2_pretrained.safetensors", "分类主干(四头挂在它上面);缺失=**全静默**随机初始化"),
    ):
        # 训练时模型按**相对 cwd** 读这两个文件,所以这里也按 cwd 查(与真实失败条件一致);
        # 若 cwd 下没有但包根下有,说明只是自检脚本被从别处调用,给出明确提示而非误报。
        at_cwd = os.path.exists(name)
        at_root = os.path.exists(os.path.join(_PKG_ROOT, name))
        if at_cwd:
            # 🔴 **存在 ≠ 可加载**:人工上传中断、或代理返回 HTML 错误页存成同名文件,
            # 都会产生"文件在、内容是垃圾"的状态。而 `pretrained_loader.load_file`
            # 失败时只打一行 [WARNING] 就**静默回退随机初始化**——自检若只查 exists,
            # 这里会显示 OK,训练却在用随机主干,35 epoch 后才发现。故真解析一次。
            size_mb = os.path.getsize(name) / 2**20
            try:
                from safetensors.torch import load_file as _load_st
                _n = len(_load_st(name))
                report("ok", f"{name}  ({size_mb:.1f} MB, {_n} 个张量,可加载)")
            except ImportError:
                report("warn", f"{name} 存在({size_mb:.1f} MB)但 safetensors 未安装,无法验内容",
                       "训练时才会暴露损坏。`pip install safetensors` 后重跑本项。")
            except Exception as e:
                report("fail", f"{name} 存在但**无法解析**({size_mb:.1f} MB)",
                       f"{type(e).__name__}: {e}\n"
                       f"多半是上传中断/传成了错误页。{why}\n"
                       f"→ 注意:训练时这个错误会被吞成一行 WARNING 然后**随机初始化**"
                       f"照常跑完,不会崩。必须重传这个文件。")
        elif at_root:
            report("fail", f"{name} 不在当前工作目录下(但包根 {_PKG_ROOT} 下有)",
                   f"{why}\n→ 训练必须 `cd {_PKG_ROOT}` 后再启动,否则加载不到权重。")
        else:
            report("fail", f"缺 {name}", f"{why}\n包根 {_PKG_ROOT} 下也没有 —— 文件没上传全?")

    # ── [2b] warm-start 底座(只在传了 --resume-from 时查)───────────────────
    # 位置在数据检查**之前**是有意的:底座校验与数据路径无关,而 [3]/[4]/[5] 任一
    # 失败都会提前 return —— 放在后面会导致"数据没就位时底座永远验不到",而那恰恰是
    # 最该先验掉的(数据可以等,底座传错了得重新上传)。
    # 训练里 `load_warm_start` 在**建完模型之后**才调用(train_official.py:439),
    # 底座与架构不匹配会 RuntimeError —— 但那时已花了几分钟建模型 + 装数据。
    # 这里用同一套判据(strict=False 后 missing_keys 是否 ⊆ 四头前缀)提前验一遍。
    if args.resume_from:
        section("[2b] warm-start 底座 checkpoint")
        try:
            import torch
            from src.models.multi_task_model import MultiTaskThyroidModel
            if not os.path.exists(args.resume_from):
                report("fail", f"底座文件不存在:{args.resume_from}")
            else:
                size_mb = os.path.getsize(args.resume_from) / 2**20
                state = torch.load(args.resume_from, map_location="cpu", weights_only=False)
                if isinstance(state, dict) and "model" in state \
                        and "location_head.0.weight" not in state:
                    state = state["model"]          # 与 load_warm_start 同款拆包
                report("ok", f"可加载({size_mb:.1f} MB,{len(state)} 个键)")
                # pretrained=False:只比键集,不联网、不读 safetensors
                m = MultiTaskThyroidModel(pretrained=False)
                r = m.load_state_dict(state, strict=False)
                head_pfx = MultiTaskThyroidModel.OFFICIAL_HEAD_PREFIXES
                bad = [k for k in r.missing_keys if not k.startswith(head_pfx)]
                if bad:
                    report("fail", f"底座缺 {len(bad)} 个**非四头**键 → 训练会 RuntimeError 拒载",
                           f"前 8 个:{bad[:8]}\n"
                           f"说明这个 checkpoint 与当前架构不匹配,换一个或去掉 --resume-from。")
                else:
                    report("ok", f"键集匹配(四头 {len(r.missing_keys)} 键随机初始化,属预期)"
                                 + (f";另有 {len(r.unexpected_keys)} 个多余键会被忽略"
                                    if r.unexpected_keys else ""))
        except Exception:
            import traceback
            report("fail", "底座校验抛异常", traceback.format_exc())

    # ── [3] 数据路径 ────────────────────────────────────────────────────
    section("[3] 官方数据路径")
    root = args.data_root or DEFAULT_ROOT
    if os.path.isdir(root):
        report("ok", f"数据根目录 {root}")
    else:
        report("fail", f"数据根目录不存在:{root}")
        for probe in ("/2026aicompetition", "/2026aicompetition/datasets"):
            if os.path.isdir(probe):
                report("info", f"但 {probe} 存在,内容:",
                       "\n".join(sorted(os.listdir(probe))[:20]))
        return summarize()

    subdirs = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    report("ok" if subdirs else "fail", f"一级子目录(PatientId)共 {len(subdirs)} 个",
           f"前 3 个:{subdirs[:3]}" if subdirs else "空目录")

    # 🔴 磁盘余量 —— 35 epoch × 每个 checkpoint ~117MB,加日志与临时文件需 ≥6GB。
    #    满盘的表现是训练跑到第一次存 checkpoint 才崩(几十分钟后),而不是启动即失败。
    try:
        import shutil
        free_gb = shutil.disk_usage(root).free / 2**30
        if free_gb < 6:
            report("fail", f"磁盘剩余仅 {free_gb:.1f} GB",
                   "35 epoch 的 checkpoint + 日志约需 6GB。现在清理,别等训到一半崩。")
        elif free_gb < 15:
            report("warn", f"磁盘剩余 {free_gb:.1f} GB", "够跑但不宽裕,留意 checkpoint 累积。")
        else:
            report("ok", f"磁盘剩余 {free_gb:.1f} GB")
    except Exception as e:
        report("warn", "磁盘余量探测失败", str(e))

    # 🔴 JSONL 日志目录可写性 —— V2.1 强制路径,`JsonlLogger.__init__` 才会 makedirs+open,
    #    而那发生在建模型、warm-start、跑完一轮 baseline validate **之后**。
    _log_dir = args.log_dir or "/2026aicompetition/workspace/logs"
    try:
        os.makedirs(_log_dir, exist_ok=True)
        _probe = os.path.join(_log_dir, ".preflight_write_probe")
        with open(_probe, "w") as f:
            f.write("ok")
        os.remove(_probe)                       # 只读原则:探针立即删除,不留痕
        report("ok", f"日志目录可写 {_log_dir}")
    except Exception as e:
        report("fail", f"日志目录不可写 {_log_dir}",
               f"{type(e).__name__}: {e}\n"
               f"训练要跑完一轮 validate 才会碰它,那时才崩 → 现在就修。\n"
               f"若这不是正式训练(试训),可给 train_official.py 传 --log-dir logs 绕开。")

    xlsx = args.xlsx
    if not xlsx:
        cands = glob.glob(os.path.join(root, "*.xlsx")) + \
                glob.glob(os.path.join(os.path.dirname(root), "*.xlsx")) + \
                glob.glob(os.path.join(root, "**", "*.xlsx"), recursive=True)[:10]
        cands = [c for c in dict.fromkeys(cands) if not os.path.basename(c).startswith("~$")]
        if cands:
            xlsx = cands[0]
            report("info", f"自动选用 xlsx:{xlsx}",
                   f"全部候选:\n" + "\n".join(cands) if len(cands) > 1 else "")
        else:
            report("fail", "未找到任何 xlsx", "请用 --xlsx 显式指定")
            return summarize()
    if os.path.exists(xlsx):
        report("ok", f"标注表 {xlsx}  ({os.path.getsize(xlsx) / 2**20:.1f} MB)")
    else:
        report("fail", f"标注表不存在:{xlsx}")
        return summarize()

    # ── [4] 表头与列名命中(2026-08-14 崩溃点,最关键)──────────────────────
    section("[4] 表头解析与列名命中  ← 上次就是崩在这")
    try:
        import pandas as pd
        xl = pd.ExcelFile(xlsx)
        report("ok", f"sheet 列表:{xl.sheet_names}")
        sheet = args.sheet if args.sheet in xl.sheet_names else xl.sheet_names[0]
        if sheet != args.sheet:
            report("warn", f"指定的 sheet `{args.sheet}` 不存在,改用 `{sheet}`")

        # header=1:官方 xlsx 是合并单元格表头,真实列名在第2行
        df = pd.read_excel(xlsx, sheet_name=sheet, header=1, dtype=str)
        report("ok", f"读表成功(header=1):{df.shape[0]} 行 × {df.shape[1]} 列")

        print("\n       ── 全部真实列名(逐个 repr,暴露前导空格/全角字符)──")
        for i, c in enumerate(df.columns):
            print(f"       {i:3d}  {c!r}")

        from src.dataset import official_loader as O
        colmap = O._resolve_columns(df.columns)

        print("\n       ── 逻辑字段 → 命中的真实列名 ──")
        CRITICAL = ("patient_id", "roi_name")
        SUPERVISION = ("malignancy", "lesion_loc", "extrathy_ext", "clin_t_stage", "tirads")
        missed = []
        for logical, col in colmap.items():
            if col is not None:
                report("ok", f"{logical:16s} → {col!r}")
            elif logical in CRITICAL:
                report("fail", f"{logical:16s} → 未命中(**训练无法启动**)",
                       f"候选名 {O._FIELD_ALIASES[logical]}")
                missed.append(logical)
            elif logical in SUPERVISION:
                report("fail", f"{logical:16s} → 未命中(**该头全程零监督且不报错**)",
                       f"候选名 {O._FIELD_ALIASES[logical]}")
                missed.append(logical)
            else:
                report("warn", f"{logical:16s} → 未命中(可选列,不影响四头训练)")

        # 🔴 未命中时直接把修复方案算好、打印成可粘贴的代码 —— 让你能当场自己改,
        #    不必回传等开发者。见文末「你可以自己改什么」。
        if missed:
            _suggest_alias_fix(O, df.columns, missed, df)

        # 命中列的取值分布 —— 确认解析器认得这些写法
        print("\n       ── 监督列取值分布(前 8 种)──")
        for logical in SUPERVISION:
            col = colmap.get(logical)
            if col is None:
                continue
            vc = df[col].value_counts(dropna=True)
            miss = df[col].isna().sum()
            top = ", ".join(f"{v!r}×{n}" for v, n in list(vc.items())[:8])
            print(f"       {logical:16s} 非空 {len(df) - miss}/{len(df)}  缺失 {miss / len(df) * 100:.1f}%")
            print(f"       {'':16s} {top}")
    except Exception as e:
        import traceback
        report("fail", "表头解析失败", traceback.format_exc())
        return summarize()

    # ── [5] 样本构建 dry-run(真正的端到端验证)──────────────────────────
    section("[5] 样本构建 dry-run  ← 能不能真的拼出训练样本")
    try:
        samples = O.build_official_samples(root, xlsx, sheet)
        if not samples:
            report("fail", "构造出 0 个样本", "xlsx 有数据但一个都没落地,查上面的路径/命中")
            return summarize()

        n_dyn = sum(1 for s in samples if s.get("is_dynamic"))
        # is_main 在 labels 子字典里(不是顶层),取错会恒为 0。
        n_main = sum(1 for s in samples if s.get("labels", {}).get("is_main"))
        report("ok", f"构造出 {len(samples)} 个样本",
               f"静态 {len(samples) - n_dyn} / 动态 {n_dyn};主病灶(M) {n_main}")

        # 我方既有记录:主表约 7900 行、主病灶 4877 行、静态命中率 94% → 合理量级 4000~5500。
        # 只在"表看起来是官方规模"时才判量级,否则小型测试表会被误报。
        if len(df) > 1000 and not (2000 <= len(samples) <= 9000):
            report("warn", f"样本数 {len(samples)} 超出预期量级(4000~5500)",
                   "列名可能全命中但数据行错位(如官方换了表头结构),请报告这个数字")

        print("\n       ── 各监督头的有效标签数(valid=1 才产生梯度)──")
        for key, label in (("valid_malignancy", "良恶性"), ("valid_lesion_loc", "位置"),
                           ("valid_extrathy_ext", "ETE"), ("valid_clin_t_stage", "临床T"),
                           ("valid_tirads", "TI-RADS")):
            n = sum(1 for s in samples if s.get("labels", {}).get(key))
            pct = n / len(samples) * 100
            # 阈值用**绝对数**兜底,不只看百分比:样本总数少时百分比会虚高,
            # 而 200 条以下的监督量对 35 epoch 的训练等于没训、验证也无区分力。
            # 已知真实量级:良恶性 ~4784、其余三头 ~4015-4030、TI-RADS ~4740。
            if n == 0:
                lvl, why = "fail", "**零有效标签 = 该头训了等于没训**(列名命中但解析全失败?看上面的取值分布)"
            elif n < 200:
                lvl, why = "fail", f"有效标签仅 {n} 条,不足以训练(真实量级应是 4000+),多半是解析器没认出这列的写法"
            elif pct < 10:
                lvl, why = "warn", "占比偏低,确认是否符合预期缺失率"
            else:
                lvl, why = "ok", ""
            report(lvl, f"{label:8s} 有效 {n:5d}/{len(samples)} ({pct:.1f}%)", why)
    except Exception as e:
        import traceback
        report("fail", "样本构建抛异常", traceback.format_exc())

    return summarize()


def summarize() -> int:
    section("自检结论")
    print(f"       OK {ok_count} / WARN {warn_count} / FAIL {fail_count}")
    if fail_count:
        print("""
       🔴 有 FAIL 项,**先别启动训练**。按下面分流:

       ① 列名未命中 → 上面「🔧 修复建议」已经把最可能的真实列名排好序了。
          确认语义后**自己改** src/dataset/official_loader.py 的 _FIELD_ALIASES,
          改完重跑本脚本,全绿即可开训 —— 不用回传等我。
          (只改候选列名字符串;别动其它逻辑。)

       ② 缺 safetensors / CUDA 不可用 / 数据路径找不到 → 是环境或上传问题,
          按提示里的路径自查,多半是工作目录不对或文件没传全。

       ③ 有效标签数为 0、样本数量级异常、或修复建议里没有任何相似列 →
          这类说明我方对数据的理解有偏差,**把本脚本完整输出发回给我**,
          尤其是 [4] 里的全部真实列名列表,那是修复的唯一依据。""")
        return 1
    if warn_count:
        print("\n       🟡 有 WARN 项,可以训练。把输出发回来我看一眼即可,不必等回复。")
        return 0
    print("\n       ✅ 全部通过,直接启动训练。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
