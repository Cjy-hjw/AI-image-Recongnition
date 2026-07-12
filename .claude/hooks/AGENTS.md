# .claude/hooks/AGENTS.md — Claude Code Hook 系统

> 本目录不属于 ML 竞赛交付物，是 Claude Code 会话本身的自动化护栏（Gate2 审查
> 强制执行 + 若干轻量提醒）。改这里任何一个文件前，先读本文件——这套 hook
> 互相依赖同一份路径判定/标记文件格式，单点改坏会导致"审查已过但 commit 仍被
> 拦"或"高风险改动完全没触发审查"这类隐蔽故障，且往往要等下一次真的漏审了才
> 会被发现。

## Package Structure

| 文件 | 触发时机 | 类型 | 作用 |
|:---|:---|:---|:---|
| `gate2_paths.py` | 被其余 4 个 Gate2 hook 导入 | 库(非 hook) | **单一事实源**:`is_high_risk()` 路径判定、`to_repo_relative()` 路径归一化、pending/pass 标记文件读写。三处真实教训都记在本文件顶部 docstring 里(见下)。 |
| `gate2_reminder.py` | PostToolUse，Edit/Write 后 | 提醒式 | 命中高风险文件 → 写入 `.gate2_pending.json` 待审清单 + 提醒审查，待审数量越多提醒越强硬。 |
| `gate2_escalate_block.py` | PreToolUse，Edit/Write 前 | **阻断式**(exit 2) | 待审清单堆到阈值(`ESCALATE_THRESHOLD=3`)后，挡新的高风险文件 Edit/Write——只挡"堆新账"，不挡"修正清单里已在案的文件"。 |
| `gate2_enforce.py` | PreToolUse，`git commit` 前 | **阻断式**(exit 2) | 暂存区含未过审(或标记已过期>1小时)的高风险文件 → 挡 commit，stderr 喂回模型上下文逼迫先做 Gate2。 |
| `mark_gate2_pass.py` | 人工/模型主动调用 | CLI 脚本 | 审查通过后执行 `python .claude/hooks/mark_gate2_pass.py <files...>`，写 `.gate2_pass.json`(1小时有效)+ 从 pending 清单摘除。 |
| `modularity_check.py` | PostToolUse，Edit/Write 后 | 提醒式 | `src/` 下 `.py` 超 300 行提醒拆分(rules.md §6)，不阻断——行数不能机械决定该怎么拆。 |
| `ci_push_reminder.py` | PostToolUse，`git push` 后 | 提醒式 | 提醒核对 `gh run list` 的 headSha 与本轮 commit 一致(rules.md §2)。 |
| `post_edit_check.py` | PostToolUse，`.py` 改动后 | 提醒式 | `py_compile` 语法检查 + 提示对应 `tests/test_xxx.py` 精准测试命令(不自动跑全量,太慢)。 |
| `proxy_check.py` | PreToolUse，`git push/fetch/gh` 前 | 提醒式(**曾是阻断式，已回退**) | 提醒设代理，见下方"三次教训"之外的独立踩坑记录。 |

## Gate2 状态机

```
Edit/Write 高风险文件
   → gate2_reminder.py 记入 pending
   → [pending 数 ≥ 3 时] gate2_escalate_block.py 挡后续新高风险 Edit/Write
   → 人工/模型审查通过 → mark_gate2_pass.py 写 pass + 从 pending 摘除
   → git commit → gate2_enforce.py 校验 pass 覆盖暂存区全部高风险文件 → 放行
```

pass 标记 1 小时过期，防止用旧审查混过新一轮改动；pending 清单跨工具调用持久化在磁盘（`.claude/.gate2_pending.json`），不是内存态。

## 三次真实教训（`gate2_paths.py` 顶部 docstring 原文，此处仅索引，改动前务必读原文）

1. **（2026-07-10 早）高风险目录清单本身不全**：`train_thyroidxl.py` 在 `scripts/`、`model_runner.py` 在 `src/inference/`，两者都不在旧版清单里，hook 完全没触发。→ 扩容 `HIGH_RISK_DIR_PREFIXES` + 新增 `HIGH_RISK_SCRIPT_PATTERNS` 按文件名模式判 `scripts/` 下的脚本。
2. **（2026-07-10 晚）绝对路径 vs 相对路径不一致**：Edit/Write 传入绝对路径，`git diff --cached --name-only` 吐相对路径，两边独立 normalize 导致永不相等，出现"审查已过但 commit 仍被拦"。→ 统一收口到 `to_repo_relative()`，所有 hook 写 pending/pass 前都先转换。
3. **（2026-07-11，用户审查发现）`is_high_risk()` 无条件排除所有非 `.py` 文件**：`requirements.txt`/`Dockerfile`/`scripts/entrypoint.sh` 这类改错代价是"整个交付容器不可用"的文件，因为不是 `.py` 而被系统性排除在 Gate2 覆盖之外——当轮 `requirements.txt` 新增依赖版本锁定全程未经审查。→ 新增 `HIGH_RISK_EXACT_FILES` 精确匹配清单，`is_high_risk()` 先判精确匹配再判 `.py` 扩展名，不再用扩展名无差别放行。

**方法论**：这三次教训的共性——风险判定标准本身会漏，且往往是"改动类型是新的、判定逻辑没跟上"这种方式漏的（新目录/新路径格式/新文件类型），不是"忘了检查"。每次发现漏判都优先反问"判定规则本身的分类维度够不够全"，而不是只加一条特例了事。

## 独立踩坑：`proxy_check.py` 阻断式 → 提醒式回退（2026-07-10）

曾短暂把"未设代理时挡 git push/fetch/gh"升级为阻断式(exit 2)，验证过程中被自己反例打脸：正则只匹配命令**字符串**里的子串，分不清是真的在执行 `git push` 还是这个词组只是出现在一条 `echo '...git push...'` 的测试脚本文本里，含此类文本的合法命令被误拦。阻断式下假阳性代价从"多余提醒"变成"合法命令被拦下"，遂回退为提醒式。**教训**：字符串正则匹配 shell 命令文本，天然无法区分"即将执行的命令"与"引号/echo 内的字面文本"，除非真正解析 shell 语法——阻断式 hook 的假阳性成本远高于提醒式，升级前先掂量这个成本。

## 修改本目录任何文件前的检查清单

1. 改 `gate2_paths.py` 的判定逻辑（`HIGH_RISK_DIR_PREFIXES`/`HIGH_RISK_SCRIPT_PATTERNS`/`HIGH_RISK_EXACT_FILES`/`is_high_risk()`）→ 4 个下游 hook（`gate2_reminder.py`/`gate2_escalate_block.py`/`gate2_enforce.py`/`mark_gate2_pass.py`）全部导入它，改完至少手动跑一遍 `python -c "from gate2_paths import is_high_risk; print(is_high_risk('<改动涉及的具体路径>'))"` 验证判定结果符合预期，不要只改完就当完事。
2. 新增/改动 pending/pass 标记文件格式 → 检查是否所有读写点都经过 `to_repo_relative()`，防止重蹈教训 2。
3. 想把某个提醒式 hook 升级为阻断式 → 先问自己"这个判定条件有没有假阳性可能"，参考 `proxy_check.py` 的教训。
4. 本目录改动本身**不落在** `gate2_paths.is_high_risk()` 的判定范围内（`.claude/` 不在 `HIGH_RISK_DIR_PREFIXES`/`HIGH_RISK_EXACT_FILES` 里），不会触发 Gate2 强制审查——这是有意的（hook 系统本身的改动风险跟 ML 交付代码是两类问题），但意味着改这里全靠自觉，出错只能靠"下次踩坑才发现"，改动前务必手动验证。
