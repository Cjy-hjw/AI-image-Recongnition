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
| `gate2_enforce.py` | PreToolUse，`git commit` 前 | **阻断式**(exit 2) | 暂存区含未过审(或标记已过期>1小时)的高风险文件 → 挡 commit，stderr 喂回模型上下文逼迫先做 Gate2。含 `CRITICAL_INFRA_EXACT_FILES` 时额外要求 `review_mode=="deep"`，`"self"` 不放行(见教训4)。**命令含 `git add` 或 `commit -a/--all` 时改用工作区全量改动作超集判定**——PreToolUse 早于 `git add` 执行，只读暂存区会被一行式 commit 绕开(见教训5)。 |
| `mark_gate2_pass.py` | 人工/模型主动调用 | CLI 脚本 | 审查通过后执行 `python .claude/hooks/mark_gate2_pass.py [--review-mode self\|deep] <files...>`，写 `.gate2_pass.json`(含 `review_mode`，1小时有效)+ 从 pending 清单摘除。 |
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

## 五次真实教训（`gate2_paths.py` 顶部 docstring 原文，此处仅索引，改动前务必读原文）

1. **（2026-07-10 早）高风险目录清单本身不全**：`train_thyroidxl.py` 在 `scripts/`、`model_runner.py` 在 `src/inference/`，两者都不在旧版清单里，hook 完全没触发。→ 扩容 `HIGH_RISK_DIR_PREFIXES` + 新增 `HIGH_RISK_SCRIPT_PATTERNS` 按文件名模式判 `scripts/` 下的脚本。
2. **（2026-07-10 晚）绝对路径 vs 相对路径不一致**：Edit/Write 传入绝对路径，`git diff --cached --name-only` 吐相对路径，两边独立 normalize 导致永不相等，出现"审查已过但 commit 仍被拦"。→ 统一收口到 `to_repo_relative()`，所有 hook 写 pending/pass 前都先转换。
3. **（2026-07-11，用户审查发现）`is_high_risk()` 无条件排除所有非 `.py` 文件**：`requirements.txt`/`Dockerfile`/`scripts/entrypoint.sh` 这类改错代价是"整个交付容器不可用"的文件，因为不是 `.py` 而被系统性排除在 Gate2 覆盖之外——当轮 `requirements.txt` 新增依赖版本锁定全程未经审查。→ 新增 `HIGH_RISK_EXACT_FILES` 精确匹配清单，`is_high_risk()` 先判精确匹配再判 `.py` 扩展名，不再用扩展名无差别放行。
4. **（2026-07-12，用户审查发现）"高风险"未区分审查深度**：entrypoint.sh 崩溃兜底改动虽标记过 Gate2，实际只走了主会话自审，未走 `/code-review`+reviewer 子agent；事后补审在真实 bash 环境实测复现一个信号转发竞态 bug（详见 `scripts/AGENTS.md` §7.1.1）。标记文件此前只记"哪些文件过了"，不记"以什么深度过的"。→ 新增 `CRITICAL_INFRA_EXACT_FILES`（entrypoint.sh/Dockerfile/`_src_tree_hash.sh`）+ `requires_deep_review()`，`mark_gate2_pass.py` 新增 `--review-mode self|deep`，`gate2_enforce.py` 对这个子集强制要求 `deep`。

5. **（2026-07-29，本会话实测发现）`git add … && git commit …` 一行式完全绕开阻断**：`gate2_enforce.py` 是 PreToolUse hook，在 Bash 命令**执行前**触发，此时同一条命令里的 `git add` 尚未运行，`git diff --cached` 看到的是**空暂存区** → `high_risk_staged` 为空 → `exit 0` 放行。同一份改动，先 `git add` 再单独 `git commit` 会正确 `exit 2` 阻断，写成一行就放行——而一行式恰恰是最常用的写法。实证：commit `1bafb84` 改了 `src/inference/gatekeeper_runner.py`+`postprocess.py`（均高风险），当时 `.gate2_pass.json` 已过期 **396 小时**，仍成功提交；事后补跑 `/code-review` 查出 6 条问题，含一条自引入的回归。`git commit -a/-am/--all` 同理（提交时才暂存）。→ 新增 `_files_for_review(cmd)`：命令含 `git add` 或 `commit -a/--all` 时，改用**工作区全量改动**（已暂存 ∪ tracked 已改 ∪ untracked）作保守超集判定，宁可多拦不可漏放；`--amend` 不含裸 `-a` 簇故不误伤。8 个用例回归验证（5 种绕过写法全拦、`--amend`/非 commit 命令不误伤）。

**方法论**：教训1-3的共性——风险判定标准本身会漏，多是"改动类型是新的、判定逻辑没跟上"（新目录/新路径格式/新文件类型），不是"忘了检查"。教训4是新的一类——判定范围本身没漏，但"过了 Gate2"这个二元状态掩盖了审查深度不够的问题。**教训5又是新的一类——判定逻辑和范围都对，但 hook 观测的时机（PreToolUse）与被观测状态的产生时机（命令执行中 `git add`）错位，导致闸门对最常见写法形同虚设**。写阻断式 hook 时必须问第三个问题：**"我读的这个状态，在 hook 触发的那一刻真的已经成立了吗？"**——PreToolUse 看到的是命令**执行前**的世界，任何"命令自己会改变、然后再被它依赖"的状态都不能直接读，必须按命令意图推算超集。

## 独立踩坑：`proxy_check.py` 阻断式 → 提醒式回退（2026-07-10）

曾短暂把"未设代理时挡 git push/fetch/gh"升级为阻断式(exit 2)，验证过程中被自己反例打脸：正则只匹配命令**字符串**里的子串，分不清是真的在执行 `git push` 还是这个词组只是出现在一条 `echo '...git push...'` 的测试脚本文本里，含此类文本的合法命令被误拦。阻断式下假阳性代价从"多余提醒"变成"合法命令被拦下"，遂回退为提醒式。**教训**：字符串正则匹配 shell 命令文本，天然无法区分"即将执行的命令"与"引号/echo 内的字面文本"，除非真正解析 shell 语法——阻断式 hook 的假阳性成本远高于提醒式，升级前先掂量这个成本。

## 修改本目录任何文件前的检查清单

1. 改 `gate2_paths.py` 的判定逻辑（`HIGH_RISK_DIR_PREFIXES`/`HIGH_RISK_SCRIPT_PATTERNS`/`HIGH_RISK_EXACT_FILES`/`CRITICAL_INFRA_EXACT_FILES`/`is_high_risk()`/`requires_deep_review()`）→ 4 个下游 hook（`gate2_reminder.py`/`gate2_escalate_block.py`/`gate2_enforce.py`/`mark_gate2_pass.py`）全部导入它，改完至少手动跑一遍 `python -c "from gate2_paths import is_high_risk, requires_deep_review; print(is_high_risk('<路径>'), requires_deep_review('<路径>'))"` 验证判定结果符合预期，不要只改完就当完事。
2. 新增/改动 pending/pass 标记文件格式 → 检查是否所有读写点都经过 `to_repo_relative()`，防止重蹈教训 2。
3. 想把某个提醒式 hook 升级为阻断式 → 先问自己"这个判定条件有没有假阳性可能"，参考 `proxy_check.py` 的教训。
4. **改任何 PreToolUse 阻断式 hook → 必须问"我读的状态，在 hook 触发那一刻真的已经成立了吗？"**——PreToolUse 看到的是命令**执行前**的世界。若被读状态会被这条命令自己改变（如 `git add` 改暂存区、`mkdir` 造目录），直接读就会漏判，必须按命令意图推算保守超集。见教训5。
5. **改完阻断式 hook 必须实测绕过写法**，不能只测"正常路径能拦"。教训5 的 hook 逻辑本身完全正确、单独喂它也确实 exit 2，漏洞只在真实调用姿势下暴露。测试时注意：测试命令字符串里若含 `git commit` 字面量，会触发 hook 拦住测试命令本身——用变量拼接（如 `"com"+"mit"`）规避。
4. 本目录改动本身**不落在** `gate2_paths.is_high_risk()` 的判定范围内（`.claude/` 不在 `HIGH_RISK_DIR_PREFIXES`/`HIGH_RISK_EXACT_FILES` 里），不会触发 Gate2 强制审查——这是有意的（hook 系统本身的改动风险跟 ML 交付代码是两类问题），但意味着改这里全靠自觉，出错只能靠"下次踩坑才发现"，改动前务必手动验证。
