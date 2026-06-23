# Repository Operational Rules & Verification Protocol

## 1. Resource Pre-checks (Avoid Resource Contention)
Before launching any training script on the server, you **MUST** run:
```bash
nvidia-smi
ps aux | grep python
```
* **Rule**: If other users have active training jobs utilizing the same GPU resources (high VRAM/compute usage), notify the team or wait for resources to clear. Do not double-schedule.

## 2. Compile & Shape Verification & Code Review (Pre-flight Checks)
Before deploying or starting full training, you **MUST** execute the double-gate pre-flight checks:
1. **Gate 1: Quantifiable Code Verification (Test Scripts)**: Run `verify_model.py` to perform a single forward/backward pass with standard normalized input dummy tensors. It must successfully check tensor shape compatibility and verify valid, non-zero, non-NaN gradient generation.
2. **Gate 2: Non-quantifiable Logic Review (Subagent Review)**: For complex mathematical pipelines (such as dual minimax optimization, variational mutual information decoupling, parameter registration in optimizers, gradient leaking across subnets), the agent **MUST** spawn a specialist `reviewer` subagent to audit code changes and confirm dynamic logic correctness.
*   **Protocol Reference**: The detailed quality gates, assertion guidelines, and review templates for dynamic gradient routing, optimizer binding, and BatchNorm freeze are defined in **`QA_VP.md`**. You MUST read and enforce `QA_VP.md` as the safety sub-protocol of this rulebook.
*   **Zero-Hallucination Diagnostic Loop (自诊断闭环)**:
    * If `verify_model.py` fails any Gate (Exit Code != 0), the AI agent **MUST NOT** output standard raw stack tracebacks to the human developer. Instead, it must print a structured diagnostic report (containing the failed Gate, numerical discrepancy evidence, and recovery guidance), execute patches locally in a sandbox, and verify correctness before requesting a git diff review.
* **Rule**: Never run standard training without both Gate 1 (Test verification) and Gate 2 (Subagent review) passing first.
* **OOM Preventative Rule → `src/models/AGENTS.md` §4**: See model architecture constraints for spatial attention resolution limits.

## 3. Early Failure & Short-circuiting (Early Stopping Validation)
To prevent wasting GPU hours on diverging models, we enforce a training gatekeeper rule:
* **Rule**: If `val_dice` is `< 0.10` or classification metrics are essentially random after Epoch 3, the training script must auto-abort. Do not let dead runs finish.

## 4. GradScaler & AMP Rules → `src/training/AGENTS.md` §2
*See `src/training/AGENTS.md` §2 "GradScaler Protocol" for mixed precision rules.*
## 5. Execution & Lifecycle Management
* **Rule (Long-term Runs)**: If a training session is confirmed to be stable (i.e. model compiles, gradient flows, and metrics like Dice/Loss improve steadily over the first 5+ epochs), you **MAY** yield and stop the task session to save token resource. If it is a quick test, you must wait and monitor it inline.
* **Rule (Deep Research Gate)**: Before any major architectural iteration or complex optimization strategy change, the agent **MUST** generate a `提示词.md` and instruct the user to run a Deep Research query (via o1/o3-mini or specialized agents) to fetch the latest medical imaging constraints.
* **Rule (Deep Research Grounding)**: To prevent mathematical formulation flaws, grid/shape alignment mismatches, and numpy/pytorch type coercion bugs (e.g. bitwise ufunc errors), the agent **MUST** feed the exact relevant codebase implementation snippets (especially morphological renderers, data shapes, and tensor slicing indices) as grounding context into the Deep Research query prompt. Abstract theoretical queries are strictly discouraged.
* **Rule (Dynamic Maintenance)**: Both `rules.md` and `AGENTS.md` must be dynamically updated as dependencies, paths, or constraints evolve.

## 6. Code Modularity & Maintainability
* **Rule (Functional Decomposition)**: Every Python module must maintain a clear single responsibility. When a file exceeds 300 lines or contains more than 3 distinct functional domains (e.g., model architecture + physical priors + classification heads), it **MUST** be decomposed into separate modules.
* **Implementation Standard**: 
  - Use `__init__.py` re-export shims to maintain backward compatibility
  - Group related classes/functions by domain (e.g., `losses/segmentation.py`, `losses/classification.py`)
  - Keep training logic separate from model definitions (e.g., `src/training/` vs `src/models/`)

## 7. Official Category Constraint (Project-Level Business Rule)
* **Rule (Official Category Constraint)**: As per 2026 official regulations, Track 7 does NOT offer a "Self-built Model Group" (自建模型组). Our team is strictly registered under the **"Own Model Group" (自有模型组)**. The deployment must package our custom gatekeeper, TaskAlignmentAdaptor cross-task alignment, and ConceptEmbedding CBM report generator into a standard offline inference container via `predict.py`.
