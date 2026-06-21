# AI-Assisted Quality Assurance & Verification Protocol (QA-VP)

This document defines the production-grade safety gates for all deep learning implementations, loss functions, training pipelines, and dataset loaders in this repository.

---

## 1. NON-NEGOTIABLE INDUSTRIAL VERIFICATION STANDARDS

All deep learning implementations MUST be verified by quantifiable automated tests (e.g. `scripts/verify_model.py`):

### 1.1 Dynamic Gradient Routing Auditing (Backpropagation Isolation)
*   **Safety Risk**: When training multi-task models or using auxiliary constraint estimators (e.g., Variational CLUB for mutual information minimization, adversarial heads, or secondary decoders), gradients can leak back into the backbone encoder, leading to representation collapse or instability.
*   **Rule**: Any information-theoretic or adversarial decoupling gate MUST implement strict gradient isolation (via `.detach()`).
*   **Verification Unit Test**:
    ```python
    for name, p in model.encoder.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum() < 1e-7, f"Gradient leakage detected in parameter {name}: grad sum = {p.grad.abs().sum().item()}"
    ```

### 1.2 Optimizer Binding Checkpoints (Dual & Custom Parameters)
*   **Safety Risk**: Custom loss formulations (e.g., AUC-Margin Lagrangian dual minimization, adaptive focal parameters) introduce auxiliary parameters (like `a`, `b`, `alpha` in `AUCMLossApprox`) that must be optimized alongside model weights. If these parameters are omitted from the optimizer, frozen during training stage transitions, or updated with incorrect scaling, optimization fails silently.
*   **Rule**: All learnable variables in losses/regularization modules must be explicitly registered and verified in the active optimizer.
*   **AMP & GradScaler Safeguard**: If using `torch.cuda.amp.GradScaler`, gradients must be unscaled using `scaler.unscale_(optimizer)` before any manual gradient modification (such as inverting gradients for dual maximization: `alpha.grad.data.neg_()`) to avoid scale misalignment or NaN propagation.

### 1.3 Numerical Safety & Autocast/FP16 Guarding
*   **Safety Risk**: Custom loss functions with mathematical calculations like squares, logs, divisions, or exponentials are prone to immediate overflow (`Inf`) or underflow (`NaN`) when run in Mixed Precision (FP16), where the maximum float value is $65504$.
*   **Rule**: All custom loss formulations must be protected against out-of-range value overflows.
*   **Logit Clamping**: Clamp inputs to exponential/quadratic operations. For instance, in AUC-Margin or Asymmetric Loss, clamp logits difference to a safe interval: `logits_diff = torch.clamp(logits_diff, min=-15.0, max=15.0)`.
*   **Epsilon Guarding**: Ensure all denominators and log arguments are protected: `torch.log(prob + 1e-7)` or `denom + 1e-6`.
*   **Autocast Unit Test**: Simulate forward and backward passes under autocast using extreme inputs (e.g. large logits like `+1000.0` or `-1000.0`). Assert that the loss value and computed gradients remain finite and contain zero `NaN` or `Inf` values.

### 1.4 Data Leakage & Domain Shift Auditing
*   **Safety Risk**: Patient overlap between splits causes false confidence in metrics. Dynamic statistic recalculation on validation data leaks future statistics.
*   **Rule**: Patient IDs in train, val, and holdout splits must be completely disjoint. The dataset partition MUST be strictly grouped by the DICOM-based `PatientID` (or patient ID hashes), and never split randomly by raw image filenames or index numbers.
*   **Verification**:
    ```python
    assert set(train_patient_ids).isdisjoint(set(val_patient_ids)), "Leakage: patient ID overlap!"
    ```
    Normalization mean/std values must be statically computed on the train split only and applied as constants.

### 1.5 Loss Reduction & Shape Contracts (损失函数维度契约)
*   **Safety Risk**: Multi-task learning combines various losses (BCE, Dice, AUC-Margin, vCLUB). If sub-losses have different reduction outputs (e.g. element-wise losses of shape `[B]` vs reduced scalar losses `[]`), PyTorch will silently apply implicit broadcasting, inflating the loss tensor to `[B, B]` and computing corrupt gradients without raising errors.
*   **Rule**: All sub-losses combined in MultiTaskLoss MUST be explicitly reduced to scalars (via `.mean()` or `.sum()`) and checked before summation.
*   **Verification**:
    ```python
    # Assert each task loss is a zero-dimensional scalar tensor before addition
    for k, loss_tensor in loss_dict.items():
        assert loss_tensor.dim() == 0, f"Loss '{k}' has shape {loss_tensor.shape}, must be a 0-D scalar!"
    ```

### 1.6 PACS DICOM HIPAA Anonymization Standards (医疗数据脱敏规范)
*   **Safety Risk**: Deploying inference containers in PACS networks risks leaking HIPAA Protected Health Information (PHI) if raw DICOM headers (containing PatientName, PatientID, PatientBirthDate, InstitutionName) are written to disk or sent outside the sandbox.
*   **Rule**: Any DICOM input parsed by the prediction pipeline MUST have all HIPAA PHI headers explicitly anonymized before writing results.
*   **Required Fields to Anonymize**:
    - `PatientName` -> Set to `"ANONYMIZED"`
    - `PatientID` -> Set to `"ANONYMIZED"`
    - `PatientBirthDate` -> Set to `"00000101"`
    - `ReferringPhysicianName` -> Set to `"ANONYMIZED"`
    - `InstitutionName` -> Set to `"ANONYMIZED"`

---

## 2. SUBAGENT CODE REVIEW CHECKLIST (NON-QUANTIFIABLE LOGIC GATES)

Audit all PyTorch/training code changes against this checklist:

### 2.1 Signs and Polarity Auditing (Minimax & Objective Alignment)
*   **Minimax Dual Variables**: In minimax optimization (e.g. AUC-Margin), verify that the dual variable gradient is correctly inverted before standard optimization step:
    ```python
    alpha.grad.data.neg_()
    ```
*   **Loss Terms Sign**: Double check the sign of all terms in the multi-task loss. Maximization targets (like entropy maximization) must have a negative coefficient. Minimization targets (L1/L2 penalties, mutual information minimization bounds like vCLUB) must have a positive coefficient.

### 2.2 PyTorch Broadcasting and Shape Safeguards
*   **Implicit Broadcasting**: Explicitly call `.squeeze()`, `.unsqueeze()`, or `.view()` to make dimensions match identically before operations. Prevent `[B, 1] - [B]` implicit broadcasting to `[B, B]`.
*   **Loss Reduction Alignment**: Verify that intermediate loss functions return the expected shapes. If a custom dynamic scheduler (like FAMO) expects a list of scalar task losses, ensure each task loss is reduced (e.g., `.mean()`) before passing it to the scheduler.
*   **Broadcast Assertions**: Verify that all broadcasting is prevented by adding defensive dimensions checks:
    ```python
    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape} will trigger broadcast!"
    ```

---

## 3. COMPLIANCE & QUALITY GATE ENFORCEMENT WORKFLOW

Before any code modification is merged, submitted, or deployed:
1.  **Ruff & Pyright Validation**: Ensure no syntax, typing, or formatting errors exist.
2.  **Forward/Backward Test execution**: Run `python scripts/verify_model.py` and verify all Gate checks pass.
3.  **Checklist Audit**: Perform a self-review or spawn a `reviewer` subagent pointing to this protocol.
4.  **Zero Placeholder Rule**: Do not use placeholders or mocked outputs. All code must be fully operational.
