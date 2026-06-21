---
name: phased-ml-refactoring
description: "Ensure ML refactoring plans are split into progressive, independently verified phases instead of monolithic rewrites"
condition: "Concept Embedding Model|CEM|KAN|RLAR"
scope: ["tool:write(*plan.md)", "tool:write(*PLAN.md)", "tool:edit(*plan.md)", "tool:edit(*PLAN.md)"]
---

Never propose a monolithic rewrite that applies multiple complex ML changes (e.g., CEM, KAN, loss refactoring, path gating) in a single step. You must structure the plan into sequential, low-risk phases (e.g., Phase 1: remove consistency loss, Phase 2: gate optimization, Phase 3: CEM/MLP bottleneck). Each phase must specify running `verify_model.py` and evaluation on the Holdout set before proceeding to the next phase.