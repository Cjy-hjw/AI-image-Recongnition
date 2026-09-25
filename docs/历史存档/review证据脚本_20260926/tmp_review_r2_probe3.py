# -*- coding: utf-8 -*-
"""R2 review probe (part 3): training-observation wiring for the ROI concept head.

Does the monitored quantity actually exist, and does anything consume it?

Run: D:\\program\\Python\\anaconda\\envs\\Demo_D\\python.exe tmp_review_r2_probe3.py
"""
import inspect
import os
import re

import src.training.loop.validate_step as VS
import scripts.train_official as TO

print("=== A. validate_roi_epoch: what it returns ===")
src = inspect.getsource(TO.validate_roi_epoch)
print(src[:800])

print("\n=== B. does validate_roi_epoch / any train path run the concept loss? ===")
full = inspect.getsource(TO)
for pat in ("compute_roi_concept_losses", "roi_concepts", "concept_n_valid", "roi_concept"):
    hits = [i for i, l in enumerate(full.splitlines(), 1)
            if pat in l]
    print(f"  {pat:28} lines: {hits}")

print("\n=== C. run_roi_epoch: what it passes into compute_roi_head_losses ===")
s2 = inspect.getsource(TO.run_roi_epoch)
print(s2[s2.find("loss, detail"):s2.find("scaler.scale")])

print("\n=== D. _roi_detail_extra ===")
print(inspect.getsource(TO._roi_detail_extra))

print("\n=== E. multitask forward: criterion path with concept weight ===")
import src.models.losses.multitask as MT
sf = inspect.getsource(MT.MultiTaskLoss.forward)
i = sf.find("compute_roi_head_losses")
print(sf[i - 400:i + 700])

print("\n=== F. does validate_step (image path) ever call roi concept? ===")
vs = inspect.getsource(VS)
for pat in ("compute_roi_head_losses", "roi_concepts", "roi_labels"):
    hits = [i for i, l in enumerate(vs.splitlines(), 1) if pat in l]
    print(f"  {pat:26} lines: {hits}")

print("DONE probe3")
