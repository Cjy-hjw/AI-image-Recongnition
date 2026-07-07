"""决定性对拍: probe(AUC0.705) vs 训练脚本 linear_head(0.50-0.55) 谁是真凶。
同一进程/同一 checkpoint/同一 z_latent 提取方式，四组对比全部打印 AUC。"""
import os, sys, time
sys.path.insert(0, "/home/cjy/AI_image_Recognition")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.dataset.pathology_dataset import (
    build_pathology_samples, patient_grouped_split, PathologyDataset)
from src.models.multi_task_model import MultiTaskThyroidModel

DEV = "cuda"
CKPT = "checkpoints/malignancy_pretrain_best.pth"
IMG = 512
BS = 24
torch.manual_seed(42); np.random.seed(42)

# ── 数据: 与训练脚本完全相同的 patient_grouped_split(seed=42, val_frac=0.2) ──
samples = build_pathology_samples("data/27021604_extract")
train_s, val_s = patient_grouped_split(samples, val_frac=0.2, seed=42)
train_ds = PathologyDataset(train_s, img_size=IMG, transform=None)  # 提特征关掉aug
val_ds   = PathologyDataset(val_s,   img_size=IMG, transform=None)
print(f"[data] train={len(train_ds)} val={len(val_ds)} "
      f"train_cls={train_ds.class_counts()} val_cls={val_ds.class_counts()}", flush=True)

# 患者不重叠自检
tr_pat = {s["patient_key"] for s in train_s}
va_pat = {s["patient_key"] for s in val_s}
print(f"[split] train_pat={len(tr_pat)} val_pat={len(va_pat)} "
      f"overlap={len(tr_pat & va_pat)}", flush=True)

config = {"data": {"img_size": IMG}}
model = MultiTaskThyroidModel(backbone_name="efficientnet_b2", pretrained=True, config=config).to(DEV)
ck = torch.load(CKPT, map_location=DEV)
sd = ck["model"] if "model" in ck else ck
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"[load] epoch={ck.get('epoch')} ckpt_val_auc={ck.get('val_auc')} "
      f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def extract(ds, bn_mode):
    """bn_mode: 'eval'=running统计(evaluate用) / 'train'=batch统计(训练前向用)。"""
    if bn_mode == "eval":
        model.eval()
    else:
        model.train()  # BN 用当前 batch 统计，等价训练时的前向
    loader = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=8, pin_memory=True)
    Z, Y = [], []
    with torch.no_grad():
        for imgs, _m, labels in loader:
            imgs = imgs.to(DEV)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(imgs)
            Z.append(out["z_latent"].float().cpu().numpy())
            Y.append(labels["malignancy"].numpy())
    return np.concatenate(Z), np.concatenate(Y)


t0 = time.time()
Ztr_e, Ytr = extract(train_ds, "eval")
Zva_e, Yva = extract(val_ds, "eval")
Ztr_t, _   = extract(train_ds, "train")
Zva_t, _   = extract(val_ds, "train")
print(f"[feat] done in {time.time()-t0:.0f}s  Ztr={Ztr_e.shape} Zva={Zva_e.shape} "
      f"train_pos={int(Ytr.sum())} val_pos={int(Yva.sum())}", flush=True)


def probe_fit_eval(Ztr, Ytr, Zva, Yva, tag):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(Ztr, Ytr)
    auc_va = roc_auc_score(Yva, clf.predict_proba(Zva)[:, 1])
    auc_tr = roc_auc_score(Ytr, clf.predict_proba(Ztr)[:, 1])
    print(f"[{tag}] train_auc={auc_tr:.4f}  val_auc={auc_va:.4f}", flush=True)
    return auc_va


def train_head(Ztr, Ytr, Zva, Yva, tag, epochs=100):
    """离线 BN1d(352)+Linear(352,2) + AdamW + CE，纯特征训练。"""
    Zt = torch.tensor(Ztr, dtype=torch.float32, device=DEV)
    Yt = torch.tensor(Ytr, dtype=torch.long, device=DEV)
    Zv = torch.tensor(Zva, dtype=torch.float32, device=DEV)
    head = nn.Sequential(nn.BatchNorm1d(352), nn.Linear(352, 2)).to(DEV)
    # 类权重(反频率)与训练脚本一致
    n0, n1 = (Yt == 0).sum().item(), (Yt == 1).sum().item()
    w = torch.tensor([len(Yt)/(2*n0), len(Yt)/(2*n1)], device=DEV)
    ce = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(head.parameters(), lr=8e-5, weight_decay=0.05)
    for ep in range(1, epochs + 1):
        head.train()
        opt.zero_grad()
        loss = ce(head(Zt), Yt)
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == 1:
            head.eval()
            with torch.no_grad():
                ptr = torch.softmax(head(Zt), 1)[:, 1].cpu().numpy()
                pva = torch.softmax(head(Zv), 1)[:, 1].cpu().numpy()
            print(f"  [{tag}] ep{ep:3d} loss={loss.item():.4f} "
                  f"train_auc={roc_auc_score(Ytr,ptr):.4f} val_auc={roc_auc_score(Yva,pva):.4f}",
                  flush=True)


print("\n========== 对拍A: probe方式 (train拟合/val评估, 无泄漏 patient split) ==========", flush=True)
print(">> BN=eval (running统计, evaluate()用的前向):", flush=True)
probe_fit_eval(Zva_e, Yva, Zva_e, Yva, "A-sanity-val-on-val-eval")  # 上界参考(过拟合)
a_eval = probe_fit_eval(Ztr_e, Ytr, Zva_e, Yva, "A-probe-eval")
print(">> BN=train (batch统计, 训练前向用的):", flush=True)
a_train = probe_fit_eval(Ztr_t, Ytr, Zva_t, Yva, "A-probe-train")

print("\n========== 对拍B: 离线模拟训练头 BN1d+Linear+AdamW+CE ==========", flush=True)
print(">> 特征用 BN=eval:", flush=True)
train_head(Ztr_e, Ytr, Zva_e, Yva, "B-head-eval")
print(">> 特征用 BN=train (与训练脚本前向一致):", flush=True)
train_head(Ztr_t, Ytr, Zva_t, Yva, "B-head-train")

print("\n========== 对拍C(关键): 复现原probe的泄漏setup — StratifiedKFold随机分val集内部 ==========", flush=True)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sc = cross_val_score(clf, Zva_e, Yva, cv=cv, scoring="roc_auc")
print(f"[C-LEAKY-probe] val内5fold随机分 mean={sc.mean():.4f} std={sc.std():.4f} "
      f"folds={np.round(sc,4).tolist()}", flush=True)

print("\n========== 对拍D(对照): 同样StratifiedKFold随机分, 但跑在无泄漏的按患者train上 ==========", flush=True)
# 说明: train集也是按患者分出来的, 但这里在 train 集"内部"再随机KFold(会泄漏),
# 与C对比看是否 train 集内部随机CV也虚高 -> 佐证KFold随机分本身制造泄漏
sc2 = cross_val_score(clf, Ztr_e, Ytr, cv=cv, scoring="roc_auc")
print(f"[D-LEAKY-probe-on-train] train内5fold随机分 mean={sc2.mean():.4f} std={sc2.std():.4f}", flush=True)

print("\n========== 结论数字汇总 ==========", flush=True)
print(f"无泄漏probe(train->val, BN eval)     : {a_eval:.4f}", flush=True)
print(f"无泄漏probe(train->val, BN train)    : {a_train:.4f}", flush=True)
print(f"泄漏probe(val内随机KFold, 复现0.705) : {sc.mean():.4f}", flush=True)
print("[DONE]", flush=True)
