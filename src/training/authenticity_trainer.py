import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _tensor_to_gray_uint8(img_tensor):
    """将 [C,H,W] 或 [H,W] float tensor (0~1) 转为灰度 uint8 HxW numpy 数组。"""
    t = img_tensor.detach().cpu()
    if t.ndim == 3:
        # 取均值合并通道
        t = t.mean(dim=0)
    arr = (t * 255.0).clamp(0, 255).numpy().astype(np.uint8)
    return arr


def _build_feature_set(dataset, max_samples, synth_ratio=0.5, label_offset=0):
    """
    遍历 dataset，提取真实图像特征 (label=0)，
    并对其中 synth_ratio 比例的图像额外合成 phantom (label=1) 和 synthetic (label=2)。
    返回 (features_list, labels_list)。
    """
    from src.gatekeeper.authenticity import extract_authenticity_features
    from src.gatekeeper.synthesis import make_phantom_like, make_synthetic_like

    n = min(len(dataset), max_samples)
    features, labels = [], []

    for i in range(n):
        try:
            item = dataset[i]
            img_tensor = item[0]  # [C,H,W] 或 [H,W]
            gray = _tensor_to_gray_uint8(img_tensor)

            # 真实图像特征
            feat_real = extract_authenticity_features(gray)
            features.append(feat_real)
            labels.append(0)

            # 对部分图像合成虚假样本，保持类别近似均衡
            if i < int(n * synth_ratio):
                seed = i + label_offset

                # Phantom 样本 (label=1)
                try:
                    phantom = make_phantom_like(gray, seed)
                    feat_ph = extract_authenticity_features(phantom)
                    features.append(feat_ph)
                    labels.append(1)
                except Exception:
                    pass

                # Synthetic 样本 (label=2)
                try:
                    synthetic = make_synthetic_like(gray, seed)
                    feat_syn = extract_authenticity_features(synthetic)
                    features.append(feat_syn)
                    labels.append(2)
                except Exception:
                    pass

        except Exception:
            continue

    return features, labels


def train_authenticity_classifier(train_dataset, val_dataset, device, save_path):
    from src.gatekeeper.authenticity import AuthenticityClassifier

    print("--> 构建真实图像特征集 (train) ...")
    train_feats, train_labels = _build_feature_set(
        train_dataset, max_samples=300, synth_ratio=0.5, label_offset=0
    )

    print("--> 构建真实图像特征集 (val) ...")
    val_feats, val_labels = _build_feature_set(
        val_dataset, max_samples=100, synth_ratio=0.5, label_offset=10000
    )

    # 若样本极少，至少保证能运行
    if len(train_feats) == 0:
        print("[WARNING] train_dataset 为空，使用随机初始化特征作为占位。")
        train_feats = [np.zeros(8, dtype=np.float32)]
        train_labels = [0]

    X_train = np.stack(train_feats, axis=0).astype(np.float32)  # (N, 8)
    y_train = np.array(train_labels, dtype=np.int64)

    # 特征标准化：以训练集均值/方差为基准
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    X_train = (X_train - feat_mean) / (feat_std + 1e-6)

    if len(val_feats) > 0:
        X_val = np.stack(val_feats, axis=0).astype(np.float32)
        y_val = np.array(val_labels, dtype=np.int64)
        X_val = (X_val - feat_mean) / (feat_std + 1e-6)
    else:
        X_val, y_val = X_train[:1], y_train[:1]

    print(
        f"--> 特征集大小: train={len(y_train)} (real={int((y_train==0).sum())}, "
        f"phantom={int((y_train==1).sum())}, synthetic={int((y_train==2).sum())}), "
        f"val={len(y_val)}"
    )

    # 转为 tensor
    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    # 初始化模型
    clf = AuthenticityClassifier(input_dim=8, num_classes=3)
    clf.to(device)

    optimizer = optim.Adam(clf.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 40
    batch_size = 64
    n_train = Xt.shape[0]

    print(f"--> 开始训练 AuthenticityClassifier，epochs={num_epochs} ...")
    for epoch in range(num_epochs):
        clf.train()
        # 每 epoch 随机打乱
        perm = torch.randperm(n_train, device=device)
        Xt_shuf = Xt[perm]
        yt_shuf = yt[perm]

        epoch_loss = 0.0
        steps = 0
        for start in range(0, n_train, batch_size):
            xb = Xt_shuf[start: start + batch_size]
            yb = yt_shuf[start: start + batch_size]

            optimizer.zero_grad()
            out = clf(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / max(steps, 1)
            print(f"    Epoch [{epoch+1:02d}/{num_epochs}] loss={avg_loss:.4f}")

    # 验证集评估
    clf.eval()
    with torch.no_grad():
        logits = clf(Xv)
        preds = logits.argmax(dim=1)
        correct = (preds == yv).sum().item()
        total = yv.shape[0]
        acc = correct / total if total > 0 else 0.0

        # 每类正确率
        for cls_id, cls_name in enumerate(["real", "phantom", "synthetic"]):
            mask = yv == cls_id
            if mask.sum() > 0:
                cls_acc = (preds[mask] == yv[mask]).sum().item() / mask.sum().item()
                print(
                    f"    class {cls_id} ({cls_name}): "
                    f"{int(mask.sum())} samples, acc={cls_acc:.3f}"
                )

    print(f"--> Val Accuracy: {correct}/{total} = {acc:.4f}")

    # 保存权重 + 特征标准化参数(推理端 predict 必须用同一组 mean/std,否则
    # 训练/推理特征分布不一致导致分类失效)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "state_dict": clf.state_dict(),
            "feat_mean": feat_mean.astype(np.float32),
            "feat_std": feat_std.astype(np.float32),
        },
        save_path,
    )
    print(f"--> Saved Authenticity Classifier weights (+norm stats) to {save_path}")
