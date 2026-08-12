import torch


def gate1_shapes(model, x):
    """Forward Pass & Output Shape Sanity."""
    try:
        model.eval()
        with torch.no_grad():
            out = model(x)

        assert out["mask"].shape == (2, 1, 512, 512), f"Bad mask shape: {out['mask'].shape}"
        assert out["malignancy"].shape == (2, 2), f"Bad malignancy shape: {out['malignancy'].shape}"
        assert out["composition"].shape == (2, 4), f"Bad composition shape: {out['composition'].shape}"
        assert out["echogenicity"].shape == (2, 4), f"Bad echogenicity shape: {out['echogenicity'].shape}"
        assert out["shape"].shape == (2, 2), f"Bad shape shape: {out['shape'].shape}"
        assert out["margin"].shape == (2, 4), f"Bad margin shape: {out['margin'].shape}"
        assert out["foci"].shape == (2, 4), f"Bad foci shape: {out['foci'].shape}"

        # --- Phase 2 三头接口(阶段 2.0 新增) ---
        # 用 model 常量与输入分辨率推导,而非硬编码 352/16 —— 改 backbone 或 img_size 时
        # 断言自动跟随,不会退化成"永远成立"的假门。
        _, _, H, W = x.shape
        expect_spatial = (2, model.CLS_POOLED_DIM, H // 32, W // 32)
        assert (
            out["stage4_spatial"].shape == expect_spatial
        ), f"Bad stage4_spatial shape: {out['stage4_spatial'].shape} != {expect_spatial}"
        assert out["soft_mask_detached"].shape == (
            2, 1, H, W,
        ), f"Bad soft_mask_detached shape: {out['soft_mask_detached'].shape}"
        assert out["nodule_scale_detached"].shape == (
            2, 1,
        ), f"Bad nodule_scale_detached shape: {out['nodule_scale_detached'].shape}"

        # 值域契约:soft_mask 是 sigmoid 输出、nodule_scale 是归一化等效直径,
        # 二者都必须落在 [0,1] 且有限(FP16 求和溢出会在此暴露为 inf/nan)。
        for key in ("soft_mask_detached", "nodule_scale_detached"):
            t = out[key]
            assert torch.isfinite(t).all(), f"{key} contains non-finite values"
            assert (
                t.min() >= 0.0 and t.max() <= 1.0
            ), f"{key} out of [0,1]: min={t.min().item()}, max={t.max().item()}"

        print("✔ GATE 1: Shape verification passed.")
        return True
    except Exception as e:
        print(f"✘ GATE 1 Failed: {e}")
        return False
