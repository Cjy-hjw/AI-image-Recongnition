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
        print("✔ GATE 1: Shape verification passed.")
        return True
    except Exception as e:
        print(f"✘ GATE 1 Failed: {e}")
        return False
