import os

from safetensors.torch import load_file


def load_local_safetensors(
    model, checkpoint_path, *, in_chans=1, stem_key="conv_stem.weight"
) -> bool:
    """Load local safetensors weights into *model*.

    Returns False when the checkpoint is missing or loading fails.
    Converts a 3-channel stem to single-channel when in_chans == 1.
    """
    if not os.path.exists(checkpoint_path):
        return False

    print(
        f"--> Found offline local backbone checkpoint: {checkpoint_path}. Loading weights locally."
    )
    try:
        state_dict = load_file(checkpoint_path)
        if (
            stem_key in state_dict
            and state_dict[stem_key].shape[1] == 3
            and in_chans == 1
        ):
            state_dict[stem_key] = state_dict[stem_key].sum(dim=1, keepdim=True)
        model.load_state_dict(state_dict, strict=False)
        print("--> Successfully loaded offline local backbone pretrained weights.")
        return True
    except Exception as e:
        print(
            f"[WARNING] Failed to load offline backbone weights: {e}. Falling back to default timm initialization."
        )
        return False
