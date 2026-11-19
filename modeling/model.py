"""
The classifier: a ViT with average pooling and a linear head, built exactly like MOMO's
fine-tuning model (timm VisionTransformer, global average pool + fc_norm, see kerner-lab/MOMO).

    uv run python -m modeling.model --download     # fetch the MOMO checkpoint once (needs internet)
"""
import argparse
import json
from pathlib import Path

import timm
import torch

from dataset_creation.extract import ROOT

WEIGHTS_DIR = ROOT / "checkpoints"
MOMO_REPO = "Mirali33/MOMO"
MOMO_FILE = "vit-b-16/momo.pth"
MOMO_PATH = WEIGHTS_DIR / MOMO_FILE
ARCH = "vit_base_patch16_224"
NEW_KEYS = {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}   # trained from scratch, as in MOMO


def download_momo() -> Path:
    from huggingface_hub import hf_hub_download
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=MOMO_REPO, filename=MOMO_FILE, local_dir=WEIGHTS_DIR))


def _encoder_state(path: Path) -> dict:
    """Encoder weights from an MAE-style checkpoint: drop the decoder, mask token and final norm."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k.removeprefix("module.").removeprefix("encoder."): v for k, v in state.items()}
    return {k: v for k, v in state.items()
            if not k.startswith(("decoder", "mask_token")) and not k.startswith("norm.")}


def build_model(init: str = "momo", checkpoint: Path | None = None, arch: str = ARCH,
                num_classes: int = 4, drop_path: float = 0.0) -> tuple[torch.nn.Module, dict]:
    """ViT classifier, either from MOMO weights (init="momo") or random weights (init="random").
    Returns the model and a report of what was loaded."""
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes,
                              global_pool="avg", fc_norm=True, drop_path_rate=drop_path)
    report = {"init": init, "arch": arch}
    if init == "random":
        return model, report

    path = Path(checkpoint or MOMO_PATH)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m modeling.model --download` (needs internet)")
    state = _encoder_state(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    report |= {"checkpoint": str(path), "loaded": len(state) - len(unexpected),
               "missing": sorted(missing), "unexpected": sorted(unexpected)}
    if set(missing) - NEW_KEYS or unexpected:                  # anything else means the weights did not fit
        raise RuntimeError(f"checkpoint does not match the model:\n{json.dumps(report, indent=2)}")
    torch.nn.init.trunc_normal_(model.head.weight, std=2e-5)   # MOMO / MAE head initialisation
    torch.nn.init.zeros_(model.head.bias)
    return model, report


def param_groups(model, weight_decay: float, layer_decay: float) -> list[dict]:
    """Layer-wise learning-rate decay (BEiT/MAE/MOMO): deeper blocks learn faster than early ones.
    No weight decay on biases, norms, position embedding and class token."""
    n_layers = len(model.blocks) + 1
    no_decay = set(model.no_weight_decay())
    groups = {}
    for name, p in model.named_parameters():
        if name in ("cls_token", "pos_embed") or name.startswith("patch_embed"):
            layer = 0
        elif name.startswith("blocks."):
            layer = int(name.split(".")[1]) + 1
        else:
            layer = n_layers                                    # fc_norm and head
        decay = 0.0 if (p.ndim == 1 or name in no_decay) else weight_decay
        key = (layer, decay)
        groups.setdefault(key, {"params": [], "weight_decay": decay,
                                "lr_scale": layer_decay ** (n_layers - layer)})["params"].append(p)
    return list(groups.values())


def features_and_logits(model, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled embedding (input to the head) and class scores in one pass."""
    emb = model.forward_head(model.forward_features(x), pre_logits=True)
    return emb, model.head(emb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download:
        print("saved", download_momo())
        model, report = build_model("momo")
        print(f"checkpoint fits: {report['loaded']} tensors loaded, new: {report['missing']}")