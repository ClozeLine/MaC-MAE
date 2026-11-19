"""
Fine-tune the ViT classifier, with MOMO's published fine-tuning settings as defaults.

    uv run python -m modeling.train --name momo_s0 --seed 0
    uv run python -m modeling.train --name momo_s0 --seed 0 --max-minutes 14   # stop in time, resume later

Rerunning the same command resumes an unfinished run from its last epoch; a finished run is
skipped. Everything is written to runs/<name>/:
    config.json             all settings, environment, checkpoint load report
    log.txt                 everything printed
    history.csv             per epoch: learning rate, losses, validation metrics, time, GPU memory
    val_epochs/epoch_XXX.npy   validation class probabilities after every epoch (float16)
    checkpoints/last.pt     full training state for resuming (model, optimiser, scaler, RNG, counters)
    checkpoints/best.pt     model weights of the epoch with the best validation macro-F1
    predictions.csv         best model on train / val / test: label, prediction, class probabilities
    embeddings.npy          best model's pooled features for every crater (float16), same row order
    metrics.json            all metrics, overall and per size bin / period
"""
import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_creation.split import DATASET_CSV
from modeling.data import CraterDataset
from modeling.features import groups
from modeling.metrics import evaluate, scores
from modeling.model import ARCH, build_model, features_and_logits, param_groups
from modeling.runs import CLASSES, mark_finished, run_dir, save_predictions, write_config

DEFAULTS = dict(                     # MOMO main_finetune.py defaults unless noted
    init="momo", arch=ARCH, checkpoint=None, size=224,
    epochs=100, patience=10,         # MOMO: patience 5; 10 is less likely to stop on noise
    batch_size=32, lr=1e-4, min_lr=1e-6, warmup_epochs=0,
    weight_decay=0.05, layer_decay=0.75, drop_path=0.0, max_grad_norm=1.0,
    label_smoothing=0.0,             # MAE fine-tuning recipe uses 0.1 (with drop_path 0.1)
    class_weights="balanced", select_on="macro_f1",   # MOMO selects on val loss; macro-F1 is our metric
    workers=8, amp=True, seed=0,
)


class Tee:
    """Print to the terminal and to log.txt."""
    def __init__(self, path):
        self.file, self.stdout = open(path, "a", buffering=1), sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id: int):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def lr_at(epoch: int, cfg: dict) -> float:
    """Half-cycle cosine after optional warmup, per epoch (MOMO lr_sched.adjust_learning_rate)."""
    if epoch < cfg["warmup_epochs"]:
        return cfg["lr"] * epoch / cfg["warmup_epochs"]
    t = (epoch - cfg["warmup_epochs"]) / (cfg["epochs"] - cfg["warmup_epochs"])
    return cfg["min_lr"] + (cfg["lr"] - cfg["min_lr"]) * 0.5 * (1 + math.cos(math.pi * t))


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """'balanced' weights, n / (k * n_c), as sklearn's compute_class_weight used by MOMO."""
    counts = np.bincount(labels, minlength=len(CLASSES))
    return torch.tensor(len(labels) / (len(CLASSES) * counts), dtype=torch.float32)


@torch.no_grad()
def predict(model, loader, device, amp, with_embeddings=False):
    """Class probabilities (and optionally embeddings) for every item, in dataset order."""
    model.eval()
    probs, embs = [], []
    for x, _, _ in loader:
        with autocast(device, amp):
            emb, logits = features_and_logits(model, x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), 1).cpu())
        if with_embeddings:
            embs.append(emb.float().cpu())
    probs = torch.cat(probs).numpy()
    return (probs, torch.cat(embs).numpy()) if with_embeddings else probs


def autocast(device, amp):
    return torch.autocast("cuda", dtype=torch.float16) if (amp and device.type == "cuda") else nullcontext()


def loader(ds, cfg, shuffle, seed=0, persistent=False):
    """Training loaders are rebuilt each epoch with an epoch-specific seed, so a resumed run
    sees the same data order as an uninterrupted one."""
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=cfg["workers"],
                      pin_memory=torch.cuda.is_available(), worker_init_fn=worker_seed, generator=g,
                      persistent_workers=persistent and cfg["workers"] > 0)


def train(cfg: dict, name: str, max_minutes: float | None):
    t_start = time.time()
    out = run_dir(name)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    (out / "val_epochs").mkdir(exist_ok=True)
    sys.stdout = Tee(out / "log.txt")

    if (out / "predictions.csv").exists():
        print(f"{name} is already finished; delete runs/{name} to retrain")
        return
    last_path, resuming = ckpt_dir / "last.pt", (ckpt_dir / "last.pt").exists()
    if resuming:
        saved = json.loads((out / "config.json").read_text())
        changed = {k: (saved.get(k, DEFAULTS.get(k)), v) for k, v in cfg.items()      # settings added later
                   if saved.get(k, DEFAULTS.get(k)) != v and k != "workers"}           # count as their default
        if changed:
            raise RuntimeError(f"settings differ from the run being resumed: {changed}")

    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(DATASET_CSV)
    tables = {s: data[data.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    train_ds = CraterDataset(tables["train"], cfg["size"], train=True)
    val_loader = loader(CraterDataset(tables["val"], cfg["size"]), cfg, shuffle=False, persistent=True)

    model, load_report = build_model(cfg["init"], cfg["checkpoint"], cfg["arch"], drop_path=cfg["drop_path"])
    model.to(device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg["weight_decay"], cfg["layer_decay"]), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    weights = class_weights(train_ds.labels) if cfg["class_weights"] == "balanced" else None
    criterion = torch.nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device),
                                          label_smoothing=cfg["label_smoothing"])   # val loss stays unsmoothed

    state = {"epoch": 0, "best": -math.inf, "best_epoch": -1, "bad_epochs": 0, "seconds": 0.0}
    if resuming:
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        if device.type == "cuda" and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        print(f"resuming {name} after epoch {state['epoch']} (best val {cfg['select_on']} "
              f"{state['best']:.4f} at epoch {state['best_epoch']})")
    else:
        env = {"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "n": {s: len(t) for s, t in tables.items()},
               "class_weights": None if weights is None else weights.tolist(), "load_report": load_report}
        write_config(out, dict(cfg, name=name, environment=env))
        print(f"training {name} on {env['gpu'] or 'CPU'}: {env['n']}")

    history_path = out / "history.csv"
    epoch_times = []
    while state["epoch"] < cfg["epochs"] and state["bad_epochs"] < cfg["patience"]:
        epoch, t0 = state["epoch"], time.time()
        lr = lr_at(epoch, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr * g["lr_scale"]

        model.train()
        train_loader = loader(train_ds, cfg, shuffle=True, seed=cfg["seed"] * 1000 + epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        total, n = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, cfg["amp"]):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            total, n = total + loss.item() * len(y), n + len(y)

        probs = predict(model, val_loader, device, cfg["amp"])
        np.save(out / "val_epochs" / f"epoch_{epoch:03d}.npy", probs.astype(np.float16))
        y_val = tables["val"].label.to_numpy()
        pred = np.array(CLASSES)[probs.argmax(1)]
        s = scores(y_val, pred, probs)
        val_loss = float(torch.nn.functional.cross_entropy(
            torch.log(torch.from_numpy(probs).clamp_min(1e-8)), torch.from_numpy(y_val - 1),
            weight=weights).item())

        improved = s[cfg["select_on"]] > state["best"]
        if improved:
            state.update(best=float(s[cfg["select_on"]]), best_epoch=epoch, bad_epochs=0)
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            state["bad_epochs"] += 1
        seconds = time.time() - t0
        epoch_times.append(seconds)
        state["epoch"], state["seconds"] = epoch + 1, state["seconds"] + seconds

        row = {"epoch": epoch, "lr": lr, "train_loss": total / n, "val_loss": val_loss,
               **{f"val_{k}": s[k] for k in ["macro_f1", "accuracy", "balanced_accuracy", "qwk", "mae", "within_one"]},
               "best": improved, "seconds": round(seconds, 1),
               "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None}
        pd.DataFrame([row]).to_csv(history_path, mode="a", header=not history_path.exists(), index=False)
        print(f"epoch {epoch:3d} | lr {lr:.2e} | train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"val macro-F1 {s['macro_f1']:.4f} | QWK {s['qwk']:.4f} | {seconds:.0f}s"
              + ("  *best" if improved else ""))

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "state": state,
                    "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                            "python": random.getstate(),
                            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}},
                   last_path)

        if max_minutes and (time.time() - t_start + max(epoch_times)) / 60 > max_minutes:
            print(f"time limit: stopping after epoch {epoch}; rerun the same command to resume")
            return

    stop = "early stopping" if state["bad_epochs"] >= cfg["patience"] else "max epochs"
    print(f"{stop} after {state['epoch']} epochs; best epoch {state['best_epoch']} "
          f"(val {cfg['select_on']} {state['best']:.4f}); predicting with the best model")

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    all_table = pd.concat([tables["train"], tables["val"], tables["test"]], ignore_index=True)
    probs, emb = predict(model, loader(CraterDataset(all_table, cfg["size"]), cfg, shuffle=False),
                         device, cfg["amp"], with_embeddings=True)
    pred = save_predictions(out, all_table.crater_id, all_table.split, all_table.label, probs)
    np.save(out / "embeddings.npy", emb.astype(np.float16))
    result = evaluate(pred, groups())
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float))
    cfg_saved = json.loads((out / "config.json").read_text())
    cfg_saved["result"] = {"stop": stop, "epochs": state["epoch"], "best_epoch": state["best_epoch"],
                           "train_hours": round(state["seconds"] / 3600, 2)}
    (out / "config.json").write_text(json.dumps(cfg_saved, indent=2, default=str))
    mark_finished(out)
    print(f"test macro-F1 {result['test']['all']['macro_f1']:.4f} | QWK {result['test']['all']['qwk']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run folder name, e.g. momo_s0")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop after the last epoch that fits in this time (resume later)")
    for k, v in DEFAULTS.items():
        kind = (lambda s: s.lower() in ("1", "true", "yes")) if isinstance(v, bool) else (type(v) if v is not None else str)
        ap.add_argument("--" + k.replace("_", "-"), type=kind, default=v)
    args = vars(ap.parse_args())
    name, max_minutes = args.pop("name"), args.pop("max_minutes")
    train(args, name, max_minutes)


if __name__ == "__main__":
    main()