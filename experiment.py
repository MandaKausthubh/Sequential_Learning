import os
import time
import yaml
import argparse
import csv
import warnings

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from peft import LoraConfig

# Import your model and datasets
# Adjust import paths as needed for your project
from models.ViT_Base32 import ViTBaseModel
from datasets.VisionDatasets import (
    ImageNet, ImageNetV2MF, ImageNetA, ImageNetR, ImageNetSketch
)

from utils.accumulate import accumulate_subspaces_fast


# ------------------------
# Helpers
# ------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(task_cfg: dict, global_cfg: dict):
    """Factory for datasets defined in the YAML."""
    ds_type = task_cfg["type"]
    root = os.path.expandvars(task_cfg["root"])
    class_map = task_cfg.get("class_map_path", task_cfg.get("class_map_path", None))
    split = task_cfg.get("split", None)

    # We return a dataset instance (not a dataloader) so caller can wrap it
    if ds_type == "ImageNet":
        assert split is not None, "ImageNet requires split in YAML"
        return ImageNet(root=root, split=split, transform=None)
    elif ds_type == "ImageNetV2MF" or ds_type == "ImageNetV2" or ds_type == "ImageNetV2MF":
        # For legacy support accept several type names
        return ImageNetV2MF(root=root, transform=None)
    elif ds_type == "ImageNetA":
        assert class_map is not None, "ImageNetA requires class_map_path"
        return ImageNetA(root=root, class_map_path=class_map, transform=None)
    elif ds_type == "ImageNetR":
        assert class_map is not None, "ImageNetR requires class_map_path"
        return ImageNetR(root=root, class_map_path=class_map, transform=None)
    elif ds_type == "ImageNetSketch" or ds_type == "Sketch":
        assert class_map is not None, "ImageNetSketch requires class_map_path"
        return ImageNetSketch(root=root, class_map_path=class_map, transform=None)
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")


def safe_get_subspace(model, task_name: str, rank_k: int, device: torch.device):
    """
    Try to obtain task Hessian subspace (Q_new, L_new) from model.
    1) call model.compute_task_subspace(task_name, rank_k)
    2) call model.estimate_hessian_subspace(task_name, rank_k)
    3) fallback: produce a random orthonormal Q and small decreasing eigenvalues.
    NOTE: The user should implement (1) or (2) in the model to compute real subspace.
    """
    # prefer user-implemented methods
    for method_name in ("compute_task_subspace", "estimate_hessian_subspace", "compute_hessian_subspace"):
        fn = getattr(model, method_name, None)
        if callable(fn):
            print(f"[subspace] Using model.{method_name} to estimate Q_new,L_new")
            Q_new, L_new = fn(task_name=task_name, rank_k=rank_k)
            # move to device
            Q_new = Q_new.to(device)
            L_new = L_new.to(device)
            return Q_new, L_new

    # fallback: create random orthonormal Q_new
    warnings.warn(
        "No subspace estimation method found on model. "
        "Using random orthonormal fallback for Q_new/L_new. "
        "Replace this with your model.compute_task_subspace implementation for real experiments."
    )
    # n should match number of tunable parameters used by nostalgia: try to fetch model parameter count
    n_params = sum(p.numel() for p in model._get_trainable_params())
    if n_params == 0:
        raise RuntimeError("Model has zero trainable params; cannot create fallback subspace.")

    # Create random Q of shape (n_params, rank_k)
    rank_k = min(rank_k, n_params)
    # torch.manual_seed(int(time.time()) % (2 ** 32))
    Q_rand = torch.randn(n_params, rank_k, device=device, dtype=torch.float32)
    Q_new, _ = torch.linalg.qr(Q_rand)  # orthonormal columns
    # Synthetic eigenvalues (descending)
    L_new = torch.linspace(1.0, 0.1, steps=rank_k, device=device, dtype=torch.float32)
    return Q_new, L_new


def write_results_csv(out_path: str, rows: list):
    header = None
    if os.path.exists(out_path):
        # append
        with open(out_path, "a", newline="") as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow(r)
        return

    # new file
    if len(rows) == 0:
        return
    header = ["experiment", "task", "epoch", "metric", "value", "time"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)


def collate_fn(batch):
    images = [item["image"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"image": images, "label": labels}


# ------------------------
# Main experiment runner
# ------------------------

def run_experiment(cfg: dict):
    exp_name = cfg.get("experiment_name", f"exp_{int(time.time())}")
    out_dir = cfg.get("output_dir", "./exp_outputs")
    os.makedirs(out_dir, exist_ok=True)

    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    set_seed(cfg.get("seed", 0))

    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("model_name", "google/vit-base-patch16-224")
    use_peft = bool(model_cfg.get("use_peft", False))
    peft_raw = model_cfg.get("peft", None)
    peft_config = None if not use_peft or peft_raw is None else LoraConfig(
            r=peft_raw.get("r", 8),
            lora_alpha=peft_raw.get("lora_alpha", 16),
            lora_dropout=peft_raw.get("lora_dropout", 0.0),
            bias=peft_raw.get("bias", "none"),
            target_modules=peft_raw.get("target_modules", []),
        )
    user_nostalgia = bool(model_cfg.get("user_nostalgia", False))
    lanczos_r = int(model_cfg.get("lanczos_r", 16))

    # instantiate model
    print(f"[model] Instantiating model {model_name} use_peft={use_peft} nostalgia={user_nostalgia}")
    model = ViTBaseModel(
        model_name=model_name,
        lanczos_r=lanczos_r,
        use_peft=use_peft,
        peft_config=peft_config,
        user_nostalgia=user_nostalgia,
    )
    model.to(device)

    # Add a 1000-class head by default (ImageNet). If you have varied class counts,
    # you can adapt this to read per-task num_classes from YAML.
    model.add_task("imagenet", 1000)
    # It's fine to add heads lazily as needed per dataset below as well.

    data_cfg = cfg.get("data", {})
    batch_size = int(data_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 4))

    trainer_cfg = cfg.get("trainer", {})
    trainer_kwargs = {}
    # map some common trainer args
    if trainer_cfg.get("gpus") is not None:
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = int(trainer_cfg["gpus"])
    if trainer_cfg.get("precision") is not None:
        trainer_kwargs["precision"] = int(trainer_cfg["precision"])

    # instantiate a single trainer that can be reused per-task
    # Note: per-task epochs are controlled by the YAML task entries
    trainer = Trainer(
        logger=pl.loggers.CSVLogger(save_dir=out_dir, name=exp_name),
        max_epochs=trainer_cfg.get("max_epochs", None),
        **trainer_kwargs
    )

    # Nostalgia/accumulate state
    nostalgia_cfg = cfg.get("nostalgia", {})
    rank_k = int(nostalgia_cfg.get("rank_k", 32))
    acc_device = torch.device(cfg.get("device", "cpu"))

    Q_accum = None
    L_accum = None

    results_rows = []

    # iterate tasks
    for task in cfg.get("tasks", []):
        task_name = task["name"]
        ds_type = task["type"]
        print(f"\n=== Running task {task_name} (type={ds_type}) ===")

        # build dataset and dataloader
        dataset = build_dataset(task, cfg)
        num_classes = task.get("num_classes", 1000)  # default to 1000
        # add or overwrite head for this task
        if task_name not in model.task_dict:
            model.add_task(task_name, num_classes)
        model.set_current_task_head(task_name)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn,
        )

        # per-task training epochs
        epochs = int(task.get("epochs", 1))
        # Train: using Trainer.fit (Lightning)
        print(f"[train] Fitting task={task_name} epochs={epochs} bs={batch_size}")
        # If the YAML specifies per-task max epochs that differ from trainer, we temporarily override
        # trainer.max_epochs = epochs # type: ignore
        trainer.fit(model, train_loader, ckpt_path=None)  # type: ignore

        # After training: estimate task subspace Q_new, L_new
        print("[subspace] Estimating task subspace...")
        Q_new, L_new = safe_get_subspace(model, task_name=task_name, rank_k=rank_k, device=acc_device)

        # Accumulate with previous Q_accum/L_accum
        print("[accumulate] Merging subspaces...")
        Q_accum, L_accum, diag = accumulate_subspaces_fast(
            Q_old=Q_accum,
            L_old=L_accum,
            Q_new=Q_new,
            L_new=L_new,
            rank_k=rank_k,
            device=acc_device,
            use_cpu_for_eig=nostalgia_cfg.get("use_cpu_for_eig", False),
            use_svd_fallback=nostalgia_cfg.get("use_svd_fallback", False),
            eps=nostalgia_cfg.get("eps", 1e-12),
            verbose=1,
        )
        # Save accumulated subspace into model for NostalgiaOptimizer usage
        model.nostalgia_Q = Q_accum
        model.nostalgia_scaling = L_accum

        # Reconfigure optimizers if necessary (Lightning will call configure_optimizers when needed).
        # Evaluate on validation/test splits if present in YAML 'eval' field
        if "eval" in task:
            # user-specified eval sets (list of dataset dicts)
            for eval_set in task["eval"]:
                eval_ds = build_dataset(eval_set, cfg)
                eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                print(f"[eval] Running evaluation on {eval_set.get('name', 'eval_set')}")
                metrics = trainer.validate(model, dataloaders=eval_loader, verbose=False)  # returns list of dicts
                # Write metrics to CSV
                tstamp = time.time()
                for m in metrics:
                    for k, v in m.items():
                        results_rows.append([exp_name, task_name, epochs, k, float(v), tstamp])
        else:
            # generic validation: try to run model validation if possible
            print("[eval] No explicit eval sets provided for this task in YAML. Skipping per-task eval.")

        # Save a checkpoint for the task
        ckpt_path = os.path.join(out_dir, f"{exp_name}_{task_name}.ckpt")
        print(f"[save] saving checkpoint to {ckpt_path}")
        trainer.save_checkpoint(ckpt_path)

    # final: run global evaluations if present in top-level config
    global_eval = cfg.get("global_eval", [])
    for eval_cfg in global_eval:
        print(f"[global_eval] Running evaluation: {eval_cfg.get('name', 'global_eval')}")
        eval_ds = build_dataset(eval_cfg, cfg)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        metrics = trainer.test(model, dataloaders=eval_loader, verbose=False)
        tstamp = time.time()
        for m in metrics:
            for k, v in m.items():
                results_rows.append([exp_name, "global_eval", "", k, float(v), tstamp])

    # write results to CSV
    out_csv = os.path.join(out_dir, f"{exp_name}_results.csv")
    write_results_csv(out_csv, results_rows)
    print(f"[done] Experiment finished. Results at {out_csv}")


# ------------------------
# CLI
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", type=str, help="Path to experiment YAML config")
    args = parser.parse_args()

    with open(args.yaml, "r") as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg)


if __name__ == "__main__":
    main()
