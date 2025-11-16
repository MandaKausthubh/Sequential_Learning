import torch
import numpy as np
from pathlib import Path
from hessian_eigenthings import compute_hessian_eigenthings


def store_eigens(
    model: torch.nn.Module,
    data_loader,
    loss_fn,
    save_path: str,
    top_k: int = 10,
    max_iters: int = 100,
    full_dataset: bool = True,
    max_possible_gpu_samples: int = 1024,
    fp16: bool = False,
    task_name: str = "task1",
    task_id: int = 1,
):
    """
    Compute top-k Hessian eigenpairs for *trainable* parameters of `model`
    using hessian_eigenthings, then store all relevant data in .npz.
    """

    model.eval()

    # ---------------------------------------------------------
    # 1. Identify the EXACT parameter set that HessianEigenthings will use
    # ---------------------------------------------------------
    # These params are exactly what eigenvectors correspond to.
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found (requires_grad=True).")

    # Total params = flatten of all trainable parameters
    n_params = sum(p.numel() for p in trainable_params)

    # ---------------------------------------------------------
    # 2. Compute Hessian eigenthings (top-k)
    # ---------------------------------------------------------
    eigenvalues, eigenvectors = compute_hessian_eigenthings(
        model=model,
        dataloader=data_loader,
        loss=loss_fn,
        num_eigenthings=top_k,
        full_dataset=full_dataset,
        mode="lanczos",
        max_iters=max_iters,
        max_possible_gpu_samples=max_possible_gpu_samples,
        fp16=fp16,
    )

    # eigenvectors: list of k flattened 1D tensors of length n_params
    eigvec_tensors = [v.detach().cpu().reshape(-1) for v in eigenvectors]

    # Q matrix = stack eigenvectors as columns
    Q = torch.stack(eigvec_tensors, dim=1).cpu().numpy()
    Q_fp16 = Q.astype(np.float16)  # store compactly

    # eigenvalues (store in fp32 for precision)
    Lambda_fp32 = torch.as_tensor(eigenvalues, dtype=torch.float32).cpu().numpy()

    # ---------------------------------------------------------
    # 3. Build param_slices and param_names
    # ---------------------------------------------------------
    # MUST match trainable_params ordering.
    param_slices = []
    param_names = []
    offset = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        numel = p.numel()
        start, end = offset, offset + numel
        param_slices.append((start, end))
        param_names.append(name)
        offset += numel

    assert offset == n_params, "Eigenvector length does not match flattened params."

    param_slices = np.asarray(param_slices, dtype=np.int64)
    param_names = np.asarray(param_names, dtype=object)

    # ---------------------------------------------------------
    # 4. Metadata
    # ---------------------------------------------------------
    dtype_info = {
        "Q_dtype": "float16",
        "Lambda_dtype": "float32",
        "n_params": int(n_params),
        "top_k": int(top_k),
    }

    settings = {
        "top_k": int(top_k),
        "max_iters": int(max_iters),
        "full_dataset": bool(full_dataset),
        "max_possible_gpu_samples": int(max_possible_gpu_samples),
        "fp16_used_in_hvp": bool(fp16),
    }

    task_info = {
        "task_id": int(task_id),
        "task_name": str(task_name),
    }

    # ---------------------------------------------------------
    # 5. Save to compressed .npz
    # ---------------------------------------------------------
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        Q_fp16=Q_fp16,
        Lambda_fp32=Lambda_fp32,
        param_slices=param_slices,
        param_names=param_names,
        dtype_info=dtype_info,
        settings=settings,
        task_info=task_info,
    )

    print(f"[store_eigens] Saved Hessian eigenspace to {save_path}")
