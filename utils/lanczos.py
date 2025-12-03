import torch
import warnings
from torch.utils.data import DataLoader, Subset
from hessian_eigenthings import compute_hessian_eigenthings


def safe_get_subspace(
    model,
    task_name: str,
    rank_k: int,
    device: torch.device,
    *,
    subset_size: int = 10000,
    batch_size: int = 32,
    criterion=None,
    dataloader=None,
):
    """
    Compute Hessian subspace (Q_new, L_new) using:
        - model.compute_task_subspace (if provided)
        - model.estimate_hessian_subspace (if provided)
        - model.compute_hessian_subspace (if provided)
        - OTHERWISE: use compute_hessian_eigenthings (HVP + Lanczos)
        - fallback: random orthonormal Q (last resort)
    """

    # --- 0. PRIORITY: User-provided subspace method ---------------------------
    for method_name in ("compute_task_subspace", "estimate_hessian_subspace", "compute_hessian_subspace"):
        fn = getattr(model, method_name, None)
        if callable(fn):
            print(f"[subspace] Using model.{method_name} to estimate Q_new,L_new")
            Q_new, L_new = fn(task_name=task_name, rank_k=rank_k)
            Q_new = Q_new.to(device)
            L_new = L_new.to(device)
            return Q_new, L_new

    # --- 1. Use compute_hessian_eigenthings (recommended default) ------------
    if dataloader is None:
        raise ValueError(
            "safe_get_subspace requires a dataset/dataloader to compute Hessian eigenthings "
            "because no model-level subspace method was found."
        )

    if criterion is None:
        # Default to cross-entropy
        from torch.nn import CrossEntropyLoss
        criterion = CrossEntropyLoss()

    print(f"[subspace] Using compute_hessian_eigenthings (HVP + Lanczos)")

    # 1.1 Build subset for Hessian computation
    dataset = dataloader.dataset
    N = len(dataset)
    subset_size = min(subset_size, N)

    indices = torch.randperm(N)[:subset_size]
    subset = Subset(dataset, indices.tolist())
    subsetloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataloader.collate_fn if hasattr(dataloader, "collate_fn") else None,
    )

    # 1.2 Get number of parameters (must match Nostalgia optimizer)
    params = list(model._get_trainable_params())
    n_params = sum(p.numel() for p in params)

    # 1.3 Compute Hessian eigenpairs
    eigenvals, eigenvecs = compute_hessian_eigenthings(
        model=model,
        dataloader=subsetloader,
        criterion=criterion,
        num_eigenthings=rank_k,
        mode="lanczos",               # required for large models
        max_possible_gpu_samples=subset_size,
        use_gpu=(device.type == "cuda"),
    )

    # eigenvals: list/array of k scalars
    # eigenvecs: list of k "flattened" parameter vectors

    # 1.4 Stack eigenvectors into Q_new
    #     eigenvecs = list of length k, each is shape (n_params,)
    Q_new = torch.stack(eigenvecs, dim=1)   # (n_params, k)
    L_new = torch.tensor(eigenvals, dtype=torch.float32, device=device)  # (k,)

    Q_new = Q_new.to(device).float()
    Q_new, _ = torch.linalg.qr(Q_new)       # ensure orthonormal columns

    return Q_new, L_new

    # --- 2. Fallback (never used unless eigenthings fails) -------------------
    warnings.warn(
        "[subspace] compute_hessian_eigenthings failed; using random fallback"
    )

    rank_k = min(rank_k, n_params)
    Q_rand = torch.randn(n_params, rank_k, device=device, dtype=torch.float32)
    Q_fallback, _ = torch.linalg.qr(Q_rand)
    L_fallback = torch.linspace(1.0, 0.1, steps=rank_k, device=device, dtype=torch.float32)
    return Q_fallback, L_fallback
