import torch
import time
from typing import Optional, Tuple, Dict

def _tprint(msg: str, verbose: int, level: int = 1):
    if verbose >= level:
        print(msg)




@torch.no_grad()
def accumulate_subspaces_fast(
    Q_old: Optional[torch.Tensor],
    L_old: Optional[torch.Tensor],
    Q_new: torch.Tensor,
    L_new: torch.Tensor,
    rank_k: int,
    *,
    device: Optional[torch.device] = None,
    use_cpu_for_eig: bool = False,
    use_svd_fallback: bool = False,
    eps: float = 1e-12,
    verbose: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Efficient ACCUMULATE merging of (Q_old, L_old) and (Q_new, L_new).
    Returns (Q_next, L_next, diagnostics).

    Args:
      Q_old: (n, k_old) or None
      L_old: (k_old,) or None
      Q_new: (n, k_new)
      L_new: (k_new,)
      rank_k: desired output rank (k)
    Options:
      device: where intermediate B/Q_live should live (defaults to Q_new.device)
      use_cpu_for_eig: move small S to CPU and run torch.linalg.eigh there (sometimes faster/stable)
      use_svd_fallback: if eigh fails, try SVD on S
      eps: tiny regularizer for numeric safety
      verbose: 0 = silent, 1 = essential logs, 2 = detailed timings
    Returns:
      Q_next: (n, rank_k) orthonormal columns (dtype = Q_new.dtype)
      L_next: (rank_k,) eigenvalues (dtype = L_new.dtype)
      diagnostics: dict with timings and shapes
    """

    t0 = time.time()
    dtype = Q_new.dtype
    n = Q_new.shape[0]
    device = device or Q_new.device

    # handle trivial base-case
    if Q_old is None or L_old is None:
        k = min(rank_k, Q_new.shape[1])
        Qc = Q_new[:, :k]
        Lc = L_new[:k]
        # ensure orthonormal (QR) in output dtype/device
        Qc = Qc.to(device).to(dtype).contiguous()
        Qc, _ = torch.linalg.qr(Qc) if Qc.shape[1] > 0 else (Qc, None)
        diag = {
            "m": Qc.shape[1],
            "time_total": time.time() - t0,
            "note": "base_case_only_new"
        }
        return Qc, Lc.to(dtype), diag

    # shapes
    k_old = Q_old.shape[1]
    k_new = Q_new.shape[1]
    m = k_old + k_new

    _tprint(f"[ACC] merging k_old={k_old}, k_new={k_new} -> m={m}", verbose, 1)

    # Move inputs to the compute device, ensure contiguous
    Q_old = Q_old.to(device).contiguous()
    Q_new = Q_new.to(device).contiguous()
    L_old = L_old.to(device).contiguous()
    L_new = L_new.to(device).contiguous()

    # t_prep = time.time()

    # 1) Form M = [Q_old, Q_new] but avoid copying large memory: cat then QR
    # Reduced QR: compute orthonormal basis B of M
    M = torch.cat([Q_old, Q_new], dim=1)  # (n, m)
    # If m is small (<= rank_k), we can skip some steps later, but still compute B.
    # QR in reduced mode
    t_qr0 = time.time()
    B, _ = torch.linalg.qr(M)    # B: (n, m), orthonormal
    t_qr = time.time() - t_qr0
    _tprint(f"[ACC] QR completed (n={n}, m={m}) in {t_qr:.4f}s", verbose, 2)

    # 2) Compute small matrix S = B^T (H_old + H_new) B using eigen-decompositions only
    #    Using H_old ≈ Q_old diag(L_old) Q_old^T, H_new ≈ Q_new diag(L_new) Q_new^T
    t_proj0 = time.time()
    # A_old = B^T Q_old  -> (m, k_old)
    A_old = B.T @ Q_old
    # A_new = B^T Q_new  -> (m, k_new)
    A_new = B.T @ Q_new

    # Form S (m x m)
    S = torch.zeros((m, m), device=device, dtype=torch.float32)  # work in float32 for stability
    if k_old > 0:
        S += (A_old @ torch.diag(L_old.to(torch.float32))) @ A_old.T
    if k_new > 0:
        S += (A_new @ torch.diag(L_new.to(torch.float32))) @ A_new.T
    # small regularizer for numerical stability
    S = S + eps * torch.eye(m, device=device, dtype=S.dtype)
    t_proj = time.time() - t_proj0
    _tprint(f"[ACC] Built S (m={m}x{m}) in {t_proj:.4f}s", verbose, 2)

    # 3) Eigendecompose S (tiny). Optionally move to CPU and use float32 for stability.
    t_eig0 = time.time()
    # prefer CPU if requested (useful when GPU small-m eig has overhead)
    if use_cpu_for_eig:
        cpu_S = S.cpu()
        try:
            L_s, U_small = torch.linalg.eigh(cpu_S)  # ascending order
            L_s = L_s.to(device)
            U_small = U_small.to(device)
        except Exception as e:
            _tprint(f"[ACC] CPU eigh failed: {e}", verbose, 1)
            if use_svd_fallback:
                _tprint("[ACC] Falling back to SVD on CPU", verbose, 1)
                U_small, S_svals, _ = torch.svd(cpu_S)
                # SVD returns U,S,V: here we map to eigenvectors via U
                L_s = (S_svals ** 2).to(device)
                U_small = U_small.to(device)
            else:
                raise
    else:
        try:
            # compute on device in float32 to be safe
            L_s, U_small = torch.linalg.eigh(S)  # ascending order
        except Exception as e:
            _tprint(f"[ACC] GPU eigh failed: {e}", verbose, 1)
            if use_svd_fallback:
                _tprint("[ACC] Falling back to SVD on device", verbose, 1)
                U_small, S_svals, _ = torch.svd(S)
                L_s = (S_svals ** 2)
            else:
                # try CPU fallback
                _tprint("[ACC] Falling back to CPU eigh", verbose, 1)
                cpu_S = S.cpu()
                L_s, U_small = torch.linalg.eigh(cpu_S)
                L_s = L_s.to(device)
                U_small = U_small.to(device)

    t_eig = time.time() - t_eig0
    _tprint(f"[ACC] Eigendecomposition done in {t_eig:.4f}s (m={m})", verbose, 2)

    # 4) Select top-k eigenpairs (largest)
    # note: L_s is ascending; take last rank_k elements
    k_take = min(rank_k, L_s.shape[0])
    L_top = L_s[-k_take:].to(dtype)           # (k_take,)
    U_top = U_small[:, -k_take:]              # (m, k_take)

    # 5) Reconstruct Q_next = B @ U_top  (n, k_take)
    t_recon0 = time.time()
    Q_next = (B @ U_top).to(dtype)            # cast back to original dtype
    # final orthonormalize to remove numerical drift
    if Q_next.shape[1] > 0:
        Q_next, _ = torch.linalg.qr(Q_next)
    t_recon = time.time() - t_recon0
    _tprint(f"[ACC] Reconstructed Q_next (k={k_take}) in {t_recon:.4f}s", verbose, 2)

    # diagnostics
    t_total = time.time() - t0
    diag = {
        "n": n,
        "k_old": k_old,
        "k_new": k_new,
        "m": m,
        "k_out": k_take,
        "time_total": t_total,
        "time_qr": t_qr,
        "time_proj": t_proj,
        "time_eig": t_eig,
        "time_recon": t_recon,
        "device": str(device),
        "dtype": str(dtype),
    }
    if verbose >= 1:
        _tprint(f"[ACC] Done. total={t_total:.4f}s | m={m} -> k={k_take}", verbose, 1)
    if verbose >= 2:
        _tprint(f"[ACC] diag={diag}", verbose, 2)

    return Q_next, L_top, diag


# -------------------------
# Minimal unit-check / example
# -------------------------
if __name__ == "__main__":
    # small sanity test
    n = 1024
    k_old = 8
    k_new = 8
    rank_k = 8

    # random orthonormal Q_old, Q_new
    torch.manual_seed(0)
    Qold_rand = torch.randn(n, k_old, device="cpu")
    Qold, _ = torch.linalg.qr(Qold_rand)
    Qnew_rand = torch.randn(n, k_new, device="cpu")
    Qnew, _ = torch.linalg.qr(Qnew_rand)

    Lold = torch.linspace(1.0, 0.5, k_old)
    Lnew = torch.linspace(0.8, 0.3, k_new)

    Qnext, Lnext, info = accumulate_subspaces_fast(
        Q_old=Qold, L_old=Lold, Q_new=Qnew, L_new=Lnew, rank_k=rank_k,
        device=torch.device("cpu"),
        use_cpu_for_eig=False,
        verbose=2
    )

    print("Qnext.shape:", Qnext.shape)
    print("Lnext.shape:", Lnext.shape)
    print("diag:", info)
