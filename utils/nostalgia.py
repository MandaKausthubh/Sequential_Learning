import torch
from torch.optim import Optimizer
from typing import List, Optional, Tuple, Iterable


def _flatten_tensors_to_buffer(params: List[torch.Tensor], buffer: torch.Tensor) -> None:
    """
    Copy param.grad (if present) flattened into preallocated 1D buffer in-order.
    Missing grads -> zeros.
    """
    offset = 0
    for p in params:
        n = p.numel()
        if p.grad is None:
            buffer[offset: offset + n].zero_()
        else:
            # ensure we read grad as contiguous 1D
            buffer[offset: offset + n].copy_(p.grad.detach().view(-1))
        offset += n


def _scatter_buffer_to_param_grads(params: List[torch.Tensor], buffer: torch.Tensor) -> None:
    """
    Scatter flattened buffer back into p.grad for each parameter.
    We assign p.grad to a view of the buffer slice to avoid extra copies if possible.
    """
    offset = 0
    for p in params:
        n = p.numel()
        chunk = buffer[offset: offset + n]
        grad_view = chunk.view_as(p)
        p.grad = grad_view
        offset += n


class NostalgiaOptimizer:
    """
    Nostalgia optimizer wrapper.

    Usage pattern:
      base_opt = torch.optim.Adam(lora_params, lr=1e-4)
      opt = NostalgiaOptimizer(base_opt, params=list(lora_params), device=device)

    Key methods:
      - step(): call after loss.backward(); will replace .grad on params with projected grads
                and then call base_optimizer.step()
      - set_Q(Q, scale=None): set low-rank basis Q (shape [n, k]) and optional scaling (k,)
      - zero_grad(): zero the gradients (for convenience)
    """

    def __init__(self,
                 base_optimizer: Optimizer,
                 params: Iterable[torch.nn.Parameter],
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32):
        """
        base_optimizer: an already-initialized torch optimizer that was created on the same params
        params: list/iterable of parameters that belong to base_optimizer (these are the parameters
                whose gradients will be flattened and projected)
        device: device to keep flat buffers / Q on (default: device of first param)
        dtype: numeric dtype for the internal projection (float32 recommended)
        """
        self.base = base_optimizer
        self.state = self.base.state
        # store params list (preserve order)
        self.params: List[torch.Tensor] = list(params)

        if len(self.params) == 0:
            raise ValueError("params must be a non-empty iterable of parameters.")

        self.device = device or self.params[0].device
        self.dtype = dtype

        # precompute total dimension and slices
        self._slices: List[Tuple[int, int]] = []
        offset = 0
        for p in self.params:
            n = p.numel()
            self._slices.append((offset, offset + n))
            offset += n
        self.total_dim = offset

        # flat gradient buffer (single contiguous tensor)
        self.flat_grad = torch.zeros(self.total_dim, dtype=self.dtype, device=self.device)

        # low-rank basis Q (None or [n, k]) and optional scaling vector (k,)
        self.Q: Optional[torch.Tensor] = None
        self.Q_t_contig: Optional[torch.Tensor] = None
        self.scaling: Optional[torch.Tensor] = None
        self._eps = 1e-12

    def set_Q(self, Q: Optional[torch.Tensor], scaling: Optional[torch.Tensor] = None, keep_dtype: Optional[torch.dtype] = None):
        """
        Provide orthonormal basis Q with shape [n, k] (columns are eigenvectors).
        Optionally provide scaling of shape [k] which will multiply the c = Q^T g coefficients
        before reconstruction (effectively (I - Q diag(scaling) Q^T) projection).

        Q: None to disable projection.
        scaling: None or 1D tensor, length k.
        keep_dtype: if provided, cast Q to this dtype (recommended: torch.bfloat16 or float32).
        """
        if Q is None:
            self.Q = None
            self.Q_t_contig = None
            self.scaling = None
            return

        if Q.shape[0] != self.total_dim:
            raise ValueError(f"Q has {Q.shape[0]} rows but expected {self.total_dim}.")

        # move to device and dtype
        tgt_dtype = keep_dtype or self.dtype
        Q = Q.to(self.device).to(tgt_dtype)
        self.Q = Q.contiguous()
        self.Q_t_contig = self.Q.t().contiguous()

        if scaling is not None:
            s = scaling.to(self.device).to(self.Q.dtype).reshape(-1)
            if s.shape[0] != self.Q.shape[1]:
                raise ValueError("scaling length must match k (Q.shape[1])")
            self.scaling = s
        else:
            self.scaling = None

    def _project_flat_grad(self) -> None:
        """
        Replace self.flat_grad with its projected version in-place.
        Uses float32 accumulations internally for numeric safety.
        """
        if self.Q is None:
            return  # nothing to do

        # use float32 accumulation to be stable (cast Q/g temporarily if needed)
        # g is flat_grad
        g = self.flat_grad

        # ensure dtypes for matmuls: use higher precision for ops
        mat_dtype = torch.float32 if g.dtype in (torch.float16, torch.bfloat16) else g.dtype
        g_acc = g.to(mat_dtype)

        # Q^T g  -> shape [k]
        Q_t = self.Q_t_contig.to(mat_dtype)  # type: ignore
        c = torch.matmul(Q_t, g_acc)  # [k]

        if self.scaling is not None:
            c = c * self.scaling.to(mat_dtype)

        # reconstruct Q c  -> shape [n]
        Q_acc = self.Q.to(mat_dtype)
        recon = torch.matmul(Q_acc, c)  # [n]

        # subtract and write back (cast to original dtype)
        g_projected = (g_acc - recon).to(g.dtype)

        # in-place replace flat_grad (avoid new allocation if same device/dtype)
        self.flat_grad.copy_(g_projected)

    def step(self, closure=None):
        """
        Should be called after loss.backward(). This will:
          1. flatten current .grad into flat buffer
          2. project the flat gradient (in-place)
          3. scatter the projected buffer back to param.grad
          4. call base_optimizer.step()

        Returns whatever base_optimizer.step() returns (typically None)
        """
        # 1. flatten current grads into buffer (on self.device)
        # ensure buffer is on correct device & dtype
        if self.flat_grad.device != self.device or self.flat_grad.dtype != self.dtype:
            self.flat_grad = torch.zeros(self.total_dim, dtype=self.dtype, device=self.device)

        # populate flat_grad from params' .grad (this will copy)
        _flatten_tensors_to_buffer(self.params, self.flat_grad)

        # 2. project in-place
        self._project_flat_grad()

        # 3. scatter back to param.grad (we replace p.grad with a view into a contiguous tensor)
        # To avoid lifetime issues, create a contiguous copy that base optimizer won't mutate.
        # We already use self.flat_grad as contiguous; scatter uses views into it.
        _scatter_buffer_to_param_grads(self.params, self.flat_grad)

        # 4. call underlying optimizer
        return self.base.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        """
        Convenience wrapper to zero grads on the parameters (delegates to base optimizer).
        """
        return self.base.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        """
        Save base optimizer state and Q/scaling (moved to CPU to make checkpoint portable).
        """
        state = {
            "base_state": self.base.state_dict(),
            "Q": None if self.Q is None else self.Q.cpu(),
            "scaling": None if self.scaling is None else self.scaling.cpu()
        }
        return state

    def load_state_dict(self, state: dict):
        """
        Load optimizer state including Q.
        """
        self.base.load_state_dict(state["base_state"])
        Q = state.get("Q", None)
        scaling = state.get("scaling", None)
        if Q is None:
            self.set_Q(None)
        else:
            self.set_Q(Q.to(self.device), scaling=scaling.to(self.device) if scaling is not None else None)


