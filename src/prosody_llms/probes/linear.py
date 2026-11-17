"""Linear probes for every layer from one pass, on the GPU (PyTorch).

``LinearStats`` accumulates, per layer, the feature sums, the Gram matrix and the
per-class feature sums in float64, for the 3-way labels. The 2-way statistics are exact
sums of the 3-way ones (class 1 = classes 1 + 2), so one pass serves both tasks.

``solve`` returns ridge (regression onto one-hot labels) or LDA probes for a whole grid
of regularisation values from one eigendecomposition per layer: both need the inverse
of ``a * C + b * I`` for a standardised covariance ``C``, which the eigenvectors of ``C``
give for every ``(a, b)`` at once. ``prosody_llms.probes.ridge`` is the plain NumPy
reference implementation; the tests check that both agree.
"""

from __future__ import annotations

from dataclasses import dataclass

FAMILIES = ("ridge", "lda")
DEFAULT_GRID = {
    # lambda added to the correlation matrix
    "ridge": (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    # shrinkage towards the identity
    "lda": (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.6, 0.9),
}


class LinearStats:
    def __init__(self, n_layers: int, dim: int, n_classes: int = 3, device=None):
        import torch

        self.n_layers, self.dim, self.n_classes = n_layers, dim, n_classes
        f64 = dict(dtype=torch.float64, device=device)
        self.n = 0
        self.sx = torch.zeros(n_layers, dim, **f64)
        self.xx = torch.zeros(n_layers, dim, dim, **f64)
        self.xy = torch.zeros(n_layers, dim, n_classes, **f64)
        self.sy = torch.zeros(n_classes, **f64)

    def update(self, hidden, labels) -> None:
        """``hidden``: (layers, words, dim) tensor; ``labels``: (words,) with -1 = skip."""
        import torch

        labels = torch.as_tensor(labels, device=self.xx.device)
        keep = labels >= 0
        if not bool(keep.any()):
            return
        y = torch.nn.functional.one_hot(labels[keep], self.n_classes).to(torch.float64)
        self.n += int(keep.sum())
        self.sy += y.sum(0)
        for layer in range(self.n_layers):  # one layer at a time keeps peak memory low
            x = hidden[layer][keep.to(hidden.device)].to(device=self.xx.device, dtype=torch.float64)
            self.sx[layer] += x.sum(0)
            self.xx[layer] += x.T @ x
            self.xy[layer] += x.T @ y

    def collapse(self, n_classes: int) -> "LinearStats":
        """Statistics for the 2-way task (labels 1 and 2 merged); 3 returns ``self``."""
        if n_classes == self.n_classes:
            return self
        if (self.n_classes, n_classes) != (3, 2):
            raise ValueError("only 3-way -> 2-way is supported")
        import torch

        out = LinearStats.__new__(LinearStats)
        out.n_layers, out.dim, out.n_classes, out.n = self.n_layers, self.dim, 2, self.n
        out.sx, out.xx = self.sx, self.xx
        out.xy = torch.stack([self.xy[..., 0], self.xy[..., 1] + self.xy[..., 2]], dim=-1)
        out.sy = torch.stack([self.sy[0], self.sy[1] + self.sy[2]])
        return out

    def state_dict(self) -> dict:
        """CPU copy for checkpoints. The Gram matrices are symmetric, so only their upper
        triangles are stored (half the size: 6.8 GB instead of 13.6 GB for a 64-layer,
        5120-dimensional model)."""
        import torch

        iu = torch.triu_indices(self.dim, self.dim, device=self.xx.device)
        xx = torch.empty(self.n_layers, iu.shape[1], dtype=torch.float64)
        for layer in range(self.n_layers):
            xx[layer] = self.xx[layer][iu[0], iu[1]].cpu()
        return {"n": self.n, "sx": self.sx.cpu(), "xx_triu": xx, "xy": self.xy.cpu(),
                "sy": self.sy.cpu(), "dim": self.dim}

    def load_state_dict(self, state: dict) -> None:
        import torch

        self.n = int(state["n"])
        for k in ("sx", "xy", "sy"):
            getattr(self, k).copy_(state[k])
        iu = torch.triu_indices(self.dim, self.dim, device=self.xx.device)
        for layer in range(self.n_layers):
            m = torch.zeros(self.dim, self.dim, dtype=torch.float64, device=self.xx.device)
            m[iu[0], iu[1]] = state["xx_triu"][layer].to(self.xx.device)
            self.xx[layer] = m + m.T - torch.diag(torch.diagonal(m))

    def to_numpy_accumulator(self):
        """The same statistics as a ``prosody_llms.probes.ridge.RidgeAccumulator``."""
        from prosody_llms.probes.ridge import RidgeAccumulator

        acc = RidgeAccumulator(self.n_layers, self.dim, self.n_classes)
        acc.n = self.n
        acc.sx, acc.xx = self.sx.cpu().numpy(), self.xx.cpu().numpy()
        acc.xy, acc.sy = self.xy.cpu().numpy(), self.sy.cpu().numpy()
        return acc


@dataclass
class ProbeGrid:
    """Probes for every layer and grid value: scores = x @ weights + bias."""

    family: str
    grid: tuple[float, ...]
    weights: object  # (layers, grid, dim, classes) float64 tensor
    bias: object  # (layers, grid, classes)

    def predict_layer(self, layer: int, x):
        """``x``: (words, dim) -> class indices (grid, words)."""
        import torch

        x = x.to(self.weights.device, torch.float64)
        scores = torch.einsum("wd,gdc->gwc", x, self.weights[layer]) + self.bias[layer][:, None]
        return scores.argmax(-1)


def solve(stats: LinearStats, family: str, grid: tuple[float, ...] | None = None,
          device=None) -> ProbeGrid:
    """``device``: where to solve and keep the probes (default: where the statistics are).
    The statistics can live on the CPU while each layer is solved on the GPU."""
    import torch

    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    grid = tuple(grid or DEFAULT_GRID[family])
    if stats.n < 2:
        raise ValueError("not enough data")
    n = stats.n
    n_c = stats.sy
    if family == "lda" and bool((n_c == 0).any()):
        raise ValueError(f"every class needs examples, got {n_c.tolist()}")
    device = torch.device(device) if device is not None else stats.xx.device
    weights = torch.zeros(stats.n_layers, len(grid), stats.dim, stats.n_classes,
                          dtype=torch.float64, device=device)
    bias = torch.zeros(stats.n_layers, len(grid), stats.n_classes, dtype=torch.float64,
                       device=device)
    n_c = n_c.to(device)
    for layer in range(stats.n_layers):
        sx, xx, xy = (t[layer].to(device) for t in (stats.sx, stats.xx, stats.xy))
        mu = sx / n
        if family == "ridge":
            cov = xx / n - torch.outer(mu, mu)
            target = xy / n - torch.outer(mu, n_c / n)  # centred cross products
        else:
            mu_c = xy / n_c  # (d, C) class means
            cov = xx / n - (mu_c * (n_c / n)) @ mu_c.T  # pooled within-class
            target = mu_c
        sd = torch.sqrt(torch.clamp(torch.diagonal(cov), min=1e-12))
        corr = cov / torch.outer(sd, sd)
        try:
            evals, evecs = torch.linalg.eigh(corr)
        except RuntimeError:  # e.g. cuSOLVER cannot start because the GPU is full
            evals, evecs = (t.to(corr.device) for t in torch.linalg.eigh(corr.cpu()))
        evals = torch.clamp(evals, min=0.0)
        proj = evecs.T @ (target / sd[:, None])  # (d, C) in the eigenbasis
        for g, value in enumerate(grid):
            denom = evals + value if family == "ridge" else (1 - value) * evals + value
            w_std = evecs @ (proj / denom[:, None])
            w = w_std / sd[:, None]
            weights[layer, g] = w
            if family == "ridge":
                bias[layer, g] = n_c / n - mu @ w
            else:
                bias[layer, g] = (-0.5 * ((target / sd[:, None]) * w_std).sum(0)
                                  + torch.log(n_c / n))
    return ProbeGrid(family, grid, weights, bias)
