"""Ridge-regression probes for all layers from one pass over the data.

For every layer the accumulator keeps the sums needed for ridge regression onto one-hot
labels (feature sums, the Gram matrix, and feature-label cross products). Features are
standardised implicitly, so one penalty works across layers and models. Accumulation is
in float64 to keep the Gram matrices well conditioned over millions of words.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RidgeAccumulator:
    def __init__(self, n_layers: int, dim: int, n_classes: int):
        self.n_layers, self.dim, self.n_classes = n_layers, dim, n_classes
        self.n = 0
        self.sx = np.zeros((n_layers, dim))
        self.xx = np.zeros((n_layers, dim, dim))
        self.xy = np.zeros((n_layers, dim, n_classes))
        self.sy = np.zeros(n_classes)

    def update(self, features: np.ndarray, labels: np.ndarray) -> None:
        """``features``: (layers, words, dim); ``labels``: (words,) with -1 = not scored."""
        keep = labels >= 0
        if not keep.any():
            return
        x = np.asarray(features, dtype=np.float64)[:, keep]
        y = np.eye(self.n_classes)[labels[keep]]
        self.n += int(keep.sum())
        self.sx += x.sum(axis=1)
        self.xx += np.einsum("lwd,lwe->lde", x, x)
        self.xy += np.einsum("lwd,wc->ldc", x, y)
        self.sy += y.sum(axis=0)

    def merge(self, other: "RidgeAccumulator") -> None:
        """Add another accumulator's statistics (for sharded or resumed runs)."""
        self.n += other.n
        self.sx += other.sx
        self.xx += other.xx
        self.xy += other.xy
        self.sy += other.sy

    def state(self) -> dict:
        return {"n": self.n, "sx": self.sx, "xx": self.xx, "xy": self.xy, "sy": self.sy}

    @classmethod
    def from_state(cls, state: dict) -> "RidgeAccumulator":
        layers, dim, classes = state["xy"].shape
        acc = cls(layers, dim, classes)
        acc.n = int(state["n"])
        for k in ("sx", "xx", "xy", "sy"):
            setattr(acc, k, np.array(state[k], dtype=np.float64))
        return acc

    def solve(self, alpha: float) -> "RidgeProbe":
        """Ridge solution for every layer with penalty ``alpha`` on standardised features."""
        if self.n < 2:
            raise ValueError("not enough data")
        n = self.n
        mu = self.sx / n  # (L, d)
        ymu = self.sy / n  # (C,)
        cov = self.xx / n - np.einsum("ld,le->lde", mu, mu)  # (L, d, d)
        sd = np.sqrt(np.clip(np.einsum("ldd->ld", cov), 1e-12, None))
        corr = cov / np.einsum("ld,le->lde", sd, sd)
        xy_c = self.xy / n - np.einsum("ld,c->ldc", mu, ymu)  # (L, d, C)
        xy_s = xy_c / sd[:, :, None]
        eye = np.eye(self.dim)
        w_std = np.stack([np.linalg.solve(corr[i] + (alpha / n) * eye, xy_s[i])
                          for i in range(self.n_layers)])
        w = w_std / sd[:, :, None]
        b = ymu[None, :] - np.einsum("ld,ldc->lc", mu, w)
        return RidgeProbe(w, b, alpha)

    def solve_lda(self, shrinkage: float = 0.1) -> "RidgeProbe":
        """Linear discriminant analysis from the same statistics.

        The class sums are the columns of ``xy``. The pooled within-class covariance is
        standardised and shrunk towards the identity by ``shrinkage`` (0 to 1). Unlike
        ridge regression onto one-hot labels, LDA does not "mask" a middle class when the
        classes are ordered along one direction (as prominence 0 < 1 < 2 is).
        """
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be between 0 and 1")
        n_c = self.sy
        if (n_c == 0).any():
            raise ValueError(f"every class needs examples, got counts {n_c.tolist()}")
        n = self.n
        mu_c = self.xy / n_c[None, None, :]  # (L, d, C)
        within = self.xx / n - np.einsum("ldc,lec,c->lde", mu_c, mu_c, n_c / n)
        sd = np.sqrt(np.clip(np.einsum("ldd->ld", within), 1e-12, None))
        corr = within / np.einsum("ld,le->lde", sd, sd)
        mu_s = mu_c / sd[:, :, None]
        eye = np.eye(self.dim)
        w_s = np.stack([np.linalg.solve((1 - shrinkage) * corr[i] + shrinkage * eye, mu_s[i])
                        for i in range(self.n_layers)])
        b = -0.5 * np.einsum("ldc,ldc->lc", mu_s, w_s) + np.log(n_c / n)[None, :]
        return RidgeProbe(w_s / sd[:, :, None], b, shrinkage)


@dataclass
class RidgeProbe:
    weights: np.ndarray  # (layers, dim, classes)
    bias: np.ndarray  # (layers, classes)
    alpha: float

    def scores(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        return np.einsum("lwd,ldc->lwc", x, self.weights) + self.bias[:, None, :]

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.scores(features).argmax(-1)
