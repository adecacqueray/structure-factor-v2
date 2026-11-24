"""
The unit torus [0, 1)^D and its dual lattice Z^D.

    integer_grid, ball, half_ball, half_cube, cube_wave_vectors
                            integer points of Z^D: grids, balls, cubes, one representative
                            per pair {k, -k}
    isotropic_wave_vectors  low modes + isotropically sampled high modes (D >= 4)

Balls are given by an integer squared radius r2: ball(r2, D) = {k in Z^D : |k|^2 <= r2}.
They are built shell by shell (cylinders over the (D-1)-ball), so memory never exceeds the
size of the output, even for millions of vectors.
"""
from __future__ import annotations

import numpy as np


# ── Integer points ────────────────────────────────────────────────────────────

def integer_grid(axis, D: int) -> np.ndarray:
    """Cartesian product axis^D, as a (len(axis)^D, D) array in 'ij' order."""
    grids = np.meshgrid(*([np.asarray(axis)] * D), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def _isqrt(x):
    """Exact floor(sqrt(x)) of non-negative integers."""
    x = np.asarray(x, dtype=np.int64)
    z = np.floor(np.sqrt(x.astype(np.float64))).astype(np.int64)
    z -= z * z > x
    z += (z + 1) * (z + 1) <= x
    return z


def _cylinders(counts, lo):
    """Concatenation of the integer ranges [lo[i], lo[i] + counts[i]), int32."""
    starts = np.cumsum(counts) - counts
    z = np.arange(int(counts.sum()), dtype=np.int32)
    z -= np.repeat(starts, counts).astype(np.int32)
    z += np.repeat(np.asarray(lo, dtype=np.int32), counts)
    return z


def ball(r2: int, D: int) -> np.ndarray:
    """All k in Z^D with |k|^2 <= r2, int32, in lexicographic order."""
    if D == 1:
        K = int(_isqrt(r2))
        return np.arange(-K, K + 1, dtype=np.int32)[:, None]
    head = ball(r2, D - 1)
    m = _isqrt(r2 - np.sum(head.astype(np.int64) ** 2, axis=1))
    counts = 2 * m + 1
    return np.concatenate([np.repeat(head, counts, axis=0), _cylinders(counts, -m)[:, None]], axis=1)


def _half_ball_last_positive(r2: int, D: int) -> np.ndarray:
    """Half-ball whose representatives have their *last* non-zero coordinate positive."""
    if D == 1:
        return np.arange(1, int(_isqrt(r2)) + 1, dtype=np.int32)[:, None]
    head = ball(r2, D - 1)                                  # last coordinate > 0
    m = _isqrt(r2 - np.sum(head.astype(np.int64) ** 2, axis=1))
    upper = np.concatenate([np.repeat(head, m, axis=0), _cylinders(m, np.ones_like(m))[:, None]], axis=1)
    flat = _half_ball_last_positive(r2, D - 1)              # last coordinate = 0
    flat = np.concatenate([flat, np.zeros((len(flat), 1), dtype=np.int32)], axis=1)
    return np.concatenate([upper, flat], axis=0)


def half_ball(r2: int, D: int) -> np.ndarray:
    """One k per pair {k, -k} with 0 < |k|^2 <= r2 (first non-zero coordinate > 0), int32,
    in lexicographic order."""
    k = _half_ball_last_positive(int(r2), D)
    first = k[np.arange(len(k)), np.argmax(k != 0, axis=1)]
    k = k * np.sign(first)[:, None].astype(np.int32)
    return k[np.lexsort(k.T[::-1])]


def half_cube(K: int, D: int) -> np.ndarray:
    """One k per pair {k, -k} with 0 < |k|_inf <= K (first non-zero coordinate > 0), int64."""
    k = integer_grid(np.arange(-int(K), int(K) + 1, dtype=np.int64), D)
    first = k[np.arange(len(k)), np.argmax(k != 0, axis=1)]
    return k[first > 0]


def cube_wave_vectors(K: int, D: int, stride: int = 1, K_in: int | None = None) -> np.ndarray:
    """The wave vectors of `generalised_nufft.cube_sums`: n = stride * m, 0 < |m|_inf <= K,
    |m|_inf > K_in (if given), one per pair {n, -n}, int64."""
    m = half_cube(K, D)
    if K_in is not None:
        m = m[np.abs(m).max(axis=1) > K_in]
    return stride * m


# ── Wave vectors ──────────────────────────────────────────────────────────────

def isotropic_wave_vectors(k_low: int, k_max: float, D: int, n_high: int) -> np.ndarray:
    """Every mode with |k| <= k_low, plus n_high modes drawn isotropically in k_low < |k| < k_max
    (rounded to Z^D, deduplicated, origin removed)."""
    low = half_ball(k_low ** 2, D)
    dirs = np.random.normal(size=(n_high, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = np.random.uniform(k_low, k_max, size=(n_high, 1))
    high = np.rint(radii * dirs).astype(int)
    k = np.concatenate([low, high], axis=0)
    return np.unique(k[np.any(k != 0, axis=1)], axis=0)
