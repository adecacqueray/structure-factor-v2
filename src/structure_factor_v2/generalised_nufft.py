"""
Exhaustive Fourier sums F(n) = sum_j c_j exp(2i pi n.x_j) on cubes of Z^D, in any dimension,
with finufft / cufinufft. One n per pair {n, -n}.

Cube of stride s
    n = s m,  |m|_inf <= K.  Since exp(2i pi s m.x) = exp(2i pi m.(s x mod 1)),
        F(s m) = sum_j c_j exp(2i pi m.(s x_j mod 1)):
    the same NUFFT as for s = 1, on the points s x_j mod 1 (the s^D sub-cubes of side 1/s
    folded onto [0, 1)^D). Exact in floating point when s is a power of 2.

Multiscale (`multiscale_cube_sums`)
    level 0: every mode |n|_inf <= K; level l >= 1: the modes of 2^l Z^D with
    2^(l-1) K < |n|_inf <= 2^l K (stride 2^l, inner cube |m|_inf <= K // 2 dropped: it is
    already covered, at a finer resolution, by the previous levels). Every level costs the
    same as level 0.

D <= 3
    One type-1 NUFFT of the points on the centred cube of modes [-K, K]^D.

D >= 4: 3D slices
    Write m = (a, c), a in Z^3 (first three coordinates), c in Z^(D-3) (the others), and
    x_j = (u_j, v_j) accordingly. Then
        F(a, c) = sum_j [c_j exp(2i pi c.v_j)] exp(2i pi a.u_j),
    so for a fixed c the slice {(a, c) : |a|_inf <= K} is one 3D NUFFT of the 3D points u_j,
    with modulated weights, on the cube [-K, K]^3. Every slice has the same size: one NUFFT
    plan serves batches of slices (n_trans). c runs over one representative per pair
    {c, -c} plus c = 0: (2K + 1)^(D-3) / 2 slices.

The cost is ~ (2K + 1)^(D-3) / 2 NUFFTs of N points per level (one for D <= 3).
"""
from __future__ import annotations

import numpy as np

from .backend import (NUFFT_BACKENDS, NUFFT_CHUNK, _complex, _load, _real, resolve_backend,
                      resolve_precision, resolve_tolerance)
from .torus import half_cube, integer_grid

__all__ = ["cube_sums", "multiscale_cube_sums", "level_strides"]

MAX_BATCH_MODES = 2 ** 23           # modes per batch of transforms (a single one may exceed it)
MAX_BATCH_WEIGHTS = 2 ** 24         # entries of the (n_trans, N) weight array of a batch
MAX_BATCH = 64                      # max number of transforms per batch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _grid(b: int, d: int) -> np.ndarray:
    """Wave vectors of a centred cube of b^d modes (b odd), in finufft's (modeord=0) order."""
    return integer_grid(np.arange(-(b // 2), b - (b // 2)), d).astype(np.int64)


def _slices(D: int, K: int) -> np.ndarray:
    """Slice offsets c in Z^(D-3): c = 0 first, then one per pair {c, -c}, |c|_inf <= K."""
    if D <= 3:
        return np.zeros((1, 0), dtype=np.int64)
    return np.concatenate([np.zeros((1, D - 3), dtype=np.int64), half_cube(K, D - 3)])


def _batch_size(N: int, b: int, d: int, n_tasks: int) -> int:
    return max(1, min(MAX_BATCH, MAX_BATCH_WEIGHTS // max(N, 1), MAX_BATCH_MODES // b ** d, n_tasks))


def level_strides(K: int, n_levels: int) -> list:
    """[(stride, K_in)] of the levels of `multiscale_cube_sums` (K_in = None: no hole)."""
    return [(1, None)] + [(2 ** l, int(K) // 2) for l in range(1, n_levels)]


class _BatchedNufft:
    """Type-1 NUFFTs of fixed points u (N, d) in [0, 1)^d, for batches of weight vectors.

    __call__(w (B, N) complex xp array, b) -> (B, b^d) complex128 numpy array, modes of the
    centred cube of side b in finufft's modeord=0 order (as `_grid(b, d)`).
    """

    def __init__(self, u, backend, precision, tol):
        self.lib = _load(backend)
        if backend == "cufinufft":
            import cupy
            self.xp, self.to_np = cupy, cupy.asnumpy
        else:
            self.xp, self.to_np = np, np.asarray
        self.N, self.d = u.shape
        self.dtype, self.tol = _complex(precision), tol
        self.chunks = [(s, min(s + NUFFT_CHUNK, self.N)) for s in range(0, self.N, NUFFT_CHUNK)]
        self.u = [[self.xp.asarray(np.ascontiguousarray(2.0 * np.pi * u[s:e, a], dtype=_real(precision)))
                   for a in range(self.d)] for s, e in self.chunks]
        self.plans = {}

    def _plans(self, b, B):
        if (b, B) not in self.plans:
            plans = []
            for x in self.u:
                plan = self.lib.Plan(1, (b,) * self.d, n_trans=B, eps=self.tol, isign=1,
                                     dtype=self.dtype)
                plan.setpts(*x)
                plans.append(plan)
            self.plans[(b, B)] = plans
        return self.plans[(b, B)]

    def __call__(self, w, b):
        B = w.shape[0]
        out = None
        for plan, (s, e) in zip(self._plans(b, B), self.chunks):
            f = plan.execute(self.xp.ascontiguousarray(w[:, s:e]))
            out = f if out is None else out + f
        return self.to_np(out).reshape(B, -1).astype(np.complex128)


# ── Public API ────────────────────────────────────────────────────────────────

def cube_sums(
    points: np.ndarray,
    K: int,
    weights: np.ndarray | None = None,
    stride: int = 1,
    K_in: int | None = None,
    backend: str = "auto",
    device: str = "auto",
    precision: str = "float32",
    nufft_tolerance: float | str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """
    F(n) = sum_j c_j exp(2i pi n.x_j) on every wave vector n = stride * m, 0 < |m|_inf <= K
    (and |m|_inf > K_in if given), exactly (up to the NUFFT tolerance).

    Parameters
    ----------
    points          : (N, D) array in [0, 1)^D, any D.
    K               : half-side of the cube of m, integer.
    weights         : (N,) real weights c_j, default 1 (complex weights: D <= 3 only).
    stride          : n = stride * m (a power of 2 keeps the folding s x mod 1 exact).
    K_in            : drop the inner cube |m|_inf <= K_in (None: keep everything).
    backend         : "auto" | "finufft" | "cufinufft".
    device          : "auto" | "cpu" | "gpu", for backend="auto".
    precision       : "float32" | "float64".
    nufft_tolerance : relative tolerance of (cu)finufft, float or "auto".

    Returns
    -------
    kint : (M, D) int64, one wave vector per pair {n, -n}, first non-zero coordinate
           positive (unordered; uses F(-n) = conj(F(n)), so complex weights need D <= 3).
    F    : (M,) complex128.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, np.asarray(points).shape[-1])
    N, D = pts.shape
    d = min(D, 3)
    K, stride = int(K), int(stride)
    if K < 1 or stride < 1:
        raise ValueError(f"K and stride must be positive integers, got K={K}, stride={stride}")
    K_in = -1 if K_in is None else int(K_in)
    if K_in >= K:
        return np.zeros((0, D), dtype=np.int64), np.zeros(0, dtype=np.complex128)
    precision = resolve_precision(precision)
    backend = resolve_backend(backend, d, device)
    if backend not in NUFFT_BACKENDS:
        raise ValueError(f"cube sums need finufft or cufinufft, got backend {backend!r}")
    tol = resolve_tolerance(nufft_tolerance, precision)
    c = np.ones(N) if weights is None else np.asarray(weights).reshape(N)
    if np.iscomplexobj(c) and D > 3:
        raise ValueError("complex weights are only supported for D <= 3")
    if stride > 1:                                   # fold the s^D sub-cubes onto [0, 1)^D
        pts = np.mod(stride * pts, 1.0)

    b = 2 * K + 1
    C = _slices(D, K)
    grid = _grid(b, d)
    grid_inf = np.abs(grid).max(axis=1)
    head = grid[np.arange(len(grid)), np.argmax(grid != 0, axis=1)] > 0   # half space, m != 0
    engine = _BatchedNufft(pts[:, :d], backend, precision, tol)
    xp = engine.xp
    v_x, c_x = xp.asarray(pts[:, d:]), xp.asarray(c)
    B = _batch_size(N, b, d, len(C))

    ks, Fs = [], []
    for s in range(0, len(C), B):
        batch = np.arange(s, min(s + B, len(C)))
        w = xp.zeros((B, N), dtype=engine.dtype)         # padded to B: one plan
        if D > d:                                        # slice c: weights c_j exp(2i pi c.v_j)
            phase = v_x @ xp.asarray(C[batch].T, dtype=float)
            w[:len(batch)] = c_x[None, :] * xp.exp(2j * np.pi * phase.T)
        else:
            w[0] = c_x
        Fb = engine(w, b)[:len(batch)]
        for t, i in enumerate(batch):
            ci = C[i]
            c_inf = int(np.abs(ci).max()) if ci.size else 0
            ok = head.copy() if c_inf == 0 else np.ones(len(grid), dtype=bool)
            if c_inf <= K_in:                            # hole: drop |m|_inf <= K_in
                ok &= grid_inf > K_in
            if ok.any():
                m = grid[ok]
                ks.append(np.concatenate([m, np.repeat(ci[None], len(m), axis=0)], axis=1))
                Fs.append(Fb[t][ok])
    if not ks:
        return np.zeros((0, D), dtype=np.int64), np.zeros(0, dtype=np.complex128)
    k, F = np.concatenate(ks), np.concatenate(Fs)
    flip = k[np.arange(len(k)), np.argmax(k != 0, axis=1)] < 0   # first non-zero coordinate > 0
    k[flip] *= -1
    F[flip] = np.conj(F[flip])                           # never happens for D <= 3
    return stride * k, F


def multiscale_cube_sums(
    points: np.ndarray,
    K: int,
    n_levels: int = 1,
    weights: np.ndarray | None = None,
    backend: str = "auto",
    device: str = "auto",
    precision: str = "float32",
    nufft_tolerance: float | str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Level 0: every mode 0 < |n|_inf <= K. Level l = 1 .. n_levels - 1: the modes of 2^l Z^D
    with 2^(l-1) K < |n|_inf <= 2^l K. No mode is computed twice. Same parameters and
    returns as `cube_sums`.
    """
    ks, Fs = [], []
    for stride, K_in in level_strides(K, n_levels):
        k, F = cube_sums(points, K, weights, stride, K_in, backend, device, precision,
                         nufft_tolerance)
        ks.append(k)
        Fs.append(F)
    return np.concatenate(ks), np.concatenate(Fs)
