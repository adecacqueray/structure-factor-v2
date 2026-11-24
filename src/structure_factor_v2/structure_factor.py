"""
Structure factor estimation.

The structure factor is estimated from the scattering intensity.
Two modes, selected by ``trust_boundary``
-----------------------------------------
trust_boundary=True (default, reference)
    The boundary of [0, 1)^D is trusted: plain scattering intensity
    S(n) = |sum_j exp(2i pi n.x_j)|^2 / N on the integer wave vectors of [0, 1)^D.
    Exact for point sets that "self compensate" for boundary effects,
    e.g. because they were sampled with a periodic procedure on the unit torus.

trust_boundary=False
    The boundary is not trusted, e.g. because the point set is not periodic
    (which biases the estimate at low k).
    A compensation based on tapering is then applied:

    1. Truncation: keep only the points of the inner cube [m, 1-m]^D
    2. Tapering: weight each point by a separable Kaiser-Bessel taper
    See https://numpy.org/doc/2.5/reference/generated/numpy.kaiser.html
    3. Debiasing with the estimated density rho = N (points per unit volume)

Wave vectors
------------
kunit = (rho |W|)^(1/D) is the inter-particle wave number (N^(1/D) with trust_boundary=True),
K = floor(kunit). One wave vector per pair {n, -n} (S(-n) = S(n)).

D <= CFG.max_exhaustive_D (6): cubes, by `generalised_nufft.multiscale_cube_sums`
    level 0                 : every mode |n|_inf <= K, exactly. By Minkowski's theorem this
                              cube (volume 2^D N) contains a Bragg peak of every lattice of
                              N points on the torus.
    level l = 1 .. L        : the modes of 2^l Z^D with 2^(l-1) K < |n|_inf <= 2^l K, i.e. the
                              same NUFFT on the points 2^l x mod 1 (the 2^(lD) sub-cubes of
                              side 2^-l summed together). L = ceil(log2(CFG.k_max)).
    (cu)finufft, in 3D slices for D >= 4; a direct-sum backend computes the same modes.
D > CFG.max_exhaustive_D
    every mode with |n| <= k_low (k_low set by resolution * CFG.low_mode_fraction), plus an
    isotropic sample up to CFG.k_max * kunit, all by direct sums; finufft is not used.
"""

from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass, field

from .backend import (
    NUFFT_BACKENDS,
    direct_sums,
    is_available,
    resolve_backend,
    resolve_precision,
)
from .generalised_nufft import level_strides, multiscale_cube_sums
from .torus import cube_wave_vectors, isotropic_wave_vectors


@dataclass
class StructureFactorConfig:
    margin: float = 0.1
    max_beta: float = 15.0                 # Kaiser taper: leakage ~exp(-2 beta) ~ 1e-13
    k_max: float = 2.0                     # in units of kunit: cubes up to |n|_inf <= 2^ceil(log2 k_max) K
    max_exhaustive_D: int = 6             # D <= this: cubes of modes (see module docstring)
    low_mode_fraction: float = 0.25        # D > max_exhaustive_D: share of exhaustive low modes
    radial_bins: dict = field(default_factory=lambda: {1: (10, 50), 2: (5, 25), 3: (2, 6)})  # (width, start) in |n|^2
    max_display: int = 30_000              # plot_structure_factor: max raw S(k) drawn (peaks always drawn)
    smoothing: float = 0.01                # gaussian smoothing width, fraction of the log-k range
    BRAG_PEAK: float = 100.0                # brag_peaks: S(k) >= BRAG_PEAK is a Bragg peak
    brag_max_work: float = 1e11            # brag_peaks: max NUFFT work (transforms x points)

CFG = StructureFactorConfig()

__all__ = ["structure_factor", "structure_factor_and_average", "tapered_kmin", "auto_beta", "n_levels"]


# ── Kaiser-Bessel taper on [0, 1] ─────────────────────────────────────────────

def _kaiser_norm(beta: float) -> float:
    """sqrt(int_0^1 I0(beta sqrt(1 - (2x - 1)^2))^2 dx), by Gauss-Legendre quadrature."""
    u, w = np.polynomial.legendre.leggauss(400)          # u in [-1, 1], x = (u + 1) / 2
    return float(np.sqrt(0.5 * np.sum(w * np.i0(beta * np.sqrt(1.0 - u ** 2)) ** 2)))


def _kaiser_weights(pts: np.ndarray, beta: float) -> np.ndarray:
    """Separable taper h(x_j) (normalised: int_[0,1]^D h^2 = 1), shape (N,)."""
    u = 2.0 * np.clip(pts, 0.0, 1.0) - 1.0
    w = np.i0(beta * np.sqrt(1.0 - u ** 2)) / _kaiser_norm(beta)
    return np.prod(w, axis=1)


def _kaiser_ft(kint: np.ndarray, beta: float) -> np.ndarray:
    """H(n) = int_[0,1]^D h(x) exp(2i pi n.x) dx, analytic, shape (M,).

    Uses int_{-1}^{1} I0(beta sqrt(1-u^2)) e^{i w u} du = 2 sinh(sqrt(beta^2 - w^2)) / sqrt(beta^2 - w^2).
    """
    norm = _kaiser_norm(beta)
    H = np.ones(len(kint), dtype=complex)
    for d in range(kint.shape[1]):
        n = kint[:, d].astype(float)
        r = np.sqrt((beta ** 2 - (np.pi * n) ** 2).astype(complex))
        safe = np.abs(r) > 1e-12
        g = np.ones_like(r)                                   # limit sinh(r)/r -> 1
        g[safe] = np.sinh(r[safe]) / r[safe]
        H *= np.real(g) * np.cos(np.pi * n) / norm           # (-1)^n from the shift x -> u
    return H


def auto_beta(n_points: int, D: int, margin: float) -> float:
    """Kaiser parameter used when beta=None.

    The spectral window of the taper spans ~beta/pi modes, while the truncated window
    only holds n_axis = (rho|W|)^(1/D) modes per axis up to the inter-particle scale.
    We take beta = min(15, pi * n_axis / 4): leakage ~1e-13 whenever there are enough
    points (typically D <= 3), otherwise the main lobe is capped at a quarter of the
    useful range and the leakage floor rises to ~exp(-2 beta) (curse of dimension:
    certifying a low bulk S(k) in high D requires a huge number of points).
    """
    n_axis = (n_points * (1.0 - 2.0 * margin) ** D) ** (1.0 / D)
    return float(min(CFG.max_beta, np.pi * n_axis / 4.0))


def tapered_kmin(n_points: int, D: int, margin: float | None = None, beta: float | None = None) -> float:
    """Smallest trustworthy normalised wave number (k = 1 <-> |n| = (rho|W|)^(1/D))
    when trust_boundary=False (truncated + tapered estimate)."""
    margin = CFG.margin if margin is None else margin
    if beta is None:
        beta = auto_beta(n_points, D, margin)
    n_exp = n_points * (1.0 - 2.0 * margin) ** D
    return (beta / np.pi + 1.0) / n_exp ** (1.0 / D)


# ── Wave vectors and backends ─────────────────────────────────────────────────

def n_levels(k_max: float | None = None) -> int:
    """Number of cube levels (stride 1, 2, 4, ...) needed to reach k_max * kunit."""
    k_max = CFG.k_max if k_max is None else k_max
    return 1 + max(0, math.ceil(math.log2(k_max) - 1e-12))


def _resolve_backends(backend: str, D: int, device: str):
    """(nufft backend or None, direct-sum backend) used for dimension D.

    D <= max_exhaustive_D: "auto" or a (cu)finufft backend -> (cu)finufft for every cube;
    a direct-sum backend -> used everywhere. D > max_exhaustive_D: direct sums only.
    """
    if D <= CFG.max_exhaustive_D:
        if backend == "auto" or backend in NUFFT_BACKENDS:
            nufft = resolve_backend(backend, 3, device)
            direct = "cupy" if nufft == "cufinufft" and is_available("cupy") else "numpy"
            return nufft, direct
        return None, resolve_backend(backend, D, device)
    if backend in NUFFT_BACKENDS:
        raise ValueError(f"backend {backend!r} is only used for D <= CONFIG.max_exhaustive_D "
                         f"({CFG.max_exhaustive_D}), got D={D}")
    return None, resolve_backend(backend, D, device)


# ── Core ──────────────────────────────────────────────────────────────────────

def _structure_factor_core(points, resolution, device, precision, trust_boundary,
                           margin, beta, backend, nufft_tolerance):
    """Return (kint, S, n_exp): wave vectors of the analysed window, S, expected count rho|W|."""
    pts = np.asarray(points, dtype=float).reshape(-1, np.asarray(points).shape[-1])
    N, D = pts.shape
    taper = not trust_boundary

    if taper:
        margin = CFG.margin if margin is None else margin
        side = 1.0 - 2.0 * margin
        if not 0.0 < side <= 1.0:
            raise ValueError("margin must be in [0, 0.5)")
        keep = np.all((pts >= margin) & (pts < 1.0 - margin), axis=1)
        pts = (pts[keep] - margin) / side
        n_exp = N * side ** D                      # promised density x volume of the window
        if beta is None:
            beta = auto_beta(N, D, margin)
        weights = _kaiser_weights(pts, beta)
    else:
        n_exp = float(N)
        weights = np.ones(len(pts))

    kunit = float(n_exp ** (1.0 / D))
    kmax = CFG.k_max * kunit
    precision = resolve_precision(precision)
    nufft, direct = _resolve_backends(backend, D, device)

    if D <= CFG.max_exhaustive_D:                  # cubes: all modes, then stride 2, 4, ...
        K = max(1, int(math.floor(kunit * (1.0 + 1e-12))))
        L = n_levels()
        if nufft is not None:
            kint, F = multiscale_cube_sums(pts, K, L, weights, nufft, device, precision,
                                           nufft_tolerance)
        else:
            kint = np.concatenate([cube_wave_vectors(K, D, s, K_in)
                                   for s, K_in in level_strides(K, L)])
            F = direct_sums(pts, weights, kint, direct, precision)
    else:                                          # low modes + isotropic sample of the others
        k_low = max(int((resolution * CFG.low_mode_fraction) ** (1.0 / D)), 1)
        if kmax <= k_low:
            kmax = k_low + 1
        kint = isotropic_wave_vectors(k_low, kmax, D, int(resolution * (1.0 - CFG.low_mode_fraction)))
        kint = kint.astype(np.int64)
        F = direct_sums(pts, weights, kint, direct, precision)

    if taper:
        F = F - n_exp * _kaiser_ft(kint, beta)
    S = np.abs(F) ** 2 / n_exp

    isnt0 = ~np.all(kint == 0, axis=1)
    kint, S = kint[isnt0], S[isnt0]
    order = np.argsort((kint ** 2).sum(axis=1), kind="stable")
    return kint[order], S[order], n_exp


def _gaussian_filter1d(x, sigma, truncate=4.0):
    """Gaussian smoothing of a 1D array using a truncated kernel."""
    radius = int(np.ceil(truncate * sigma))
    if radius == 0:
        return x.copy()
    kernel_x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel /= kernel.sum()
    xp = np.pad(x, radius, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(xp, 2 * radius + 1)
    return windows @ kernel


# ── Public API ────────────────────────────────────────────────────────────────

def structure_factor(
    points: np.ndarray,
    resolution: int = 30_000,
    device="auto",
    precision="float64",
    trust_boundary: bool = True,
    margin: float | None = None,
    beta: float | None = None,
    backend: str = "auto",
    nufft_tolerance: float | str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate S(k) via scattering intensity.

    Parameters
    ----------
    points         : (N, D) array of point coordinates in [0, 1)^D.
    resolution     : D > CONFIG.max_exhaustive_D only: total number of wave vectors.
                     Ignored for D <= CONFIG.max_exhaustive_D (cubes of modes, see the
                     module docstring).
    device         : "auto" | "cpu" | "gpu": for backend="auto" only, "cpu" never picks
                     cufinufft / cupy.
    precision      : "float32" | "float64". Use float64 to resolve S below ~1e-6.
    trust_boundary : True (default): plain estimate on the whole cube.
                     False: truncate to [margin, 1 - margin]^D, Kaiser-taper, debias
                     with the promised density (see module docstring).
    margin         : width removed on each side (trust_boundary=False only),
                     default structure_factor_v2.CONFIG.margin (0.1).
    beta           : Kaiser parameter (trust_boundary=False only). Larger -> less
                     leakage (~exp(-2 beta)) but coarser resolution (~beta/pi modes).
                     None (default) -> auto_beta(N, D, margin).
    backend        : "auto" | "finufft" | "cufinufft" | "keops" | "torch" | "numpy" | "cupy"
                     (see `structure_factor_v2.backend`).
                     D <= CONFIG.max_exhaustive_D: "auto" or (cu)finufft -> (cu)finufft for
                     every cube; a direct-sum backend computes the same modes (exact, slow:
                     cost ~ 2^D N^2 per level).
                     D > CONFIG.max_exhaustive_D: direct sums only ("auto": cupy or numpy).
    nufft_tolerance: relative tolerance of (cu)finufft, float or "auto"
                     (1e-5 in float32, 1e-12 in float64). Ignored by the direct sums (exact).

    Returns
    -------
    kint : (M, D) int array of wave vectors of the analysed window (k = 2 pi kint),
           one per pair {n, -n} (S(-n) = S(n)), sorted by norm. With trust_boundary=False
           they refer to the rescaled truncated window.
    S    : (M,) array.

    Note
    ----
    Points must lie in the unit hypercube [0, 1)^D. This is not checked.
    """
    kint, S, _ = _structure_factor_core(points, resolution, device, precision,
                                        trust_boundary, margin, beta, backend, nufft_tolerance)
    return kint, S


def structure_factor_and_average(
    points,
    resolution: int = 30_000,
    min_val: float = 1e-30,
    precision="float64",
    trust_boundary: bool = True,
    margin: float | None = None,
    beta: float | None = None,
    backend: str = "auto",
    nufft_tolerance: float | str = "auto",
    device: str = "auto",
):
    """
    Structure factor of a point set, raw and radially averaged, ready to plot.

    resolution : as in ``structure_factor`` (D > CONFIG.max_exhaustive_D only).
    min_val    : floor of the returned values (log plots).
    ``precision``, ``trust_boundary``, ``margin``, ``beta``, ``backend``, ``nufft_tolerance``
    and ``device`` behave as in ``structure_factor``.

    Returns
    -------
    kraw, Sraw     : normalised wave numbers and raw S values (subsampled for display).
    kgroup, Sgroup : radially averaged then log-k smoothed curve.

    Wave numbers are normalised so that k = 1 corresponds to |n| = (rho |W|)^(1/D),
    i.e. to the inter-particle distance, in both modes.
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])
    D = pts.shape[1]
    kint, Sraw, n_exp = _structure_factor_core(pts, resolution, device,
                                               precision, trust_boundary, margin, beta, backend,
                                               nufft_tolerance)
    kunit = n_exp ** (1.0 / D)
    k2_ = (kint ** 2).sum(axis=1)

    if D in CFG.radial_bins:
        groupbin, groupstart = CFG.radial_bins[D]
        mask = k2_ >= groupstart
        k2_[mask] = groupbin * (k2_[mask] // groupbin)

    kraw = np.sqrt(k2_) / kunit
    k2group_, inverse = np.unique(k2_, return_inverse=True)
    counts = np.bincount(inverse)
    Sgroup = (np.bincount(inverse, weights=Sraw) / counts).clip(min=min_val)
    kgroup = np.sqrt(k2group_) / kunit
    Sraw = Sraw.clip(min=min_val)

    logk = np.log(kgroup)
    logk_u = np.linspace(logk[0], logk[-1], 1000)
    S_u = np.interp(logk_u, logk, Sgroup)
    sigma_pixels = (logk[-1] - logk[0]) * CFG.smoothing / (logk_u[1] - logk_u[0])
    Sgroup = np.interp(logk, logk_u, _gaussian_filter1d(S_u, sigma_pixels, truncate=4.0)).clip(min=min_val)
    Sraw = Sraw.clip(min=Sgroup.min())
    return kraw, Sraw, kgroup, Sgroup
