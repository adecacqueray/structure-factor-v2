"""
Lattice point sets in [0, 1)^D: cheap, well spread, and a quick reference for the structure
factor (a jittered lattice is hyperuniform; an unjittered one is a crystal, S = 0 except on
its Bragg peaks).

    korobov_lattice    rank-1 lattice with the best spectral test among random generators
    jittered_lattice   Korobov lattice, randomly shifted, every point uniform in its own cell
    curled_lattice     Korobov lattice deformed by a random divergence-free (curl noise) flow

>>> import structure_factor_v2 as sf
>>> x = sf.jittered_lattice(10_000, 2)
>>> sf.plot_structure_factor(x)

Korobov lattice: x_t = t z / N mod 1 with z = (1, a, a^2, ..., a^(D-1)) mod N. The generator a
maximises the length of the shortest non-zero vector of the dual lattice {h : h.z = 0 mod N},
estimated by an LLL reduction (O(D^3) per candidate; for D <= 6 its approximation factor is
<= 2.37, so the first reduced vector is virtually always the shortest).
Sloan & Joe (1994), Lattice Methods for Multiple Integration.

Jittered lattice: each point is moved uniformly in a fundamental cell P of the lattice L (the
parallelepiped of an LLL-reduced basis, volume 1/N). The expected structure factor is exactly 
S(k) = 1 - |phi_P(k)|^2 ~ k^2.

Curled lattice: curl noise jittering (Bærentzen, Frisvad & Martínez 2023, reference below):
the lattice points are moved along the streamlines of a random divergence-free vector field.
The flow preserves volume, so the points stay well separated, while the lattice anisotropy
(Bragg peaks) is smeared out: the result is a blue noise point set. Two departures from the
original method:
- the starting lattice is our Korobov lattice instead of a triangular (2D) / cubic (3D) grid.
  The method only needs well spread starting points on a lattice, and a good Korobov lattice
  is close to the densest lattice packing (its generator maximises the shortest dual vector);
- the field is a sum of random Fourier modes u_m cos(2 pi k_m.x + phase_m) with integer
  wave vectors k_m (|k_m| ~ N^(1/D) / noise_scale) and polarisations u_m orthogonal to k_m,
  instead of the curl of a Perlin / sparse convolution noise. Each mode is exactly
  divergence-free (in 2D it is the curl of a stream function, in 3D of a vector potential),
  in any dimension D, and exactly periodic on the torus: no boundary effect.

Reference
---------
J. A. Bærentzen, J. R. Frisvad, and J. Martínez. Curl Noise Jittering. SIGGRAPH Asia 2023
Conference Papers, article 88, 2023. https://doi.org/10.1145/3610548.3618163
Code: https://github.com/jonasmb/curlnoisejittering (AGPL-3.0; not used here, the
implementation below is independent).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
from numpy.typing import NDArray

__all__ = ["korobov_lattice", "jittered_lattice", "curled_lattice", "curl_noise"]


@dataclass
class CurlNoiseConfig:
    """Defaults of curled_lattice (see the module docstring for how they were tuned)."""
    dt_base: float = 0.45              # dt = dt_base + dt_per_dim * D (RMS displacement, spacings)
    dt_per_dim: float = 0.15
    noise_scale: float = 5.0           # wavelength of the field, in spacings N^(-1/D) ...
    min_wave_number: float = 1.25      # ... capped so that k_0 = N^(1/D) / noise_scale >= this
    n_modes: int = 256
    bandwidth: float = 0.3
    n_steps: int = 1
    point_chunk: int = 50_000          # points per evaluation of the field

    def dt(self, D: int) -> float:
        return self.dt_base + self.dt_per_dim * D

    def scale(self, N: int, D: int) -> float:
        return min(self.noise_scale, N ** (1.0 / D) / self.min_wave_number)


curl_noise = CurlNoiseConfig()


def jittered_lattice(N: int, D: int, seed=None, max_candidates: int | None = 100) -> NDArray:
    """
    A Korobov lattice of N points, randomly translated, each point moved uniformly in its own
    fundamental cell (LLL-reduced parallelepiped, see the module docstring): exactly one point
    per cell, no Bragg peak. seed: int, numpy Generator or None.
    """
    rng = np.random.default_rng(seed)
    z = _korobov_generator(N, D, rng, max_candidates)
    cell = _reduced_cell(z, N)                                       # (D, D), rows: basis of L
    x = _lattice_points(z, N) + rng.random(D) + rng.random((N, D)) @ cell
    return x % 1.0


def curled_lattice(N: int, D: int, noise_scale: float | None = None, dt: float | None = None,
                   n_modes: int | None = None, bandwidth: float | None = None,
                   n_steps: int | None = None, seed=None,
                   max_candidates: int | None = 100) -> NDArray:
    """
    A Korobov lattice of N points, randomly shifted, then moved along a random periodic
    divergence-free field (curl noise jittering, see the module docstring). Any D; tuned for
    D = 2 .. 5. Every None parameter takes its default from `curl_noise`.

    noise_scale : wavelength of the field, in lattice spacings N^(-1/D) (larger: smoother
                  flow, but towards a warped crystal). Default min(5, N^(1/D) / 1.25).
    dt          : length of the flow = RMS displacement, in lattice spacings (not the dt of
                  the paper). Smaller keeps Bragg peaks (below ~0.4 + 0.15 D), larger brings
                  the points closer to each other. Default 0.45 + 0.15 D.
    n_modes     : number of random Fourier modes of the field (default 256; 64 is enough in
                  2D/3D and faster).
    bandwidth   : relative spread of |k_m| around N^(1/D) / noise_scale (default 0.3).
    n_steps     : RK4 steps (default 1: the paper's direct method; more: closer to the exact flow).
    seed        : int, numpy Generator or None.
    """
    cfg = curl_noise
    noise_scale = cfg.scale(N, D) if noise_scale is None else noise_scale
    dt = cfg.dt(D) if dt is None else dt
    n_modes = cfg.n_modes if n_modes is None else n_modes
    bandwidth = cfg.bandwidth if bandwidth is None else bandwidth
    n_steps = cfg.n_steps if n_steps is None else n_steps
    rng = np.random.default_rng(seed)
    z = _korobov_generator(N, D, rng, max_candidates)
    x = (_lattice_points(z, N) + rng.random(D)) % 1.0
    k, u, phase = _solenoidal_modes(N, D, noise_scale, n_modes, bandwidth, rng)
    h = dt * N ** (-1.0 / D) / n_steps
    chunk = cfg.point_chunk
    for s in range(0, N, chunk):
        y = x[s:s + chunk]
        for _ in range(n_steps):
            a = h * _velocity(y, k, u, phase)
            b = h * _velocity(y + a / 2, k, u, phase)
            c = h * _velocity(y + b / 2, k, u, phase)
            d = h * _velocity(y + c, k, u, phase)
            y = y + (a + 2 * b + 2 * c + d) / 6
        x[s:s + chunk] = y
    return x % 1.0


def _solenoidal_modes(N: int, D: int, noise_scale: float, n_modes: int, bandwidth: float, rng):
    """Integer wave vectors k (M, D), amplitudes u (M, D) with u_m . k_m = 0, phases (M,);
    the field sum_m u_m cos(2 pi k_m.x + phase_m) has unit RMS norm."""
    k0 = N ** (1.0 / D) / noise_scale
    dirs = rng.normal(size=(n_modes, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = k0 * np.abs(1.0 + bandwidth * rng.normal(size=(n_modes, 1)))
    k = np.rint(radii * dirs)
    k = k[np.any(k != 0, axis=1)]
    if len(k) == 0:
        raise ValueError(f"noise_scale={noise_scale} is too large for N={N}: no wave vector left")
    u = rng.normal(size=k.shape)
    u -= (u * k).sum(1, keepdims=True) / (k * k).sum(1, keepdims=True) * k   # u_m orthogonal to k_m
    u *= np.sqrt(2.0 / len(k)) / np.linalg.norm(u, axis=1, keepdims=True)
    return k, u, rng.uniform(0.0, 2.0 * np.pi, len(k))


def _velocity(x, k, u, phase):
    """sum_m u_m cos(2 pi k_m.x + phase_m), shape of x."""
    return np.cos(2.0 * np.pi * (x @ k.T) + phase) @ u


# ── Korobov lattice ───────────────────────────────────────────────────────────

def korobov_lattice(N: int, D: int, shift: bool = True, seed=None,
                    max_candidates: int | None = 100) -> NDArray:
    """
    N points of a Korobov lattice, generator chosen among max_candidates random ones (all if
    None), with a random toroidal shift if `shift`. seed: int, numpy Generator or None
    (reproducible for a given int).
    """
    rng = np.random.default_rng(seed)
    points = _lattice_points(_korobov_generator(N, D, rng, max_candidates), N)
    if shift:
        points = (points + rng.random(D)) % 1.0
    return points


def _korobov_generator(N: int, D: int, rng, max_candidates: int | None) -> np.ndarray:
    """Best generator z among max_candidates random ones (all if None)."""
    candidates = [a for a in range(2, N) if gcd(a, N) == 1]
    if max_candidates is not None and len(candidates) > max_candidates:
        candidates = sorted(rng.choice(candidates, size=max_candidates, replace=False))
    best = max(candidates, key=lambda a: _shortest_dual_vector(_generator(a, N, D), N))
    return _generator(int(best), N, D)


def _lattice_points(z: np.ndarray, N: int) -> np.ndarray:
    """The N points t z / N mod 1."""
    return (np.arange(N, dtype=np.float64)[:, None] * (z.astype(np.float64) / N)) % 1.0


def _reduced_cell(z: np.ndarray, N: int) -> np.ndarray:
    """LLL-reduced basis (rows) of the primal lattice L = Z z/N + Z^D, covolume 1/N."""
    basis = N * np.eye(len(z), dtype=np.int64)              # rows N z/N = z, N e_2, ..., N e_D
    basis[0] = z                                            # (z_0 = 1: generates N L)
    return _lll(basis).astype(np.float64) / N


def _generator(a: int, N: int, D: int) -> np.ndarray:
    """z = (1, a, a^2, ..., a^(D-1)) mod N."""
    z = np.empty(D, dtype=np.int64)
    z[0] = 1
    for j in range(1, D):
        z[j] = (z[j - 1] * a) % N
    return z


def _shortest_dual_vector(z: np.ndarray, N: int) -> float:
    """Length of the first LLL-reduced vector of the dual lattice {h : h.z = 0 mod N}."""
    basis = np.eye(len(z), dtype=np.int64)                  # rows, determinant N
    basis[0, 0] = N
    basis[1:, 0] = (-z[1:]) % N
    return float(np.linalg.norm(_lll(basis)[0]))


def _gram_schmidt(B: np.ndarray):
    D = len(B)
    B_star = np.zeros_like(B)
    mu = np.zeros((D, D))
    for i in range(D):
        B_star[i] = B[i]
        for j in range(i):
            mu[i, j] = B[i] @ B_star[j] / (B_star[j] @ B_star[j])
            B_star[i] -= mu[i, j] * B_star[j]
    return B_star, mu


def _lll(B: np.ndarray, delta: float = 0.99) -> np.ndarray:
    """LLL reduction of the lattice spanned by the rows of B."""
    B = B.astype(np.float64).copy()
    k = 1
    while k < len(B):
        _, mu = _gram_schmidt(B)
        for j in range(k - 1, -1, -1):                      # size reduction
            q = round(mu[k, j])
            if q != 0:
                B[k] -= q * B[j]
        B_star, mu = _gram_schmidt(B)
        if B_star[k] @ B_star[k] >= (delta - mu[k, k - 1] ** 2) * (B_star[k - 1] @ B_star[k - 1]):
            k += 1                                          # Lovasz condition holds
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            k = max(1, k - 1)
    return np.round(B).astype(np.int64)
