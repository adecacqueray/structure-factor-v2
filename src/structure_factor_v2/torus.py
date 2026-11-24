"""
The unit torus [0, 1)^D and its dual lattice Z^D.

    integer_grid, half_cube, cube_wave_vectors
                            integer points of Z^D: grids, cubes, one representative per
                            pair {k, -k}
    orbit_wave_vectors      the same number of wave vectors on each orbit |k|^2 = r2 (high D)
"""
from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np


# ── Integer points ────────────────────────────────────────────────────────────

def integer_grid(axis, D: int) -> np.ndarray:
    """Cartesian product axis^D, as a (len(axis)^D, D) array in 'ij' order."""
    grids = np.meshgrid(*([np.asarray(axis)] * D), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


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


# ── Orbits |k|^2 = r2 ─────────────────────────────────────────────────────────

def _shapes(r2: int, D: int, top: int | None = None) -> list:
    """Non-increasing tuples of positive integers, at most D of them, with sum of squares r2."""
    if r2 == 0:
        return [()]
    if D == 0:
        return []
    top = math.isqrt(r2) if top is None else min(top, math.isqrt(r2))
    out = []
    for a in range(top, 0, -1):
        out += [(a,) + s for s in _shapes(r2 - a * a, D - 1, a)]
    return out


def _shape_count(shape: tuple, D: int) -> int:
    """Number of k in Z^D whose sorted non-zero |coordinates| are `shape`."""
    p = len(shape)
    n = math.perm(D, p) * 2 ** p
    for m in Counter(shape).values():
        n //= math.factorial(m)
    return n


def _canonical(k: np.ndarray) -> np.ndarray:
    """Representative of {k, -k} whose first non-zero coordinate is positive."""
    first = k[np.arange(len(k)), np.argmax(k != 0, axis=1)]
    return k * np.where(first < 0, -1, 1)[:, None]


def orbit_size(r2: int, D: int) -> int:
    """Number of k in Z^D with |k|^2 = r2."""
    return sum(_shape_count(s, D) for s in _shapes(int(r2), D))


def _enumerate_orbit(r2: int, D: int) -> np.ndarray:
    """Every k with |k|^2 = r2, one per pair {k, -k} (small orbits only)."""
    rows = []
    for shape in _shapes(r2, D):
        p = len(shape)
        vals = np.array(sorted(set(itertools.permutations(shape))), dtype=np.int64)   # (P, p)
        signs = np.array(list(itertools.product((1, -1), repeat=p)), dtype=np.int64)   # (2^p, p)
        sv = (vals[:, None, :] * signs[None, :, :]).reshape(-1, p)
        for pos in itertools.combinations(range(D), p):
            k = np.zeros((len(sv), D), dtype=np.int64)
            k[:, pos] = sv
            rows.append(k)
    if not rows:
        return np.zeros((0, D), dtype=np.int64)
    k = np.concatenate(rows)
    return np.unique(_canonical(k), axis=0)


def _sample_orbit(r2: int, D: int, n: int, rng) -> np.ndarray:
    """n distinct k with |k|^2 = r2 (one per pair {k, -k}), uniform on the orbit: a shape is
    drawn with probability proportional to its size, then random positions and signs."""
    shapes = _shapes(r2, D)
    counts = np.array([float(_shape_count(s, D)) for s in shapes])
    prob = counts / counts.sum()
    found = np.zeros((0, D), dtype=np.int64)
    for _ in range(50):
        m = 2 * (n - len(found)) + 16
        which = rng.choice(len(shapes), size=m, p=prob)
        k = np.zeros((m, D), dtype=np.int64)
        for i in np.unique(which):
            rows = np.flatnonzero(which == i)
            p = len(shapes[i])
            pos = np.argsort(rng.random((len(rows), D)), axis=1)[:, :p]   # random ordered slots
            sgn = rng.choice((-1, 1), size=(len(rows), p))
            k[rows[:, None], pos] = sgn * np.array(shapes[i], dtype=np.int64)
        found = np.unique(np.concatenate([found, _canonical(k)]), axis=0)
        if len(found) >= n:
            break
    if len(found) > n:
        found = found[rng.choice(len(found), n, replace=False)]
    return found


def orbit_wave_vectors(r2_max: int, D: int, n_max: int, seed=0) -> np.ndarray:
    """For every orbit |k|^2 = r2, r2 = 1 .. r2_max, the same number n_max // r2_max of wave
    vectors (at least 1), one per pair {k, -k}, drawn uniformly on the orbit; every vector
    of the orbits that hold fewer. int64, sorted by |k|^2."""
    rng = np.random.default_rng(seed)
    r2_max = int(r2_max)
    per_orbit = max(1, int(n_max) // max(r2_max, 1))
    out = []
    for r2 in range(1, r2_max + 1):
        size = orbit_size(r2, D) // 2
        if size == 0:
            continue
        out.append(_enumerate_orbit(r2, D) if size <= per_orbit
                   else _sample_orbit(r2, D, per_orbit, rng))
    if not out:
        return np.zeros((0, D), dtype=np.int64)
    return np.concatenate(out)
