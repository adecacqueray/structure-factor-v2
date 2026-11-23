"""
structure_factor_v2: structure factor S(k) of point sets in the unit cube [0, 1)^D, and plots.

>>> import structure_factor_v2 as sf
>>> points = sf.jittered_lattice(10_000, 2)          # a hyperuniform test pattern
>>> kint, S = sf.structure_factor(points)            # raw scattering intensity
>>> sf.plot_structure_factor(points)                 # log-log plot, radial average
>>> sf.plot_structure_factor(points, trust_boundary=False)   # non-periodic window: Kaiser taper
>>> sf.plot(points)                                  # scatter plot of the points
>>> sf.plot_structure_factor(points, backend="keops", precision="float64")

Layout
------
structure_factor  estimation of S(k): scattering intensity, Kaiser taper, radial averages
viz               plot (point sets), plot_structure_factor
lattices          Korobov, jittered and curled lattices (test patterns, warm starts)
torus             integer wave vectors of Z^D (grids, balls, isotropic sampling)
backend           Fourier sums: finufft, cufinufft, keops, torch, numpy, cupy; check_gpu

Tunable constants live in `structure_factor_v2.CONFIG` (editable in place).
"""
from .backend import BACKENDS, available_backends, check_gpu
from .lattices import curled_lattice, jittered_lattice, korobov_lattice
from .structure_factor import (
    CFG as CONFIG,
    auto_beta,
    structure_factor,
    structure_factor_and_average,
    tapered_kmin,
)
from .viz import plot, plot_structure_factor

__all__ = [
    "structure_factor", "structure_factor_and_average", "tapered_kmin", "auto_beta",
    "plot", "plot_structure_factor",
    "jittered_lattice", "curled_lattice", "korobov_lattice",
    "check_gpu", "available_backends", "BACKENDS", "CONFIG",
]

__version__ = "2.0.0"
