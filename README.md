# structure-factor-v2

[![PyPI](https://img.shields.io/pypi/v/structure-factor-v2.svg)](https://pypi.org/project/structure-factor-v2/)
[![Python](https://img.shields.io/pypi/pyversions/structure-factor-v2.svg)](https://pypi.org/project/structure-factor-v2/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/adecacqueray/structure-factor-v2/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-adecacqueray%2Fstructure--factor--v2-181717?logo=github)](https://github.com/adecacqueray/structure-factor-v2)

**Structure factor $S(k)$ of point sets in the unit cube $[0, 1)^D$**, estimated by the
scattering intensity, with fast transforms (finufft / cufinufft) and ready-made log-log plots.

The structure factor measures the density fluctuations of a point set in Fourier space
(it is the power spectrum, up to normalisation). For a hyperuniform (blue-noise) pattern,
$S(k) \to 0$ as $k \to 0$; for an order-$\alpha$ hyperuniform pattern $S(k) \sim k^\alpha$.

---

## Installation

```bash
pip install structure-factor-v2              # CPU: numpy + finufft
#pip install structure-factor-v2[gpu12]      # GPU: [gpu12] or [gpu13], cupy + cufinufft
```

---

## Quick start

```python
import structure_factor_v2 as sf

x = sf.jittered_lattice(10_000, 2)           # (N, D) points in [0, 1)^D, hyperuniform

sf.plot(x)                                   # scatter plot
sf.plot_structure_factor(x)                  # log-log S(k), raw values + radial average

kint, S = sf.structure_factor(x)             # raw estimate: integer wave vectors (M, D), S (M,)
k, S, kgroup, Sgroup = sf.structure_factor_and_average(x)   # normalised k, raw + smoothed
```

A jittered lattice (a Korobov lattice, randomly shifted, every point moved uniformly within its
cell) is a cheap hyperuniform pattern: $S(k) \sim k^2$ at small $k$. Compare with i.i.d. points
($S \equiv 1$) and with the bare lattice (a crystal: $S = 0$ except on its Bragg peaks):

```python
import numpy as np
sf.plot_structure_factor(np.random.rand(10_000, 2))          # Poisson: S = 1
sf.plot_structure_factor(sf.korobov_lattice(10_000, 2))      # crystal
```

A curled lattice goes further: the Korobov lattice is moved along a random periodic
divergence-free flow (curl noise jittering, Bærentzen, Frisvad & Martínez, SIGGRAPH Asia
2023). The flow preserves volume, so the points stay well separated, and the Bragg peaks are
smeared out: a blue noise point set, with a much lower $S(k)$ at small $k$ than the jittered
lattice. Any dimension (tuned in 2D and 3D), about 3 s for $10^5$ points.

```python
x = sf.curled_lattice(10_000, 2)                             # noise_scale=5.0, dt=1.2
sf.plot_structure_factor(x)
```

Wave numbers returned by `structure_factor_and_average` (and shown by
`plot_structure_factor`) are normalised so that $k = 1$ is the inter-particle wave
number $N^{1/D}$.

---

## Two estimators: `trust_boundary`

| `trust_boundary` | When                                                        | Estimator |
|------------------|-------------------------------------------------------------|-----------|
| `True` (default) | the points are periodic on the unit torus                   | $S(n) = \lvert\sum_j e^{2i\pi n \cdot x_j}\rvert^2 / N$ on the integer wave vectors $n$ |
| `False`          | the points are a window cut out of a larger, non-periodic pattern | points restricted to $[m, 1-m)^D$, separable Kaiser–Bessel taper, debiased with the density |
| `"both"`         | (plots only) compare the two                                |           |

```python
sf.plot_structure_factor(x, trust_boundary=False)              # margin m = 0.1, beta automatic
sf.plot_structure_factor(x, trust_boundary=False, margin=0.05, beta=10)
sf.plot_structure_factor(x, trust_boundary="both")
```

With the taper, leakage decays like $e^{-2\beta}$ while the resolution at small $k$
coarsens (about $\beta/\pi$ modes). `beta=None` picks
$\beta = \min(15, \pi\, n_{\text{axis}} / 4)$, see `sf.auto_beta`; nothing is plotted
below the resolution limit `sf.tapered_kmin(N, D, margin, beta)`.

---

## Backends, precision and tolerance

Every function takes `backend`, `precision` and `nufft_tolerance`:

| `backend`     | method                                   | dimensions | accuracy |
|---------------|------------------------------------------|------------|----------|
| `"finufft"`   | type-1 non-uniform FFT, CPU              | D <= 3     | `nufft_tolerance` |
| `"cufinufft"` | type-1 non-uniform FFT, GPU (+ cupy)     | D <= 3     | `nufft_tolerance` |
| `"keops"`     | direct sums, pykeops (GPU if available)  | any        | exact |
| `"torch"`     | direct sums, torch (cuda if available)   | any        | exact |
| `"numpy"`     | direct sums, numpy, chunked              | any        | exact |
| `"cupy"`      | direct sums, cupy, chunked               | any        | exact |

`"auto"` (default): (cu)finufft for D <= 3 (cufinufft when it works), direct sums with cupy or
numpy above; `device="cpu"` keeps "auto" off the GPU. `keops` and `torch` are optional
installs, only used when asked for (KeOps counts as working only if it compiles a test
reduction).

| Dimension | Wave vectors |
|-----------|--------------|
| $D \le 3$ | every integer mode of the cube up to $2 N^{1/D}$ |
| $D \ge 4$ | all low modes + isotropically sampled high modes (`resolution`) |

```python
sf.plot_structure_factor(x, backend="keops")                  # exact sums
sf.plot_structure_factor(x, precision="float64", nufft_tolerance=1e-14)
sf.available_backends(D=2)                                    # what works here
sf.check_gpu()
```

`nufft_tolerance="auto"` means 1e-5 in float32 and 1e-12 in float64; float32 sets a floor on
the smallest resolvable S (float64 for S below ~1e-6 at large N).

Tunable constants (margin, $k_{\max}$, radial bins, smoothing...) are in `sf.CONFIG`, editable in place:

```python
sf.CONFIG.k_max = 3.0          # wave numbers up to 3 N^(1/D)
```

---

## API

| Function | Returns |
|----------|---------|
| `structure_factor(points, resolution=2000, device="auto", precision="float32", trust_boundary=True, margin=None, beta=None, backend="auto", nufft_tolerance="auto")` | `kint (M, D)`, `S (M,)`, sorted by $\lvert n\rvert$ |
| `structure_factor_and_average(points, resolution=20000, min_val=1e-20, precision="float32", trust_boundary=True, margin=None, beta=None, backend="auto", nufft_tolerance="auto", device="auto")` | `kraw, Sraw, kgroup, Sgroup` |
| `plot_structure_factor(points, resolution=20000, precision="float32", ax=None, return_fig=False, title=None, trust_boundary=True, margin=None, beta=None, backend="auto", nufft_tolerance="auto", device="auto", **plot_kw)` | `None`, or `(fig, ax)` |
| `plot(points, auto_zoom=False, max_points=30000, ax=None, return_fig=False, figsize=(8, 8), **scatter_kw)` | `None`, or `(fig, ax)` |
| `jittered_lattice(N, D, seed=None, max_candidates=100)`, `korobov_lattice(N, D, shift=True, seed=None, max_candidates=100)` | `(N, D)` test point sets |
| `curled_lattice(N, D, noise_scale=5.0, dt=1.2, n_modes=256, bandwidth=0.3, n_steps=1, seed=None, max_candidates=100)` | `(N, D)` blue noise point set |
| `auto_beta(n_points, D, margin)`, `tapered_kmin(n_points, D, margin=None, beta=None)` | Kaiser parameter, smallest trustworthy $k$ |
| `check_gpu()`, `available_backends(D=None)`, `BACKENDS` | diagnostic, backends that work here |

Points must lie in $[0, 1)^D$; this is not checked.

---

## References

- **Korobov lattices**: I. H. Sloan and S. Joe. *Lattice Methods for Multiple Integration.*
  Oxford University Press, 1994.
- **Curl noise jittering**: J. A. Bærentzen, J. R. Frisvad, and J. Martínez. *Curl Noise
  Jittering.* SIGGRAPH Asia 2023 Conference Papers, article 88, 2023.
  [doi:10.1145/3610548.3618163](https://doi.org/10.1145/3610548.3618163),
  code: [jonasmb/curlnoisejittering](https://github.com/jonasmb/curlnoisejittering).
  `curled_lattice` starts from a Korobov lattice instead of their triangular / cubic grids,
  and uses a random Fourier divergence-free field (exactly periodic, any dimension) instead
  of the curl of a Perlin noise; independent implementation.
- **finufft / cufinufft**: A. H. Barnett, J. F. Magland, and L. af Klinteberg.
  *A parallel non-uniform fast Fourier transform library based on an "exponential of
  semicircle" kernel.* SIAM J. Sci. Comput. 41(5), C479–C504, 2019.
  See [finufft](https://github.com/flatironinstitute/finufft).
- **structure-factor**: the original package of the same group,
  [For-a-few-DPPs-more/structure-factor](https://github.com/For-a-few-DPPs-more/structure-factor).

Used by [blue-sampler](https://pypi.org/project/blue-sampler/), which re-exports
`plot`, `plot_structure_factor`, `structure_factor` and `structure_factor_and_average`.

---

## License

MIT, see [LICENSE](LICENSE).
