"""plot (point sets) and plot_structure_factor."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .structure_factor import structure_factor_and_average, tapered_kmin


def _show_or_return(fig, axes, return_fig: bool):
    plt.tight_layout()
    if return_fig:
        return fig, axes
    plt.show()
    return None


def plot(
    points: NDArray,
    auto_zoom: bool = False,
    max_points: int = 30_000,
    ax: plt.Axes | None = None,
    return_fig: bool = False,
    figsize: tuple = (8, 8),
    **scatter_kw,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Scatter plot of a point set (N, D), or (N, K, D) flattened. Only the first 3 coordinates
    are shown (a 3D plot if D >= 3).

    auto_zoom : show a random sub-box holding about max_points points (large sets stay readable).
    ax : existing axes (with projection="3d" for 3D data). Otherwise a new figure of figsize.
    return_fig : return (fig, ax) instead of calling plt.show().
    **scatter_kw : forwarded to ax.scatter (color, s, alpha...).
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])
    D = min(pts.shape[-1], 3)
    pts = pts[:, :D]

    if auto_zoom and len(pts) > max_points:
        # keep a random sub-box holding ~max_points points
        unit = (pts - pts.min()) / (pts.max() - pts.min())
        zoom = (max_points / len(pts)) ** (1.0 / D)
        lo = (1 - zoom) * np.random.rand(D)
        pts = pts[((lo <= unit) & (unit <= lo + zoom)).all(axis=1)]

    kw = dict(s=10_000 / len(pts), color="black")
    kw.update(scatter_kw)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        if D == 2:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    if D == 2:
        ax.scatter(pts[:, 0], pts[:, 1], **kw)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **kw)

    ax.set_axis_off()
    return _show_or_return(fig, ax, return_fig)


def plot_structure_factor(
    points: NDArray,
    resolution: int = 20_000,
    precision: str = "float32",
    ax: plt.Axes | None = None,
    return_fig: bool = False,
    title: str = None,
    trust_boundary: bool | str = True,
    margin: float | None = None,
    beta: float | None = None,
    backend: str = "auto",
    nufft_tolerance: float | str = "auto",
    device: str = "auto",
    **plot_kw,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Log-log plot of the radially averaged structure factor S(k) of points in [0, 1)^D.

    S is estimated by the scattering intensity on the integer wave vectors up to 2 N^(1/D);
    k is normalised so that k = 1 is the inter-particle wave number N^(1/D). Raw values are
    shown in grey, their smoothed radial average as a curve. For an order-alpha hyperuniform
    pattern S(k) ~ k^alpha at small k.

    Parameters
    ----------
    points : (N, D) array in the unit cube (not checked).
    resolution : number of wave vectors shown. For D >= 4 (no fast transform) only
        resolution / 10 are evaluated: ask for 200_000 to get 20_000.
    precision : "float32", or "float64" (much slower) to resolve S below ~1e-10.
    ax, return_fig : draw on existing axes; return (fig, ax) instead of showing.
    title : default depends on trust_boundary.
    trust_boundary : True, False or "both".
        True: plain estimate, for patterns that are periodic on the unit torus.
        False: for a window cut out of a larger pattern. The points are restricted to
        [m, 1 - m)^D (m = margin) and the estimate is debiased with a Kaiser taper.
        "both": plot both estimates.
    margin : default structure_factor_v2.CONFIG.margin (0.1), trust_boundary=False only.
    beta : Kaiser parameter, trust_boundary=False only. Larger: less leakage (~exp(-2 beta)),
        coarser resolution at small k. None: automatic (`structure_factor.auto_beta`).
    backend, nufft_tolerance, device : Fourier sums, as in `structure_factor`.
    **plot_kw : forwarded to ``ax.loglog`` for the smoothed curve.
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])
    N, D = pts.shape

    if isinstance(trust_boundary, str):
        if trust_boundary != "both":
            raise ValueError("trust_boundary must be True, False or 'both'")
        modes = [True, False]
        both = True
    else:
        modes = [bool(trust_boundary)]
        both = False

    # --- Axes setup ---
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    ax.set_axisbelow(True)
    ax.grid(True, which="both", alpha=0.4, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")

    for trust in modes:
        # --- Compute structure factor and averages ---
        k, S, kgroup, Sgroup = structure_factor_and_average(
            pts, resolution=resolution, precision=precision,
            trust_boundary=trust, margin=margin, beta=beta,
            backend=backend, nufft_tolerance=nufft_tolerance, device=device,
        )
        if not trust:
            # nothing is shown below the resolution limit of the taper
            kmin = tapered_kmin(N, D, margin, beta)
            keep, keep_g = k >= kmin, kgroup >= kmin
            k, S, kgroup, Sgroup = k[keep], S[keep], kgroup[keep_g], Sgroup[keep_g]
            if len(kgroup) < 2:
                warnings.warn("trust_boundary=False: no wave number above the resolution "
                              "limit (too few points in the truncated window); curve not drawn.")
                continue

        # --- Scatter (raw values) ---
        # single mode: neutral grays; 'both': same hue as the smoothed curve,
        # bluish gray (trusted, tab:blue) / orangish gray (tapered, tab:orange)
        if not both:
            c_light, c_dark = "lightgray", "gray"
        elif trust:
            c_light, c_dark = "#b8c4d0", "#6b7f94"
        else:
            c_light, c_dark = "#d8c7b6", "#9c8068"
        Sgroup_interp = np.exp(np.interp(np.log(k), np.log(kgroup), np.log(Sgroup)))
        bigS = S >= 30 * Sgroup_interp
        ax.scatter(k[~bigS], S[~bigS], s=5, color=c_light, alpha=0.6, zorder=2)
        ax.scatter(k[bigS], S[bigS], s=20, color=c_dark, alpha=1.0, zorder=2)

        # --- Smoothed curve ---
        curve_kw = {"color": "tab:blue", "zorder": 3, "label": None,
                    "marker": "o", "markersize": 2, "linewidth": 2}
        curve_kw.update(plot_kw)
        if both:
            curve_kw["label"] = "boundary trusted" if trust else "truncated + taper"
            if not trust:
                curve_kw["color"] = "tab:orange"
                curve_kw["zorder"] = 4
        ax.loglog(kgroup, Sgroup, **curve_kw)

    if ax.get_legend_handles_labels()[1]:
        ax.legend()

    # --- Reference lines and axis limits (only when we own the figure) ---
    if own_fig:
        ax.axvline(1, color="green", linewidth=2, alpha=0.2, zorder=1)
        ax.axhline(1, color="green", linewidth=2, alpha=0.2, zorder=1)
        ax.set_xlim(right=2)
        ax.set_ylim(top=2)
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$S(k)$")

    if both:
        default_title = "Structure factor (log-log): boundary trusted vs truncated + taper"
    elif not modes[0]:
        default_title = "Structure factor (log-log, truncated + Kaiser taper)"
    else:
        default_title = "Structure factor (log-log, scattering intensity)"
    ax.set_title(title or default_title)
    return _show_or_return(fig, ax, return_fig)