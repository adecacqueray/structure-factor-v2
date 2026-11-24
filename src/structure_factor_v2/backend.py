"""
Backends of the Fourier sums F(n) = sum_j c_j exp(2i pi n.x_j), for points x (N, D) in
[0, 1)^D, real weights c (N,) and integer wave vectors n (M, D).

    backend       method                                   dimensions  accuracy
    "finufft"     type-1 non-uniform FFT, CPU               D <= 3      nufft_tolerance
    "cufinufft"   type-1 non-uniform FFT, GPU (+ cupy)      D <= 3      nufft_tolerance
    "keops"       direct sums, pykeops (GPU if available)   any         exact
    "torch"       direct sums, torch (cuda if available)    any         exact
    "numpy"       direct sums, numpy, chunked               any         exact
    "cupy"        direct sums, cupy, chunked                any         exact

resolve_backend("auto", D): D <= 3 -> cufinufft if it works, else finufft; D >= 4 -> cupy if
it works, else numpy. device="cpu" restricts "auto" to finufft / numpy. keops and torch are
optional and only used when asked for. `check_gpu` reports what is usable.

The (cu)finufft calls here are limited to D <= 3; `generalised_nufft` extends them to any D
(3D slices), and `structure_factor` uses it for D <= CONFIG.max_exhaustive_D.
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
from functools import lru_cache

import numpy as np

BACKENDS = ("finufft", "cufinufft", "keops", "torch", "numpy", "cupy")
NUFFT_BACKENDS = ("finufft", "cufinufft")
NUFFT_MAX_DIM = 3
PRECISIONS = ("float32", "float64")
AUTO_NUFFT_TOLERANCE = {"float32": 1e-6, "float64": 1e-12}
CHUNK_ELEMS = 8_000_000                     # direct sums: (points x wave vectors) per chunk
NUFFT_CHUNK = 400_000                       # points per 3D NUFFT call

_MODULES = {
    "finufft": ("finufft",),
    "cufinufft": ("cufinufft", "cupy"),
    "keops": ("pykeops.numpy",),
    "torch": ("torch",),
    "numpy": ("numpy",),
    "cupy": ("cupy",),
}
_PIP = {
    "finufft": "pip install finufft",
    "cufinufft": "pip install cufinufft cupy-cuda12x  (or structure-factor-v2[gpu12])",
    "keops": "pip install pykeops",
    "torch": "pip install torch",
    "numpy": "pip install numpy",
    "cupy": "pip install cupy-cuda12x",
}


# ── Loading and selection ─────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _try_load(backend: str):
    """(module, None) if the backend imports and works (tested once), else (None, error)."""
    try:
        mods = [importlib.import_module(m) for m in _MODULES[backend]]
        if backend in ("cufinufft", "cupy"):
            cp = mods[-1]
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device found")
        if backend == "keops":                 # pykeops often imports where it cannot compile
            x = np.random.rand(10, 3).astype(np.float32)
            LazyTensor = mods[0].LazyTensor
            ((LazyTensor(x[:, None, :]) - LazyTensor(x[None, :, :])) ** 2).sum(-1).sum_reduction(axis=1)
        return mods[0], None
    except Exception as exc:  # noqa: BLE001 (any failure means "unusable")
        return None, exc


def _load(backend: str):
    mod, err = _try_load(backend)
    if mod is None:
        raise ImportError(f"structure_factor_v2: backend {backend!r} is not available "
                          f"({err}). Install it with `{_PIP[backend]}`.") from None
    return mod


def is_available(backend: str) -> bool:
    return _try_load(backend)[0] is not None


def available_backends(D: int | None = None) -> list:
    """Backends that import and work (and, if D is given, that support dimension D)."""
    out = [b for b in BACKENDS if is_available(b)]
    if D is not None and D > NUFFT_MAX_DIM:
        out = [b for b in out if b not in NUFFT_BACKENDS]
    return out


def resolve_backend(backend: str, D: int, device: str = "auto") -> str:
    """The backend actually used: checks a named one, picks one for "auto"."""
    if backend == "auto":
        gpu = device.lower() in ("auto", "gpu", "cuda")
        if D <= NUFFT_MAX_DIM:
            return "cufinufft" if gpu and is_available("cufinufft") else "finufft"
        return "cupy" if gpu and is_available("cupy") else "numpy"
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, expected 'auto' or one of {BACKENDS}")
    if backend in NUFFT_BACKENDS and D > NUFFT_MAX_DIM:
        raise ValueError(f"the {backend} backend only supports D <= {NUFFT_MAX_DIM} (got D={D})")
    _load(backend)
    return backend


def resolve_precision(precision: str) -> str:
    precision = "float64" if precision in ("float64", "double") else precision
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    return precision


def resolve_tolerance(nufft_tolerance, precision: str) -> float:
    if isinstance(nufft_tolerance, str):
        if nufft_tolerance != "auto":
            raise ValueError(f"nufft_tolerance must be a positive float or 'auto', got {nufft_tolerance!r}")
        return AUTO_NUFFT_TOLERANCE[precision]
    if not nufft_tolerance > 0:
        raise ValueError(f"nufft_tolerance must be positive, got {nufft_tolerance!r}")
    return float(nufft_tolerance)


def _real(precision):
    return np.float64 if precision == "float64" else np.float32


def _complex(precision):
    return np.complex128 if precision == "float64" else np.complex64


# ── Fourier sums ──────────────────────────────────────────────────────────────

def nufft_grid_sums(pts, c, n_modes: int, backend: str, precision: str, tol: float) -> np.ndarray:
    """F on the centred cube of n_modes^D modes (finufft modeord=0 order), flattened, by a
    type-1 NUFFT ("finufft" or "cufinufft"). Returns complex128 (n_modes^D,)."""
    nufft = _load(backend)
    if backend == "cufinufft":
        import cupy as xp
        to_np = xp.asnumpy
    else:
        xp, to_np = np, np.asarray
    N, D = pts.shape
    x = [xp.asarray(np.ascontiguousarray(2.0 * np.pi * pts[:, d], dtype=_real(precision)))
         for d in range(D)]
    cc = xp.asarray(np.ascontiguousarray(c, dtype=_complex(precision)))
    shape = (n_modes,) * D
    fn = getattr(nufft, f"nufft{D}d1")
    if D < 3:
        fk = fn(*x, cc, n_modes if D == 1 else shape, eps=tol, isign=1)
    else:
        fk = xp.zeros(shape, dtype=_complex(precision))
        for s in range(0, N, NUFFT_CHUNK):
            fk += fn(*[a[s:s + NUFFT_CHUNK].copy() for a in x], cc[s:s + NUFFT_CHUNK].copy(),
                     shape, eps=tol, isign=1)
    return to_np(fk).ravel().astype(np.complex128)


def direct_sums(pts, c, kint, backend: str, precision: str) -> np.ndarray:
    """F on the wave vectors kint (M, D) by direct sums ("keops", "torch", "numpy", "cupy").
    Returns complex128 (M,)."""
    real = _real(precision)
    pts = np.ascontiguousarray(pts, dtype=real)
    c = np.ascontiguousarray(c, dtype=real)
    k2pi = np.ascontiguousarray(2.0 * np.pi * np.asarray(kint), dtype=real)
    if backend == "keops":
        LazyTensor = _load("keops").LazyTensor
        phase = (LazyTensor(pts[:, None, :]) * LazyTensor(k2pi[None, :, :])).sum(-1, keepdim=True)
        w = LazyTensor(c.reshape(-1, 1, 1))
        re = np.asarray((w * phase.cos()).sum(0)).ravel()
        im = np.asarray((w * phase.sin()).sum(0)).ravel()
        return re.astype(np.float64) + 1j * im.astype(np.float64)

    if backend == "torch":
        torch = _load("torch")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dt = getattr(torch, np.dtype(real).name)
        asx = lambda a: torch.as_tensor(a, dtype=dt, device=dev)   # noqa: E731
        to_np = lambda a: a.cpu().numpy()                          # noqa: E731
        cos, sin = torch.cos, torch.sin
    elif backend in ("numpy", "cupy"):
        xp = _load(backend)
        asx = lambda a: xp.asarray(a, dtype=real)                  # noqa: E731
        to_np = xp.asnumpy if backend == "cupy" else np.asarray
        cos, sin = xp.cos, xp.sin
    else:
        raise ValueError(f"{backend!r} is not a direct-sum backend")

    kt, ct = asx(k2pi), asx(c)
    M = len(k2pi)
    re, im = asx(np.zeros(M)), asx(np.zeros(M))
    chunk = max(1, CHUNK_ELEMS // max(M, 1))
    for s in range(0, len(pts), chunk):
        phase = asx(pts[s:s + chunk]) @ kt.T                        # (n, M)
        re += ct[s:s + chunk] @ cos(phase)
        im += ct[s:s + chunk] @ sin(phase)
    return to_np(re).astype(np.float64) + 1j * to_np(im).astype(np.float64)


# ── Diagnostic ────────────────────────────────────────────────────────────────

def check_gpu() -> None:
    """Print which backends work: the GPU stack (CuPy, cuFINUFFT) and the optional KeOps, torch."""
    print("GPU support diagnostic\n----------------------")
    print("NVIDIA GPU detected." if _nvidia_gpu() else "No NVIDIA GPU detected.")
    for backend in ("cupy", "cufinufft", "keops", "torch"):
        mod, err = _try_load(backend)
        print(f"{backend}: {'OK' if mod is not None else f'unavailable ({err})'}")
    print(f"auto backend: D <= 3 -> {resolve_backend('auto', 2)}, D >= 4 -> {resolve_backend('auto', 4)}")


def _nvidia_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        return out.returncode == 0 and bool(out.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False
