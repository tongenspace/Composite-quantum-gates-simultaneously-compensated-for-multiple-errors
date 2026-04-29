import math
from typing import List, Tuple

TwoPi = 2.0 * math.pi


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, FuncFormatter

import json, os, datetime as dt
from pathlib import Path
import numpy as np

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.circuit.library import XGate, SXGate, RZGate
from qiskit.circuit import Measure

def plot_contoursnotog(results1, order=3, range=0.2, N_POINTS=1000, name='', operator='Time_robust_x'):

    # axes grids
    x = np.linspace(-range, range, N_POINTS)   # detuning
    y = np.linspace(-range, range, N_POINTS)   # error (duration or amplitude)
    X, Y = np.meshgrid(x, y)

    # LaTeX + fonts + bigger tick labels
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    # data (ensure float, mask non-positive for log scale)
    Z = np.asarray(results1, dtype=float).T
    Z = np.where(Z > 0, Z, np.nan)

    # Labeled iso-levels (same as before)
    exponents = np.arange(-4, 0)              # -4, -3, -2, -1
    iso_levels = 10.0 ** exponents

    # Filled background: include one decade below to show "≤1e-4" clearly
    vmin = 10.0 ** (exponents.min() - 1)      # 1e-5
    vmax = 10.0 ** (exponents.max())          # 1e-1
    filled_bounds = np.logspace(np.log10(vmin), np.log10(vmax), 5)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Filled contour (log scale) so regions are unambiguous
    CF = ax.contourf(
        X, Y, Z,
        levels=filled_bounds,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        #cmap="viridis",
        extend="min",     # show extension below vmin in colorbar
    )

    # Overlay isolines exactly at 10^{-4..-1}
    CS = ax.contour(X, Y, Z, levels=iso_levels, colors="k", linewidths=1.2)
    ax.clabel(
        CS, inline=True,
        fmt=lambda v: rf"$10^{{{int(np.round(np.log10(v)))}}}$",
        fontsize=15
    )

    # Colorbar with LaTeX decade ticks
    decade_fmt = FuncFormatter(lambda v, pos: rf"$10^{{{int(np.round(np.log10(v)))}}}$")
    cbar = fig.colorbar(CF, ax=ax, pad=0.012, fraction=0.06, ticks=iso_levels, format=decade_fmt)
    cbar.set_label(r"\textbf{Infidelity}", fontsize=18)
    cbar.ax.tick_params(which="both", labelsize=16)

    # Axis labels
    if operator == 'Time_robust_x':
        ax.set_ylabel(r"Pulse duration error", fontsize=19)
    else:
        ax.set_ylabel(r"Rabi frequency error", fontsize=19)
    ax.set_xlabel(r"Detuning $\Delta$", fontsize=19)

    # Ticks, spines
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    for s in ax.spines.values():
        s.set_linewidth(0.6)

    plt.tight_layout()

    # Save (create folder if needed)
    outdir = f"Plots\\{operator}"
    os.makedirs(outdir, exist_ok=True)
    outname = name if name else f"N={order}_times_robust_non_rob"
    plt.savefig(os.path.join(outdir, f"{outname}.svg"), dpi=300, bbox_inches="tight")
    plt.close()



def _wrap_phase(phi: float) -> float:
    """Map phase to [0, 2π)."""
    phi = phi % TwoPi
    return phi if phi >= 0 else phi + TwoPi

def _split_into_2pi_segments(theta: float, phase: float) -> List[Tuple[float, float]]:
    """Represent a rotation angle as k copies of 2π plus a remainder (all with same phase)."""
    segs = []
    while theta >= TwoPi - 1e-12:
        segs.append((TwoPi, _wrap_phase(phase)))
        theta -= TwoPi
    if theta > 1e-12:
        segs.append((theta, _wrap_phase(phase)))
    return segs

def corpse_pulse(theta, base_phase=0.0, n1=1, n2=1, n3=0):
    delta = np.arcsin(np.sin(theta/2)/2.0)

    θ1 = 2*π*n1 + theta/2 - delta
    θ2 = 2*π*n2 - 2*delta
    θ3 = 2*π*n3 + theta/2 - delta

    thetas = np.array([θ1, θ2, θ3])
    phis   = np.array([base_phase, base_phase+π, base_phase])
    phis_teta = np.concatenate([phis/np.pi,thetas/np.pi])
    return phis_teta

# ---------- SCROFULOUS (pulse-length error robust) ----------

def arcsinc(y: float, tol: float = 1e-12, max_iter: int = 100) -> float:
    """
    Inverse of sinc on (0, π]: find x in (0,π] such that sin(x)/x = y.
    Assumes 0 ≤ y ≤ 1. Uses bisection (monotone on this interval).
    """
    if not (0.0 <= y <= 1.0):
        raise ValueError("arcsinc expects 0 ≤ y ≤ 1.")
    lo, hi = 1e-12, math.pi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = math.sin(mid) / mid
        if abs(val - y) < tol:
            return mid
        if val > y:
            lo = mid   # need larger x
        else:
            hi = mid   # need smaller x
    return 0.5 * (lo + hi)

def scrofulous(theta: float, base_phase: float = 0.0) -> List[Tuple[float, float]]:
    """
    SCROFULOUS (Short Composite ROtation For Undoing Length Over and Under Shoot)
    robust to pulse-length (amplitude) error.
    Returns [(θ1, φ1), (π, φ2), (θ1, φ1)] in radians.
    """
    # θ1 = arcsinc( 2 cos(θ/2) / π ),  θ2 = π
    y = (2.0 * math.cos(theta / 2.0)) / math.pi
    # clamp numerical roundoff
    y = max(0.0, min(1.0, y))
    th1 = arcsinc(y)
    th2 = math.pi

    # φ1 = φ3 = arccos( -π cos θ1 / (2 θ1 sin(θ/2)) )
    denom = (2.0 * th1 * math.sin(theta / 2.0))
    if abs(denom) < 1e-14:
        raise ValueError("SCROFULOUS undefined for this θ (denominator ~ 0).")
    arg1 = -math.pi * math.cos(th1) / denom
    arg1 = max(-1.0, min(1.0, arg1))
    phi1 = math.acos(arg1)

    # φ2 = φ1 − arccos( -π / (2 θ1) )
    arg2 = -math.pi / (2.0 * th1)
    arg2 = max(-1.0, min(1.0, arg2))
    phi2 = phi1 - math.acos(arg2)

    return np.concatenate([[base_phase + phi1,base_phase + phi2,base_phase + phi1],[th1,th2,th1]])/np.pi


import math
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass

# ---------- Primitive analytic building blocks ----------

def sk1(theta: float, base_phase: float = 0.0) -> List[Tuple[float, float]]:
    """
    SK1 (first-order Solovay–Kitaev composite) – cancels amplitude error to O(ε^2).
    Sequence:
        R_φ(θ), R_{φ+ϕ}(2π), R_{φ−ϕ}(2π)
      with ϕ = arccos( -θ / (4π) ).
    """
    ϕ = math.acos(-theta / (4.0 * math.pi))
    return [
        (theta,          base_phase),
        (2.0 * math.pi,  base_phase + ϕ),
        (2.0 * math.pi,  base_phase - ϕ),
    ]

def bb1(theta: float, base_phase: float = 0.0, symmetric: bool = True) -> List[Tuple[float, float]]:
    """
    BB1 ≡ B2 (Wimperis broadband) – cancels amplitude error through O(ε^3).
    Symmetric 5-pulse form:
        R_φ(θ/2), R_{φ+ϕ}(π), R_{φ+3ϕ}(2π), R_{φ+ϕ}(π), R_φ(θ/2)
      with ϕ = arccos( -θ / (4π) ).
    Asymmetric 4-pulse variant provided if symmetric=False.
    """
    ϕ = math.acos(-theta / (4.0 * math.pi))
    if symmetric:
        return [
            (0.5 * theta,    base_phase),
            (math.pi,        base_phase + ϕ),
            (2.0 * math.pi,  base_phase + 3.0 * ϕ),
            (math.pi,        base_phase + ϕ),
            (0.5 * theta,    base_phase),
        ]
    else:
        return [
            (theta,          base_phase),
            (math.pi,        base_phase + ϕ),
            (2.0 * math.pi,  base_phase + 3.0 * ϕ),
            (math.pi,        base_phase + ϕ),
        ]

# ---------- Generic gate infidelity model vs amplitude error ----------
# This is a portable SU(2) simulator (no JAX) to *design* phases.
# You can still *execute* with your JAX kernel later.

def Rx(angle: float, phase: float) -> np.ndarray:
    """Rotation by 'angle' about equatorial axis at azimuth 'phase' (radians)."""
    # Axis n = (cos φ, sin φ, 0); generator = -i (n·σ)/2
    c = math.cos(angle/2.0)
    s = math.sin(angle/2.0)
    nx, ny = math.cos(phase), math.sin(phase)
    return np.array([
        [c - 1j*0.0, -1j*s*(nx - 1j*ny)],
        [-1j*s*(nx + 1j*ny), c + 1j*0.0]
    ], dtype=np.complex128)

def composite_U_amplitude_error(seq: List[Tuple[float, float]], eps: float) -> np.ndarray:
    """
    Apply sequence with amplitude scale (1+eps): angle_i -> (1+eps)*angle_i.
    """
    U = np.eye(2, dtype=np.complex128)
    for ang, ph in seq:
        U = Rx((1.0+eps)*ang, ph) @ U
    return U

def avg_gate_infidelity(U: np.ndarray, V: np.ndarray) -> float:
    Ud = np.conjugate(U.T)
    tr = np.trace(Ud @ V)
    d = 2.0
    F_avg = (np.abs(tr)**2 + d) / (d*(d+1.0))
    return 1.0 - F_avg

def target_gate(theta: float, base_phase: float = 0.0) -> np.ndarray:
    return Rx(theta, base_phase)

# ---------- Taylor conditions for amplitude error cancellation ----------

def amplitude_error_conditions(seq: List[Tuple[float, float]],
                               theta: float,
                               base_phase: float,
                               order: int) -> np.ndarray:
    """
    Compute derivative conditions of average gate infidelity r(ε) at ε=0 up to 'order'.
    We enforce ∂^k r / ∂ε^k |_{0} = 0 for k = 1..order.
    """
    V = target_gate(theta, base_phase)
    # finite-difference derivatives at eps=0
    # use symmetric differences with shrinking step
    ks = list(range(1, order+1))
    derivs = []
    for k in ks:
        h = 1e-5  # base step
        # compute k-th derivative via central differences recursively (simple formula)
        # For robustness, use Richardson extrapolation on first derivative building blocks
        def r_eps(ε):
            U = composite_U_amplitude_error(seq, ε)
            return avg_gate_infidelity(U, V)
        # central finite difference for k-th derivative (simple implementation)
        # D1 ≈ (r(h)-r(-h))/(2h), D2 ≈ (r(h)-2r(0)+r(-h))/h^2, etc.
        if k == 1:
            val = (r_eps(h) - r_eps(-h)) / (2*h)
        elif k == 2:
            val = (r_eps(h) - 2*r_eps(0.0) + r_eps(-h)) / (h**2)
        elif k == 3:
            val = (r_eps(2*h) - 2*r_eps(h) + 2*r_eps(-h) - r_eps(-2*h)) / (2*h**3)
        elif k == 4:
            val = (r_eps(2*h) - 4*r_eps(h) + 6*r_eps(0.0) - 4*r_eps(-h) + r_eps(-2*h)) / (h**4)
        else:
            # crude fallback using numpy's polynomial fit around 0
            pts = np.array([-2*h, -h, 0.0, h, 2*h])
            ys = np.array([r_eps(e) for e in pts])
            coeffs = np.polyfit(pts, ys, 4)  # degree-4 fit around 0
            # derivative of order k at 0 from polynomial coefficients
            poly = np.poly1d(coeffs)
            dpoly = poly.deriv(m=k)
            val = dpoly(0.0)
        derivs.append(val)
    return np.array(derivs)

# ---------- Bn / SKn synthesis via nonlinear solve ----------

@dataclass
class SynthesisResult:
    name: str
    theta: float
    order: int
    phases: np.ndarray
    sequence: List[Tuple[float, float]]

def _bn_template(theta: float, base_phase: float, n: int,
                 symmetric: bool = True) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Template for B_n (broadband) with n 'corrector blocks'. Generalizes BB1 pattern:
      Symmetric:  [θ/2] + Σ_k [π at φ_k, 2π at ψ_k, π at φ_k] + [θ/2]
      Asymmetric: [θ]   + Σ_k [...] (less phase-symmetric).
    Return initial sequence with placeholder phases (0) and a structure index list:
      indices of entries whose phases are free variables to optimize, in order:
      [φ1, ψ1, φ2, ψ2, ..., φn, ψn] (symmetric case).
    """
    seq: List[Tuple[float, float]] = []
    free_indices: List[int] = []

    if symmetric:
        seq.append((0.5*theta, base_phase))  # front half
    else:
        seq.append((theta, base_phase))      # full theta first

    # Add n corrector blocks: (π)_φk, (2π)_ψk, (π)_φk
    for k in range(n):
        # φ_k
        free_indices.append(len(seq))
        seq.append((math.pi, 0.0))
        # ψ_k
        free_indices.append(len(seq))
        seq.append((2.0*math.pi, 0.0))
        # φ_k again
        seq.append((math.pi, 0.0))

    if symmetric:
        seq.append((0.5*theta, base_phase))  # back half

    return seq, free_indices

def _apply_phases(seq: List[Tuple[float, float]],
                  free_indices: List[int],
                  vars_vec: np.ndarray,
                  base_phase: float) -> List[Tuple[float, float]]:
    out = list(seq)
    # map variables to phases (relative to base_phase)
    for i, idx in enumerate(free_indices):
        ang, _ = out[idx]
        out[idx] = (ang, base_phase + vars_vec[i])
    # also mirror φ_k into the third member of each block
    # because each block is (π)_φk, (2π)_ψk, (π)_φk
    # We need to set those φ_k duplicates equal automatically:
    b = 0
    for k in range(len(free_indices)//2):
        phi_idx = free_indices[b]
        psi_idx = free_indices[b+1]
        # duplicate φ into the third member (which is phi_idx+2 within the block)
        out[phi_idx+2] = (out[phi_idx+2][0], out[phi_idx][1])
        b += 2
    return out

def synthesize_Bn(theta: float, n: int,
                  base_phase: float = 0.0,
                  symmetric: bool = True,
                  ϕ_init: Optional[np.ndarray] = None) -> SynthesisResult:
    """
    Broadband B_n synthesis: choose 2n phases to kill amplitude-error derivatives
    up to order n at ε=0 (i.e., r'(0)=...=r^(n)(0)=0).
    For n=1 this produces SK1-like; for n=2 it reproduces BB1 (one valid solution).
    For n>2 it numerically solves a nonlinear system; multiple solutions may exist.
    """
    import scipy.optimize as opt

    # Build template and variable vector
    tmpl, free = _bn_template(theta, base_phase, n, symmetric=symmetric)
    m = len(free)  # = 2n
    if ϕ_init is None:
        # Good heuristic: start near BB1 pattern generalization
        # set all φ_k = arccos(-θ/(4π)), ψ_k = 3*φ_k
        phi0 = math.acos(-theta/(4.0*math.pi))
        x0 = np.array([phi0 if i % 2 == 0 else 3.0*phi0 for i in range(m)], dtype=float)
    else:
        x0 = np.asarray(ϕ_init, dtype=float)
        if x0.size != m:
            raise ValueError(f"ϕ_init must have size {m} for n={n}")

    def residuals(x):
        seq = _apply_phases(tmpl, free, x, base_phase)
        derivs = amplitude_error_conditions(seq, theta, base_phase, order=n)
        return derivs  # we want these ≈ 0

    sol = opt.least_squares(residuals, x0, method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=5000)

    seq_opt = _apply_phases(tmpl, free, sol.x, base_phase)
    return SynthesisResult(name=f"B{n}", theta=theta, order=n,
                           phases=sol.x.copy(), sequence=seq_opt)

# ---------- SK_n via recursive commutator (Solovay–Kitaev-style) ----------
# We provide a practical SK_n that nests SK1-correctors.
# This is a constructive (not unique) recipe that increases cancellation order.
# For n=1 returns classical SK1; for n>1 we recursively sandwich correction blocks.

def skn(theta: float, n: int, base_phase: float = 0.0) -> List[Tuple[float, float]]:
    """
    Recursive SK_n generator.
    Heuristic construction: SK_{n+1}(θ) = SK_n(θ/2) ∘ C_n ∘ SK_n(θ/2),
    where C_n is an SK1-like corrector with phases optimized (numerically)
    to null the next derivative. For n=1 this reduces to SK1.
    """
    if n <= 1:
        return sk1(theta, base_phase)
    # split rotation and insert a corrector
    left  = skn(theta/2.0, n-1, base_phase)
    # synthesize a single-block B_n corrector for the next order (n)
    # using the numerical solver with one block (n=1) but targetting higher derivative via weighted residual
    res = synthesize_Bn(theta=0.0, n=1, base_phase=base_phase, symmetric=True)  # a pure corrector (net ~ identity)
    corrector = res.sequence  # three pulses around identity area (π,2π,π)
    right = skn(theta/2.0, n-1, base_phase)
    return left + corrector + right

# ---------- Convenience front-end ----------

def bn(theta: float, n: int, base_phase: float = 0.0, symmetric: bool = True) -> List[Tuple[float, float]]:
    """
    Front-end for B_n:
      - n=2 returns analytic BB1 (symmetric 5-pulse) exactly.
      - n=1 returns SK1 (3-pulse) as a minimal broadband-ish corrector.
      - n>2 uses nonlinear synthesis to find phases canceling up to order n.
    """
    if n == 1:
        return sk1(theta, base_phase)
    if n == 2:
        return bb1(theta, base_phase, symmetric=True)
    # numerical synthesis for n>2
    res = synthesize_Bn(theta=theta, n=n, base_phase=base_phase, symmetric=symmetric)
    return res.sequence



def knill_5pulse(theta=np.pi/2, base_phase=0.0):
    """
    Knill-type 5-pulse robust rotation.
    For θ=π/2 this is a widely used robust choice; for general θ we scale the three “π/2” pulses to θ/2.
    """
    thetas = np.array([theta/2, np.pi,theta/2, np.pi,theta/2])
    phis   = np.array([np.pi/6, 0.0, np.pi/2, 0.0, np.pi/6]) + base_phase
    phis_teta = [(theta, phi) for theta, phi in zip(thetas, phis)]
    return phis_teta


import numpy as np

π = np.pi

def _f_j(j: int) -> int:
    """
    TS scaling factor f_j with f1 = 1 and
    f_j = (2^(2j-1) - 2) * f_{j-1}.
    """
    if j < 1:
        raise ValueError("j must be ≥ 1")
    f = 1
    for k in range(2, j+1):
        f *= (2**(2*k - 1) - 2)
    return f

def _phi_j(theta: float, j: int) -> float:
    """
    φ_j = arccos( -θ / (8π f_j) ).
    """
    fj = _f_j(j)
    x = -theta / (8.0 * π * fj)
    if abs(x) > 1:
        raise ValueError(f"|theta| must be ≤ 8π f_j = {8*π*fj:.6g} for order n=2j; got {theta:.6g}")
    return float(np.arccos(x))

def _phi_B_from_phi_j(phi_j: float, j: int, theta: float) -> float:
    """
    Broadband mapping: cos(φ_B) = 2 cos(φ_j).
    (Equivalently cos φ_B = -θ / (4π f_j).)
    """
    fj = _f_j(j)
    x = -theta / (4.0 * π * fj)   # equals 2*cos(phi_j), but numerically steadier
    x = max(-1.0, min(1.0, x))
    return float(np.arccos(x))

def _S1(theta_scale: float, phi1: float, phi2: float):
    """Primitive block S1(φ1, φ2, m): [mπ, 2mπ, mπ] at phases [φ1, φ2, φ1]."""
    return [
        (theta_scale * π,        phi1),
        (2.0 * theta_scale * π,  phi2),
        (theta_scale * π,        phi1),
    ]

def _repeat(seq, times: int):
    out = []
    for _ in range(times):
        out.extend(seq)
    return out

def _SB_recursive(n: int, phiB: float, m: int):
    """
    Broadband version of S_n built by applying the paper's replacement
    only at the S1 level:  S1(φj, -φj, m) → S1(φB, -φB + 4φB*((m/2) mod 2), m/2).
    Works for m = ± 2^k integers that appear in TS recursion.
    """
    if n == 1:
        if m % 2 != 0:
            raise ValueError("Internal: m must be even at S1 level in TS recursion.")
        m2 = m // 2
        odd = (abs(m2) % 2 == 1)        # (m/2) mod 2 ∈ {0,1}
        phi2 = (-phiB) + (4.0 * phiB if odd else 0.0)
        return _S1(m2, phiB, phi2)

    # S_n = [S_{n-1}(m)]^{4^{n-1}}  ∘  S_{n-1}(-2m)  ∘  [S_{n-1}(m)]^{4^{n-1}}
    left  = _SB_recursive(n-1, phiB,  m)
    mid   = _SB_recursive(n-1, phiB, -2*m)
    right = _SB_recursive(n-1, phiB,  m)
    return _repeat(left, 4**(n-1)) + mid + _repeat(right, 4**(n-1))

def bn_thetas_phis(theta: float, n: int, base_phase: float = 0.0, include_base: bool = True):
    """
    Return (thetas, phis) for the higher-order broadband B_n correction
    targeting a rotation R_{base_phase}(theta).

    Parameters
    ----------
    theta : float
        Target rotation angle in radians for the underlying ideal pulse.
    n : int
        Even order of broadband sequence (2, 4, 6, ...). Uses TS→B mapping.
    base_phase : float
        Phase (axis in xy-plane) of the target rotation (default 0 = x-axis).
    include_base : bool
        If True, append the base pulse (theta, base_phase) at the end.

    Returns
    -------
    thetas : np.ndarray
    phis   : np.ndarray
        Lists of angles and phases for the composite sequence, in order.

    Notes
    -----
    Construction follows:
      - TS passband P_{2j} = S_j(φ_j, -φ_j, 2) · M_{base}(θ)
      - Broadband B_{2j}: replace each S1(φ_j, -φ_j, m) by
        S1(φ_B, -φ_B + 4 φ_B * ((m/2) mod 2), m/2),
        with cos φ_B = 2 cos φ_j.  Only even orders appear in TS. 
    """
    if n % 2 != 0 or n < 2:
        raise ValueError("B_n via TS exists for even n ≥ 2 (i.e., n=2,4,6,...)")
    j = n // 2

    # Compute φ_j and the broadband φ_B (domain-checked)
    phi_j = _phi_j(theta, j)                    # φ_j = arccos( -θ/(8π f_j) )
    phi_B = _phi_B_from_phi_j(phi_j, j, theta)  # cos φ_B = 2 cos φ_j

    # Build the broadband correction S_j^B with seed m=2 (as in P_{2j} definition)
    corr = _SB_recursive(j, phi_B, m=2)

    # Append the base pulse if requested
    if include_base:
        corr = corr + [(theta, base_phase)]

    thetas, phis = map(np.array, zip(*corr)) if corr else (np.array([]), np.array([]))
    return thetas, phis
