"""The equations of covis-change-detection-theory.pdf, one function each.

Pure numpy, no I/O: every other file of this method calls these, and
test_theory.py checks them against the numbers printed in the PDF.
Equation numbers (eq. N) and sections (§N) refer to the PDF.
"""
from __future__ import annotations

import math

import numpy as np


# ─── §3 dynamism model ───

def survival(a: float, b: float, dt: float) -> float:
    """eq. 3: P(no move during dt), averaged over lambda ~ Gamma(a, b) (Lomax survival)."""
    return float((b / (b + dt)) ** a)


def hazard(a: float, b: float, dt: float) -> float:
    """§3.3: h(dt) = a / (b + dt), falls with time."""
    return a / (b + dt)


def median_time_to_move(a: float, b: float) -> float:
    """§3.3 item 4: b (2^(1/a) - 1)."""
    return b * (2 ** (1 / a) - 1)


def init_rate(rate_guess: float, kappa: float) -> tuple[float, float]:
    """§3.4: a = kappa pseudo-moves, b = kappa / rate_guess pseudo-days."""
    return float(kappa), float(kappa / rate_guess)


def logit(p: float, eps: float = 1e-9) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ─── §4 belief ───

def voxel_llr(p_hit_present: float = 0.85, p_hit_absent: float = 0.08) -> float:
    """§4.2: l = ln p(hit | x=1) / p(hit | x=0) (~2.36); a through outcome counts -l."""
    return math.log(p_hit_present / p_hit_absent)


def kish_alpha(n, rho_c: float):
    """eq. 6: alpha = n_eff / n = 1 / (1 + (n - 1) rho_c) for n correlated observations."""
    n = np.asarray(n, dtype=np.float64)
    return 1 / (1 + np.maximum(n - 1, 0) * rho_c)


def clip_logodds(L, l_max: float):
    """§4.4."""
    return np.clip(L, -l_max, l_max)


# ─── §5 relocalisation ───

def ransac_iterations(inlier_ratio: float, p: float = 0.99, s: int = 3) -> int:
    """eq. 12: samples needed to draw one all-inlier minimal set with probability p."""
    w = inlier_ratio ** s
    if w <= 0:
        return 10 ** 9
    if w >= 1:
        return 1
    return int(math.ceil(math.log(1 - p) / math.log(1 - w)))


# ─── §6 signed residual ───

def depth_sigma(z, sigma0: float, k1: float, k2: float):
    """eq. 14 generalised: sigma(z) = sigma0 + k1 z + k2 z^2.
    The PDF has sigma0 + k z^2 (stereo / structured light: error grows with
    disparity); monocular depth networks are off by a share of the distance,
    which is the linear term. k1 = 0 gives the PDF's form."""
    z = np.asarray(z, dtype=np.float64)
    return sigma0 + k1 * z + k2 * z * z


HIT, THROUGH, OCCLUDED = 1, 2, 3


def classify_residual(r, sigma, signed: bool = True):
    """eq. 15: r = measured - stored depth.
    hit |r| <= 2 sigma; through r > 2 sigma (the ray passed the voxel's place:
    empty); occluded r < -2 sigma (something in front: no evidence).
    signed=False is DoVSG's |dz| test (ablation 6): occluded counts as through."""
    r = np.asarray(r)
    out = np.full(r.shape, HIT, np.uint8)
    out[r > 2 * sigma] = THROUGH
    out[r < -2 * sigma] = THROUGH if not signed else OCCLUDED
    return out


# ─── §7 co-visibility gate ───

def noisy_or(w, P, h):
    """eq. 16: rho = 1 - prod_j (1 - w_j P_j h_j). Arrays over neighbours
    (and optionally a leading axis, e.g. frames: reduces the last axis)."""
    w, P, h = (np.asarray(x, dtype=np.float64) for x in (w, P, h))
    if w.size == 0:
        return np.zeros(h.shape[:-1]) if h.ndim > 1 else 0.0
    return 1 - np.prod(1 - np.clip(w * P * h, 0, 1), axis=-1)


def gated_absence(x, rho, mode: str = "tempered"):
    """Absence evidence x (< 0) of one frame, trusted with probability rho.
    tempered: rho x (eq. 17, the PDF's choice); mixture: ln(rho e^x + 1 - rho) (eq. 18)."""
    x, rho = np.asarray(x, dtype=np.float64), np.asarray(rho, dtype=np.float64)
    if mode == "tempered":
        return rho * x
    if mode == "mixture":
        return np.log(rho * np.exp(x) + 1 - rho)
    raise ValueError(mode)


# ─── §8 decision ───

def chow_thresholds(c_false_removal: float, c_missed_change: float, c_defer: float) -> tuple[float, float]:
    """eq. 19: absent if P < C_D / C_FR, present if P > 1 - C_D / C_MC, else defer."""
    return c_defer / c_false_removal, 1 - c_defer / c_missed_change


def receptacle_prior(counts: dict, receptacle: str, n_receptacles: int, beta: float) -> float:
    """eq. 20: Dirichlet-categorical posterior predictive p(parent = c | i)."""
    total = sum(counts.values())
    return (counts.get(receptacle, 0) + beta) / (total + n_receptacles * beta)


# ─── §9 rate learning ───

def update_stayed(a: float, b: float, dt: float) -> tuple[float, float]:
    """eq. 21: no move during dt is conjugate."""
    return a, b + dt


def moved_moments(a: float, b: float, dt: float) -> tuple[float, float]:
    """eq. 22: E[lambda | moved at least once], E[lambda^2 | moved]."""
    S = (b / (b + dt)) ** a
    r = b / (b + dt)
    m1 = a / b * (1 - S * r) / (1 - S)
    m2 = a * (a + 1) / b ** 2 * (1 - S * r * r) / (1 - S)
    return m1, m2


def update_moved(a: float, b: float, dt: float) -> tuple[float, float]:
    """eq. 23: the Gamma with the posterior's first two moments."""
    m1, m2 = moved_moments(a, b, dt)
    var = m2 - m1 * m1
    return m1 * m1 / var, m1 / var


def update_moved_naive(a: float, b: float, dt: float) -> tuple[float, float]:
    """§9.2: the old (a + 1, b + dt), biased when lambda dt is not small (ablation 4)."""
    return a + 1, b + dt


def forget(a: float, b: float, gamma: float) -> tuple[float, float]:
    """§9.3: same mean, larger variance."""
    return gamma * a, gamma * b
