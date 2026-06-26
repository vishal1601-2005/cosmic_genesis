"""
physics/cosmology.py — Friedmann equation solver and cosmic thermodynamics.

The universe's expansion is governed by:

    H² = (ȧ/a)² = (8πG/3) ρ_total − k/a² + Λ/3

For a flat universe (k=0) in the radiation-dominated epoch:
    H = 1/(2t),   a ∝ t^{1/2},   T ∝ 1/a ∝ t^{-1/2}

In matter domination:
    H = 2/(3t),   a ∝ t^{2/3},   T ∝ 1/a

We track a(t), T(t), H(t) using JAX-JIT-compiled leapfrog integration
so we can fast-forward / slow-down the cosmic clock interactively.

The energy density is:
    ρ_total = ρ_radiation + ρ_matter + ρ_dark_energy + ρ_curvature(=0)

    ρ_radiation = (π²/30) g_*(T) T⁴     (Stefan-Boltzmann, relativistic DOF)
    ρ_matter    = ρ_m0 · (a_0/a)³
    ρ_Λ         = Λ/(8πG) = const
"""

from __future__ import annotations
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from config import (
    M_PLANCK_RED, OMEGA_M, OMEGA_LAMBDA, OMEGA_B, OMEGA_DM,
    T_CMB_K, H0_KM_S_MPC, LAMBDA_QCD_GEV,
)

# Conversion: km/s/Mpc → s⁻¹
H0_SI = H0_KM_S_MPC * 1e3 / (3.0857e22)    # ~2.2×10⁻¹⁸ s⁻¹
# Critical density today
RHO_CRIT0_GEV4 = 3 * H0_SI**2 / (8 * math.pi * 6.674e-11)  # kg/m³ (SI)

# ── Effective degrees of freedom g*(T) ───────────────────────
# g*(T) counts relativistic species at temperature T.
# We use a piecewise approximation from the Standard Model:
#   T >> 100 GeV: g* = 106.75  (all SM particles)
#   T ~ 100 GeV:  g* = 96     (after Higgs decoupling)
#   T ~ 0.3 GeV:  g* = 17.25  (quarks+gluons confine, pions remain)
#   T ~ 0.1 MeV:  g* = 10.75  (after e+e- annihilation: γ+3ν)
#   T << 0.1 MeV: g* = 3.36   (photons + neutrinos)

G_STAR_TABLE = np.array([
    #  T (GeV)       g*(T)     g*S(T)
    [1e6,            106.75,   106.75],
    [1e4,            106.75,   106.75],
    [1e2,            96.00,    96.00 ],
    [1e1,            86.25,    86.25 ],
    [5.00,           75.75,    75.75 ],
    [1.00,           61.75,    61.75 ],
    [0.30,           21.25,    21.25 ],   # QCD transition
    [0.15,           17.25,    17.25 ],
    [0.05,           10.75,    10.75 ],
    [1e-3,           10.75,    10.75 ],
    [5e-4,           3.36,     3.91  ],   # e+e- annihilation
    [1e-5,           3.36,     3.91  ],
    [1e-10,          3.36,     3.91  ],
])

def g_star(T_GeV: float) -> float:
    """Effective relativistic DOF at temperature T_GeV."""
    logT = math.log10(max(T_GeV, 1e-15))
    logT_table = np.log10(G_STAR_TABLE[:, 0])
    # Interpolate in log space
    return float(np.interp(-logT, -logT_table, G_STAR_TABLE[:, 1]))

def g_star_s(T_GeV: float) -> float:
    """Entropic DOF g*S(T)."""
    logT = math.log10(max(T_GeV, 1e-15))
    logT_table = np.log10(G_STAR_TABLE[:, 0])
    return float(np.interp(-logT, -logT_table, G_STAR_TABLE[:, 2]))


# ── Energy density ─────────────────────────────────────────────
def rho_radiation_GeV4(T_GeV: float) -> float:
    """ρ_rad = (π²/30) g*(T) T⁴  in GeV⁴."""
    return (math.pi**2 / 30) * g_star(T_GeV) * T_GeV**4

def rho_matter_GeV4(a: float, a0: float = 1.0) -> float:
    """ρ_matter = ρ_m0 (a0/a)³.  We set ρ_m0 from Ω_m H0² Mₚ²."""
    rho_m0 = OMEGA_M * 3 * (H0_SI * 6.58e-25)**2 * M_PLANCK_RED**2  # rough GeV⁴
    return rho_m0 * (a0 / a)**3

def rho_lambda_GeV4() -> float:
    """ρ_Λ = Ω_Λ · 3Mₚ²H₀²."""
    rho_crit0 = 3 * (H0_SI * 6.58e-25)**2 * M_PLANCK_RED**2
    return OMEGA_LAMBDA * rho_crit0

def hubble_rate_GeV(T_GeV: float, a: float) -> float:
    """
    H² = (8π/3Mₚ²) [ρ_rad(T) + ρ_m(a) + ρ_Λ]
    Returns H in GeV (natural units).
    """
    rho = rho_radiation_GeV4(T_GeV) + rho_matter_GeV4(a) + rho_lambda_GeV4()
    H_sq = (8 * math.pi / (3 * M_PLANCK_RED**2)) * rho
    return math.sqrt(max(H_sq, 0))

def temperature_from_a(a: float, T0_GeV: float = T_CMB_K * 8.617e-14,
                        a0: float = 1.0) -> float:
    a = max(a, 1e-50)
    """
    T(a) = T0 (a0/a) · [g*S(T0)/g*S(T)]^{1/3}
    We solve iteratively (since g*S depends on T).
    """
    T_naive = T0_GeV * (a0 / a)
    for _ in range(8):
        gS0 = g_star_s(T0_GeV)
        gST = g_star_s(T_naive)
        T_naive = T0_GeV * (a0 / a) * (gS0 / gST) ** (1/3)
    return T_naive

def time_from_temperature_s(T_GeV: float) -> float:
    """
    Radiation-dominated approximation:
    t ≈ 0.301 g*^{-1/2} Mₚ / T²
    in natural units, then convert to seconds.
    """
    gst = g_star(T_GeV)
    t_GeV_inv = 0.301 * gst**(-0.5) * M_PLANCK_RED / T_GeV**2
    # 1 GeV⁻¹ = 6.582×10⁻²⁵ s
    return t_GeV_inv * 6.582e-25


# ── Cosmic clock (integrable in real time) ────────────────────
class CosmicClock:
    """
    Tracks the game's current position in cosmic time.

    The player can fast-forward / slow-down. We integrate the
    Friedmann equation step by step so a(t), T(t) are always consistent.

    log_a is used as the integration variable (more numerically stable).
    """
    def __init__(self):
        # Start at very early times: T ~ 10¹⁹ GeV, a ~ 10⁻²⁸
        self.log_a  = -28.0   # log10(a), a0=1 today
        self.T_GeV  = 1e19    # temperature
        self.t_s    = 5.4e-44 # Planck time
        self.epoch_id = 0
        self.time_scale = 1.0  # simulation speed multiplier

    @property
    def a(self) -> float:
        return 10 ** max(-50, min(0, self.log_a))

    @property
    def H_GeV(self) -> float:
        return hubble_rate_GeV(self.T_GeV, self.a)

    @property
    def H_inv_s(self) -> float:
        """Hubble time in seconds."""
        H_GeV = self.H_GeV
        return 6.582e-25 / H_GeV if H_GeV > 0 else 1e40

    def step(self, dt_game_s: float):
        dt_game_s = min(dt_game_s, 0.05)
        """
        Advance the clock by dt_game_s of game time (not cosmic time!).
        The cosmic time advances by dt_game_s * time_scale * cosmic_time_factor.
        """
        # How many cosmic seconds per game second
        cosmic_dt_s = dt_game_s * self.time_scale * self._cosmic_rate()
        # Integrate: ȧ = a·H → d(log a)/dt = H
        H = self.H_GeV / 6.582e-25   # convert to s⁻¹
        d_loga = H * cosmic_dt_s / math.log(10)
        self.log_a += d_loga
        self.t_s += cosmic_dt_s
        # Update temperature from scale factor
        self.T_GeV = temperature_from_a(self.a)
        # Update epoch
        self._update_epoch()

    def _cosmic_rate(self) -> float:
        """
        Maps epoch to cosmic time rate for playability.
        Early epochs are extremely compressed; BBN is real-time.
        """
        rates = {0: 1e6,  1: 1e4,  2: 100,  3: 10,
                 4: 1e3, 5: 1.0, 6: 1e8, 7: 1e12}
        return rates.get(self.epoch_id, 1e6)

    def _update_epoch(self):
        T = self.T_GeV
        if   T > 1e16:  self.epoch_id = 0
        elif T > 1e2:   self.epoch_id = 1
        elif T > 0.2:   self.epoch_id = 2
        elif T > 1e-3:  self.epoch_id = 3
        elif T > 1e-4:  self.epoch_id = 4
        elif T > 1e-5:  self.epoch_id = 5
        elif T > 1e-10: self.epoch_id = 6
        else:           self.epoch_id = 7

    def display_time(self) -> str:
        """Human-readable cosmic time string."""
        t = self.t_s
        if   t < 1e-36: return f"{t:.2e} s (Planck epoch)"
        elif t < 1e-6:  return f"{t:.2e} s"
        elif t < 1.0:   return f"{t*1e3:.1f} ms"
        elif t < 60:    return f"{t:.1f} s"
        elif t < 3600:  return f"{t/60:.1f} min"
        elif t < 86400: return f"{t/3600:.1f} hr"
        elif t < 3.15e7: return f"{t/86400:.0f} days"
        elif t < 3.15e10: return f"{t/3.15e7:.2g} yr"
        elif t < 3.15e13: return f"{t/3.15e10:.2g} kyr"
        elif t < 3.15e16: return f"{t/3.15e13:.2g} Myr"
        else:            return f"{t/3.15e16:.2g} Gyr"

    def display_temperature(self) -> str:
        T = self.T_GeV
        T_K = T / 8.617e-14   # GeV → K
        if T >= 1e3:    return f"{T:.2e} GeV"
        elif T >= 1:    return f"{T:.1f} GeV"
        elif T >= 1e-3: return f"{T*1e3:.0f} MeV"
        elif T >= 1e-6: return f"{T*1e6:.0f} keV"
        else:           return f"{T_K:.2e} K"

    def set_epoch(self, epoch_id: int):
        """Jump to a specific epoch (for testing / chapter select)."""
        T_targets = {0:1e19, 1:1e14, 2:1e2, 3:0.15, 4:1e-3, 5:5e-4, 6:3e-10, 7:3e-12}
        self.T_GeV = T_targets.get(epoch_id, 1e19)
        # Rough a from T: T ∝ 1/a → a ~ T0/T (T0=CMB today ~2.35×10⁻¹³ GeV)
        T0 = T_CMB_K * 8.617e-14
        self.log_a = math.log10(T0 / self.T_GeV)
        self.t_s = time_from_temperature_s(self.T_GeV)
        self.epoch_id = epoch_id
        self.time_scale = {0:1e6, 1:1e4, 2:100, 3:10, 4:5, 5:1, 6:50, 7:1000}[epoch_id]
