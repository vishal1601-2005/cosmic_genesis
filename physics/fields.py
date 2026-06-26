"""
physics/fields.py — JAX-accelerated scalar field solvers.

Three scalar fields are simulated on a 2D spatial grid (the screen plane),
each with its own potential and equation of motion:

1. INFLATON  φ(x,t)
   ─────────────────
   Equation of motion (with Hubble friction):
       φ̈ + 3H φ̇ = ∇²φ/a² − ∂V/∂φ

   Potential (chaotic inflation):
       V(φ) = ½ m²φ²     (quadratic — simplest)
   or  V(φ) = λ(φ²−v²)²  (double-well — more dramatic)

   The field rolls from large φ → minimum, driving exponential expansion.
   Quantum fluctuations δφ ≈ H/2π are added as white noise each Hubble time.

2. AXION  a(x,t) = f_a × θ(x,t)
   ────────────────────────────────
   Misalignment mechanism:
       θ̈ + 3H θ̇ = ∇²θ/a² − m_a² sin(θ)

   This is the sine-Gordon equation with Hubble friction.
   Initial condition: θ(x,0) = θ_i (misalignment angle, set by player)
   The field oscillates at ω ≈ m_a after T drops below m_a.

3. HIGGS  h(x,t)  (electroweak sector)
   ──────────────────────────────────────
   Simplified real scalar:
       ḧ + 3H ḣ = ∇²h/a² − μ²h − λh³

   At T > T_EW: μ²(T) = μ²₀(T²/T²_EW − 1) > 0 → symmetric phase (h=0)
   At T < T_EW: μ²(T) < 0 → symmetry breaking, h → v = 246 GeV

All fields are solved on a (GRID, GRID) spatial grid using JAX's jit+vmap.
The field values are exported as float32 textures for the volume shader.
"""

from __future__ import annotations
import math
from functools import partial
from typing import NamedTuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    JAX_OK = True
except ImportError:
    JAX_OK = False
    import numpy as jnp   # numpy fallback

from config import (
    M_PLANCK_RED, ALPHA_PRIME, V_EW_GEV,
)

# ── Grid parameters ────────────────────────────────────────────
GRID     = 256        # spatial resolution (GRID × GRID)
DX       = 1.0 / GRID  # spatial step in normalised coords
DT_FIELD = 0.002      # field time step (in units where H₀=1)


# ── State containers ───────────────────────────────────────────
class InflatonState(NamedTuple):
    phi:     "jax.Array"   # (GRID, GRID) field values
    phi_dot: "jax.Array"   # (GRID, GRID) time derivatives
    t:       float


class AxionState(NamedTuple):
    theta:     "jax.Array"  # (GRID, GRID) misalignment angle
    theta_dot: "jax.Array"
    t:         float


class HiggsState(NamedTuple):
    h:     "jax.Array"   # (GRID, GRID) Higgs field
    h_dot: "jax.Array"
    t:     float


# ══════════════════════════════════════════════════════════════
#  INFLATON FIELD
# ══════════════════════════════════════════════════════════════

def inflaton_potential(phi: float, model: str = "quadratic",
                        m_sq: float = 1e-12,
                        lam: float = 0.1, v: float = 1.0) -> float:
    """
    V(φ):
        quadratic:   ½ m² φ²
        double_well: λ(φ² − v²)²
        starobinsky: (3/4)m²(1 − e^{-√(2/3)φ})² (Starobinsky R² inflation)
    """
    if model == "quadratic":
        return 0.5 * m_sq * phi ** 2
    elif model == "double_well":
        return lam * (phi**2 - v**2)**2
    elif model == "starobinsky":
        x = 1.0 - math.exp(-math.sqrt(2/3) * phi)
        return 0.75 * m_sq * x**2
    return 0.5 * m_sq * phi**2


if JAX_OK:
    @jit
    def _inflaton_step(phi: jnp.ndarray, phi_dot: jnp.ndarray,
                       H: float, a: float, m_sq: float = 1e-12) -> tuple:
        """
        One leapfrog step for the inflaton field.
        ∂V/∂φ = m²φ  (quadratic potential)
        """
        dt = DT_FIELD
        # Laplacian with periodic BCs
        lap = (jnp.roll(phi,  1, 0) + jnp.roll(phi, -1, 0)
             + jnp.roll(phi,  1, 1) + jnp.roll(phi, -1, 1)
             - 4 * phi) / (DX * DX)

        dV_dphi = m_sq * phi   # ∂V/∂φ for quadratic potential

        # EOM: φ̈ = ∇²φ/a² - 3Hφ̇ - ∂V/∂φ
        phi_ddot = lap / (a * a) - 3.0 * H * phi_dot - dV_dphi
        phi_dot_new = phi_dot + dt * phi_ddot
        phi_new     = phi     + dt * phi_dot_new

        return phi_new, phi_dot_new

    @jit
    def _axion_step(theta: jnp.ndarray, theta_dot: jnp.ndarray,
                    H: float, a: float, m_a_sq: float) -> tuple:
        """
        Sine-Gordon with Hubble friction:
        θ̈ + 3Hθ̇ = ∇²θ/a² − m_a² sin(θ)
        """
        dt  = DT_FIELD
        lap = (jnp.roll(theta,  1, 0) + jnp.roll(theta, -1, 0)
             + jnp.roll(theta,  1, 1) + jnp.roll(theta, -1, 1)
             - 4 * theta) / (DX * DX)

        theta_ddot  = lap / (a * a) - 3.0 * H * theta_dot - m_a_sq * jnp.sin(theta)
        theta_dot_n = theta_dot + dt * theta_ddot
        theta_n     = theta     + dt * theta_dot_n
        return theta_n, theta_dot_n

    @jit
    def _higgs_step(h: jnp.ndarray, h_dot: jnp.ndarray,
                    H: float, a: float,
                    mu_sq: float, lam: float = 0.13) -> tuple:
        """
        Mexican hat (Higgs) field:
        ḧ = ∇²h/a² - 3Hḣ - μ²(T)h - λh³
        """
        dt  = DT_FIELD
        lap = (jnp.roll(h,  1, 0) + jnp.roll(h, -1, 0)
             + jnp.roll(h,  1, 1) + jnp.roll(h, -1, 1)
             - 4 * h) / (DX * DX)

        dV = mu_sq * h + lam * h**3
        h_ddot  = lap / (a * a) - 3.0 * H * h_dot - dV
        h_dot_n = h_dot + dt * h_ddot
        h_n     = h     + dt * h_dot_n
        return h_n, h_dot_n

else:
    # Numpy fallbacks (CPU only)
    def _laplacian_np(f):
        return (np.roll(f,1,0)+np.roll(f,-1,0)+np.roll(f,1,1)+np.roll(f,-1,1)-4*f)/DX**2

    def _inflaton_step(phi, phi_dot, H, a, m_sq=1e-12):
        dt = DT_FIELD
        lap = _laplacian_np(phi)
        phi_ddot  = lap/(a*a) - 3*H*phi_dot - m_sq*phi
        phi_dot_n = phi_dot + dt*phi_ddot
        phi_n     = phi     + dt*phi_dot_n
        return phi_n, phi_dot_n

    def _axion_step(theta, theta_dot, H, a, m_a_sq):
        dt = DT_FIELD
        lap = _laplacian_np(theta)
        theta_ddot  = lap/(a*a) - 3*H*theta_dot - m_a_sq*np.sin(theta)
        theta_dot_n = theta_dot + dt*theta_ddot
        theta_n     = theta     + dt*theta_dot_n
        return theta_n, theta_dot_n

    def _higgs_step(h, h_dot, H, a, mu_sq, lam=0.13):
        dt = DT_FIELD
        lap = _laplacian_np(h)
        dV    = mu_sq*h + lam*h**3
        h_ddot  = lap/(a*a) - 3*H*h_dot - dV
        h_dot_n = h_dot + dt*h_ddot
        h_n     = h     + dt*h_dot_n
        return h_n, h_dot_n


# ══════════════════════════════════════════════════════════════
#  FIELD MANAGER — holds and advances all scalar fields
# ══════════════════════════════════════════════════════════════

class FieldManager:
    """
    Manages all scalar field simulations.
    Each epoch uses different active fields.

    Exports field_texture() as a (GRID, GRID, 4) RGBA float32 array
    for the volume.frag shader to read as density/emission data.

    Channels:
      R = inflaton φ (normalised)
      G = axion θ (normalised)
      B = Higgs h (normalised)
      A = total field energy density (for brightness)
    """

    def __init__(self, grid: int = GRID, seed: int = 42):
        self.grid = grid
        rng = np.random.default_rng(seed)

        # Inflaton: start near large field value (slow-roll regime)
        phi0 = 3.5 + rng.standard_normal((grid, grid)) * 0.05
        self.inflaton = InflatonState(
            phi=jnp.array(phi0, dtype=jnp.float32),
            phi_dot=jnp.zeros((grid, grid), dtype=jnp.float32),
            t=0.0,
        )

        # Axion: misalignment angle θ_i (set by player, default π/3)
        self._theta_i = math.pi / 3.0
        theta0 = np.full((grid, grid), self._theta_i, dtype=np.float32)
        theta0 += rng.standard_normal((grid, grid)).astype(np.float32) * 0.02
        self.axion = AxionState(
            theta=jnp.array(theta0),
            theta_dot=jnp.zeros((grid, grid), dtype=jnp.float32),
            t=0.0,
        )

        # Higgs: symmetric phase initially (h=0 + small fluctuations)
        h0 = rng.standard_normal((grid, grid)).astype(np.float32) * 0.01
        self.higgs = HiggsState(
            h=jnp.array(h0),
            h_dot=jnp.zeros((grid, grid), dtype=jnp.float32),
            t=0.0,
        )

        self._texture_cache: np.ndarray | None = None
        self._dirty = True

        # Quantum noise key (JAX)
        self._noise_key = jax.random.PRNGKey(seed) if JAX_OK else None

    def set_axion_angle(self, theta_i: float):
        """Player sets the initial axion misalignment angle θ_i."""
        self._theta_i = float(theta_i)
        theta0 = np.full((self.grid, self.grid), theta_i, dtype=np.float32)
        self.axion = AxionState(
            theta=jnp.array(theta0),
            theta_dot=jnp.zeros((self.grid, self.grid), dtype=jnp.float32),
            t=self.axion.t,
        )
        self._dirty = True

    def step(self, epoch_id: int, H: float, a: float,
              T_GeV: float, n_steps: int = 4):
        """
        Advance the active field(s) for the current epoch by n_steps × DT_FIELD.
        """
        for _ in range(n_steps):
            if epoch_id == 1:
                # Inflation: evolve inflaton
                m_sq = 1e-12
                phi_n, phid_n = _inflaton_step(
                    self.inflaton.phi, self.inflaton.phi_dot, H, a, m_sq
                )
                # Add quantum fluctuations δφ ≈ H/2π each step
                if JAX_OK:
                    self._noise_key, subkey = jax.random.split(self._noise_key)
                    noise = jax.random.normal(subkey, (self.grid, self.grid)) * (H / (2*math.pi))
                    phi_n = phi_n + noise * DT_FIELD
                self.inflaton = InflatonState(phi=phi_n, phi_dot=phid_n,
                                              t=self.inflaton.t + DT_FIELD)

            elif epoch_id == 2:
                # Baryogenesis / Higgs: evolve Higgs field
                # Temperature-dependent mass squared:
                # μ²(T) = μ²₀(T²/T²_EW − 1)
                T_EW  = 100.0   # GeV
                mu0sq = -1.0    # negative below T_EW
                mu_sq = mu0sq * ((T_GeV / T_EW)**2 - 1.0)
                h_n, hd_n = _higgs_step(
                    self.higgs.h, self.higgs.h_dot, H, a, mu_sq
                )
                self.higgs = HiggsState(h=h_n, h_dot=hd_n,
                                         t=self.higgs.t + DT_FIELD)

            elif epoch_id == 4:
                # Axion condensation
                # m_a² ≈ Λ_QCD⁴/f_a² (simplified)
                f_a   = 1e12 / 1e9   # in GeV (f_a ~ 10¹² GeV)
                m_a_sq = (0.217**4) / (f_a**2)   # GeV²
                th_n, thd_n = _axion_step(
                    self.axion.theta, self.axion.theta_dot, H, a, m_a_sq
                )
                self.axion = AxionState(theta=th_n, theta_dot=thd_n,
                                         t=self.axion.t + DT_FIELD)

        self._dirty = True

    def field_texture(self, epoch_id: int) -> np.ndarray:
        """
        Export (GRID, GRID, 4) RGBA float32 texture for GPU upload.
        This is sampled by the volume.frag shader as the density field.
        """
        if not self._dirty and self._texture_cache is not None:
            return self._texture_cache

        G  = self.grid
        tex = np.zeros((G, G, 4), dtype=np.float32)

        if epoch_id == 1:
            phi = np.array(self.inflaton.phi)
            phi_n = (phi - phi.min()) / (phi.max() - phi.min() + 1e-8)
            phid  = np.array(self.inflaton.phi_dot)
            energy= 0.5 * phid**2 + 0.5 * 1e-12 * phi**2
            energy_n = (energy / (energy.max() + 1e-8)).clip(0, 1)
            tex[:,:,0] = phi_n        # R = φ
            tex[:,:,3] = energy_n     # A = energy density (for glow)

        elif epoch_id == 2:
            h    = np.array(self.higgs.h)
            h_n  = (h - h.min()) / (h.max() - h.min() + 1e-8)
            hd   = np.array(self.higgs.h_dot)
            energy = 0.5 * hd**2 + 0.13 * (h**2 - 1.0)**2
            energy_n = (energy / (energy.max() + 1e-8)).clip(0, 1)
            tex[:,:,2] = h_n          # B = Higgs field
            tex[:,:,3] = energy_n

        elif epoch_id == 4:
            theta = np.array(self.axion.theta)
            # Axion potential: V ∝ 1 - cos(θ)
            V = 1.0 - np.cos(theta)
            V_n = (V / (V.max() + 1e-8)).clip(0, 1)
            tex[:,:,1] = V_n          # G = axion potential
            tex[:,:,3] = V_n * 0.5

        else:
            # Other epochs: flat grey background
            tex[:,:,3] = 0.1

        self._texture_cache = tex
        self._dirty = False
        return tex

    def inflaton_value_at(self, x: float, y: float) -> float:
        """Sample inflaton field at normalised position (x,y) ∈ [0,1]²."""
        ix = int(x * self.grid) % self.grid
        iy = int(y * self.grid) % self.grid
        phi = np.array(self.inflaton.phi)
        return float(phi[iy, ix])

    def axion_relic_density(self) -> float:
        """
        Estimate the axion relic density fraction Ω_a h²
        from the current misalignment angle θ_i.
        Ω_a h² ≈ 0.12 × (θ_i/1)² × (f_a/10¹² GeV)^{7/6}
        """
        f_a_over_ref = 1.0   # assuming f_a = 10¹² GeV
        return 0.12 * (self._theta_i)**2 * f_a_over_ref**(7/6)

    def higgs_vev(self) -> float:
        """Current Higgs vacuum expectation value ⟨h⟩ in GeV."""
        h = np.array(self.higgs.h)
        return float(np.mean(np.abs(h))) * V_EW_GEV

    def is_inflating(self) -> bool:
        """True if slow-roll conditions are satisfied: ε = (Mₚ/√2)(V′/V)² < 1."""
        phi = np.array(self.inflaton.phi)
        phi_mean = float(np.mean(phi))
        if abs(phi_mean) < 1e-6:
            return False
        # ε ≈ 2Mₚ²/φ² for chaotic inflation
        epsilon = 2.0 * M_PLANCK_RED**2 / (phi_mean**2 + 1e-8)
        return epsilon < 1.0

    def n_efolds(self) -> float:
        """
        Approximate number of e-folds elapsed:
        N_e ≈ φ_i²/(4Mₚ²) − φ_f²/(4Mₚ²)  (chaotic inflation)
        """
        phi = np.array(self.inflaton.phi)
        phi_mean = float(np.mean(phi))
        phi_init = 3.5   # initial value
        return max(0.0, (phi_init**2 - phi_mean**2) / (4 * M_PLANCK_RED**2))
