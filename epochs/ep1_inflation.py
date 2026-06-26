"""
epochs/ep1_inflation.py — Inflation epoch (t = 10⁻³⁶ → 10⁻³² s).

GAMEPLAY:
  The inflaton field φ is a glowing scalar condensate filling the screen.
  The player sees the potential V(φ) as a height field — a hill whose slope
  the field rolls down.

  MECHANICS:
    - A draggable "ball" sits on the potential V(φ) = ½m²φ²
    - Player can nudge φ left/right (changing the initial field value)
    - As φ rolls: space expands (zoom-out effect), quantum fluctuations appear
    - Each fluctuation is a tiny coloured ripple — these are the CMB seeds
    - The number of e-folds N_e is tracked: need N_e ≥ 60 to proceed
    - If player stops inflation too early: FORBIDDEN → "insufficient e-folds"
    - Reheating: φ oscillates at minimum → energy transferred to radiation
    - Density perturbations δρ/ρ ≈ H/2πφ̇ displayed per fluctuation

  VISUAL:
    - Field rendered as a 2D colour map (R=φ value, density=|∇φ|²)
    - Quantum fluctuations: tiny glowing spots that stretch and freeze
    - Potential hill: rendered as a semi-transparent overlay curve
    - e-fold counter, Hubble radius, slow-roll parameter ε in HUD
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field as dc_field
import numpy as np

from config import M_PLANCK_RED, EPOCHS
from physics.fields import FieldManager, GRID
from physics.cosmology import CosmicClock


@dataclass
class QuantumFluctuation:
    """A single frozen-in quantum fluctuation — seed of a future structure."""
    x: float          # position in field space (normalised -1..1)
    y: float
    amplitude: float  # δφ/H
    phase: float
    frozen: bool = False       # True once it exits the Hubble radius
    age: float = 0.0
    color: tuple = (0.6, 0.5, 1.0)
    # Render
    radius: float = 0.008
    emission_str: float = 1.5
    particle_type_int: int = 0  # generic
    color_rgb: tuple = (0.6, 0.5, 1.0)
    emission_rgb: tuple = (0.5, 0.4, 0.9)
    metallic: float = 0.0
    roughness: float = 0.6
    age_s: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    z: float = 0.0
    selected: bool = False

    def update(self, dt: float, H: float, a: float):
        self.age   += dt
        self.age_s += dt
        # Physical size grows with expansion: r ∝ a
        if not self.frozen:
            self.radius = 0.008 + self.age * 0.012 * a
            # Freeze when physical wavelength > Hubble radius 1/H
            comoving_k = 1.0 / max(self.radius, 1e-6)
            if comoving_k < H * a:
                self.frozen = True
                self.emission_str = 3.0  # brighten on freeze
        else:
            # Frozen: stays fixed, slowly dims
            self.emission_str = max(0.8, self.emission_str - dt * 0.2)


class InflationEpoch:
    """Full state of the inflation epoch simulation."""

    def __init__(self, clock: CosmicClock, fields: FieldManager):
        self.clock  = clock
        self.fields = fields
        self.clock.set_epoch(1)

        # Inflaton state
        self.phi        = 3.5    # initial field value (in Mₚ units, slow-roll regime)
        self.phi_dot    = -0.01  # slowly rolling
        self.a          = 1e-28  # scale factor
        self.H          = 0.0    # Hubble rate
        self.n_efolds   = 0.0    # e-folds elapsed
        self.epsilon    = 0.0    # slow-roll parameter ε = (Mₚ/√2 × V′/V)²
        self.reheating  = False
        self.complete   = False

        # Quantum fluctuations
        self.fluctuations: list[QuantumFluctuation] = []
        self._fluct_timer = 0.0
        self._FLUCT_INTERVAL = 0.12   # seconds between new fluctuations

        # Perturbation amplitude (power spectrum)
        self.delta_rho_over_rho = 0.0

        # Player-adjustable: can nudge φ
        self.player_nudge = 0.0

        # Statistics
        self.n_frozen = 0   # fluctuations that have exited Hubble radius
        self.forbidden_events: list[dict] = []

    # ── Physics step ──────────────────────────────────────
    def update(self, dt: float):
        if self.complete:
            return

        # Advance cosmic clock
        self.clock.step(dt)

        # Inflaton EOM with Hubble friction:
        # φ̈ + 3H φ̇ + V′(φ) = 0
        # V = ½m²φ²  →  V′ = m²φ
        m_sq = 1e-12   # in Mₚ units

        # Compute H from Friedmann:
        # H² = (φ̇²/2 + V) / (3Mₚ²)
        V        = 0.5 * m_sq * self.phi**2
        rho_inf  = 0.5 * self.phi_dot**2 + V
        H_sq     = rho_inf / 3.0   # (in units where Mₚ=1)
        self.H   = math.sqrt(max(H_sq, 0))

        dV_dphi  = m_sq * self.phi + self.player_nudge
        phi_ddot = -3.0 * self.H * self.phi_dot - dV_dphi
        self.phi_dot += dt * phi_ddot
        self.phi     += dt * self.phi_dot

        # Scale factor
        d_loga   = self.H * dt
        self.a  *= math.exp(d_loga)
        self.n_efolds += d_loga / math.log(10) * math.log(10)
        # Correct: N_e = ∫H dt
        self.n_efolds = (math.log(self.a) - math.log(1e-28))

        # Slow-roll parameter ε = ½(V′/V)² × Mₚ²
        if abs(V) > 1e-20:
            self.epsilon = 0.5 * (dV_dphi / V)**2
        else:
            self.epsilon = 999.0

        # Density perturbation amplitude
        phi_dot_sq = max(1e-20, self.phi_dot**2)
        self.delta_rho_over_rho = self.H / (2 * math.pi * math.sqrt(phi_dot_sq))

        # Advance field texture
        self.fields.step(1, self.H, self.a, self.clock.T_GeV, n_steps=2)

        # Spawn quantum fluctuations
        self._fluct_timer += dt
        if self._fluct_timer >= self._FLUCT_INTERVAL and self.H > 0:
            self._fluct_timer = 0.0
            self._spawn_fluctuation()

        # Update existing fluctuations
        for f in self.fluctuations:
            f.update(dt, self.H, self.a)
        self.n_frozen = sum(1 for f in self.fluctuations if f.frozen)

        # Check inflation end (ε ≥ 1 → slow-roll violated)
        if self.epsilon >= 1.0 and not self.reheating:
            self.reheating = True

        # Check reheating complete
        if self.reheating and abs(self.phi) < 0.1:
            self.complete = True

        # Keep fluctuation count manageable
        if len(self.fluctuations) > 300:
            self.fluctuations = self.fluctuations[-200:]

    def _spawn_fluctuation(self):
        """Spawn a quantum fluctuation δφ ~ H/2π."""
        delta_phi = self.H / (2 * math.pi)
        hue = random.uniform(220, 290)   # blue-violet
        col = self._hue_to_rgb(hue)
        x = random.uniform(-9, 9)
        y = random.uniform(-5, 5)
        f = QuantumFluctuation(
            x=x, y=y,
            amplitude=delta_phi,
            phase=random.uniform(0, 2*math.pi),
            color=col,
            color_rgb=col,
            emission_rgb=tuple(min(1.0, c*1.4) for c in col),
            emission_str=1.5 + delta_phi * 100,
            radius=0.006,
            z=random.uniform(-2, 2),
        )
        self.fluctuations.append(f)

    def attempt_stop_inflation(self) -> dict:
        """Player tries to stop inflation early — FORBIDDEN if N_e < 60."""
        if self.n_efolds < 58.0:
            deficit = 60.0 - self.n_efolds
            ev = {
                "allowed": False,
                "reason": "Insufficient inflation",
                "equation": f"N_e = {self.n_efolds:.1f} < 60 required",
                "why": (
                    f"Only {self.n_efolds:.1f} e-folds have elapsed — "
                    f"{deficit:.1f} short of the minimum ~60 needed to solve "
                    "the horizon and flatness problems. The observable universe "
                    "was not in causal contact before inflation, and spatial "
                    "curvature Ω_k is not driven to zero."
                ),
                "fx": "forbidden_efolds",
                "reason_type": 2,  # energy-type colour (cold blue)
            }
            self.forbidden_events.append(ev)
            return ev
        return {"allowed": True, "reason": "Inflation complete"}

    def nudge_phi(self, direction: float):
        """Player nudges the inflaton (±1 = push up/down the potential)."""
        self.player_nudge = direction * 1e-13
        # Reset after one frame
        import threading
        def _reset():
            import time; time.sleep(0.05)
            self.player_nudge = 0.0
        threading.Thread(target=_reset, daemon=True).start()

    @property
    def V_phi(self) -> float:
        return 0.5 * 1e-12 * self.phi**2

    @property
    def slow_roll_satisfied(self) -> bool:
        return self.epsilon < 1.0

    @property
    def hubble_radius(self) -> float:
        return 1.0 / self.H if self.H > 0 else 1e10

    def get_render_particles(self) -> list:
        return self.fluctuations

    @staticmethod
    def _hue_to_rgb(hue_deg: float) -> tuple:
        h = hue_deg / 60.0
        x = 1.0 - abs(h % 2 - 1)
        if   h < 1: r,g,b = 1,x,0
        elif h < 2: r,g,b = x,1,0
        elif h < 3: r,g,b = 0,1,x
        elif h < 4: r,g,b = 0,x,1
        elif h < 5: r,g,b = x,0,1
        else:       r,g,b = 1,0,x
        return (r, g, b)

    def narrator_text(self) -> str:
        if self.reheating:
            return "Inflation ends. The inflaton oscillates. Reheating begins. The hot Big Bang."
        if self.n_efolds >= 60:
            return f"N_e = {self.n_efolds:.0f} e-folds. The universe is flat. The horizon problem is solved."
        if self.epsilon < 0.01:
            return f"Slow roll: ε = {self.epsilon:.4f} ≪ 1. φ = {self.phi:.2f} Mₚ. Rolling..."
        return f"N_e = {self.n_efolds:.1f} / 60 · δρ/ρ = {self.delta_rho_over_rho:.2e}"
