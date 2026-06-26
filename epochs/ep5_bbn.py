"""
epochs/ep5_bbn.py — Big Bang Nucleosynthesis epoch (t = 1–200 s, T ~ 1–0.01 MeV).

GAMEPLAY:
  The player has a real-time 3-minute window (mirroring the actual ~3 min
  of BBN in the universe) to fuse as many nuclei as possible.

  - Protons and neutrons drift across the field
  - Player drag-selects and releases to attempt fusion
  - Each fusion attempt goes through InteractionEngine
  - Forbidden attempts show the physics explanation
  - A "deuterium bottleneck" timer shows: at T > 70 keV, D photodissociates
  - Score = primordial He-4 mass fraction achieved (target: Y_p = 0.245)

REAL PHYSICS TRACKED:
  - n/p ratio: starts at 1:6 at T=1 MeV (from weak freeze-out)
    n/p = exp(−Δm/T) = exp(−1.293 MeV / T)
  - Deuterium bottleneck: D forms only when T < 70 keV
  - Chain reactions: D+D→³He+n→⁴He, D+D→T+p→⁴He
  - Free neutron decay: τ_n = 879 s (neutrons decay if not bound)
  - Final Y_p = 2(n/p)/(1 + n/p) × (He-4 fraction from chain)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
try:
    import jax.numpy as jnp
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from config import PARTICLES, EPOCHS
from physics.interactions import Particle, InteractionEngine, InteractionResult
from physics.cosmology import CosmicClock, temperature_from_a, time_from_temperature_s


# ── Nuclear binding energies (MeV) ────────────────────────────
BINDING = {
    "deuterium":  2.224,    # p+n → D, Q = 2.224 MeV
    "helium3":    7.718,    # D+p → ³He, Q = 5.494 MeV
    "tritium":    8.482,    # D+n → T, Q = 6.258 MeV
    "helium4":   28.296,    # D+D → ⁴He (via chains), Q = 23.85 MeV
    "lithium7":  39.245,
}

# Reaction rates (simplified)
# σv in cm³/s at T_9 (temperature in units of 10⁹ K)
def reaction_rate_pn(T_MeV: float) -> float:
    """p + n → D + γ rate (cm³/s), approximate."""
    if T_MeV <= 0: return 0
    # Gamow peak cross section × velocity, rough parameterisation
    return 4.55e-20 / T_MeV**0.5 * math.exp(-0.0 / T_MeV)

def reaction_rate_DD(T_MeV: float) -> float:
    """D + D → products rate."""
    if T_MeV <= 0: return 0
    E_G = 0.66   # Gamow energy (MeV) for DD
    return 3.9e-18 * math.exp(-3 * (E_G / T_MeV) ** (1/3)) / T_MeV**(2/3)

def np_ratio(T_MeV: float) -> float:
    """
    n/p ratio from weak interaction equilibrium:
    n/p = exp(−Δm/T) where Δm = m_n − m_p = 1.293 MeV.
    Freeze-out at T_f ≈ 0.8 MeV: n/p = exp(−1.293/0.8) ≈ 0.2 (1:5)
    After freeze-out, n/p only decreases due to neutron decay.
    """
    delta_m = 1.293   # MeV
    if T_MeV > 0.8:   # above freeze-out: equilibrium
        return math.exp(-delta_m / T_MeV)
    else:              # below freeze-out: frozen-in value × decay
        r_freeze = math.exp(-delta_m / 0.8)
        t_s = time_from_temperature_s(T_MeV * 1e-3)  # convert MeV to GeV
        t_freeze = time_from_temperature_s(0.8e-3)
        tau_n = 879.6   # neutron lifetime (s)
        return r_freeze * math.exp(-(t_s - t_freeze) / tau_n)


@dataclass
class NuclearParticle:
    """A nucleus / nucleon in the BBN field."""
    species: str          # 'proton', 'neutron', 'deuterium', 'helium3', etc.
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    id: int = 0
    age_s: float = 0.0
    selected: bool = False
    fusing: bool = False  # currently animating a fusion event
    decay_timer: float = 0.0  # for free neutron decay

    @property
    def charge(self) -> int:
        return {"proton":1,"neutron":0,"deuterium":1,"helium3":2,
                "tritium":0,"helium4":2,"lithium7":3}.get(self.species, 0)

    @property
    def baryon_number(self) -> int:
        return {"proton":1,"neutron":1,"deuterium":2,"helium3":3,
                "tritium":3,"helium4":4,"lithium7":7}.get(self.species, 0)

    @property
    def mass_GeV(self) -> float:
        return {"proton":0.9383,"neutron":0.9396,"deuterium":1.8756,
                "helium3":2.8089,"tritium":2.8089,"helium4":3.7274,
                "lithium7":6.5355}.get(self.species, 1.0)

    @property
    def color_rgb(self) -> tuple:
        return {
            "proton":   (0.94, 0.75, 0.20),
            "neutron":  (0.55, 0.62, 0.72),
            "deuterium":(0.35, 0.75, 0.94),
            "helium3":  (0.40, 0.88, 0.55),
            "tritium":  (0.90, 0.55, 0.20),
            "helium4":  (0.50, 0.92, 0.40),
            "lithium7": (0.88, 0.70, 0.92),
        }.get(self.species, (0.8, 0.8, 0.8))

    @property
    def radius(self) -> float:
        A = self.baryon_number
        return 0.010 + 0.004 * A**(1/3)   # nuclear radius ~ A^{1/3}


@dataclass
class FusionEvent:
    """A completed or ongoing fusion event for visual replay."""
    x: float
    y: float
    product: str
    Q_MeV: float
    t_created: float
    equation: str
    fx_radius: float = 0.0
    alpha: float = 1.0


@dataclass
class ForbiddenEvent:
    """A forbidden interaction attempt for visual display."""
    x: float
    y: float
    reason: str
    equation: str
    law: str
    t_created: float
    alpha: float = 1.0
    reason_type: int = 0    # maps to shader colour


class BBNEpoch:
    """
    Full state of the BBN epoch simulation.

    Manages:
      - Field of nuclear particles drifting + colliding
      - Real-time temperature T(t) from CosmicClock
      - n/p ratio tracking + neutron decay
      - Player fusion interactions
      - Score: He-4 mass fraction Y_p
      - 3-minute countdown
    """

    def __init__(self, clock: CosmicClock, n_particles: int = 80):
        self.clock = clock
        self.clock.set_epoch(5)

        self.T_MeV    = 1.0      # start at T = 1 MeV
        self.t_game_s = 0.0      # game clock (0 → 180 s real time)
        self.time_limit = 180.0  # 3 minutes
        self.game_over  = False
        self.victory    = False

        # Physics counters
        self.n_protons   = 0
        self.n_neutrons  = 0
        self.n_deuterium = 0
        self.n_helium4   = 0
        self.n_helium3   = 0
        self.n_tritium   = 0
        self.n_lithium7  = 0
        self.total_baryons = 0

        # Particle field
        self.field: list[NuclearParticle] = []
        self._next_id = 0
        self._init_field(n_particles)

        # Events
        self.fusion_events:    list[FusionEvent]    = []
        self.forbidden_events: list[ForbiddenEvent] = []

        # Interaction engine
        self.engine = InteractionEngine(current_epoch=5)

        # Selected particles for interaction
        self.selection: list[NuclearParticle] = []

    def _spawn(self, species: str, x: float = None, y: float = None) -> NuclearParticle:
        if x is None: x = random.uniform(-0.9, 0.9)
        if y is None: y = random.uniform(-0.9, 0.9)
        speed = 0.08 + random.gauss(0, 0.03)
        angle = random.uniform(0, 2 * math.pi)
        p = NuclearParticle(
            species=species, x=x, y=y,
            vx=speed*math.cos(angle), vy=speed*math.sin(angle),
            id=self._next_id,
            decay_timer=879.6 if species=="neutron" else 1e10,
        )
        self._next_id += 1
        return p

    def _init_field(self, n: int):
        """
        Initialise field with correct n/p ratio at T = 1 MeV.
        n/p = exp(−1.293/1.0) ≈ 0.275 → roughly 1 neutron per 3.6 protons.
        """
        self.total_baryons = n
        ratio = np_ratio(self.T_MeV)
        n_n = int(n * ratio / (1 + ratio))
        n_p = n - n_n
        self.n_protons  = n_p
        self.n_neutrons = n_n

        for _ in range(n_p):
            self.field.append(self._spawn("proton"))
        for _ in range(n_n):
            self.field.append(self._spawn("neutron"))

    def update(self, dt_real: float):
        """Advance simulation by dt_real seconds of real time."""
        if self.game_over:
            return

        self.t_game_s += dt_real
        if self.t_game_s >= self.time_limit:
            self.game_over = True
            self._compute_final_score()
            return

        # Advance cosmic temperature
        # Map 180 s game time to t ~ 1 s → 200 s cosmic time
        t_cosmic = 1.0 + (self.t_game_s / self.time_limit) * 199.0
        self.T_MeV = self._cosmic_T(t_cosmic)

        # Move particles (simple Langevin + walls)
        self._move_particles(dt_real)

        # Neutron decay
        self._neutron_decay(dt_real)

        # Automatic fusion near threshold (background rate)
        self._background_reactions(dt_real)

        # Age fusion / forbidden events
        for ev in self.fusion_events:
            ev.alpha = max(0, ev.alpha - dt_real * 0.4)
        for ev in self.forbidden_events:
            ev.alpha = max(0, ev.alpha - dt_real * 0.6)
        self.fusion_events    = [e for e in self.fusion_events    if e.alpha > 0]
        self.forbidden_events = [e for e in self.forbidden_events if e.alpha > 0]

    def _cosmic_T(self, t_s: float) -> float:
        """
        T(t) in radiation dominated epoch:
        T ≈ (45 Mₚ² / 16π³ g*)^{1/4} · t^{-1/2}
        In MeV: T ≈ 1.307 / √t_s  (for g*=10.75 at T~1 MeV)
        """
        return max(0.001, 1.307 / math.sqrt(max(t_s, 0.01)))

    def _move_particles(self, dt: float):
        for p in self.field:
            p.x += p.vx * dt
            p.y += p.vy * dt
            # Wall bounce
            if abs(p.x) > 0.92: p.vx *= -1; p.x = math.copysign(0.92, p.x)
            if abs(p.y) > 0.92: p.vy *= -1; p.y = math.copysign(0.92, p.y)
            # Thermal noise
            p.vx += random.gauss(0, 0.008) * dt
            p.vy += random.gauss(0, 0.008) * dt
            # Speed cap
            spd = math.hypot(p.vx, p.vy)
            if spd > 0.35: p.vx, p.vy = p.vx/spd*0.35, p.vy/spd*0.35
            p.age_s += dt

    def _neutron_decay(self, dt: float):
        """Free neutrons decay: n → p + e⁻ + ν̄_e, τ = 879.6 s."""
        decayed = []
        for p in self.field:
            if p.species == "neutron":
                p.decay_timer -= dt
                if p.decay_timer <= 0:
                    decayed.append(p)
        for p in decayed:
            self.field.remove(p)
            proton = self._spawn("proton", x=p.x, y=p.y)
            self.field.append(proton)
            self.n_neutrons = max(0, self.n_neutrons - 1)
            self.n_protons += 1
            self._add_fusion_event(p.x, p.y, "proton", 1.293 - 0.511,
                "n → p + e⁻ + ν̄_e  (neutron β-decay, τ = 879.6 s)")

    def _background_reactions(self, dt: float):
        """
        Automatic background reactions based on temperature-dependent rates.
        The player's manual fusions are on top of this.
        """
        T = self.T_MeV
        if T < 0.07:  # below deuterium bottleneck: D survives
            rate_pn = reaction_rate_pn(T) * dt * 0.001
            # Probabilistically fuse nearby p+n pairs
            protons  = [p for p in self.field if p.species == "proton"]
            neutrons = [p for p in self.field if p.species == "neutron"]
            for pr in protons[:3]:
                for nt in neutrons[:3]:
                    d = math.hypot(pr.x - nt.x, pr.y - nt.y)
                    if d < 0.12 and random.random() < rate_pn * 0.5:
                        self._do_fusion(pr, nt, "deuterium",
                            BINDING["deuterium"],
                            "p + n → D + γ  (Q = 2.224 MeV)")

    def attempt_interaction(self, p1: NuclearParticle, p2: NuclearParticle) -> InteractionResult:
        """
        Player attempts to fuse p1 and p2.
        Returns InteractionResult with full physics explanation.
        """
        particle_a = Particle(species=p1.species, x=p1.x, y=p1.y,
                              energy_GeV=p1.mass_GeV)
        particle_b = Particle(species=p2.species, x=p2.x, y=p2.y,
                              energy_GeV=p2.mass_GeV)
        result = self.engine.attempt([particle_a, particle_b])

        mx = (p1.x + p2.x) / 2
        my = (p1.y + p2.y) / 2

        if result.allowed:
            product = result.products[0] if result.products else None
            if product:
                Q = self._Q_value(p1.species, p2.species, product)
                self._do_fusion(p1, p2, product, Q, result.equation)
        else:
            reason_type = self._reason_type(result.forbidden_law)
            ev = ForbiddenEvent(
                x=mx, y=my,
                reason=result.forbidden_reason,
                equation=result.forbidden_equation,
                law=result.forbidden_law,
                t_created=self.t_game_s,
                reason_type=reason_type,
            )
            self.forbidden_events.append(ev)

        return result

    def _do_fusion(self, p1: NuclearParticle, p2: NuclearParticle,
                   product: str, Q_MeV: float, equation: str):
        """Execute a fusion, removing reactants and spawning product."""
        mx = (p1.x + p2.x) / 2
        my = (p1.y + p2.y) / 2

        # Remove reactants
        for p in [p1, p2]:
            if p in self.field:
                self.field.remove(p)

        # Spawn product
        daughter = self._spawn(product, x=mx, y=my)
        daughter.vx = (p1.vx + p2.vx) * 0.5
        daughter.vy = (p1.vy + p2.vy) * 0.5
        self.field.append(daughter)

        # Update counters
        self._update_counts()

        # Record event
        self._add_fusion_event(mx, my, product, Q_MeV, equation)

    def _add_fusion_event(self, x, y, product, Q_MeV, equation):
        self.fusion_events.append(FusionEvent(
            x=x, y=y, product=product, Q_MeV=Q_MeV,
            t_created=self.t_game_s, equation=equation,
            fx_radius=0.0,
        ))

    def _update_counts(self):
        counts = {}
        for p in self.field:
            counts[p.species] = counts.get(p.species, 0) + 1
        self.n_protons   = counts.get("proton",   0)
        self.n_neutrons  = counts.get("neutron",  0)
        self.n_deuterium = counts.get("deuterium",0)
        self.n_helium3   = counts.get("helium3",  0)
        self.n_tritium   = counts.get("tritium",  0)
        self.n_helium4   = counts.get("helium4",  0)
        self.n_lithium7  = counts.get("lithium7", 0)

    def _Q_value(self, s1: str, s2: str, product: str) -> float:
        """Q = (m_initial - m_final) × 931.5 MeV/u (approximate)."""
        m_i = (PARTICLES.get(s1, {}).get("mass_GeV", 0) or 0)
        m_i += (PARTICLES.get(s2, {}).get("mass_GeV", 0) or 0)
        m_f = (PARTICLES.get(product, {}).get("mass_GeV", 0) or 0)
        return max(0, (m_i - m_f) * 1000)  # MeV

    def _reason_type(self, law: str) -> int:
        law = law.lower()
        if "charge" in law: return 0
        if "colour" in law or "confinement" in law: return 1
        if "energy" in law: return 2
        if "epoch" in law: return 3
        if "baryon" in law: return 4
        if "gso" in law: return 5
        return 1  # default red

    def _compute_final_score(self):
        """Compute Y_p = 4·n(He4) / (n_total_baryons) at end."""
        self._update_counts()
        baryon_weighted = (
            4 * self.n_helium4 +
            3 * (self.n_helium3 + self.n_tritium) +
            2 * self.n_deuterium +
            7 * self.n_lithium7 +
            self.n_protons + self.n_neutrons
        )
        if baryon_weighted > 0:
            self.Y_p = 4 * self.n_helium4 / baryon_weighted
        else:
            self.Y_p = 0.0
        self.victory = (0.20 < self.Y_p < 0.30)  # within range of real Y_p ≈ 0.245

    @property
    def Y_p_current(self) -> float:
        """Current primordial He-4 mass fraction."""
        total = sum(p.baryon_number for p in self.field)
        he4   = sum(p.baryon_number for p in self.field if p.species == "helium4")
        return (4 * he4 / total) if total > 0 else 0.0

    @property
    def time_remaining(self) -> float:
        return max(0, self.time_limit - self.t_game_s)

    @property
    def deuterium_bottleneck_active(self) -> bool:
        return self.T_MeV > 0.07

    @property
    def np_ratio_current(self) -> float:
        if self.n_protons == 0: return 0
        return self.n_neutrons / self.n_protons
