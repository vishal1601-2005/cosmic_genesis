"""
epochs/ep2_baryogenesis.py — Baryogenesis (t = 10⁻¹² → 10⁻⁶ s, T ~ 100 GeV → 200 MeV).

GAMEPLAY:
  Quarks and antiquarks fill the screen in equal numbers.
  The player must trigger sphaleron events and exploit CP violation
  to create the matter-antimatter asymmetry η = (n_B - n_B̄)/n_γ ≈ 6×10⁻¹⁰.

  MECHANICS:
    - Quarks (colour: R/G/B) and antiquarks drift across the field
    - Click quark + antiquark → annihilation → 2 photons (allowed)
    - Click quark + quark (same colour) → FORBIDDEN (colour repulsion)
    - Click quark + quark + quark → proton if colours form singlet
    - Sphaleron events appear as glowing electroweak instantons:
        click one → ΔB = ΔL = ±3, creates 3 extra baryons
    - CP violation slider: adjust δ_CKM phase → changes η
    - Target: η ≥ 5×10⁻¹⁰ before T drops below Λ_QCD
    - At T < Λ_QCD: quarks confine → transition to epoch 3

  FORBIDDEN interactions:
    - q + q (same colour) → ⊗ "3⊗3 = 6⊕3̄ — sextet repulsive"
    - q + q (wrong epoch) → ⊗ "Below Λ_QCD quarks are confined"
    - Creating isolated colour charge → ⊗ "Colour confinement"
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field as dc_field
from typing import Optional
import numpy as np

from config import PARTICLES, INTERACTIONS, LAMBDA_QCD_GEV
from physics.interactions import Particle, InteractionEngine, InteractionResult
from physics.cosmology import CosmicClock


# ── Colour charge enum ────────────────────────────────────────
COLOURS = ["red", "green", "blue"]
ANTICOLOURS = ["antired", "antigreen", "antiblue"]


@dataclass
class QuarkParticle:
    """A quark or antiquark in the baryogenesis field."""
    species:    str          # 'u_quark','d_quark','s_quark','u_antiquark', etc.
    colour:     str          # 'red','green','blue','antired','antigreen','antiblue'
    x:          float = 0.0
    y:          float = 0.0
    z:          float = 0.0
    vx:         float = 0.0
    vy:         float = 0.0
    id:         int   = 0
    age_s:      float = 0.0
    selected:   bool  = False
    # Render
    radius:     float = 0.008
    metallic:   float = 0.0
    roughness:  float = 0.4
    emission_str: float = 2.5
    particle_type_int: int = 1   # type 1 = quark

    @property
    def is_antiquark(self) -> bool:
        return "anti" in self.species

    @property
    def charge(self) -> float:
        return PARTICLES.get(self.species, {}).get("charge", 0)

    @property
    def baryon_number(self) -> float:
        return PARTICLES.get(self.species, {}).get("B", 0)

    @property
    def color_rgb(self) -> tuple:
        """Colour charge → RGB for rendering."""
        BASE = {
            "red":       (0.95, 0.25, 0.15),
            "green":     (0.15, 0.85, 0.30),
            "blue":      (0.20, 0.45, 0.95),
            "antired":   (0.05, 0.80, 0.85),   # cyan (complementary to red)
            "antigreen": (0.85, 0.20, 0.75),   # magenta
            "antiblue":  (0.95, 0.80, 0.10),   # yellow
        }
        return BASE.get(self.colour, (0.8, 0.8, 0.8))

    @property
    def emission_rgb(self) -> tuple:
        c = self.color_rgb
        return tuple(min(1.0, v * 1.4) for v in c)

    def update(self, dt: float, T_GeV: float):
        """Thermal motion at temperature T."""
        thermal_v = math.sqrt(T_GeV / 0.001) * 0.06
        self.x   += self.vx * dt
        self.y   += self.vy * dt
        self.age_s += dt
        # Brownian thermal kicks
        self.vx  += random.gauss(0, thermal_v * 0.15) * dt
        self.vy  += random.gauss(0, thermal_v * 0.15) * dt
        # Speed cap
        spd = math.hypot(self.vx, self.vy)
        if spd > thermal_v:
            self.vx, self.vy = self.vx/spd*thermal_v, self.vy/spd*thermal_v
        # Wrap boundaries
        if abs(self.x) > 9.5: self.vx *= -0.8
        if abs(self.y) > 5.5: self.vy *= -0.8
        self.x = max(-9.5, min(9.5, self.x))
        self.y = max(-5.5, min(5.5, self.y))


@dataclass
class SphalerOn:
    """An electroweak sphaleron — EW instanton that changes B+L by ±3."""
    x:      float
    y:      float
    z:      float = 0.0
    active: bool  = True
    age_s:  float = 0.0
    radius: float = 0.05
    # Render: glowing electroweak knot
    color_rgb:    tuple = (0.90, 0.60, 0.10)
    emission_rgb: tuple = (1.00, 0.75, 0.20)
    emission_str: float = 5.0
    metallic:     float = 0.3
    roughness:    float = 0.2
    particle_type_int: int = 0
    selected:     bool  = False
    vx: float = 0.0
    vy: float = 0.0

    def update(self, dt: float):
        self.age_s += dt
        # Pulse
        self.emission_str = 4.0 + 2.0 * math.sin(self.age_s * 8.0)
        # Slow drift
        self.x += self.vx * dt
        self.y += self.vy * dt
        if abs(self.x) > 9: self.vx *= -1
        if abs(self.y) > 5: self.vy *= -1


@dataclass
class AnnihilationEvent:
    x: float; y: float
    color: tuple
    alpha: float = 1.0
    radius: float = 0.0
    t: float = 0.0

    def update(self, dt: float):
        self.t      += dt
        self.radius  = self.t * 3.5
        self.alpha   = max(0.0, 1.0 - self.t / 0.6)


class BaryogenesisEpoch:
    """Full state of the baryogenesis epoch."""

    # Sakharov conditions tracker
    class Sakharov:
        b_violation  = False   # B violation (sphaleron triggered)
        c_cp_violation = False # C and CP violation (δ_CKM ≠ 0)
        nonequilibrium = False # Non-equilibrium (EW phase transition)

    def __init__(self, clock: CosmicClock, n_quarks: int = 120):
        self.clock     = clock
        self.clock.set_epoch(2)
        self.engine    = InteractionEngine(current_epoch=2)

        # Physics
        self.T_GeV     = 100.0   # start at EW scale
        self.delta_CKM = 1.2     # CP-violation phase (radians), player-adjustable
        self.eta       = 0.0     # baryon asymmetry achieved
        self.eta_target= 6.1e-10

        # Sakharov conditions
        self.sakharov  = self.Sakharov()

        # Particles
        self.quarks: list[QuarkParticle]  = []
        self.sphalerons: list[SphalerOn]  = []
        self._next_id = 0
        self._init_quarks(n_quarks)
        self._spawn_sphalerons(4)

        # Interaction events
        self.annihilations: list[AnnihilationEvent] = []
        self.forbidden_events: list[dict] = []
        self.interaction_log: list[InteractionResult] = []

        # Selection
        self.selection: list[QuarkParticle] = []

        # Statistics
        self.n_baryons    = 0
        self.n_antibaryons= 0
        self.n_annihilations = 0
        self.n_sphalerons_triggered = 0
        self.complete = False

        # Timing
        self._sphaleron_timer = 0.0
        self._SPHALERON_INTERVAL = 8.0

    def _make_quark(self, species: str, colour: str,
                    x: float = None, y: float = None) -> QuarkParticle:
        if x is None: x = random.uniform(-9, 9)
        if y is None: y = random.uniform(-5, 5)
        spd = random.uniform(0.08, 0.25)
        ang = random.uniform(0, 2*math.pi)
        q = QuarkParticle(
            species=species, colour=colour,
            x=x, y=y,
            vx=spd*math.cos(ang), vy=spd*math.sin(ang),
            id=self._next_id,
        )
        self._next_id += 1
        return q

    def _init_quarks(self, n: int):
        """Spawn equal numbers of quarks and antiquarks."""
        species_q  = ["u_quark", "d_quark", "s_quark"]
        species_aq = ["u_antiquark", "d_antiquark"]
        for i in range(n // 2):
            sp = random.choice(species_q)
            cl = random.choice(COLOURS)
            self.quarks.append(self._make_quark(sp, cl))
        for i in range(n // 2):
            sp = random.choice(species_aq)
            cl = random.choice(ANTICOLOURS)
            self.quarks.append(self._make_quark(sp, cl))
        self._recount()

    def _spawn_sphalerons(self, n: int = 2):
        for _ in range(n):
            spd = random.uniform(0.02, 0.08)
            ang = random.uniform(0, 2*math.pi)
            self.sphalerons.append(SphalerOn(
                x=random.uniform(-8, 8),
                y=random.uniform(-4, 4),
                vx=spd*math.cos(ang),
                vy=spd*math.sin(ang),
            ))

    def update(self, dt: float):
        if self.complete:
            return

        self.clock.step(dt)
        self.T_GeV = self.clock.T_GeV * 1000   # GeV

        # Update particles
        for q in self.quarks:
            q.update(dt, self.T_GeV)
        for s in self.sphalerons:
            s.update(dt)

        # Background pair creation/annihilation at thermal rate
        self._background_processes(dt)

        # Auto-respawn sphalerons
        self._sphaleron_timer += dt
        if self._sphaleron_timer > self._SPHALERON_INTERVAL:
            self._sphaleron_timer = 0.0
            self._spawn_sphalerons(1)

        # Update visual events
        for ev in self.annihilations:
            ev.update(dt)
        self.annihilations = [e for e in self.annihilations if e.alpha > 0]

        # Compute eta
        self._recount()
        n_photons = max(1, len(self.quarks) * 10)  # rough estimate
        self.eta  = (self.n_baryons - self.n_antibaryons) / n_photons

        # Sakharov tracker
        self.sakharov.c_cp_violation   = abs(math.sin(self.delta_CKM)) > 0.01
        self.sakharov.nonequilibrium   = self.T_GeV < 110.0

        # Check transition to QCD epoch
        if self.T_GeV < LAMBDA_QCD_GEV * 1000 * 0.8:
            self.complete = True

        # Keep quark count reasonable
        if len(self.quarks) > 300:
            self.quarks = self.quarks[-200:]

    def _background_processes(self, dt: float):
        """Thermal background: random pair annihilations."""
        rate = max(0, (self.T_GeV - 10) / 100) * dt * 0.3
        quarks_list    = [q for q in self.quarks if not q.is_antiquark]
        antiquarks_list= [q for q in self.quarks if q.is_antiquark]
        n_annihilate   = int(rate * min(len(quarks_list), len(antiquarks_list)))
        for _ in range(n_annihilate):
            if not quarks_list or not antiquarks_list:
                break
            q  = random.choice(quarks_list)
            aq = random.choice(antiquarks_list)
            if q in self.quarks and aq in self.quarks:
                mx = (q.x + aq.x) / 2
                my = (q.y + aq.y) / 2
                self.quarks.remove(q)
                self.quarks.remove(aq)
                quarks_list.remove(q)
                antiquarks_list.remove(aq)
                self.annihilations.append(AnnihilationEvent(
                    x=mx, y=my, color=q.color_rgb
                ))
                self.n_annihilations += 1

        # CP violation: slight excess of quarks surviving
        # Probability of quark surviving annihilation > antiquark
        if self.sakharov.c_cp_violation and random.random() < abs(math.sin(self.delta_CKM)) * dt * 0.5:
            sp = random.choice(["u_quark", "d_quark"])
            cl = random.choice(COLOURS)
            self.quarks.append(self._make_quark(sp, cl))

    def attempt_interaction(self, particles: list) -> InteractionResult:
        """Player attempts interaction between selected particles."""
        if len(particles) < 2:
            return InteractionResult(allowed=False, forbidden_reason="Select 2 particles")

        a, b = particles[0], particles[1]

        # Both quarks?
        if not a.is_antiquark and not b.is_antiquark:
            # Same colour → forbidden
            if a.colour == b.colour:
                return InteractionResult(
                    allowed=False,
                    forbidden_reason="Same colour: 3⊗3 = 6⊕3̄ — sextet repulsive",
                    forbidden_law="SU(3) colour antisymmetry",
                    forbidden_equation="3 ⊗ 3 = 6 ⊕ 3̄: only 3̄ (antitriplet) can bind",
                    description=(
                        "Two quarks in the same colour representation combine as 3⊗3 = 6⊕3̄. "
                        "The symmetric sextet 6 is colour-repulsive and cannot form a bound state. "
                        "Only the antisymmetric antitriplet 3̄ can pair with a third quark to form "
                        "a colour-singlet baryon."
                    ),
                    fx_type="forbidden_colour",
                )
            # Different colours → need a third quark for proton
            return InteractionResult(
                allowed=False,
                forbidden_reason="Need 3 quarks for colour singlet: 3⊗3⊗3 ∋ 1",
                forbidden_law="SU(3) colour singlet requirement",
                forbidden_equation="3⊗3⊗3 = 10⊕8⊕8⊕1 — only the 1 is colour-neutral",
                description=(
                    "Two quarks alone cannot form a colour-neutral (singlet) state. "
                    "You need a third quark: 3⊗3⊗3 contains the singlet ε^{ijk}q_iq_jq_k."
                ),
                fx_type="forbidden_colour",
            )

        # Quark + antiquark → annihilation
        if not a.is_antiquark and b.is_antiquark:
            # Check colour-anticolour match
            colour_match = {
                "red": "antired", "green": "antigreen", "blue": "antiblue"
            }
            if colour_match.get(a.colour) != b.colour:
                return InteractionResult(
                    allowed=False,
                    forbidden_reason=f"Colour mismatch: {a.colour} ≠ complement of {b.colour}",
                    forbidden_law="Colour neutrality",
                    forbidden_equation="q(R) + q̄(Ḡ) → not colour-neutral",
                    description="A quark and antiquark must carry complementary colour charges to annihilate into a colour-singlet (e.g. R+R̄, G+Ḡ, B+B̄).",
                    fx_type="forbidden_colour",
                )
            mx = (a.x + b.x) / 2
            my = (a.y + b.y) / 2
            if a in self.quarks: self.quarks.remove(a)
            if b in self.quarks: self.quarks.remove(b)
            self.annihilations.append(AnnihilationEvent(x=mx, y=my, color=a.color_rgb))
            self.n_annihilations += 1
            self._recount()
            return InteractionResult(
                allowed=True,
                products=["photon", "photon"],
                equation="q + q̄ → 2γ  (Q = 2m_q c²)",
                description="Quark-antiquark annihilation into two photons. The colour charges cancel in the final state.",
                fx_type="annihilation_flash",
                energy_released_GeV=2 * (PARTICLES.get(a.species, {}).get("mass_GeV", 0) or 0.002),
            )

        # Antiquark + quark (reversed order)
        if a.is_antiquark and not b.is_antiquark:
            return self.attempt_interaction([b, a])

        return InteractionResult(allowed=False, forbidden_reason="Unknown interaction")

    def trigger_sphaleron(self, sphaleron: SphalerOn) -> InteractionResult:
        """Player clicks a sphaleron → ΔB = ΔL = +3."""
        if not sphaleron.active:
            return InteractionResult(allowed=False, forbidden_reason="Sphaleron already fired")

        sphaleron.active = False
        self.sakharov.b_violation = True
        self.n_sphalerons_triggered += 1

        # Create 3 extra quarks (baryogenesis!)
        colours = ["red", "green", "blue"]
        species = ["u_quark", "u_quark", "d_quark"]  # proton content
        for sp, cl in zip(species, colours):
            q = self._make_quark(sp, cl, x=sphaleron.x, y=sphaleron.y)
            q.vx = random.gauss(0, 0.3)
            q.vy = random.gauss(0, 0.3)
            self.quarks.append(q)

        self.sphalerons.remove(sphaleron)
        self._recount()

        return InteractionResult(
            allowed=True,
            products=["u_quark", "u_quark", "d_quark"],
            equation="Sphaleron: ΔB = ΔL = +3 · Rate Γ ∝ α_W⁴T⁴e^{-E_sph/T}",
            description=(
                "An electroweak sphaleron event violates baryon + lepton number by ±3. "
                f"CP violation (δ_CKM = {self.delta_CKM:.2f} rad) biases this toward net baryon production. "
                f"η = (n_B - n_B̄)/n_γ = {self.eta:.2e} (target: 6×10⁻¹⁰)."
            ),
            fx_type="sphaleron_burst",
            energy_released_GeV=0.0,
        )

    def set_cp_phase(self, delta: float):
        """Player adjusts the CKM CP-violation phase δ ∈ [0, 2π]."""
        self.delta_CKM = float(delta)
        self.sakharov.c_cp_violation = abs(math.sin(delta)) > 0.01

    def _recount(self):
        self.n_baryons     = sum(1 for q in self.quarks if not q.is_antiquark)
        self.n_antibaryons = sum(1 for q in self.quarks if q.is_antiquark)

    def get_render_particles(self) -> list:
        return self.quarks + self.sphalerons

    @property
    def asymmetry_achieved(self) -> bool:
        return self.eta >= self.eta_target * 0.5

    @property
    def all_sakharov(self) -> bool:
        s = self.sakharov
        return s.b_violation and s.c_cp_violation and s.nonequilibrium

    def narrator_text(self) -> str:
        if self.complete:
            return f"T < Λ_QCD. Quarks confine. η = {self.eta:.2e}. All matter that will ever exist is fixed."
        if self.all_sakharov:
            return f"All three Sakharov conditions met. η = {self.eta:.2e} (target: 6×10⁻¹⁰)."
        missing = []
        if not self.sakharov.b_violation:    missing.append("B-violation (trigger a sphaleron)")
        if not self.sakharov.c_cp_violation: missing.append("CP-violation (set δ_CKM ≠ 0)")
        if not self.sakharov.nonequilibrium: missing.append("non-equilibrium (wait for EW transition)")
        return "Sakharov: need " + " · ".join(missing[:2])
