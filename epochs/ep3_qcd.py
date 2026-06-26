"""
epochs/ep3_qcd.py — QCD confinement (t = 10⁻⁶ → 10⁻⁴ s, T ~ 200 → 1 MeV).

GAMEPLAY:
  The quark-gluon plasma cools below Λ_QCD ≈ 217 MeV. Colour flux tubes
  form between quarks and confine them into hadrons.

  MECHANICS:
    - Quarks drift in a hot plasma
    - Drag a quark away from others → colour flux tube stretches visually
    - At r > 1 fm the tube snaps → new quark-antiquark pair created from vacuum
    - Click u + d + u (3 different colours) → proton forms (sound + glow)
    - Click u + d̄ → pion forms (meson)
    - Gluons drift separately — click two gluons → glueball (unstable)
    - Temperature counter: as T → 0, plasma clears and hadron gas fills screen

  FLUX TUBE VISUAL:
    - Rendered as a bright line between quark pair (in renderer)
    - Width ∝ 1/r² (constant energy per unit length = string tension)
    - Colour = mix of the two endpoint colour charges
    - At breaking point: flash + two new quarks materialise

  FORBIDDEN:
    - Single free quark: ⊗ "Colour confinement — V(r) = κr, κ = 0.9 GeV/fm"
    - Two quarks same colour: ⊗ "3⊗3 sextet — repulsive"
    - Gluon + quark → free: ⊗ "Colour not neutral"
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field as dc_field
from typing import Optional
import numpy as np

from config import LAMBDA_QCD_GEV, ALPHA_S_MZ, PARTICLES
from physics.cosmology import CosmicClock

STRING_TENSION_GEV_PER_FM = 0.9   # κ in V(r) = κr


@dataclass
class ColoredParticle:
    """A quark or gluon in the QCD epoch."""
    species:   str      # 'u_quark','d_quark','gluon', etc.
    colour:    str      # 'red','green','blue','octet_RG', etc.
    x: float = 0.0; y: float = 0.0; z: float = 0.0
    vx: float = 0.0;  vy: float = 0.0
    id: int = 0;       age_s: float = 0.0
    selected: bool = False
    confined: bool = False   # True once inside a hadron
    partner_id: Optional[int] = None  # flux tube partner

    # Render
    radius: float = 0.007
    metallic: float = 0.0
    roughness: float = 0.4
    emission_str: float = 2.5
    particle_type_int: int = 1   # quark

    _COLOUR_RGB = {
        "red":      (0.95, 0.20, 0.10),
        "green":    (0.10, 0.88, 0.25),
        "blue":     (0.15, 0.40, 0.98),
        "antired":  (0.05, 0.85, 0.90),
        "antigreen":(0.90, 0.15, 0.80),
        "antiblue": (0.98, 0.85, 0.05),
        "octet":    (0.95, 0.55, 0.10),   # gluon
    }

    @property
    def color_rgb(self) -> tuple:
        return self._COLOUR_RGB.get(self.colour, (0.8, 0.8, 0.8))

    @property
    def emission_rgb(self) -> tuple:
        return tuple(min(1.0, c * 1.5) for c in self.color_rgb)

    def update(self, dt: float, T_GeV: float):
        thermal_v = math.sqrt(max(T_GeV / 0.15, 0.01)) * 0.12
        self.x    += self.vx * dt
        self.y    += self.vy * dt
        self.age_s += dt
        self.vx   += random.gauss(0, thermal_v * 0.1) * dt
        self.vy   += random.gauss(0, thermal_v * 0.1) * dt
        spd = math.hypot(self.vx, self.vy)
        if spd > thermal_v:
            self.vx, self.vy = self.vx/spd*thermal_v, self.vy/spd*thermal_v
        # Boundary
        if abs(self.x) > 9.5: self.vx *= -0.85
        if abs(self.y) > 5.5: self.vy *= -0.85
        self.x = max(-9.5, min(9.5, self.x))
        self.y = max(-5.5, min(5.5, self.y))


@dataclass
class FluxTube:
    """Colour flux tube between two quarks."""
    q1_id: int
    q2_id: int
    x1: float; y1: float
    x2: float; y2: float
    color: tuple = (0.8, 0.5, 0.1)
    tension_GeV_per_fm: float = STRING_TENSION_GEV_PER_FM
    broken: bool = False
    break_alpha: float = 0.0

    @property
    def length_fm(self) -> float:
        """Length in fermi (1 fm ≈ screen unit * scale)."""
        return math.hypot(self.x2-self.x1, self.y2-self.y1) * 0.3

    @property
    def energy_GeV(self) -> float:
        return self.tension_GeV_per_fm * self.length_fm

    @property
    def should_break(self) -> bool:
        """Break when energy ≥ 2 × lightest quark mass (creates new pair)."""
        return self.length_fm > 1.0   # > 1 fm: break


@dataclass
class Hadron:
    """A formed hadron (proton, neutron, pion…)."""
    species:   str
    x: float;  y: float;  z: float = 0.0
    vx: float = 0.0; vy: float = 0.0
    constituent_ids: list = dc_field(default_factory=list)
    age_s: float = 0.0
    id: int = 0

    # Render
    radius: float = 0.016
    metallic: float = 0.1
    roughness: float = 0.5
    emission_str: float = 3.5
    particle_type_int: int = 6   # nucleus type
    selected: bool = False

    _HADRON_RGB = {
        "proton":   (0.94, 0.75, 0.18),
        "neutron":  (0.52, 0.60, 0.72),
        "pion_plus":(0.68, 0.65, 0.95),
        "pion_zero":(0.85, 0.85, 0.85),
        "glueball": (0.95, 0.55, 0.10),
    }

    @property
    def color_rgb(self) -> tuple:
        return self._HADRON_RGB.get(self.species, (0.7, 0.7, 0.7))

    @property
    def emission_rgb(self) -> tuple:
        c = self.color_rgb
        return tuple(min(1.0, v * 1.3) for v in c)

    def update(self, dt: float):
        self.x     += self.vx * dt
        self.y     += self.vy * dt
        self.age_s += dt
        self.vx    += random.gauss(0, 0.02) * dt
        self.vy    += random.gauss(0, 0.02) * dt
        if abs(self.x) > 9: self.vx *= -1
        if abs(self.y) > 5: self.vy *= -1


class QCDEpoch:
    """Full state of the QCD confinement epoch."""

    def __init__(self, clock: CosmicClock, n_quarks: int = 80):
        self.clock   = clock
        self.clock.set_epoch(3)
        self.T_GeV   = LAMBDA_QCD_GEV * 1.5   # start above Λ_QCD

        self.quarks:   list[ColoredParticle] = []
        self.gluons:   list[ColoredParticle] = []
        self.hadrons:  list[Hadron]          = []
        self.flux_tubes: list[FluxTube]      = []
        self.forbidden_events: list[dict]    = []
        self.interaction_log = []

        self.selection: list[ColoredParticle] = []
        self._next_id = 0
        self._init_plasma(n_quarks)

        # Statistics
        self.n_protons  = 0
        self.n_neutrons = 0
        self.n_pions    = 0
        self.n_string_breaks = 0
        self.complete = False

        # Dragged quark (player drags to stretch flux tube)
        self.dragged: Optional[ColoredParticle] = None
        self.drag_x = 0.0
        self.drag_y = 0.0

    def _spawn_particle(self, species: str, colour: str,
                        x: float = None, y: float = None,
                        ptype: int = 1) -> ColoredParticle:
        if x is None: x = random.uniform(-8, 8)
        if y is None: y = random.uniform(-4.5, 4.5)
        spd = random.uniform(0.06, 0.2)
        ang = random.uniform(0, 2*math.pi)
        p = ColoredParticle(
            species=species, colour=colour, x=x, y=y,
            vx=spd*math.cos(ang), vy=spd*math.sin(ang),
            id=self._next_id, particle_type_int=ptype,
        )
        self._next_id += 1
        return p

    def _init_plasma(self, n: int):
        colours = ["red", "green", "blue"]
        for i in range(n // 3):
            for col in colours:
                self.quarks.append(self._spawn_particle("u_quark", col))
                self.quarks.append(self._spawn_particle("d_quark", col))
        n_gluons = n // 4
        for _ in range(n_gluons):
            self.gluons.append(self._spawn_particle("gluon", "octet", ptype=2))

    def update(self, dt: float):
        if self.complete:
            return

        self.clock.step(dt)
        self.T_GeV = max(0.001, self.clock.T_GeV * 1000)

        # Update free quarks
        for q in self.quarks:
            if not q.confined:
                q.update(dt, self.T_GeV)

        for g in self.gluons:
            g.update(dt, self.T_GeV * 0.5)

        for h in self.hadrons:
            h.update(dt)

        # Update flux tubes: check for breaking
        self._update_flux_tubes()

        # Below Λ_QCD: auto-confine nearby colour-complementary quarks
        if self.T_GeV < LAMBDA_QCD_GEV * 1.1:
            self._auto_confine(dt)

        # Track counts
        self.n_protons  = sum(1 for h in self.hadrons if h.species == "proton")
        self.n_neutrons = sum(1 for h in self.hadrons if h.species == "neutron")
        self.n_pions    = sum(1 for h in self.hadrons if "pion" in h.species)

        # Complete when most quarks confined
        free_quarks = sum(1 for q in self.quarks if not q.confined)
        if free_quarks < 6 and len(self.hadrons) > 5:
            self.complete = True

    def _update_flux_tubes(self):
        to_break = []
        for ft in self.flux_tubes:
            # Find current positions
            q1 = next((q for q in self.quarks if q.id == ft.q1_id), None)
            q2 = next((q for q in self.quarks if q.id == ft.q2_id), None)
            if q1 and q2:
                ft.x1, ft.y1 = q1.x, q1.y
                ft.x2, ft.y2 = q2.x, q2.y
                if ft.should_break:
                    to_break.append(ft)
            else:
                to_break.append(ft)  # particle gone, remove tube

        for ft in to_break:
            if ft in self.flux_tubes:
                self.flux_tubes.remove(ft)
                self._break_flux_tube(ft)

    def _break_flux_tube(self, ft: FluxTube):
        """String breaks: create new quark-antiquark pair at midpoint."""
        mx = (ft.x1 + ft.x2) / 2
        my = (ft.y1 + ft.y2) / 2
        # New q-qbar pair
        colours = ["red", "green", "blue"]
        col = random.choice(colours)
        anticol = {"red":"antired","green":"antigreen","blue":"antiblue"}[col]
        new_q  = self._spawn_particle("u_quark",    col,    mx - 0.3, my)
        new_aq = self._spawn_particle("u_antiquark", anticol, mx + 0.3, my)
        self.quarks += [new_q, new_aq]
        self.n_string_breaks += 1

    def _auto_confine(self, dt: float):
        """Below Λ_QCD: nearby RGB triplets automatically form hadrons."""
        free = [q for q in self.quarks if not q.confined]
        red_qs   = [q for q in free if q.colour == "red"]
        green_qs = [q for q in free if q.colour == "green"]
        blue_qs  = [q for q in free if q.colour == "blue"]

        # Rate: faster at lower T
        rate = max(0, (LAMBDA_QCD_GEV - self.T_GeV) / LAMBDA_QCD_GEV) * dt * 0.8

        if red_qs and green_qs and blue_qs and random.random() < rate:
            r = random.choice(red_qs)
            g = random.choice(green_qs)
            b = random.choice(blue_qs)
            # Check species: uud → proton, udd → neutron
            specs = sorted([r.species, g.species, b.species])
            if specs.count("u_quark") == 2 and specs.count("d_quark") == 1:
                product = "proton"
            elif specs.count("u_quark") == 1 and specs.count("d_quark") == 2:
                product = "neutron"
            else:
                product = "proton"   # default
            self._form_hadron([r, g, b], product)

        # Quark + antiquark → pion
        anti = [q for q in free if q.is_antiquark]
        nonanit = [q for q in free if not q.is_antiquark]
        if anti and nonanit and random.random() < rate * 0.4:
            q  = random.choice(nonanit)
            aq = random.choice(anti)
            if math.hypot(q.x-aq.x, q.y-aq.y) < 1.5:
                self._form_hadron([q, aq], "pion_plus")

    def _form_hadron(self, constituents: list, species: str):
        mx = sum(q.x for q in constituents) / len(constituents)
        my = sum(q.y for q in constituents) / len(constituents)
        for q in constituents:
            q.confined = True
        h = Hadron(
            species=species, x=mx, y=my,
            vx=random.gauss(0, 0.06), vy=random.gauss(0, 0.06),
            constituent_ids=[q.id for q in constituents],
            id=self._next_id,
            radius={"proton":0.016,"neutron":0.016,"pion_plus":0.010}.get(species, 0.012),
        )
        self._next_id += 1
        self.hadrons.append(h)

    # ── Player interactions ────────────────────────────────
    def attempt_form_hadron(self, particles: list[ColoredParticle]):
        """Player manually selects quarks to form a hadron."""
        if len(particles) == 3:
            colours = [p.colour for p in particles]
            if set(colours) == {"red", "green", "blue"}:
                specs = sorted([p.species for p in particles])
                if specs.count("u_quark") == 2 and specs.count("d_quark") == 1:
                    self._form_hadron(particles, "proton")
                    return InteractionResult(
                        allowed=True, products=["proton"],
                        equation="uud → p⁺  ε^{ijk}q_iq_jq_k, colour singlet",
                        description="Three quarks in RGB form the colour singlet εᵢⱼₖqⁱqʲqᵏ — the proton. Binding energy ≈ 938 MeV from QCD gluon field energy.",
                        fx_type="proton_form",
                    )
                elif specs.count("u_quark") == 1 and specs.count("d_quark") == 2:
                    self._form_hadron(particles, "neutron")
                    return InteractionResult(
                        allowed=True, products=["neutron"],
                        equation="udd → n⁰  ε^{ijk}q_iq_jq_k, colour singlet",
                        description="One up + two down quarks form the neutron. Free neutron lifetime τ = 879.6 s.",
                        fx_type="proton_form",
                    )
            return InteractionResult(
                allowed=False,
                forbidden_reason="Not RGB: 3⊗3⊗3 ∋ 1 requires one of each colour",
                forbidden_law="SU(3) colour singlet requirement",
                forbidden_equation="ε^{ijk}q_R^iq_G^jq_B^k — antisymmetric in colour",
                description="Three quarks must carry one each of red, green, and blue to form a colour-neutral singlet. Your selection is not RGB.",
                fx_type="forbidden_colour",
            )

        if len(particles) == 2:
            a, b = particles
            if not a.is_antiquark and b.is_antiquark:
                anticol = {"red":"antired","green":"antigreen","blue":"antiblue"}
                if anticol.get(a.colour) == b.colour:
                    self._form_hadron([a, b], "pion_plus")
                    return InteractionResult(
                        allowed=True, products=["pion_plus"],
                        equation="ud̄ → π⁺  3⊗3̄ ∋ 1 (colour singlet meson)",
                        description="A quark and its colour-conjugate antiquark form a meson. The pion is the lightest and mediates nuclear forces at long range.",
                        fx_type="hadron_form",
                    )
            return InteractionResult(
                allowed=False,
                forbidden_reason="Two quarks cannot form colour singlet alone",
                forbidden_law="SU(3) colour singlet",
                forbidden_equation="3⊗3 = 6⊕3̄ — no singlet, need q+q̄ or q+q+q",
                description="Two quarks in the same colour representation combine as 3⊗3 = 6⊕3̄. Neither the sextet nor the antitriplet alone is a colour singlet.",
                fx_type="forbidden_colour",
            )

        return InteractionResult(allowed=False, forbidden_reason="Select 2 or 3 quarks")

    def start_drag(self, particle: ColoredParticle):
        self.dragged = particle
        # Create flux tube to nearest opposite-colour quark
        partners = [q for q in self.quarks
                    if q.id != particle.id and not q.confined
                    and not q.is_antiquark == particle.is_antiquark]
        if partners:
            nearest = min(partners, key=lambda q: math.hypot(q.x-particle.x, q.y-particle.y))
            ft = FluxTube(
                q1_id=particle.id, q2_id=nearest.id,
                x1=particle.x, y1=particle.y,
                x2=nearest.x, y2=nearest.y,
                color=tuple((a+b)/2 for a,b in zip(particle.color_rgb, nearest.color_rgb)),
            )
            self.flux_tubes.append(ft)

    def drag_to(self, x: float, y: float):
        if self.dragged:
            self.dragged.x = x
            self.dragged.y = y
            self.drag_x, self.drag_y = x, y

    def end_drag(self):
        self.dragged = None

    def get_render_particles(self) -> list:
        return [q for q in self.quarks if not q.confined] + self.gluons + self.hadrons

    def narrator_text(self) -> str:
        if self.T_GeV < LAMBDA_QCD_GEV * 0.9:
            return f"T = {self.T_GeV*1000:.0f} MeV < Λ_QCD. Confinement. {self.n_protons} protons · {self.n_neutrons} neutrons · {self.n_pions} pions."
        return f"T = {self.T_GeV*1000:.0f} MeV ≈ Λ_QCD = 217 MeV. Flux tubes forming. {self.n_string_breaks} string breaks so far."
