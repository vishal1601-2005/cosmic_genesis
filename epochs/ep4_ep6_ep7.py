"""
epochs/ep4_axions.py — Axion condensation (t = 10⁻⁴ → 1 s).
epochs/ep6_recombination.py — Recombination (t = 380,000 yr).
epochs/ep7_structure.py — Structure formation (t = 100 Myr → 1 Gyr).

All three in one file to complete the epoch set.
"""

from __future__ import annotations
import math, random
from dataclasses import dataclass, field as dc_field
from typing import Optional
import numpy as np

from config import PARTICLES
from physics.cosmology import CosmicClock
from physics.fields import FieldManager


# ══════════════════════════════════════════════════════════════
#  EPOCH 4: AXION CONDENSATION
# ══════════════════════════════════════════════════════════════

@dataclass
class AxionParticle:
    """Visual representation of an axion field excitation."""
    x: float; y: float; z: float = 0.0
    vx: float = 0.0; vy: float = 0.0
    phase: float = 0.0
    amplitude: float = 0.5
    age_s: float = 0.0
    id: int = 0
    selected: bool = False
    # Render
    color_rgb:    tuple = (0.40, 0.22, 0.75)
    emission_rgb: tuple = (0.50, 0.28, 0.90)
    emission_str: float = 1.2
    radius:       float = 0.006
    metallic:     float = 0.0
    roughness:    float = 0.7
    particle_type_int: int = 4   # dark matter type

    def update(self, dt: float, m_a: float, H: float):
        # Oscillate at ω ≈ m_a with Hubble damping
        self.phase     += m_a * dt
        self.amplitude *= math.exp(-H * dt * 0.5)
        self.x         += self.vx * dt
        self.y         += self.vy * dt
        self.age_s     += dt
        # Field oscillation changes brightness
        osc = 0.5 + 0.5 * math.cos(self.phase)
        self.emission_str = 0.6 + osc * 1.0
        if abs(self.x) > 9: self.vx *= -1
        if abs(self.y) > 5: self.vy *= -1


class AxionEpoch:
    """
    Axion condensation epoch.
    Player sets misalignment angle θ_i by dragging a slider.
    Too large → overclose universe. Too small → not enough DM.
    """

    OMEGA_DM_TARGET = 0.266   # target dark matter density fraction
    F_A_GEV = 1e12             # PQ scale (GeV)
    M_A_EV  = 6e-6             # axion mass (eV)

    def __init__(self, clock: CosmicClock, fields: FieldManager):
        self.clock   = clock
        self.fields  = fields
        self.clock.set_epoch(4)

        self.theta_i   = math.pi / 3.0   # misalignment angle (player sets)
        self.theta     = self.theta_i     # current field angle
        self.theta_dot = 0.0
        self.omega_a   = 0.0             # computed relic density

        self.axions: list[AxionParticle] = []
        self._next_id = 0
        self._spawn_axions(200)

        self.forbidden_events: list[dict] = []
        self.complete = False
        self._t = 0.0

    def _spawn_axions(self, n: int):
        for _ in range(n):
            spd = random.gauss(0, 0.03)
            ang = random.uniform(0, 2*math.pi)
            self.axions.append(AxionParticle(
                x=random.uniform(-9,9), y=random.uniform(-5,5),
                vx=spd*math.cos(ang), vy=spd*math.sin(ang),
                phase=random.uniform(0, 2*math.pi),
                amplitude=abs(self.theta_i),
                id=self._next_id,
            ))
            self._next_id += 1

    def update(self, dt: float):
        if self.complete: return
        self._t += dt
        self.clock.step(dt)
        H    = self.clock.H_GeV
        m_a  = self.M_A_EV * 1e-9   # in GeV

        # Axion EOM: θ̈ + 3Hθ̇ + m_a²sin(θ) = 0
        theta_ddot  = -3*H*self.theta_dot - m_a**2 * math.sin(self.theta)
        self.theta_dot += dt * theta_ddot
        self.theta     += dt * self.theta_dot

        # Relic density: Ω_a h² ≈ 0.12 × (θ_i/1)² × (f_a/10¹² GeV)^{7/6}
        self.omega_a = 0.12 * self.theta_i**2

        # Update axion particles
        for a in self.axions:
            a.update(dt, m_a * 1e9, H)   # m_a in eV for oscillation rate

        # Update field
        self.fields.step(4, H, max(1e-28, math.exp(math.log(1e-28) + H*self._t)), self.clock.T_GeV)

        if self._t > 15.0:
            self.complete = True

    def set_misalignment(self, theta: float):
        """Player drags slider to set θ_i ∈ [0, π]."""
        self.theta_i = max(0.01, min(math.pi - 0.01, float(theta)))
        self.theta   = self.theta_i
        # Recalculate amplitudes
        for a in self.axions:
            a.amplitude = abs(self.theta_i)
        self.fields.set_axion_angle(self.theta_i)

    def omega_status(self) -> str:
        if   self.omega_a > 0.35: return "TOO MUCH — universe overclosed"
        elif self.omega_a < 0.10: return "TOO LITTLE — not enough dark matter"
        else:                     return f"Ω_a h² ≈ {self.omega_a:.3f} ✓"

    def get_render_particles(self) -> list:
        return self.axions

    def narrator_text(self) -> str:
        return (f"θ_i = {self.theta_i:.2f} rad · θ(t) = {self.theta:.3f} · "
                f"Ω_a h² = {self.omega_a:.3f} ({self.omega_status()})")


# ══════════════════════════════════════════════════════════════
#  EPOCH 6: RECOMBINATION
# ══════════════════════════════════════════════════════════════

@dataclass
class PlasmaParticle:
    """Electron or proton in the recombination epoch."""
    species:  str     # 'electron' or 'proton'
    x: float; y: float; z: float = 0.0
    vx: float = 0.0; vy: float = 0.0
    id: int = 0; age_s: float = 0.0
    bound: bool = False   # True once in hydrogen atom
    selected: bool = False

    # Render
    radius:       float = 0.005
    metallic:     float = 0.0
    roughness:    float = 0.6
    emission_str: float = 2.0
    particle_type_int: int = 0

    _RGB = {
        "electron": (0.28, 0.78, 0.55),
        "proton":   (0.94, 0.75, 0.18),
        "hydrogen": (0.62, 0.75, 0.95),
    }

    @property
    def color_rgb(self) -> tuple:
        if self.bound: return self._RGB["hydrogen"]
        return self._RGB.get(self.species, (0.7,0.7,0.7))

    @property
    def emission_rgb(self) -> tuple:
        return tuple(min(1.0, c*1.3) for c in self.color_rgb)

    def update(self, dt: float, T_K: float):
        thermal_v = math.sqrt(max(T_K / 3000, 0.01)) * 0.15
        self.x    += self.vx * dt; self.y += self.vy * dt
        self.age_s += dt
        self.vx   += random.gauss(0, thermal_v * 0.08) * dt
        self.vy   += random.gauss(0, thermal_v * 0.08) * dt
        spd = math.hypot(self.vx, self.vy)
        if spd > thermal_v: self.vx,self.vy = self.vx/spd*thermal_v,self.vy/spd*thermal_v
        if abs(self.x) > 9.5: self.vx *= -1
        if abs(self.y) > 5.5: self.vy *= -1


@dataclass
class HydrogenAtom:
    """A neutral hydrogen atom formed by recombination."""
    x: float; y: float; z: float = 0.0
    vx: float = 0.0; vy: float = 0.0
    id: int = 0; age_s: float = 0.0; selected: bool = False
    radius:       float = 0.008
    metallic:     float = 0.0
    roughness:    float = 0.6
    emission_str: float = 1.5
    particle_type_int: int = 0
    color_rgb:    tuple = (0.62, 0.75, 0.95)
    emission_rgb: tuple = (0.70, 0.82, 1.00)

    def update(self, dt: float):
        self.x    += self.vx * dt; self.y += self.vy * dt
        self.age_s += dt
        self.vx   += random.gauss(0, 0.01) * dt
        self.vy   += random.gauss(0, 0.01) * dt
        if abs(self.x) > 9: self.vx *= -1
        if abs(self.y) > 5: self.vy *= -1


class RecombinationEpoch:
    """
    Recombination epoch. Player drags electrons onto protons
    to form neutral hydrogen. The universe goes transparent.
    """
    RECOMB_T_K = 3000.0   # recombination temperature

    def __init__(self, clock: CosmicClock, n_particles: int = 120):
        self.clock = clock
        self.clock.set_epoch(6)
        self.T_K   = 4000.0
        self.plasma:   list[PlasmaParticle] = []
        self.hydrogen: list[HydrogenAtom]   = []
        self.forbidden_events: list[dict]   = []
        self.interaction_log = []
        self.selection: list[PlasmaParticle] = []
        self._next_id = 0
        self._init_plasma(n_particles)
        self.n_hydrogen  = 0
        self.ionisation  = 1.0   # fraction ionised (1=full plasma, 0=all neutral)
        self.cmb_phase   = 0.0   # 0=hot plasma, 1=transparent+CMB
        self.complete    = False

    def _spawn(self, species: str, x=None, y=None) -> PlasmaParticle:
        if x is None: x = random.uniform(-9,9)
        if y is None: y = random.uniform(-5,5)
        spd = random.uniform(0.05, 0.18)
        ang = random.uniform(0, 2*math.pi)
        p = PlasmaParticle(
            species=species, x=x, y=y,
            vx=spd*math.cos(ang), vy=spd*math.sin(ang),
            id=self._next_id,
        )
        self._next_id += 1
        return p

    def _init_plasma(self, n: int):
        for _ in range(n//2):
            self.plasma.append(self._spawn("electron"))
            self.plasma.append(self._spawn("proton"))

    def update(self, dt: float):
        if self.complete: return
        self.clock.step(dt)
        self.T_K = self.clock.T_GeV / 8.617e-14

        for p in self.plasma:
            if not p.bound:
                p.update(dt, self.T_K)
        for h in self.hydrogen:
            h.update(dt)

        # Saha equation: ionisation fraction x_e
        # x_e²/(1-x_e) = (1/η)(T/m_e)^{3/2} e^{-E_ion/T}
        E_ion_eV = 13.6
        T_eV     = self.T_K * 8.617e-5
        eta      = 6.1e-10
        if T_eV > 0:
            saha = (1.0/eta) * (T_eV/0.511e6)**1.5 * math.exp(-E_ion_eV/T_eV)
            x_e  = min(1.0, max(0.0, saha / (1 + saha + 1e-8)**0.5))
            self.ionisation = x_e

        # Auto-recombination proportional to Saha
        if self.ionisation < 0.8:
            rate = (1 - self.ionisation) * dt * 0.8
            free_e = [p for p in self.plasma if p.species=="electron" and not p.bound]
            free_p = [p for p in self.plasma if p.species=="proton"   and not p.bound]
            n_recomb = int(rate * min(len(free_e), len(free_p)))
            for _ in range(n_recomb):
                if not free_e or not free_p: break
                e = random.choice(free_e); pr = random.choice(free_p)
                if e in self.plasma and pr in self.plasma:
                    self._do_recombination(e, pr)
                    free_e.remove(e); free_p.remove(pr)

        self.n_hydrogen = len(self.hydrogen)
        total = max(1, len(self.plasma)//2 + self.n_hydrogen)
        self.cmb_phase  = min(1.0, self.n_hydrogen / total)

        if self.cmb_phase > 0.9 or self.T_K < 2700:
            self.complete = True

    def attempt_recombination(self, electron: PlasmaParticle,
                               proton: PlasmaParticle):
        """Player drags electron onto proton."""
        if self.T_K > self.RECOMB_T_K * 1.3:
            return {
                "allowed": False,
                "reason": f"T = {self.T_K:.0f} K > {self.RECOMB_T_K*1.3:.0f} K",
                "equation": "E_ion = 13.6 eV > k_BT: requires T < 3000 K",
                "why": (
                    f"At T = {self.T_K:.0f} K, the Wien tail of the photon distribution "
                    "still contains enough photons above 13.6 eV to immediately re-ionise "
                    "any hydrogen that forms. Recombination is suppressed despite "
                    "E_ion/k_B = 158,000 K — the tail extends far beyond the mean."
                ),
                "reason_type": 2,
            }
        self._do_recombination(electron, proton)
        return {
            "allowed": True,
            "equation": "e⁻ + p → H(1s) + γ  (Lyman-α, E_γ = 10.2 eV)",
            "description": "Recombination. The photon released is a Lyman-α photon. The universe becomes transparent to photons that don't carry this exact energy.",
        }

    def _do_recombination(self, e: PlasmaParticle, p: PlasmaParticle):
        mx = (e.x + p.x) / 2; my = (e.y + p.y) / 2
        e.bound = p.bound = True
        h = HydrogenAtom(
            x=mx, y=my,
            vx=(e.vx+p.vx)/2, vy=(e.vy+p.vy)/2,
            id=self._next_id,
        )
        self._next_id += 1
        self.hydrogen.append(h)

    def get_render_particles(self) -> list:
        free = [p for p in self.plasma if not p.bound]
        return free + self.hydrogen

    def narrator_text(self) -> str:
        if self.cmb_phase > 0.8:
            return "The universe is transparent. Photons stream free. The CMB is born."
        return (f"T = {self.T_K:.0f} K · ionisation = {self.ionisation:.3f} · "
                f"{self.n_hydrogen} hydrogen atoms formed")


# ══════════════════════════════════════════════════════════════
#  EPOCH 7: STRUCTURE FORMATION
# ══════════════════════════════════════════════════════════════

@dataclass
class DarkMatterHalo:
    x: float; y: float; z: float = 0.0
    vx: float = 0.0; vy: float = 0.0
    mass_solar: float = 1e10   # solar masses
    radius: float = 0.08
    id: int = 0; age_s: float = 0.0; selected: bool = False
    merging: bool = False
    color_rgb:    tuple = (0.22, 0.12, 0.42)
    emission_rgb: tuple = (0.30, 0.16, 0.55)
    emission_str: float = 1.5
    metallic:     float = 0.0
    roughness:    float = 0.9
    particle_type_int: int = 7

    @property
    def radius_render(self) -> float:
        return 0.04 + 0.015 * math.log10(max(1, self.mass_solar / 1e8))

    def update(self, dt: float):
        self.x    += self.vx * dt; self.y += self.vy * dt
        self.age_s += dt
        # Gravitational pull toward massive neighbours handled externally
        if abs(self.x) > 9: self.vx *= -0.8
        if abs(self.y) > 5: self.vy *= -0.8


@dataclass
class GasCloud:
    x: float; y: float; z: float = 0.0
    vx: float = 0.0; vy: float = 0.0
    mass_solar: float = 1e6
    id: int = 0; age_s: float = 0.0; selected: bool = False
    has_halo: bool = False
    color_rgb:    tuple = (0.35, 0.55, 0.80)
    emission_rgb: tuple = (0.45, 0.65, 0.90)
    emission_str: float = 1.2
    radius:       float = 0.025
    metallic:     float = 0.0
    roughness:    float = 0.7
    particle_type_int: int = 0

    def update(self, dt: float):
        self.x    += self.vx * dt; self.y += self.vy * dt
        self.age_s += dt
        if abs(self.x) > 9: self.vx *= -1
        if abs(self.y) > 5: self.vy *= -1


@dataclass
class Star:
    x: float; y: float; z: float = 0.0
    vx: float = 0.0; vy: float = 0.0
    mass_solar: float = 100.0   # Pop III: 100 M☉
    T_surface_K: float = 100000.0  # very hot blue star
    id: int = 0; age_s: float = 0.0; selected: bool = False
    color_rgb:    tuple = (0.75, 0.88, 1.00)
    emission_rgb: tuple = (0.85, 0.93, 1.00)
    emission_str: float = 12.0
    radius:       float = 0.020
    metallic:     float = 0.9
    roughness:    float = 0.1
    particle_type_int: int = 8

    def update(self, dt: float):
        self.x    += self.vx * dt; self.y += self.vy * dt
        self.age_s += dt
        # Stars fade as they age (simplification)
        self.emission_str = max(2.0, 12.0 - self.age_s * 0.5)


class StructureEpoch:
    """Structure formation: dark matter halos, gas clouds, first stars."""

    JEANS_MASS_SOLAR = 1e6   # minimum mass for gravitational collapse

    def __init__(self, clock: CosmicClock, n_halos: int = 40):
        self.clock = clock
        self.clock.set_epoch(7)

        self.halos:   list[DarkMatterHalo] = []
        self.clouds:  list[GasCloud]       = []
        self.stars:   list[Star]           = []
        self.forbidden_events: list[dict]  = []
        self.interaction_log = []
        self.selection: list = []
        self._next_id = 0

        self._init_halos(n_halos)
        self._init_clouds(n_halos // 2)

        self.n_mergers   = 0
        self.n_stars     = 0
        self.n_galaxies  = 0
        self.complete    = False
        self._t = 0.0

    def _spawn_halo(self, x=None, y=None, mass=None) -> DarkMatterHalo:
        if x is None: x = random.uniform(-8.5,8.5)
        if y is None: y = random.uniform(-4.8,4.8)
        if mass is None: mass = 10 ** random.uniform(8, 12)
        spd = random.uniform(0.01, 0.06)
        ang = random.uniform(0, 2*math.pi)
        h = DarkMatterHalo(
            x=x, y=y, mass_solar=mass,
            vx=spd*math.cos(ang), vy=spd*math.sin(ang),
            id=self._next_id,
            radius=0.04+0.015*math.log10(max(1,mass/1e8)),
        )
        self._next_id += 1
        return h

    def _init_halos(self, n: int):
        for _ in range(n):
            self.halos.append(self._spawn_halo())

    def _init_clouds(self, n: int):
        for _ in range(n):
            spd = random.uniform(0.02, 0.08)
            ang = random.uniform(0, 2*math.pi)
            c = GasCloud(
                x=random.uniform(-8,8), y=random.uniform(-4.5,4.5),
                vx=spd*math.cos(ang), vy=spd*math.sin(ang),
                mass_solar=10**random.uniform(4, 7),
                has_halo=False, id=self._next_id,
            )
            self._next_id += 1
            self.clouds.append(c)

    def update(self, dt: float):
        if self.complete: return
        self._t += dt
        self.clock.step(dt)

        for h in self.halos:  h.update(dt)
        for c in self.clouds: c.update(dt)
        for s in self.stars:  s.update(dt)

        # Gravitational attraction between halos
        self._gravity(dt)

        # Gas falls into halos
        self._gas_infall(dt)

        if len(self.stars) > 8:
            self.complete = True

    def _gravity(self, dt: float):
        """N-body gravity between halos (simplified O(N²))."""
        G_eff = 0.004
        for i, a in enumerate(self.halos):
            for b in self.halos[i+1:]:
                dx = b.x - a.x; dy = b.y - a.y
                r2 = dx*dx + dy*dy + 0.5
                r  = math.sqrt(r2)
                F  = G_eff * math.log10(a.mass_solar * b.mass_solar) / r2
                ax_a =  F * dx/r / math.log10(max(10, a.mass_solar))
                ay_a =  F * dy/r / math.log10(max(10, a.mass_solar))
                ax_b = -F * dx/r / math.log10(max(10, b.mass_solar))
                ay_b = -F * dy/r / math.log10(max(10, b.mass_solar))
                a.vx += ax_a*dt; a.vy += ay_a*dt
                b.vx += ax_b*dt; b.vy += ay_b*dt

    def _gas_infall(self, dt: float):
        """Gas clouds fall into nearby dark matter halos."""
        for c in self.clouds:
            if c.has_halo: continue
            for h in self.halos:
                d = math.hypot(c.x - h.x, c.y - h.y)
                if d < h.radius_render * 2.5:
                    c.has_halo = True
                    # If gas mass > Jeans mass and in halo → form star
                    if c.mass_solar > self.JEANS_MASS_SOLAR:
                        self._ignite_star(c, h)
                    break
            if not c.has_halo:
                # Drift toward nearest halo
                nearest = min(self.halos, key=lambda h: math.hypot(c.x-h.x,c.y-h.y), default=None)
                if nearest:
                    dx = nearest.x-c.x; dy = nearest.y-c.y
                    r  = math.hypot(dx,dy)+0.1
                    c.vx += 0.002*dx/r*dt*math.log10(max(10,nearest.mass_solar))
                    c.vy += 0.002*dy/r*dt*math.log10(max(10,nearest.mass_solar))

    def _ignite_star(self, cloud: GasCloud, halo: DarkMatterHalo):
        s = Star(
            x=cloud.x + random.gauss(0,0.2),
            y=cloud.y + random.gauss(0,0.2),
            vx=cloud.vx, vy=cloud.vy,
            mass_solar=min(1000, cloud.mass_solar * 0.01),
            id=self._next_id,
        )
        self._next_id += 1
        self.stars.append(s)
        self.n_stars += 1

    def attempt_merge(self, a, b) -> dict:
        """Player merges two halos."""
        if not isinstance(a, DarkMatterHalo) or not isinstance(b, DarkMatterHalo):
            return {"allowed": False, "reason": "Can only merge dark matter halos"}
        mx = (a.x + b.x)/2; my = (a.y + b.y)/2
        merged = self._spawn_halo(mx, my, a.mass_solar + b.mass_solar)
        merged.vx = (a.vx + b.vx)/2; merged.vy = (a.vy + b.vy)/2
        if a in self.halos: self.halos.remove(a)
        if b in self.halos: self.halos.remove(b)
        self.halos.append(merged)
        self.n_mergers += 1
        return {
            "allowed": True,
            "equation": "t_merge ~ (M/Ṁ)⁻¹, NFW profile, Ṁ/M ∝ (1+z)^{5/2}",
            "description": f"Halo merger. New mass: {merged.mass_solar:.2e} M☉. Gas infall triggered. Expect starburst."
        }

    def attempt_gas_collapse(self, cloud: GasCloud, no_halo: bool = False) -> dict:
        """Gas cloud collapse — forbidden without dark matter halo."""
        if no_halo or not cloud.has_halo:
            return {
                "allowed": False,
                "reason": "Jeans criterion not met without dark matter halo",
                "equation": "M_J = (5k_BT/Gm)^{3/2}(3/4πρ)^{1/2} — needs DM for ρ",
                "why": (
                    "Without a dark matter halo, the baryonic gas density ρ is too low "
                    "to exceed the Jeans mass M_J. Thermal pressure exceeds gravity — "
                    "the cloud disperses. Dark matter halos collapse first (decoupled "
                    "from radiation pressure at decoupling) and provide the gravitational "
                    "well that baryons fall into. Without dark matter, no galaxies form."
                ),
                "reason_type": 2,
            }
        self._ignite_star(cloud, self.halos[0] if self.halos else None)
        return {
            "allowed": True,
            "equation": "M > M_J → gravitational collapse → star formation",
            "description": "Gas collapses in the dark matter potential well. A Population III star ignites.",
        }

    def get_render_particles(self) -> list:
        return self.halos + [c for c in self.clouds if not c.has_halo] + self.stars

    def narrator_text(self) -> str:
        if self.n_stars > 0:
            return f"The first stars burn. {self.n_stars} Pop III stars. UV light reionises the universe."
        if self.n_mergers > 0:
            return f"{self.n_mergers} halo mergers. Galaxies assembling. Gas cooling."
        return "Dark matter halos collapse under gravity. Gas will follow. Stars will follow."
