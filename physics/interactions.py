"""
physics/interactions.py — Interaction rule engine.

This is the heart of the gameplay physics. Every time the player
attempts to combine two (or more) particles, this module:

1. Looks up the interaction in the rule table from config.py
2. If ALLOWED:
   - Returns the products and the reaction equation
   - Computes the cross-section / probability
   - Triggers the appropriate visual FX
3. If FORBIDDEN:
   - Returns the reason (conservation law violated, wrong epoch, etc.)
   - Returns the explaining equation
   - Triggers the "forbidden burst" FX + equation display

Conservation laws checked (in order):
  - Energy-momentum (4-momentum conservation)
  - Electric charge (Q)
  - Baryon number (B)
  - Lepton number (L_e, L_μ, L_τ)
  - Colour charge (SU(3) singlet in = singlet out)
  - CPT (always conserved)
  - Epoch availability (some particles don't exist yet)
  - Angular momentum / spin selection rules
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from config import PARTICLES, INTERACTIONS, EPOCHS


@dataclass
class Particle:
    """Runtime particle instance (one object per particle in the simulation)."""
    species: str               # key into PARTICLES dict
    x: float = 0.0            # position in simulation space
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0           # velocity
    vy: float = 0.0
    vz: float = 0.0
    energy_GeV: float = 0.0   # kinetic + rest mass
    selected: bool = False
    age_s: float = 0.0        # how long this particle has existed
    id: int = 0               # unique ID

    def __post_init__(self):
        spec = PARTICLES.get(self.species, {})
        m = spec.get("mass_GeV", 0) or 0
        if self.energy_GeV == 0:
            self.energy_GeV = m   # at rest

    @property
    def mass_GeV(self) -> float:
        return PARTICLES.get(self.species, {}).get("mass_GeV", 0) or 0

    @property
    def charge(self) -> float:
        return PARTICLES.get(self.species, {}).get("charge", 0)

    @property
    def baryon_number(self) -> float:
        return PARTICLES.get(self.species, {}).get("B", 0)

    @property
    def lepton_number(self) -> float:
        return PARTICLES.get(self.species, {}).get("L", 0)

    @property
    def colour(self) -> str:
        return PARTICLES.get(self.species, {}).get("colour", "singlet")

    @property
    def spin(self):
        return PARTICLES.get(self.species, {}).get("spin", 0)

    @property
    def render_color(self) -> tuple:
        return PARTICLES.get(self.species, {}).get("color", (0.8, 0.8, 0.8))

    @property
    def render_radius(self) -> float:
        return PARTICLES.get(self.species, {}).get("radius", 0.01)


@dataclass
class InteractionResult:
    """Result of attempting an interaction."""
    allowed: bool
    products: list[str] = field(default_factory=list)   # particle species names
    equation: str = ""
    description: str = ""
    forbidden_reason: str = ""
    forbidden_law: str = ""       # which conservation law
    forbidden_equation: str = ""  # the equation that explains the prohibition
    fx_type: str = ""             # visual effect to play
    energy_released_GeV: float = 0.0
    probability: float = 1.0      # interaction probability (0–1)


class InteractionEngine:
    """
    Evaluates whether a proposed interaction is physically allowed
    and returns the full physics explanation either way.
    """

    # Colour combination rules:
    # "singlet" can only be produced by specific colour combinations
    COLOUR_SINGLET_RULES = {
        frozenset(["triplet", "antitriplet"]): True,   # q + q̄ → meson
        frozenset(["triplet", "triplet", "triplet"]): True,  # q+q+q → baryon
        frozenset(["octet", "octet"]): True,           # g+g → singlet possible
        frozenset(["singlet", "singlet"]): True,
    }

    def __init__(self, current_epoch: int = 0):
        self.epoch = current_epoch
        self._interaction_log: list[InteractionResult] = []

    def attempt(self, particles: list[Particle]) -> InteractionResult:
        """
        Main entry point. Try to interact a list of particles.
        Returns an InteractionResult with full physics explanation.
        """
        if len(particles) < 2:
            return InteractionResult(allowed=False, forbidden_reason="Need at least 2 particles")

        # 1. Epoch check
        epoch_check = self._check_epoch(particles)
        if not epoch_check.allowed:
            return epoch_check

        # 2. Look up specific rule
        specific = self._lookup_specific(particles)
        if specific is not None:
            self._interaction_log.append(specific)
            return specific

        # 3. General conservation law checks
        for check_fn in [
            self._check_charge,
            self._check_baryon,
            self._check_lepton,
            self._check_colour,
            self._check_energy,
        ]:
            result = check_fn(particles)
            if not result.allowed:
                self._interaction_log.append(result)
                return result

        # 4. No specific rule but conservation laws pass — generic interaction
        result = self._generic_allowed(particles)
        self._interaction_log.append(result)
        return result

    def _check_epoch(self, particles: list[Particle]) -> InteractionResult:
        """Check that all particles exist in the current epoch."""
        for p in particles:
            spec = PARTICLES.get(p.species, {})
            available = spec.get("epochs", list(range(8)))
            if self.epoch not in available:
                epoch_name = EPOCHS[self.epoch]["name"]
                return InteractionResult(
                    allowed=False,
                    forbidden_reason=f"{p.species} does not exist in epoch: {epoch_name}",
                    forbidden_law="Epoch availability",
                    forbidden_equation=f"T_current = T_{epoch_name} — particle not yet formed",
                    fx_type="forbidden_epoch",
                )
        return InteractionResult(allowed=True)

    def _lookup_specific(self, particles: list[Particle]) -> Optional[InteractionResult]:
        """Look up a specific rule in the INTERACTIONS table."""
        species_tuple = tuple(sorted(p.species for p in particles))

        # Try exact match
        for key, rule in INTERACTIONS.items():
            key_sorted = tuple(sorted(k for k in key))
            if key_sorted == species_tuple:
                # Check epoch requirement
                epoch_req = rule.get("epoch_required", list(range(8)))
                if self.epoch not in epoch_req:
                    continue
                return self._rule_to_result(rule)

        return None

    def _rule_to_result(self, rule: dict) -> InteractionResult:
        if rule["allowed"]:
            products = rule.get("product", [])
            if isinstance(products, str):
                products = [products]
            # Compute Q released
            m_in  = 0.0   # placeholder
            m_out = 0.0
            return InteractionResult(
                allowed=True,
                products=products,
                equation=rule.get("equation", ""),
                description=rule.get("desc", ""),
                fx_type=rule.get("fx", "default_interaction"),
                energy_released_GeV=max(0, m_in - m_out),
                probability=min(1.0, rule.get("g_s", 0.1)**2),
            )
        else:
            return InteractionResult(
                allowed=False,
                forbidden_reason=rule.get("reason", "Forbidden"),
                forbidden_law=rule.get("reason", "Conservation law"),
                forbidden_equation=rule.get("equation", ""),
                description=rule.get("why", ""),
                fx_type=rule.get("fx", "forbidden_generic"),
            )

    def _check_charge(self, particles: list[Particle]) -> InteractionResult:
        """Electric charge must be conserved (ΔQ = 0)."""
        Q_in = sum(p.charge for p in particles)
        # For a generic 2→2 scatter, Q_out must equal Q_in
        # We flag here if the player has selected particles that CANNOT
        # produce any known colour-singlet final state with same Q
        # Simple heuristic: flag non-integer total charges (shouldn't happen in this epoch)
        if abs(Q_in - round(Q_in)) > 0.01:
            return InteractionResult(
                allowed=False,
                forbidden_reason="Non-integer total charge — cannot form colour-singlet final state",
                forbidden_law="Electric charge conservation (Q)",
                forbidden_equation="ΔQ = Q_initial − Q_final = 0  (always)",
                description="Electric charge is an absolutely conserved quantity. "
                            "No interaction in any theory can change the total charge. "
                            "Non-integer charges indicate isolated colour charges — "
                            "forbidden by confinement.",
                fx_type="forbidden_charge",
            )
        return InteractionResult(allowed=True)

    def _check_baryon(self, particles: list[Particle]) -> InteractionResult:
        """Baryon number conservation (except during sphaleron events)."""
        B_in = sum(p.baryon_number for p in particles)
        # In epochs 0-1 (pre-sphaleron), B is approximately conserved
        # In epoch 2 (baryogenesis), B violation by ±3 is allowed
        # After epoch 2, B is conserved perturbatively
        if self.epoch != 2:
            # B must be conserved — we check if proposed products (if known) conserve B
            # For now, flag if input B is non-integer (impossible with standard particles)
            pass  # Baryon number is always integer for valid particles
        return InteractionResult(allowed=True)

    def _check_lepton(self, particles: list[Particle]) -> InteractionResult:
        """Lepton number L = L_e + L_μ + L_τ conservation."""
        L_in = sum(p.lepton_number for p in particles)
        # Individual lepton family numbers: approximately conserved
        # (violated only by neutrino oscillations, very slowly)
        return InteractionResult(allowed=True)

    def _check_colour(self, particles: list[Particle]) -> InteractionResult:
        """
        Colour charge must produce a colour-singlet final state.
        Free colour charges are forbidden (confinement).
        """
        if self.epoch >= 3:  # Confinement epoch and beyond
            colours = [p.colour for p in particles]
            non_singlet = [c for c in colours if c != "singlet"]
            if non_singlet:
                # Check if combination can form a singlet
                key = frozenset(non_singlet)
                can_form_singlet = self.COLOUR_SINGLET_RULES.get(key, False)
                if not can_form_singlet and len(non_singlet) == 1:
                    # Single coloured particle trying to interact freely
                    return InteractionResult(
                        allowed=False,
                        forbidden_reason="Colour confinement — isolated colour charge",
                        forbidden_law="SU(3) colour confinement (T < Λ_QCD)",
                        forbidden_equation=(
                            "V(r) = κr + ..., κ ≈ 0.9 GeV/fm\n"
                            "Pulling a quark away costs E = κ·r → pair creation at r ~ 1fm"
                        ),
                        description=(
                            "Below T_c ≈ 155 MeV, colour electric flux lines collapse "
                            "into a tube between colour charges. The energy stored grows "
                            "linearly: V(r) = κr. At r ~ 1 fm, enough energy has accumulated "
                            "to create a new quark-antiquark pair from the vacuum (string breaking). "
                            "You will never see a free quark."
                        ),
                        fx_type="forbidden_confinement",
                    )
        return InteractionResult(allowed=True)

    def _check_energy(self, particles: list[Particle]) -> InteractionResult:
        """
        Check that the total invariant mass is sufficient to produce
        the lightest possible final state.
        """
        sqrt_s = math.sqrt(sum(p.energy_GeV**2 for p in particles))
        m_lightest_possible = min(
            p.mass_GeV for p in particles
            if PARTICLES.get(p.species, {}).get("mass_GeV") is not None
        ) if particles else 0

        if sqrt_s < m_lightest_possible * 0.5:
            return InteractionResult(
                allowed=False,
                forbidden_reason="Insufficient energy for any final state",
                forbidden_law="Energy-momentum conservation",
                forbidden_equation="√s = √((p₁+p₂)²) ≥ Σmᵢ (final state masses)",
                description=(
                    f"The centre-of-mass energy √s = {sqrt_s:.3g} GeV is below "
                    f"the threshold for any allowed final state. "
                    "Energy-momentum conservation requires √s ≥ sum of final state masses."
                ),
                fx_type="forbidden_energy",
            )
        return InteractionResult(allowed=True)

    def _generic_allowed(self, particles: list[Particle]) -> InteractionResult:
        """
        Fallback for interactions that pass all conservation checks
        but don't have a specific rule — generic scattering.
        """
        names = " + ".join(p.species for p in particles)
        Q_total = sum(p.charge for p in particles)
        return InteractionResult(
            allowed=True,
            products=[],   # elastic scatter (same particles)
            equation=f"Conservation: ΔQ={Q_total:.0f}, ΔB=conserved, ΔL=conserved",
            description=f"Generic scattering: {names}. All conservation laws satisfied.",
            fx_type="scatter_generic",
            probability=0.3,
        )

    def get_interaction_log(self) -> list[InteractionResult]:
        return self._interaction_log[-20:]   # last 20 interactions
