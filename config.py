"""
config.py — Cosmic Genesis: all physics constants, particle definitions,
and the complete interaction rule table (allowed + forbidden + why).

Every forbidden interaction carries:
  - reason: human-readable physics explanation
  - equation: the relevant formula that explains the prohibition
  - visual: what effect to show in the game

Units: natural units ħ = c = k_B = 1 unless stated.
Energy in GeV, length in GeV⁻¹, time in GeV⁻¹.
Conversion: 1 GeV⁻¹ ≈ 6.58 × 10⁻²⁵ s ≈ 1.97 × 10⁻¹⁶ m
"""

import math

# ══════════════════════════════════════════════════════════════
#  FUNDAMENTAL CONSTANTS (SI for display; natural units for physics)
# ══════════════════════════════════════════════════════════════
HBAR_SI        = 1.0546e-34     # J·s
C_SI           = 2.9979e8       # m/s
K_B_SI         = 1.3806e-23     # J/K
G_N_SI         = 6.674e-11      # m³/(kg·s²)
M_PLANCK_GEV   = 1.2209e19      # GeV  (reduced: M_P = 2.435×10¹⁸ GeV)
M_PLANCK_RED   = 2.435e18       # GeV  (Mₚ = √(ħc/8πG))
T_PLANCK_K     = 1.416e32       # K
T_PLANCK_S     = 5.391e-44      # s
ALPHA_EM       = 1.0 / 137.036  # fine structure constant
ALPHA_S_MZ     = 0.1181         # strong coupling at Z mass
LAMBDA_QCD_GEV = 0.217          # QCD confinement scale (GeV)
V_EW_GEV       = 246.0          # electroweak VEV (GeV)
ALPHA_PRIME    = 1.0            # string slope α′ (ℓₛ = 1 in natural units)

# ══════════════════════════════════════════════════════════════
#  COSMOLOGICAL PARAMETERS  (Planck 2018)
# ══════════════════════════════════════════════════════════════
H0_KM_S_MPC    = 67.4           # Hubble constant (km/s/Mpc)
OMEGA_M        = 0.315          # matter density fraction
OMEGA_LAMBDA   = 0.685          # dark energy fraction
OMEGA_B        = 0.049          # baryon fraction
OMEGA_DM       = 0.266          # dark matter fraction
N_EFF          = 2.99           # effective neutrino species
T_CMB_K        = 2.7255         # CMB temperature today (K)
Z_RECOMB       = 1100           # redshift of recombination
ETA_BARYON     = 6.1e-10        # baryon-to-photon ratio η
Y_P_HELIUM     = 0.245          # primordial helium mass fraction
SIGMA_8        = 0.811          # matter power spectrum normalisation

# ══════════════════════════════════════════════════════════════
#  EPOCH DEFINITIONS
# ══════════════════════════════════════════════════════════════
EPOCHS = [
    dict(
        id=0, name="String landscape",
        t_start_s=0,       t_end_s=5.4e-44,
        T_start_GeV=1e19,  T_end_GeV=1e16,
        description="Pre-geometric epoch. Spacetime has not yet condensed. "
                    "You observe vibrating strings in 10 dimensions. "
                    "The Calabi-Yau manifold is choosing its vacuum.",
        narrator="Before time, before space — only vibration. "
                 "Ten dimensions tremble at the Planck scale. "
                 "You are the first observer.",
        bg_color=(0.01, 0.01, 0.04),
    ),
    dict(
        id=1, name="Inflation",
        t_start_s=1e-36,   t_end_s=1e-32,
        T_start_GeV=1e16,  T_end_GeV=1e9,
        description="The inflaton field φ rolls down its potential. "
                    "Space expands by e^60 — quantum fluctuations stretch "
                    "to cosmic scales and become the seeds of all structure.",
        narrator="A scalar field begins to roll. Space itself inflates. "
                 "Every quantum ripple you see will one day be a galaxy.",
        bg_color=(0.02, 0.01, 0.03),
    ),
    dict(
        id=2, name="Baryogenesis",
        t_start_s=1e-12,   t_end_s=1e-6,
        T_start_GeV=100.0, T_end_GeV=0.2,
        description="The electroweak phase transition breaks SU(2)_L×U(1)_Y → U(1)_EM. "
                    "CP violation and B+L-violating sphaleron processes "
                    "create 1 extra quark per billion pairs.",
        narrator="Quarks and antiquarks annihilate in a fury. "
                 "One in a billion survives. That one is you.",
        bg_color=(0.04, 0.01, 0.01),
    ),
    dict(
        id=3, name="QCD confinement",
        t_start_s=1e-6,    t_end_s=1e-4,
        T_start_GeV=0.2,   T_end_GeV=0.001,
        description="Temperature drops below Λ_QCD ≈ 217 MeV. "
                    "Free quarks can no longer exist — colour flux tubes "
                    "confine them into protons and neutrons.",
        narrator="The strong force tightens its grip. "
                 "Colour can no longer roam free. "
                 "Protons and neutrons crystallise from the quark-gluon plasma.",
        bg_color=(0.03, 0.02, 0.01),
    ),
    dict(
        id=4, name="Axion condensation",
        t_start_s=1e-4,    t_end_s=1.0,
        T_start_GeV=1e-3,  T_end_GeV=1e-4,
        description="The Peccei-Quinn symmetry U(1)_PQ breaks at f_a ≈ 10¹² GeV. "
                    "The axion field misaligns by angle θ_i and begins oscillating. "
                    "These oscillations are cold dark matter.",
        narrator="An invisible field misaligns. No one will ever touch it. "
                 "But it outweighs all the stars. This is dark matter forming.",
        bg_color=(0.01, 0.01, 0.03),
    ),
    dict(
        id=5, name="Big Bang nucleosynthesis",
        t_start_s=1.0,     t_end_s=200.0,
        T_start_GeV=1e-3,  T_end_GeV=1e-5,
        description="In a 3-minute window, protons and neutrons fuse into "
                    "D, He-3, He-4, and Li-7. The primordial abundances "
                    "are set forever: ~75% H, ~25% He-4 by mass.",
        narrator="Three minutes to forge the light elements. "
                 "Miss the window and the universe stays hydrogen forever.",
        bg_color=(0.01, 0.02, 0.04),
        time_limit_s=180.0,  # real-time game clock
    ),
    dict(
        id=6, name="Recombination",
        t_start_s=1.2e13,  t_end_s=1.5e13,
        T_start_GeV=3e-10, T_end_GeV=2e-10,
        T_start_K=4000,    T_end_K=2700,
        description="At T ≈ 3000 K, electrons combine with protons "
                    "to form neutral hydrogen. The universe becomes transparent. "
                    "Photons stream freely — the CMB is born.",
        narrator="For 380,000 years the universe was opaque. "
                 "Then, in a moment, it cleared. "
                 "The light you see right now left then.",
        bg_color=(0.08, 0.04, 0.02),
    ),
    dict(
        id=7, name="Structure formation",
        t_start_s=3e15,    t_end_s=4e17,
        T_start_K=100,     T_end_K=2.7,
        description="Dark matter halos collapse under gravity. "
                    "Gas cools into them. First stars (Pop III, ~100 M☉) ignite "
                    "and reionise the universe. Galaxies merge.",
        narrator="Gravity sculpts the dark. Filaments, voids, halos. "
                 "Inside the halos: gas, stars, light. Inside the light: everything.",
        bg_color=(0.00, 0.00, 0.01),
    ),
]

# ══════════════════════════════════════════════════════════════
#  PARTICLE REGISTRY
# ══════════════════════════════════════════════════════════════
# Each particle has:
#   mass_GeV, charge, spin, colour, baryon_number, lepton_number,
#   isospin, strangeness, available_epochs (list of epoch ids),
#   render_color (RGB float), render_radius, glow_color
#
PARTICLES = {
    # ── String excitations (epoch 0) ──────────────────────────
    "graviton": dict(
        mass_GeV=0, charge=0, spin=2, colour="singlet",
        B=0, L=0, I3=0, S=0,
        epochs=[0,1,2,3,4,5,6,7],
        color=(0.686,0.663,0.925), radius=0.012, glow=(0.5,0.4,0.9),
        eq="m²=0, n=1 closed string, spin-2",
        desc="The massless spin-2 closed string state. Gravity. Always present.",
    ),
    "dilaton": dict(
        mass_GeV=0, charge=0, spin=0, colour="singlet",
        B=0, L=0, I3=0, S=0,
        epochs=[0],
        color=(0.365,0.792,0.647), radius=0.010, glow=(0.2,0.7,0.5),
        eq="m²=0, n=1 closed string, scalar",
        desc="Sets string coupling gₛ = e^⟨φ⟩. Stabilised by flux compactification.",
    ),
    # ── Quarks (epochs 2, 3) ──────────────────────────────────
    "u_quark": dict(
        mass_GeV=0.0022, charge=+2/3, spin=0.5, colour="triplet",
        B=1/3, L=0, I3=+0.5, S=0,
        epochs=[2,3],
        color=(0.937,0.624,0.153), radius=0.006, glow=(0.9,0.5,0.1),
        eq="m_u ≈ 2.2 MeV, Q=+2/3, colour: 3",
        desc="Up quark. Lightest quark. Two up quarks + one down = proton.",
    ),
    "d_quark": dict(
        mass_GeV=0.0047, charge=-1/3, spin=0.5, colour="triplet",
        B=1/3, L=0, I3=-0.5, S=0,
        epochs=[2,3],
        color=(0.522,0.718,0.922), radius=0.006, glow=(0.2,0.5,0.9),
        eq="m_d ≈ 4.7 MeV, Q=−1/3, colour: 3",
        desc="Down quark. One up + two down = neutron.",
    ),
    "s_quark": dict(
        mass_GeV=0.096, charge=-1/3, spin=0.5, colour="triplet",
        B=1/3, L=0, I3=0, S=-1,
        epochs=[2,3],
        color=(0.365,0.792,0.647), radius=0.007, glow=(0.1,0.8,0.5),
        eq="m_s ≈ 96 MeV, Q=−1/3, S=−1",
        desc="Strange quark. Heavier, decays via weak interaction.",
    ),
    "u_antiquark": dict(
        mass_GeV=0.0022, charge=-2/3, spin=0.5, colour="antitriplet",
        B=-1/3, L=0, I3=-0.5, S=0,
        epochs=[2,3],
        color=(0.937,0.400,0.153), radius=0.006, glow=(0.9,0.2,0.1),
        eq="m_ū ≈ 2.2 MeV, Q=−2/3, colour: 3̄",
        desc="Up antiquark. Pair-produced with u quark. Annihilates with u.",
    ),
    "d_antiquark": dict(
        mass_GeV=0.0047, charge=+1/3, spin=0.5, colour="antitriplet",
        B=-1/3, L=0, I3=+0.5, S=0,
        epochs=[2,3],
        color=(0.300,0.450,0.800), radius=0.006, glow=(0.2,0.2,0.8),
        eq="m_d̄ ≈ 4.7 MeV, Q=+1/3, colour: 3̄",
        desc="Down antiquark.",
    ),
    "gluon": dict(
        mass_GeV=0, charge=0, spin=1, colour="octet",
        B=0, L=0, I3=0, S=0,
        epochs=[2,3],
        color=(0.847,0.353,0.188), radius=0.005, glow=(0.9,0.3,0.1),
        eq="m=0, spin=1, colour: 8",
        desc="Gauge boson of SU(3)_colour. Carries colour charge — gluons self-interact.",
    ),
    # ── Hadrons (epoch 3 onward) ──────────────────────────────
    "proton": dict(
        mass_GeV=0.938, charge=+1, spin=0.5, colour="singlet",
        B=1, L=0, I3=+0.5, S=0,
        epochs=[3,4,5,6,7],
        color=(0.937,0.750,0.200), radius=0.014, glow=(0.9,0.7,0.1),
        eq="m_p = 938.3 MeV, uud, |p⟩ = |uud⟩",
        desc="The proton. Two up + one down quark. 99.95% of your body's mass is here.",
    ),
    "neutron": dict(
        mass_GeV=0.940, charge=0, spin=0.5, colour="singlet",
        B=1, L=0, I3=-0.5, S=0,
        epochs=[3,4,5,6,7],
        color=(0.600,0.650,0.700), radius=0.014, glow=(0.4,0.5,0.7),
        eq="m_n = 939.6 MeV, udd, τ_free = 879 s",
        desc="The neutron. One up + two down. Stable inside nuclei; free neutrons decay in ~15 min.",
    ),
    "pion_plus": dict(
        mass_GeV=0.140, charge=+1, spin=0, colour="singlet",
        B=0, L=0, I3=+1, S=0,
        epochs=[3],
        color=(0.686,0.663,0.925), radius=0.010, glow=(0.5,0.4,0.9),
        eq="m_π⁺ = 139.6 MeV, ud̄",
        desc="Charged pion. Lightest meson. Mediates nuclear force at long range.",
    ),
    # ── Leptons ───────────────────────────────────────────────
    "electron": dict(
        mass_GeV=5.11e-4, charge=-1, spin=0.5, colour="singlet",
        B=0, L=1, I3=-0.5, S=0,
        epochs=[2,3,4,5,6,7],
        color=(0.365,0.792,0.647), radius=0.005, glow=(0.1,0.9,0.5),
        eq="m_e = 0.511 MeV, Q=−1, Le=1",
        desc="The electron. Stable. Bound to protons by electromagnetism to make atoms.",
    ),
    "positron": dict(
        mass_GeV=5.11e-4, charge=+1, spin=0.5, colour="singlet",
        B=0, L=-1, I3=+0.5, S=0,
        epochs=[2,3,4,5,6,7],
        color=(0.937,0.400,0.200), radius=0.005, glow=(0.9,0.3,0.1),
        eq="m_e⁺ = 0.511 MeV, Q=+1, Le=−1",
        desc="The positron. Antiparticle of electron. Annihilates with electrons → 2γ.",
    ),
    "neutrino_e": dict(
        mass_GeV=1e-11, charge=0, spin=0.5, colour="singlet",
        B=0, L=1, I3=0, S=0,
        epochs=[2,3,4,5,6,7],
        color=(0.686,0.900,0.800), radius=0.003, glow=(0.3,0.9,0.7),
        eq="m_νe < 0.8 eV, Q=0, Le=1",
        desc="Electron neutrino. Barely interacts. Produced in nuclear reactions.",
    ),
    # ── Bosons ────────────────────────────────────────────────
    "photon": dict(
        mass_GeV=0, charge=0, spin=1, colour="singlet",
        B=0, L=0, I3=0, S=0,
        epochs=[2,3,4,5,6,7],
        color=(1.0,0.97,0.85), radius=0.003, glow=(1.0,0.98,0.9),
        eq="m=0, spin=1, EM gauge boson",
        desc="The photon. Massless, travels at c. The CMB is a bath of ~400 photons/cm³.",
    ),
    "higgs": dict(
        mass_GeV=125.1, charge=0, spin=0, colour="singlet",
        B=0, L=0, I3=0, S=0,
        epochs=[2],
        color=(0.937,0.900,0.200), radius=0.016, glow=(0.9,0.85,0.1),
        eq="m_H = 125.1 GeV, V(H) = μ²|H|² + λ|H|⁴",
        desc="The Higgs boson. Its VEV v=246 GeV gives mass to W, Z, quarks, leptons.",
    ),
    # ── Dark sector ───────────────────────────────────────────
    "axion": dict(
        mass_GeV=6e-12, charge=0, spin=0, colour="singlet",
        B=0, L=0, I3=0, S=0,
        epochs=[4,5,6,7],
        color=(0.522,0.400,0.800), radius=0.004, glow=(0.3,0.1,0.8),
        eq="m_a ≈ 6 μeV, f_a ≈ 10¹² GeV, L = (∂a)²/2 − m_a²f_a²(1−cos(a/f_a))",
        desc="The axion. Solves the strong CP problem. Cold dark matter if f_a ~ 10¹² GeV.",
    ),
    # ── Nuclear (epoch 5) ─────────────────────────────────────
    "deuterium": dict(
        mass_GeV=1.876, charge=+1, spin=1, colour="singlet",
        B=2, L=0, I3=0, S=0,
        epochs=[5],
        color=(0.400,0.800,0.937), radius=0.018, glow=(0.2,0.6,0.9),
        eq="m_D = 1875.6 MeV, B_D = 2.22 MeV, p+n→D+γ",
        desc="Deuterium nucleus. Binding energy only 2.22 MeV — photodissociates above T~70 keV.",
    ),
    "helium4": dict(
        mass_GeV=3.727, charge=+2, spin=0, colour="singlet",
        B=4, L=0, I3=0, S=0,
        epochs=[5,6,7],
        color=(0.590,0.769,0.349), radius=0.022, glow=(0.4,0.8,0.2),
        eq="m_⁴He = 3727.4 MeV, B=28.3 MeV, very stable",
        desc="Helium-4 nucleus (α particle). Mass fraction Y_p ≈ 0.245 set at BBN.",
    ),
    "hydrogen_atom": dict(
        mass_GeV=9.38e-1, charge=0, spin=0.5, colour="singlet",
        B=1, L=1, I3=0, S=0,
        epochs=[6,7],
        color=(0.686,0.800,0.925), radius=0.016, glow=(0.4,0.6,0.9),
        eq="E_n = −13.6 eV/n², e⁻+p→H+γ (E_ion=13.6 eV)",
        desc="Neutral hydrogen atom. Forms at T~3000 K. 75% of all baryonic matter.",
    ),
    "dark_matter_halo": dict(
        mass_GeV=None, charge=0, spin=0, colour="singlet",
        B=0, L=0, I3=0, S=0,
        epochs=[7],
        color=(0.300,0.200,0.500), radius=0.080, glow=(0.2,0.1,0.4),
        eq="δ̈ + 2Hδ̇ = 4πGρ_m δ — gravitational collapse",
        desc="Dark matter halo. Invisible but gravitationally dominant. Galaxy nursery.",
    ),
}

# ══════════════════════════════════════════════════════════════
#  INTERACTION RULES
# ══════════════════════════════════════════════════════════════
# Each rule: (particle_a, particle_b) → result
# allowed=True:  what happens, product, equation, visual FX
# allowed=False: why forbidden, what law is violated, equation, visual FX

INTERACTIONS = {
    # ── String epoch ──────────────────────────────────────────
    ("open_string", "open_string"): dict(
        allowed=True,
        product="closed_string",
        equation="V = g_s · :e^{ik·X}: (vertex operator)",
        desc="Two open strings join endpoints → closed string. "
             "Probability ∝ g_s. This is how gravity arises — "
             "the closed string ground state is the graviton.",
        fx="vertex_flash",
        epoch_required=[0],
    ),
    ("open_string_GSO_even", "open_string_GSO_odd"): dict(
        allowed=False,
        reason="GSO chirality mismatch",
        equation="GSO: (−1)^F |phys⟩ = +|phys⟩ for both strings",
        why="The GSO (Gliozzi-Scherk-Olive) projection removes the tachyon "
            "and enforces spacetime supersymmetry. States of opposite GSO parity "
            "cannot couple — their vertex operator insertion gives zero overlap "
            "in the worldsheet CFT.",
        fx="forbidden_gso",
        epoch_required=[0],
    ),
    ("closed_string_n1", "spacetime_point"): dict(
        allowed=True,
        product="graviton",
        equation="m² = 4(n−1)/α′ → m=0 at n=1",
        desc="Closed string at first excitation level n=1 is massless and spin-2. "
             "This is the graviton. It couples to the stress-energy tensor of everything.",
        fx="particle_materialise",
        epoch_required=[0],
    ),
    # ── Inflation ─────────────────────────────────────────────
    ("inflaton_high_phi", "potential_roll"): dict(
        allowed=True,
        product="inflation + density_perturbations",
        equation="H² = V(φ)/3Mₚ², δρ/ρ = (H/2πφ̇)",
        desc="Inflaton rolls slowly → exponential expansion. "
             "Quantum fluctuations δφ ~ H/2π freeze at Hubble crossing "
             "and imprint as δT/T ~ 10⁻⁵ on the CMB.",
        fx="inflation_expansion",
        epoch_required=[1],
    ),
    ("inflaton", "stop_early"): dict(
        allowed=False,
        reason="Insufficient inflation",
        equation="N_e = ∫H dt ≈ 60 required for flatness + horizon",
        why="Stopping inflation before N_e ≈ 60 e-folds means the observable "
            "universe was not in causal contact before inflation — the horizon "
            "problem returns. Also: the universe's spatial curvature Ω_k "
            "is not driven to zero, contradicting CMB observations.",
        fx="forbidden_efolds",
        epoch_required=[1],
    ),
    # ── Baryogenesis ──────────────────────────────────────────
    ("u_quark", "u_antiquark"): dict(
        allowed=True,
        product=["photon", "photon"],
        equation="uū → 2γ, σ = πα_s²/s (QCD annihilation)",
        desc="Quark-antiquark annihilation into two photons via QCD. "
             "At T~100 GeV this is in equilibrium — balanced until "
             "CP violation and sphaleron processes tilt the balance.",
        fx="annihilation_flash",
        epoch_required=[2,3],
    ),
    ("u_quark", "u_quark"): dict(
        allowed=False,
        reason="Colour antisymmetry: qq → 6 (repulsive)",
        equation="3 ⊗ 3 = 6 ⊕ 3̄: sextet is colour-repulsive",
        why="Two quarks in colour triplet 3 combine as 3⊗3 = 6⊕3̄. "
            "The symmetric sextet 6 is colour-repulsive and cannot form a "
            "bound state. Only the antisymmetric antitriplet 3̄ can bind, "
            "meaning two quarks need a third quark (in 3̄) to form a colour singlet — "
            "that's the baryon. No dibaryon made of only two quarks exists.",
        fx="forbidden_colour",
        epoch_required=[2,3],
    ),
    ("sphaleron", "baryon_number"): dict(
        allowed=True,
        product="baryon_asymmetry",
        equation="ΔB = ΔL = ±3 per sphaleron event, rate Γ ∝ α_W⁴ T⁴ e^{-E_sph/T}",
        desc="Sphaleron: an electroweak instanton that changes baryon+lepton number "
             "by ±3. At T~100 GeV, rate ≈ Hubble rate → out-of-equilibrium "
             "B-violation with CP violation → baryon asymmetry η = (n_B−n_B̄)/n_γ ≈ 6×10⁻¹⁰.",
        fx="sphaleron_burst",
        epoch_required=[2],
    ),
    ("baryon", "antibaryon"): dict(
        allowed=True,
        product=["photon", "photon", "photon"],  # 3γ from p+p̄
        equation="pp̄ → 3π⁰ → 6γ, σ ≈ πr²_p ≈ 40 mb",
        desc="Proton-antiproton annihilation. At T < Λ_QCD, "
             "this is the dominant process removing antibaryons. "
             "The 1 extra baryon per billion survives because it has no antibaryon partner.",
        fx="annihilation_fireball",
        epoch_required=[3],
    ),
    # ── QCD ───────────────────────────────────────────────────
    ("u_quark", "d_quark"): dict(
        allowed=False,
        reason="Cannot form colour singlet with only 2 quarks",
        equation="3 ⊗ 3 = 6 ⊕ 3̄ — need 3 quarks for singlet: 3⊗3⊗3 ∋ 1",
        why="Two quarks cannot form a colour-neutral (singlet) bound state. "
            "The colour decomposition 3⊗3 = 6⊕3̄ gives only a sextet (repulsive) "
            "and an antitriplet (which can bind to a third quark). "
            "A baryon requires 3 quarks: 3⊗3⊗3 = 10⊕8⊕8⊕1, "
            "and only the singlet 1 is colour-neutral and stable.",
        fx="forbidden_colour",
        epoch_required=[3],
    ),
    ("u_quark", "d_antiquark"): dict(
        allowed=True,
        product="pion_plus",
        equation="ud̄ → π⁺, m_π = 140 MeV, 3⊗3̄ ∋ 1 (colour singlet)",
        desc="A quark and antiquark can form a meson — colour 3⊗3̄ = 8⊕1 "
             "contains a singlet. The π⁺ is the lightest such state. "
             "This is how the pion gas forms just below Λ_QCD.",
        fx="hadron_form",
        epoch_required=[3],
    ),
    ("u_quark", "d_quark", "u_quark"): dict(  # 3-body
        allowed=True,
        product="proton",
        equation="uud → p⁺, colour: εᵢⱼₖ 3ⁱ⊗3ʲ⊗3ᵏ ∋ 1",
        desc="Three quarks (uud) with colours RGB form the colour singlet "
             "εᵢⱼₖ qⁱqʲqᵏ — the proton. Binding energy ≈ 938 MeV − (m_u+m_u+m_d) "
             "comes entirely from QCD gluon field energy.",
        fx="proton_form",
        epoch_required=[3],
    ),
    ("quark", "free_travel"): dict(
        allowed=False,
        reason="Colour confinement — isolated colour charge forbidden",
        equation="V(r) = −4α_s/3r + κr, κ ≈ 0.9 GeV/fm (string tension)",
        why="The QCD potential grows linearly with distance: V(r) ~ κr "
            "with string tension κ ≈ 0.9 GeV/fm. Pulling a quark away "
            "costs energy E = κ·r. At r ~ 1 fm this exceeds 2m_q, "
            "so a new quark-antiquark pair pops from the vacuum (string breaking). "
            "You can never isolate a colour charge below the deconfinement temperature T_c ~ 155 MeV.",
        fx="forbidden_confinement",
        epoch_required=[3,4,5,6,7],
    ),
    # ── BBN ───────────────────────────────────────────────────
    ("proton", "neutron"): dict(
        allowed=True,
        product="deuterium",
        equation="p + n → D + γ, Q = 2.22 MeV, σ ~ 0.3 mb at BBN",
        desc="The first step of nucleosynthesis. Binding energy 2.22 MeV "
             "means deuterium photodissociates at T > 70 keV (~8×10⁸ K). "
             "The 'deuterium bottleneck' delays BBN until T drops enough.",
        fx="fusion_glow",
        epoch_required=[5],
    ),
    ("deuterium", "deuterium"): dict(
        allowed=True,
        product="helium4",
        equation="D + D → ⁴He + γ (or D+D → ³He+n), Q = 23.8 MeV",
        desc="Deuterium fusion to helium-4. The chain D+D → ³He+n → ⁴He "
             "runs rapidly once deuterium forms. He-4 is doubly magic "
             "(Z=N=2) and very stable.",
        fx="fusion_helium",
        epoch_required=[5],
    ),
    ("proton", "proton"): dict(
        allowed=False,
        reason="No bound diproton state (²He unbound)",
        equation="²He: no bound state — Pauli exclusion + weak nuclear force too weak",
        why="Two protons (identical fermions) must have antisymmetric wavefunction. "
            "In the ¹S₀ channel (spin-singlet), the nuclear force is not quite strong "
            "enough to bind them — the virtual ²He state exists for ~10⁻²³ s "
            "then separates. The pp → D + e⁺ + νe reaction does occur but is "
            "weak-force mediated (very slow). In the BBN epoch this is "
            "negligible compared to p+n→D+γ.",
        fx="forbidden_diproton",
        epoch_required=[5],
    ),
    ("neutron", "neutron"): dict(
        allowed=False,
        reason="No bound dineutron (²n unbound)",
        equation="²n: no bound state — I=1 channel, nuclear force insufficient",
        why="Two neutrons in the only available (isospin I=1) state experience "
            "a nuclear force that is not strong enough to bind them. "
            "The ¹S₀ neutron-neutron scattering length is a_nn = −18.9 fm "
            "(negative = no bound state). The dineutron does not exist.",
        fx="forbidden_dineutron",
        epoch_required=[5],
    ),
    # ── Recombination ─────────────────────────────────────────
    ("electron", "proton"): dict(
        allowed=True,
        product="hydrogen_atom",
        equation="e⁻ + p → H(1s) + γ, E_γ = 13.6 eV (Lyman α)",
        desc="Recombination. At T ~ 3000 K, the photon energy distribution "
             "finally lacks enough photons above 13.6 eV to re-ionise "
             "the newly formed hydrogen. Neutralisation cascades through "
             "the universe in ~100,000 years — the universe becomes transparent.",
        fx="recombination_glow",
        epoch_required=[6],
    ),
    ("electron", "proton_hot"): dict(
        allowed=False,
        reason="Temperature too high — immediate re-ionisation",
        equation="E_ion = 13.6 eV > k_BT: requires T < 3000 K (k_BT < 0.26 eV)",
        why="Above T ~ 3000 K, the mean photon energy k_BT > 0.26 eV. "
            "Because the photon distribution has a Wien tail, there are still "
            "enough photons above 13.6 eV to instantly re-ionise any hydrogen "
            "that forms. Recombination is delayed far below the naive "
            "T = E_ion/k_B = 158,000 K because of this tail.",
        fx="forbidden_hot_recombination",
        epoch_required=[6],
    ),
    # ── Structure formation ───────────────────────────────────
    ("dark_matter_halo", "dark_matter_halo"): dict(
        allowed=True,
        product="merged_halo",
        equation="t_merge ~ (M/Ṁ)^{-1}, Ṁ/M ∝ (1+z)^{5/2} (NFW profile)",
        desc="Halo merger. Triggers gas infall, starburst, AGN activity. "
             "The merger rate peaks at z~2 (cosmic noon) and drives galaxy evolution.",
        fx="halo_merger",
        epoch_required=[7],
    ),
    ("gas_cloud", "gas_cloud_no_halo"): dict(
        allowed=False,
        reason="Jeans instability not met without dark matter",
        equation="M_J = (5k_BT/Gm)^{3/2} · (3/4πρ)^{1/2} — needs DM for ρ",
        why="Without a dark matter halo, the baryonic gas density is too low "
            "to exceed the Jeans mass M_J. The gas pressure gradient exceeds "
            "the gravitational force — the cloud disperses. Dark matter halos "
            "collapse first (decoupled from radiation pressure at decoupling) "
            "and provide the gravitational potential well that baryons fall into.",
        fx="forbidden_jeans",
        epoch_required=[7],
    ),
}

# ══════════════════════════════════════════════════════════════
#  RENDER SETTINGS
# ══════════════════════════════════════════════════════════════
WINDOW_W         = 1440
WINDOW_H         = 900
TARGET_FPS       = 60
MSAA_SAMPLES     = 4
BLOOM_STRENGTH   = 0.4
CHROMATIC_AB     = 0.003   # chromatic aberration amount (cinematic)
VIGNETTE_STRENGTH= 0.35

# Particle system
MAX_PARTICLES    = 200_000   # instanced draw calls
PARTICLE_LOD_NEAR= 0.020     # radius at close zoom
PARTICLE_LOD_FAR = 0.002     # radius at far zoom

# Camera
CAM_SPEED        = 0.03
CAM_ZOOM_SPEED   = 0.1
CAM_SMOOTH       = 0.08   # smoothing factor

# Epoch transition
TRANSITION_DURATION_S = 3.0   # cinematic cross-fade between epochs

# ── Theory selector (used by main.py) ─────────────────────────
THEORIES = {
    "super": {
        "name": "Type IIA Superstring",
        "dims": 10,
        "stringColors": ["#7f77dd", "#378add", "#1d9e75", "#97c459"],
        "tachyon": False,
        "fermions": True,
    },
    "bosonic": {
        "name": "Bosonic String Theory",
        "dims": 26,
        "stringColors": ["#e24b4a", "#afa9ec", "#85b7eb", "#5dcaa5"],
        "tachyon": True,
        "fermions": False,
    },
    "het": {
        "name": "Heterotic E8xE8",
        "dims": 10,
        "stringColors": ["#d85a30", "#afa9ec", "#ba7517", "#97c459"],
        "tachyon": False,
        "fermions": True,
    },
}

# ================================================
# MISSING CONSTANTS (added for compatibility)
# ================================================


N_SIGMA = 5
N_MODES = 128
G_S_DEFAULT = 0.1

# Window / Render settings (override if needed)
WINDOW_W = 1280
WINDOW_H = 720
TARGET_FPS = 60

# Other common missing ones
MAX_PARTICLES = 50000

# String simulation constants
N_SIGMA = 128
N_MODES = 32
DT_WORLDSHEET = 0.005
D_TARGET = 10
D_LARGE = 4
D_COMPACT = 6

CY_PRESETS = {"quintic":dict(h11=1,h21=101,euler=-200,label="Quintic",vacua_exp=274),"bicubic":dict(h11=19,h21=19,euler=0,label="Bicubic",vacua_exp=220),"standard":dict(h11=101,h21=1,euler=200,label="Mirror",vacua_exp=274),"het_3gen":dict(h11=3,h21=243,euler=-480,label="Het3gen",vacua_exp=500),"octic":dict(h11=2,h21=86,euler=-168,label="Octic",vacua_exp=250)}
CURRENT_CY = "quintic"
