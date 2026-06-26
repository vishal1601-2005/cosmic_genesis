# Cosmic Genesis — Interactive Universe Simulator

A cinematic, physically-accurate simulation of the entire history of the universe.
From vibrating strings at the Planck epoch to galaxy formation — every interaction
governed by real equations. You are the observer who can see through every scale.

---

## Quick Start

```bash
# 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate cosmic_genesis

# 2. Run the game (starts at the Planck epoch by default)
python main.py

# 3. Jump straight to BBN (the 3-minute nucleosynthesis window)
python main.py --epoch 5

# 4. Start with bosonic string theory (26D, tachyon vacuum)
python main.py --epoch 0 --theory bosonic

# 5. No GPU / CPU only
python main.py --no-gpu

# 6. Silent mode
python main.py --no-sound
```

---

## Controls

| Key | Action |
|-----|--------|
| `W A S D` / Arrow keys | Move camera |
| `Mouse drag (right)` | Look around |
| `Scroll wheel` | Zoom — cosmic scale → Planck scale |
| `Click particle` | Identify / first selection for interaction |
| `Click 2nd particle` | Attempt interaction (see result + physics explanation) |
| `Double-click (epoch 0)` | Split a string |
| `T` | T-duality: R → α′/R (swap winding ↔ momentum modes) |
| `S` | S-duality: gₛ → 1/gₛ (strong ↔ weak coupling) |
| `M` | Open Calabi-Yau moduli space map |
| `C` | Shift CY moduli randomly (new particle spectrum) |
| `+` / `-` | Speed up / slow down cosmic time |
| `SPACE` | Pause / resume |
| `0`–`7` | Jump directly to epoch 0–7 (debug) |
| `F1` | Help overlay |
| `ESC` | Quit |

---

## The Eight Epochs

### Epoch 0 — String Landscape (t = 0 → 10⁻⁴³ s)
Spacetime has not condensed. You observe raw vibrating strings in 10 dimensions.
The Calabi-Yau manifold is choosing its vacuum. Click two open strings to join them.
The vibrational mode that fires determines what particle will exist when geometry forms.

**Sound**: Harmonic stack of 10 slightly detuned oscillators — one per spacetime dimension.
Each mode number maps to a pitch: graviton (n=1) = tonic, first massive state (n=2) = major second.
Tachyon (n=0) = detuned, sliding downward — the unstable vacuum.

### Epoch 1 — Inflation (t = 10⁻³⁶ → 10⁻³² s)
The inflaton field φ rolls down its potential V(φ). Space expands by e⁶⁰.
Every quantum fluctuation you see will one day be a galaxy.

**Sound**: Exponential swell of noise — silent at the start, crescendoing as expansion accelerates.

### Epoch 2 — Baryogenesis (t = 10⁻¹² → 10⁻⁶ s, T ~ 100 GeV)
Quarks and antiquarks rain down. CP violation + sphaleron processes
tilt the balance: 1 extra quark per billion pairs. That 1 is all matter in the universe.

**Try**: Trigger a sphaleron event (click the glowing electroweak instanton).
**Forbidden**: Pair two quarks of the same colour → ⊗ "3⊗3 = 6⊕3̄ — sextet is repulsive"

**Sound**: Rapid stochastic clicks (pair creation rate ~10¹² Hz scaled to audio).

### Epoch 3 — QCD Confinement (t = 10⁻⁶ → 10⁻⁴ s, T ~ 150 MeV)
Temperature drops below Λ_QCD. Free quarks can no longer exist.
Drag two quarks apart — watch the colour flux tube stretch and snap.

**Try**: Combine u + d + u → proton. Combine u + d̄ → pion.
**Forbidden**: Isolate a single quark → ⊗ "V(r) = κr — string breaks, new pair appears"

**Sound**: Deep bass resonance + occasional flux-tube snap when strings break.

### Epoch 4 — Axion Condensation (t = 10⁻⁴ → 1 s)
The Peccei-Quinn symmetry U(1)_PQ breaks at f_a ≈ 10¹² GeV.
The axion field misaligns by angle θ_i and oscillates — this is dark matter.

**Sound**: Pure sine tone at f = BASE_FREQ × φ (golden ratio). Nearly inaudible but present.

### Epoch 5 — Big Bang Nucleosynthesis (t = 1 → 200 s)  ⏱ 3-MINUTE GAME
The most gameplay-intensive epoch. You have 3 real minutes — mirroring the actual
~3 cosmic minutes of BBN — to fuse as much helium as possible.

**Target**: Y_p (He-4 mass fraction) ≈ 0.245.
**Watch**: The deuterium bottleneck — D photodissociates above T = 70 keV.
Watch the neutron decay timer — free neutrons decay in τ = 879.6 s.

**Try**: p + n → D + γ (Q = 2.22 MeV). D + D → ⁴He.
**Forbidden**: p + p → ²He (no bound diproton) → ⊗ "¹S₀ nuclear force insufficient"
**Forbidden**: n + n → ²n → ⊗ "Dineutron unbound — scattering length a_nn = −18.9 fm < 0"

**Sound**: Crackling plasma at T=1 MeV → quieting to soft nuclear pops at T=0.07 MeV.
Each fusion: a thump proportional to Q-value. He-4 gets a special harmonic bloom.
Forbidden: each law has its own sonic signature.

### Epoch 6 — Recombination (t = 380,000 yr, T ~ 3000 K)
Electrons combine with protons → neutral hydrogen.
The universe becomes transparent. The CMB is born.

**The key moment**: Watch (and hear) the universe go quiet as the plasma clears.

**Sound**: Busy plasma hiss → sudden silence → pure CMB ambient tone.
Recombination sound: a chime at f ∝ log₂(13.6 eV / 1 eV) × BASE_FREQ.

### Epoch 7 — Structure Formation (t = 100 Myr → 1 Gyr)
Dark matter halos collapse under gravity. Gas cools in. First stars ignite.
Click two halos to merge them — triggers starburst and AGN.

**Forbidden**: Gas cloud without dark matter → ⊗ "Jeans criterion not met — disperses"

**Sound**: Sub-bass gravitational collapse rumble. Stellar ignition events as harmonic flashes.

---

## The Forbidden Interaction System

Every prohibited interaction shows:
1. **What law is violated** (charge conservation, colour confinement, energy threshold, etc.)
2. **The equation that explains it** (e.g. V(r) = κr for confinement)
3. **A physics explanation** (in plain language)
4. **A distinct visual + sound** per conservation law

| Violation | Visual | Sound |
|-----------|--------|-------|
| Charge conservation | Gold shockwave + X mark | Electric crackle |
| Colour confinement | Deep red radial burst | Bass rumble → abrupt silence |
| Energy threshold | Cold blue fade | Rising tone that dies mid-note |
| Epoch mismatch | Purple reversed echo | Time-reversed audio |
| No bound state | Steel grey thud | Heavy thud + distortion |
| GSO parity | Pink cancellation | Two tones cancelling |

---

## Sound Design — Physically Motivated

Every sound is synthesised from the physics:

- **String mode n** → pitch: `f = BASE_FREQ × 2^((n-1)/12)` (musical scale from mass formula)
- **Graviton** (n=1, m=0) = tonic note (A1 = 55 Hz × octave shift)
- **Tachyon** (n=0, m²<0) = detuned sliding tone (unstable vacuum)
- **Fusion thump** frequency: `f = 40 + 8 × Q_MeV` Hz (D: 58 Hz, He-4: 267 Hz)
- **Forbidden charge**: electric crackle at ~3 kHz resonance
- **Forbidden confinement**: bass rumble → silence (flux tube forms → cuts)
- **Axion ambient**: `f = 55 × φ` Hz where φ = golden ratio ≈ 1.618
- **CMB ambient**: pure tone + thermal noise hiss

---

## File Structure

```
cosmic_genesis/
├── main.py               # Game loop, input, rendering, HUD
├── config.py             # Physics constants, particle registry, interaction rules
├── environment.yml       # Conda environment
│
├── physics/
│   ├── worldsheet.py     # JAX worldsheet PDE solver (CUDA)
│   ├── calabi_yau.py     # Neural CY metric (PyTorch)
│   ├── interactions.py   # Forbidden interaction engine
│   └── cosmology.py      # Friedmann equations, cosmic clock
│
├── epochs/
│   └── ep5_bbn.py        # Big Bang Nucleosynthesis gameplay
│
├── audio/
│   └── sound_engine.py   # Procedural audio synthesis
│
└── shaders/
    ├── particle.vert/frag  # GPU particle rendering
    └── forbidden.frag      # Forbidden interaction visual
```

---

## Requirements

- **GPU**: NVIDIA (CUDA 12.x) strongly recommended — JAX worldsheet solver
- **RAM**: 8 GB minimum, 16 GB recommended
- **Python**: 3.11 (via conda)
- **OS**: Linux or Windows (macOS: no CUDA, will use CPU fallback)

---

## Physics References

- Polchinski, *String Theory* Vol. 1 & 2 — worldsheet action, mode expansion
- Weinberg, *Cosmology* — Friedmann equations, BBN, baryogenesis
- Kolb & Turner, *The Early Universe* — thermal history, axion cosmology
- Green, Schwarz & Witten, *Superstring Theory* — GSO projection, CY compactification
- Peacock, *Cosmological Physics* — structure formation, Jeans criterion
