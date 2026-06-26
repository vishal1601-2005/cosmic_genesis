"""
audio/sound_engine.py — Physically-motivated procedural audio engine.

Every sound is synthesised from first principles using the actual physics:

  STRING VIBRATION
  ─────────────────
  A string of length L under tension T has vibrational frequencies:
      fₙ = n · v / (2L)   where v = √(T/μ) (wave speed)

  In string theory with α′ = 1 and tension T = 1/(2πα′):
      m²(n) = (n-1)/α′   →   m(n) = √(n-1)

  We map the mode number n to an audible frequency logarithmically:
      f_audio(n) = f_base · 2^(n/12)   (musical semitones above base)

  This means the graviton (n=1, m=0) plays the tonic.
  The first massive excitation (n=2) plays a major second above.
  The tachyon (n=0, m²<0) plays below the tonic — detuned, unstable.

  FUSION Q-VALUE
  ───────────────
  The sound amplitude of a fusion event is proportional to Q (MeV):
      amplitude ∝ log(1 + Q_MeV)

  And the frequency of the impact thump:
      f_thump = 40 Hz + 8 · Q_MeV   (capped at 200 Hz)

  FORBIDDEN INTERACTIONS
  ───────────────────────
  Each conservation law has a distinct sonic character:
    - Charge violation: electric crackle (white noise × decaying sine burst)
    - Colour confinement: deep bass rumble + abrupt silence
    - Energy violation: rising tone that dies before completing
    - Epoch mismatch: reversed echo + pitch shift
    - Baryon violation: heavy thud with distortion
    - GSO parity: phase-inverted cancellation (destructive interference sound)

  EPOCH AMBIENCES
  ────────────────
  Each epoch has an ambient soundscape built from oscillators:
    Epoch 0 (strings): harmonic stack at 432 Hz, slowly evolving
    Epoch 1 (inflation): rising white-noise swell, very low rumble
    Epoch 2 (baryogenesis): rapid stochastic clicks (quark-antiquark pairs)
    Epoch 3 (QCD): deep bass, occasional flux-tube snap
    Epoch 4 (axions): pure sine tone at 432·φ Hz (golden ratio harmonic)
    Epoch 5 (BBN): crackling plasma → quiet nuclear pops
    Epoch 6 (recombination): busy hiss → sudden quiet → pure CMB tone
    Epoch 7 (structure): low gravitational bass, occasional stellar ignition

All synthesis uses numpy; output via pygame.mixer.Sound from raw PCM bytes.
"""

from __future__ import annotations
import math
import struct
import random
import numpy as np
import threading
from typing import Optional

try:
    import pygame
    import pygame.mixer as mixer
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# ── Audio configuration ────────────────────────────────────────
SAMPLE_RATE   = 44100    # Hz
BIT_DEPTH     = 16       # bits per sample
N_CHANNELS    = 2        # stereo
MAX_VOLUME    = 0.50     # master volume (0–1)
AMBIENT_VOL   = 0.08     # ambient soundscape volume
FX_VOL        = 0.45     # interaction FX volume

# Frequency mapping
BASE_FREQ     = 55.0     # Hz (A1 — deep bass, fits "cosmic" aesthetic)
STRING_OCTAVE = 4        # which octave string modes live in


def _to_pcm(samples: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] numpy array to 16-bit signed PCM bytes (stereo)."""
    samples = np.clip(samples, -1.0, 1.0)
    # If mono, duplicate to stereo
    if samples.ndim == 1:
        samples = np.stack([samples, samples], axis=-1)
    pcm = (samples * 32767).astype(np.int16)
    return pcm.tobytes()


def _make_sound(samples: np.ndarray) -> Optional["pygame.mixer.Sound"]:
    """Wrap numpy PCM into a pygame Sound object."""
    if not PYGAME_AVAILABLE:
        return None
    pcm = _to_pcm(samples)
    return mixer.Sound(buffer=pcm)


# ══════════════════════════════════════════════════════════════
#  SYNTHESISER PRIMITIVES
# ══════════════════════════════════════════════════════════════

def _t(duration_s: float) -> np.ndarray:
    return np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)

def _sine(freq: float, duration_s: float, amplitude: float = 1.0,
          phase: float = 0.0) -> np.ndarray:
    t = _t(duration_s)
    return amplitude * np.sin(2 * math.pi * freq * t + phase)

def _noise(duration_s: float, amplitude: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return amplitude * (rng.random(int(SAMPLE_RATE * duration_s)) * 2 - 1)

def _envelope(samples: np.ndarray, attack: float, decay: float,
               sustain: float, release: float,
               sustain_level: float = 0.6) -> np.ndarray:
    """ADSR envelope (all times in seconds, as fractions of total length)."""
    n = len(samples)
    env = np.ones(n)
    sr = SAMPLE_RATE
    a = int(attack * sr);   d = int(decay * sr)
    s = int(sustain * sr);  r = int(release * sr)
    idx = 0
    # Attack
    if a > 0 and idx + a <= n:
        env[idx:idx+a] = np.linspace(0, 1, a)
        idx += a
    # Decay
    if d > 0 and idx + d <= n:
        env[idx:idx+d] = np.linspace(1, sustain_level, d)
        idx += d
    # Sustain
    if s > 0 and idx + s <= n:
        env[idx:idx+s] = sustain_level
        idx += s
    # Release
    if r > 0 and idx + r <= n:
        env[idx:idx+r] = np.linspace(sustain_level, 0, r)
        idx += r
    env[idx:] = 0
    return samples * env

def _harmonic_stack(base_freq: float, n_harmonics: int, duration_s: float,
                    decay: float = 0.6) -> np.ndarray:
    """Sum of harmonics with falling amplitude: A_n = decay^n."""
    out = np.zeros(int(SAMPLE_RATE * duration_s))
    for n in range(1, n_harmonics + 1):
        amp = decay ** (n - 1)
        out += amp * _sine(base_freq * n, duration_s, amp)
    return out / (out.max() + 1e-8)


# ══════════════════════════════════════════════════════════════
#  STRING THEORY SOUNDS
# ══════════════════════════════════════════════════════════════

def string_vibration_sound(mode_n: int, is_closed: bool,
                           is_tachyon: bool = False) -> np.ndarray:
    """
    Synthesise the sound of a string vibrating in mode n.

    Frequency mapping:
        f = BASE_FREQ * 2^(octave + n/12)

    Closed strings are brighter (more harmonics) than open strings.
    Tachyons (mode n=0) play below base, detuned and unstable.
    """
    dur = 0.8
    if is_tachyon:
        # Detuned, dissonant — slides downward (unstable vacuum)
        f_start = BASE_FREQ * 1.5
        t = _t(dur)
        slide = np.exp(-t * 2.5)
        f_mod = f_start * slide
        # Frequency-modulated sine (pitch slides down)
        phase_acc = np.cumsum(2 * math.pi * f_mod / SAMPLE_RATE)
        sig = np.sin(phase_acc)
        # Add dissonant overtone
        sig += 0.4 * np.sin(phase_acc * 1.414)   # √2 ratio — irrational, dissonant
        noise_burst = _noise(dur, 0.15, seed=42)
        sig = sig * 0.6 + noise_burst
        return _envelope(sig, 0.01, 0.1, dur-0.15, 0.05, sustain_level=0.7)

    # Normal string: harmonic stack at mode frequency
    octave = STRING_OCTAVE + (1 if is_closed else 0)
    f_base = BASE_FREQ * (2 ** octave) * (2 ** ((mode_n - 1) / 12.0))

    if is_closed:
        # Closed string — richer timbre, more harmonics (graviton/B-field)
        sig = _harmonic_stack(f_base, 8, dur, decay=0.55)
        # Add a slight chorus (two detuned copies)
        sig += 0.3 * _harmonic_stack(f_base * 1.003, 6, dur, decay=0.5)
    else:
        # Open string — thinner, more "plucked" quality
        sig = _harmonic_stack(f_base, 5, dur, decay=0.45)

    # Bell-like envelope: sharp attack, long decay
    return _envelope(sig, 0.005, 0.15, dur * 0.3, dur * 0.5, sustain_level=0.4)


def string_join_sound(g_s: float) -> np.ndarray:
    """
    Two strings joining → vertex operator event.
    g_s controls the 'weight' (louder at strong coupling).
    Sounds like a harmonic click + resonant bloom.
    """
    dur = 0.6
    f = BASE_FREQ * 4
    bloom = _harmonic_stack(f, 6, dur, decay=0.5)
    click = _sine(f * 3, 0.02, 0.8)
    click = np.pad(click, (0, int(SAMPLE_RATE * (dur - 0.02))))
    sig = bloom * 0.7 + click * 0.3
    amp = min(1.0, 0.3 + g_s * 2.0)
    return _envelope(sig * amp, 0.003, 0.08, 0.2, 0.32, sustain_level=0.5)


def string_split_sound(g_s: float) -> np.ndarray:
    """String splitting: sharp snap then two daughter-string tones."""
    dur = 0.7
    snap_dur = 0.05
    # Sharp snap
    snap_n = int(SAMPLE_RATE * snap_dur)
    noise = np.random.randn(snap_n) * np.exp(-np.linspace(0, 8, snap_n))
    snap = np.pad(noise, (0, int(SAMPLE_RATE * (dur - snap_dur))))
    # Two daughter tones (slightly detuned from each other)
    f1 = BASE_FREQ * 4 * 1.05
    f2 = BASE_FREQ * 4 * 0.95
    d1 = _sine(f1, dur, 0.4)
    d2 = _sine(f2, dur, 0.35)
    sig = snap * 0.5 + d1 + d2
    return _envelope(sig, 0.002, 0.06, 0.25, 0.35, sustain_level=0.35)


# ══════════════════════════════════════════════════════════════
#  PARTICLE INTERACTION SOUNDS
# ══════════════════════════════════════════════════════════════

def fusion_sound(Q_MeV: float, product: str = "") -> np.ndarray:
    """
    Nuclear fusion sound.
    Q_MeV → amplitude and bass frequency (more energy = deeper, louder thump).
    """
    dur = 0.9
    f_thump = 40 + min(160, 8 * Q_MeV)   # 40–200 Hz
    amp = min(1.0, 0.2 + 0.08 * math.log(1 + Q_MeV))

    # Deep bass thump
    thump = _sine(f_thump, dur, amp)
    thump2 = _sine(f_thump * 1.5, dur, amp * 0.5)
    thump = thump + thump2

    # High harmonic 'ping' (nuclear binding energy release)
    ping = _sine(BASE_FREQ * 8, 0.15, amp * 0.35)
    ping = np.pad(ping, (int(SAMPLE_RATE * 0.02), int(SAMPLE_RATE * (dur - 0.17))))

    # He-4 gets a special resonant bloom (it's doubly magic)
    if "helium4" in product:
        bloom = _harmonic_stack(BASE_FREQ * 6, 6, dur * 0.8, decay=0.5)
        bloom = np.pad(bloom, (int(SAMPLE_RATE * 0.05), int(SAMPLE_RATE * (dur - 0.05 - dur * 0.8))))
        sig = thump + ping + bloom * amp * 0.4
    else:
        sig = thump + ping

    return _envelope(sig, 0.003, 0.12, 0.2, 0.55, sustain_level=0.4)


def annihilation_sound() -> np.ndarray:
    """Particle-antiparticle annihilation → photons. Sharp burst + dying ring."""
    dur = 0.5
    # Initial burst: broadband click
    burst_n = int(SAMPLE_RATE * 0.015)
    burst = np.random.randn(burst_n)
    burst *= np.linspace(1, 0, burst_n)

    # Ring: pure high tone that decays fast
    ring = _sine(BASE_FREQ * 16, dur, 0.6)
    ring *= np.exp(-np.linspace(0, 12, int(SAMPLE_RATE * dur)))

    burst_full = np.pad(burst, (0, int(SAMPLE_RATE * dur) - burst_n))
    return (burst_full * 0.5 + ring * 0.5) * 0.8


def proton_form_sound() -> np.ndarray:
    """Three quarks congealing into a proton: low thud + rising tone."""
    dur = 0.6
    thud = _sine(55, dur, 0.8)
    thud *= np.exp(-np.linspace(0, 8, int(SAMPLE_RATE * dur)))
    rise_t = _t(dur)
    rise_f = 200 + 400 * rise_t / dur
    rise_phase = np.cumsum(2 * math.pi * rise_f / SAMPLE_RATE)
    rise = np.sin(rise_phase) * 0.4
    rise *= np.linspace(0, 1, int(SAMPLE_RATE * dur)) * np.exp(-np.linspace(0, 4, int(SAMPLE_RATE * dur)))
    return _envelope(thud + rise, 0.002, 0.1, 0.2, 0.25, sustain_level=0.3)


def recombination_sound() -> np.ndarray:
    """
    Electron captures proton → hydrogen atom.
    Soft chime at 13.6 eV → audible frequency mapping:
    f = BASE_FREQ * log2(E_ion / 1eV) ≈ BASE_FREQ * 3.77
    """
    dur = 1.2
    f_lyman = BASE_FREQ * (2 ** 3.77)
    chime = _harmonic_stack(f_lyman, 5, dur, decay=0.5)
    return _envelope(chime * 0.6, 0.01, 0.3, 0.4, 0.4, sustain_level=0.25)


def halo_merge_sound() -> np.ndarray:
    """Two dark matter halos merging. Very low frequency rumble."""
    dur = 2.0
    f = 20 + random.uniform(0, 15)   # sub-bass
    rumble = _sine(f, dur, 0.7)
    rumble += 0.4 * _sine(f * 1.33, dur, 0.5)
    rumble += 0.15 * _noise(dur, 0.2, seed=7)
    return _envelope(rumble, 0.3, 0.4, 0.6, 0.6, sustain_level=0.5)


# ══════════════════════════════════════════════════════════════
#  FORBIDDEN INTERACTION SOUNDS
# ══════════════════════════════════════════════════════════════

def forbidden_charge_sound() -> np.ndarray:
    """
    Charge violation: electric crackle.
    Sounds like a spark discharge — white noise filtered through
    a resonant bandpass at ~3 kHz (electrical resonance).
    """
    dur = 0.4
    # White noise
    noise = _noise(dur, 0.8, seed=1)
    # Envelope: sharp attack, fast decay
    noise = _envelope(noise, 0.002, 0.05, 0.05, 0.25, sustain_level=0.15)
    # Resonant tone at electrical frequency
    buzz = _sine(3000, dur, 0.3)
    buzz *= np.exp(-np.linspace(0, 20, int(SAMPLE_RATE * dur)))
    # Negative feedback 'zap' pattern: three rapid clicks
    for i in range(3):
        offset = int(SAMPLE_RATE * i * 0.06)
        end = min(int(SAMPLE_RATE * dur), offset + int(SAMPLE_RATE * 0.02))
        if end > offset:
            n = end - offset
            noise[offset:end] += np.linspace(0.5, 0, n)
    return (noise + buzz) * 0.7


def forbidden_confinement_sound() -> np.ndarray:
    """
    Colour confinement: deep bass rumble that cuts to silence.
    Represents: flux tube forming → string tension → abrupt confinement.
    """
    dur = 0.8
    # Rising bass
    t = _t(dur)
    f_rise = 30 + 80 * t / dur
    phase = np.cumsum(2 * math.pi * f_rise / SAMPLE_RATE)
    rise = np.sin(phase) * 0.8
    # Apply: rises then hits a wall and stops
    cut = int(SAMPLE_RATE * 0.6)
    envelope = np.ones(int(SAMPLE_RATE * dur))
    envelope[cut:] = np.linspace(1, 0, int(SAMPLE_RATE * dur) - cut) ** 0.3
    rise *= envelope
    # Add string tension 'twang' at cutoff
    twang_n = int(SAMPLE_RATE * 0.1)
    twang = np.sin(np.linspace(0, math.pi * 8, twang_n)) * 0.4
    twang_full = np.zeros(int(SAMPLE_RATE * dur))
    twang_full[cut:cut + twang_n] = twang
    return (rise + twang_full) * 0.85


def forbidden_energy_sound() -> np.ndarray:
    """
    Energy violation: a tone that rises toward completion but dies mid-note.
    'Promises a harmonic it cannot deliver.'
    """
    dur = 0.5
    t = _t(dur)
    # Tone rises toward the 'goal' frequency then cuts
    f = BASE_FREQ * 4 * (1 + t / dur)   # rising glide
    phase = np.cumsum(2 * math.pi * f / SAMPLE_RATE)
    sig = np.sin(phase) * 0.6
    # Cut at 80% of duration — unfinished
    cut = int(SAMPLE_RATE * 0.4)
    env = np.ones(int(SAMPLE_RATE * dur))
    env[cut:cut + 20] = np.linspace(1, 0, 20)
    env[cut + 20:] = 0
    return sig * env


def forbidden_epoch_sound() -> np.ndarray:
    """
    Epoch mismatch: reversed, time-shifted echo.
    'This particle does not yet exist.'
    """
    dur = 0.5
    sig = _harmonic_stack(BASE_FREQ * 3, 4, dur, decay=0.5) * 0.5
    # Reverse the sound (time runs backward)
    sig_rev = sig[::-1]
    # Pitch shift down (past = lower energy)
    t = _t(dur)
    f_down = BASE_FREQ * 2 * np.exp(-t * 2)
    phase = np.cumsum(2 * math.pi * f_down / SAMPLE_RATE)
    ghost = np.sin(phase) * 0.25
    return _envelope(sig_rev + ghost, 0.05, 0.1, 0.15, 0.2, sustain_level=0.2)


def forbidden_gso_sound() -> np.ndarray:
    """
    GSO parity mismatch: two tones in perfect destructive interference
    that cancel exactly — silence appears from two sounds.
    """
    dur = 0.5
    f = BASE_FREQ * 6
    tone1 =  _sine(f, dur, 0.5)
    tone2 = -_sine(f, dur, 0.5)   # phase inverted
    # They don't cancel perfectly — there's residual noise (imperfect cancellation)
    residual = _noise(dur, 0.05, seed=3)
    sig = tone1 + tone2 + residual   # ≈ 0 + residual
    # Brief flash of both tones before cancellation
    flash = int(SAMPLE_RATE * 0.08)
    sig[:flash] = tone1[:flash] * 0.8
    return sig


FORBIDDEN_SOUNDS = {
    "charge":      forbidden_charge_sound,
    "confinement": forbidden_confinement_sound,
    "energy":      forbidden_energy_sound,
    "epoch":       forbidden_epoch_sound,
    "gso":         forbidden_gso_sound,
    "diproton":    forbidden_confinement_sound,   # reuse
    "generic":     forbidden_charge_sound,        # reuse
}


# ══════════════════════════════════════════════════════════════
#  EPOCH AMBIENT SOUNDSCAPES
# ══════════════════════════════════════════════════════════════

def ambient_strings(duration_s: float = 4.0) -> np.ndarray:
    """
    Epoch 0: Harmonic stack in 10 'dimensions' — 10 slightly detuned oscillators
    representing the 10 spacetime dimensions of superstring theory.
    Slowly evolves.
    """
    sig = np.zeros(int(SAMPLE_RATE * duration_s))
    for dim in range(10):
        f = BASE_FREQ * (1 + dim * 0.015)   # slightly detuned per dimension
        phase = dim * math.pi / 5
        tone = _sine(f, duration_s, 0.08, phase)
        sig += tone
    # Add slow modulation (worldsheet time evolving)
    t = _t(duration_s)
    mod = 0.5 + 0.5 * np.sin(2 * math.pi * 0.1 * t)   # 0.1 Hz modulation
    return sig * mod * AMBIENT_VOL


def ambient_inflation(duration_s: float = 4.0) -> np.ndarray:
    """
    Epoch 1: Exponential expansion as a swell of rising noise.
    Starts almost silent, grows continuously.
    """
    t = _t(duration_s)
    swell = np.exp(t / duration_s * 3) - 1   # exponential growth
    swell /= swell.max() + 1e-8
    noise = _noise(duration_s, 1.0, seed=5)
    # Low rumble
    rumble = _sine(25, duration_s, 0.4)
    sig = noise * swell * 0.6 + rumble * swell
    return sig * AMBIENT_VOL


def ambient_baryogenesis(duration_s: float = 4.0) -> np.ndarray:
    """
    Epoch 2: Rapid stochastic clicks — quark-antiquark pair creation/annihilation.
    ~10¹² events per second at T~100 GeV, represented as audio noise.
    """
    n = int(SAMPLE_RATE * duration_s)
    # Hot plasma: pink noise (1/f spectrum)
    white = np.random.randn(n)
    # Approximate pink noise by cascaded filters
    pink = np.zeros(n)
    b0, b1, b2 = 0, 0, 0
    for i in range(n):
        w = white[i]
        b0 = 0.99886 * b0 + w * 0.0555179
        b1 = 0.99332 * b1 + w * 0.0750759
        b2 = 0.96900 * b2 + w * 0.1538520
        pink[i] = b0 + b1 + b2 + w * 0.5362
    pink /= (np.abs(pink).max() + 1e-8)
    # Rapid burst overlay (annihilation flashes)
    burst_rate = 40   # bursts per second
    for i in range(int(duration_s * burst_rate)):
        idx = random.randint(0, n - 100)
        burst_len = 30
        pink[idx:idx + burst_len] += np.linspace(0.3, 0, burst_len)
    return np.clip(pink, -1, 1) * AMBIENT_VOL * 0.8


def ambient_qcd(duration_s: float = 4.0) -> np.ndarray:
    """
    Epoch 3: Deep bass resonance of the quark-gluon plasma cooling.
    Occasional flux-tube snap.
    """
    sig = _sine(40, duration_s, 0.5)
    sig += _sine(60, duration_s, 0.3)
    sig += _noise(duration_s, 0.1, seed=9)
    # Occasional snaps
    n = int(SAMPLE_RATE * duration_s)
    for _ in range(int(duration_s * 1.5)):
        idx = random.randint(0, n - 200)
        snap = np.random.randn(80) * np.exp(-np.linspace(0, 10, 80)) * 0.25
        if idx + 80 <= n:
            sig[idx:idx + 80] += snap
    return np.clip(sig, -1, 1) * AMBIENT_VOL


def ambient_axion(duration_s: float = 4.0) -> np.ndarray:
    """
    Epoch 4: Pure sine tone at f = BASE_FREQ × φ (golden ratio).
    The axion field oscillates at m_a ≈ 6 μeV; we map this to audio.
    Almost inaudible but felt as a presence.
    """
    phi = (1 + math.sqrt(5)) / 2   # golden ratio ≈ 1.618
    f_axion = BASE_FREQ * phi
    sig = _sine(f_axion, duration_s, 0.7)
    # Very slow amplitude modulation: the axion rolls from θ_i
    t = _t(duration_s)
    mod = 0.3 + 0.7 * np.cos(2 * math.pi * 0.2 * t) ** 2
    return sig * mod * AMBIENT_VOL * 0.7


def ambient_bbn(duration_s: float = 4.0, T_MeV: float = 1.0) -> np.ndarray:
    """
    Epoch 5: Transitions from hot plasma crackle (T=1 MeV) to quiet pops (T=0.01 MeV).
    T_MeV controls the character.
    """
    n = int(SAMPLE_RATE * duration_s)
    # Plasma component: fades with T
    plasma_amp = min(1.0, T_MeV * 2)
    plasma = np.random.randn(n) * plasma_amp
    # Nuclear pop component: increases as D forms (T < 0.07 MeV)
    pop_amp = max(0, 1 - T_MeV / 0.07) * 0.4
    pops = np.zeros(n)
    for _ in range(int(duration_s * pop_amp * 5)):
        idx = random.randint(0, n - 300)
        pop = np.sin(np.linspace(0, math.pi * 4, 200)) * 0.3
        if idx + 200 <= n:
            pops[idx:idx + 200] += pop
    # Low hum: temperature bath
    hum = _sine(80 + 60 * T_MeV, duration_s, 0.2)
    sig = plasma * 0.3 + pops + hum
    return np.clip(sig, -1, 1) * AMBIENT_VOL


def ambient_recombination(duration_s: float = 4.0,
                           phase: float = 0.0) -> np.ndarray:
    """
    Epoch 6: phase 0=hot plasma → 1=silence → 2=CMB tone.
    The most dramatic sonic transition in the game.
    """
    n = int(SAMPLE_RATE * duration_s)
    if phase < 0.5:
        # Busy plasma hiss
        sig = np.random.randn(n) * (1 - phase * 1.5)
        sig += _sine(200, duration_s, 0.2 * (1 - phase * 1.5))
    elif phase < 0.7:
        # Sudden silence (recombination: universe goes transparent)
        transition = (phase - 0.5) / 0.2
        sig = np.random.randn(n) * max(0, 0.25 * (1 - transition * 4))
    else:
        # CMB: pure thermal noise at 2.7 K mapped to a faint hiss
        # plus a pure tone representing the acoustic peak
        cmb_freq = BASE_FREQ * (2 ** 8) * 0.5   # 8th octave, half step
        sig = _sine(cmb_freq, duration_s, 0.15)
        sig += np.random.randn(n) * 0.03
    return np.clip(sig, -1, 1) * AMBIENT_VOL


def ambient_structure(duration_s: float = 4.0) -> np.ndarray:
    """
    Epoch 7: Gravitational bass as dark matter halos collapse.
    Occasional stellar ignition (brief bright flash of harmonics).
    """
    sig = _sine(20, duration_s, 0.5)
    sig += _sine(30, duration_s, 0.3)
    sig += _noise(duration_s, 0.06, seed=11)
    # Stellar ignition events
    n = int(SAMPLE_RATE * duration_s)
    for _ in range(3):
        idx = random.randint(0, n - 2000)
        star_f = BASE_FREQ * random.choice([4, 6, 8])
        star_dur = 0.4
        star = _harmonic_stack(star_f, 5, star_dur, 0.5) * 0.35
        end = min(n, idx + len(star))
        sig[idx:end] += star[:end - idx]
    return np.clip(sig, -1, 1) * AMBIENT_VOL


AMBIENT_GENERATORS = {
    0: ambient_strings,
    1: ambient_inflation,
    2: ambient_baryogenesis,
    3: ambient_qcd,
    4: ambient_axion,
    5: ambient_bbn,
    6: ambient_recombination,
    7: ambient_structure,
}


# ══════════════════════════════════════════════════════════════
#  SOUND ENGINE CLASS
# ══════════════════════════════════════════════════════════════

class SoundEngine:
    """
    Main sound engine. Manages pygame.mixer channels, ambient loops,
    and on-demand FX playback.

    Usage:
        engine = SoundEngine()
        engine.set_epoch(0)
        engine.play_string_vibration(n=1, closed=True)
        engine.play_fusion(Q_MeV=2.224, product="deuterium")
        engine.play_forbidden("confinement")
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, enabled: bool = True):
        self.enabled    = enabled and PYGAME_AVAILABLE
        self.sample_rate = sample_rate
        self._epoch     = 0
        self._ambient_channel: Optional["pygame.mixer.Channel"] = None
        self._fx_channels: list["pygame.mixer.Channel"] = []
        self._cache: dict[str, "pygame.mixer.Sound"] = {}
        self._ambient_phase = 0.0
        self._T_MeV = 1.0

        if self.enabled:
            mixer.init(frequency=sample_rate, size=-16, channels=N_CHANNELS,
                       buffer=1024)
            mixer.set_num_channels(16)
            self._ambient_channel = mixer.Channel(0)
            self._fx_channels = [mixer.Channel(i) for i in range(1, 8)]
            self._pregenerate_ambient()

    def _pregenerate_ambient(self):
        """Pre-generate ambient loops for all epochs in background thread."""
        def _gen():
            for epoch_id, gen_fn in AMBIENT_GENERATORS.items():
                try:
                    if epoch_id == 5:
                        samples = gen_fn(4.0, T_MeV=1.0)
                    elif epoch_id == 6:
                        samples = gen_fn(4.0, phase=0.0)
                    else:
                        samples = gen_fn(4.0)
                    snd = _make_sound(samples)
                    if snd:
                        self._cache[f"ambient_{epoch_id}"] = snd
                except Exception:
                    pass
        threading.Thread(target=_gen, daemon=True).start()

    def _play_on_free_channel(self, snd: "pygame.mixer.Sound", volume: float = FX_VOL):
        if not self.enabled or snd is None:
            return
        for ch in self._fx_channels:
            if not ch.get_busy():
                ch.set_volume(volume * MAX_VOLUME)
                ch.play(snd)
                return
        # All busy: use channel 1 (interrupt least important)
        self._fx_channels[0].set_volume(volume * MAX_VOLUME)
        self._fx_channels[0].play(snd)

    def set_epoch(self, epoch_id: int, T_MeV: float = 1.0):
        """Switch ambient soundscape to the given epoch."""
        self._epoch   = epoch_id
        self._T_MeV   = T_MeV
        if not self.enabled:
            return
        key = f"ambient_{epoch_id}"
        snd = self._cache.get(key)
        if snd:
            self._ambient_channel.set_volume(AMBIENT_VOL * MAX_VOLUME)
            self._ambient_channel.play(snd, loops=-1)

    def update_ambient(self, epoch_id: int, T_MeV: float = 1.0,
                        recomb_phase: float = 0.0):
        """
        Update the ambient sound to reflect current conditions.
        Call once per second or when conditions change significantly.
        """
        if not self.enabled:
            return
        if epoch_id == 5 and abs(T_MeV - self._T_MeV) > 0.01:
            self._T_MeV = T_MeV
            try:
                samples = ambient_bbn(4.0, T_MeV=T_MeV)
                snd = _make_sound(samples)
                if snd:
                    self._cache["ambient_5"] = snd
                    self._ambient_channel.play(snd, loops=-1)
            except Exception:
                pass
        elif epoch_id == 6:
            self._ambient_phase = recomb_phase
            try:
                samples = ambient_recombination(4.0, phase=recomb_phase)
                snd = _make_sound(samples)
                if snd:
                    self._cache["ambient_6_live"] = snd
                    self._ambient_channel.play(snd, loops=-1)
            except Exception:
                pass

    def play_string_vibration(self, mode_n: int, closed: bool,
                               tachyon: bool = False):
        key = f"str_{mode_n}_{closed}_{tachyon}"
        if key not in self._cache:
            samples = string_vibration_sound(mode_n, closed, tachyon)
            snd = _make_sound(samples)
            if snd:
                self._cache[key] = snd
        snd = self._cache.get(key)
        self._play_on_free_channel(snd, FX_VOL * 0.7)

    def play_string_join(self, g_s: float = 0.1):
        samples = string_join_sound(g_s)
        snd = _make_sound(samples)
        self._play_on_free_channel(snd)

    def play_string_split(self, g_s: float = 0.1):
        samples = string_split_sound(g_s)
        snd = _make_sound(samples)
        self._play_on_free_channel(snd)

    def play_fusion(self, Q_MeV: float, product: str = ""):
        samples = fusion_sound(Q_MeV, product)
        snd = _make_sound(samples)
        self._play_on_free_channel(snd, FX_VOL * 0.9)

    def play_annihilation(self):
        key = "annihilation"
        if key not in self._cache:
            snd = _make_sound(annihilation_sound())
            if snd:
                self._cache[key] = snd
        self._play_on_free_channel(self._cache.get(key))

    def play_proton_form(self):
        key = "proton_form"
        if key not in self._cache:
            snd = _make_sound(proton_form_sound())
            if snd:
                self._cache[key] = snd
        self._play_on_free_channel(self._cache.get(key))

    def play_recombination(self):
        key = "recombination"
        if key not in self._cache:
            snd = _make_sound(recombination_sound())
            if snd:
                self._cache[key] = snd
        self._play_on_free_channel(self._cache.get(key), FX_VOL * 0.8)

    def play_halo_merge(self):
        samples = halo_merge_sound()
        snd = _make_sound(samples)
        self._play_on_free_channel(snd, FX_VOL * 0.7)

    def play_forbidden(self, reason_type: str = "generic"):
        fn = FORBIDDEN_SOUNDS.get(reason_type, FORBIDDEN_SOUNDS["generic"])
        key = f"forbidden_{reason_type}"
        if key not in self._cache:
            snd = _make_sound(fn())
            if snd:
                self._cache[key] = snd
        self._play_on_free_channel(self._cache.get(key), FX_VOL * 0.85)

    def play_epoch_transition(self, from_epoch: int, to_epoch: int):
        """
        Cinematic audio for an epoch transition:
        sweep + impact + new ambient crossfade.
        """
        if not self.enabled:
            return
        dur = 1.5
        sweep = np.linspace(0, 1, int(SAMPLE_RATE * dur))
        f_sweep = BASE_FREQ * (1 + sweep * 4)
        phase = np.cumsum(2 * math.pi * f_sweep / SAMPLE_RATE)
        sig = np.sin(phase) * 0.6
        sig *= np.exp(-sweep * 3)
        snd = _make_sound(sig)
        if snd:
            self._play_on_free_channel(snd, FX_VOL)
        # Schedule epoch switch
        import threading
        def _switch():
            import time; time.sleep(dur * 0.7)
            self.set_epoch(to_epoch)
        threading.Thread(target=_switch, daemon=True).start()

    def set_master_volume(self, v: float):
        mixer.music.set_volume(v * MAX_VOLUME) if PYGAME_AVAILABLE else None

    def stop_all(self):
        if self.enabled:
            mixer.stop()
