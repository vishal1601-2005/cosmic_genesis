"""
render/narrator.py — Cinematic narrator subtitle system.

Displays the narrator text for each epoch as a slow-fade subtitle
at the bottom of the screen. Text is typed character-by-character
(typewriter effect) and held for a duration before fading.

Each epoch has a set of rotating narrations that trigger on:
  - Epoch entry (always)
  - First interaction of each type
  - Milestone events (graviton formed, He-4 produced, halo merged…)
  - Timer-based (every N seconds a new narration appears)

The text is drawn in pygame (software), composited over the GL frame.
Font: Space Mono (monospace — fits the physics aesthetic).
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from config import EPOCHS


# ── Narration database ────────────────────────────────────────
NARRATIONS: dict[int, list[str]] = {
    0: [
        "Before time, before space — only vibration.",
        "Ten dimensions tremble at the Planck scale.",
        "Each string vibrates at a frequency that will become a particle.",
        "The graviton is already here. It is always here. Gravity is unavoidable.",
        "The Calabi-Yau manifold is not yet fixed. Six dimensions choose their shape.",
        "Click two open strings. Let them join. Watch what emerges.",
        "T-duality: a universe compactified to radius R is identical to one at α′/R.",
        "S-duality: strong coupling and weak coupling are the same theory.",
        "The tachyon signals instability. The bosonic string cannot be the whole story.",
        "Superstring theory removes the tachyon. It imposes supersymmetry. It works.",
    ],
    1: [
        "A scalar field begins to roll down its potential.",
        "Space expands exponentially. Every quantum ripple stretches to cosmic scales.",
        "Sixty e-folds of inflation. The universe grows by e⁶⁰.",
        "Every fluctuation you see will one day be a galaxy.",
        "The inflaton slows. Reheating begins. The hot Big Bang.",
        "Quantum mechanics and gravity conspire to seed all structure.",
    ],
    2: [
        "Quarks and antiquarks rain down in equal numbers.",
        "One in a billion survives. That one is you. That one is everything.",
        "CP violation: the laws of physics are slightly different for matter and antimatter.",
        "Sphaleron processes: electroweak instantons that change baryon number by three.",
        "The asymmetry is written into the universe at the electroweak scale.",
        "Without this imbalance, you would not exist. The universe would be photons only.",
    ],
    3: [
        "Temperature drops below 217 MeV. The quark-gluon plasma cools.",
        "Colour can no longer roam free. Confinement begins.",
        "Flux tubes form between quarks. The string tension is 0.9 GeV per femtometre.",
        "Pull a quark away and the vacuum tears. A new pair appears.",
        "Protons and neutrons crystallise from the chaos.",
        "The strong force has hidden three quarks inside every proton ever since.",
    ],
    4: [
        "An invisible field misaligns. No one will ever touch it.",
        "But it outweighs all the stars. This is dark matter forming.",
        "The Peccei-Quinn symmetry breaks. The axion begins to roll.",
        "Its mass is six micro-electron-volts. It oscillates ten billion times per second.",
        "These oscillations are cold. They do not clump. They fill the universe.",
        "You cannot see dark matter. But without it, no galaxy would ever form.",
    ],
    5: [
        "Three minutes. That is all the time the universe has to forge its elements.",
        "Miss the window and the universe stays hydrogen forever.",
        "The deuterium bottleneck: above 70 keV, photons destroy every nucleus that forms.",
        "Neutrons are decaying. Each one lost is a neutron that cannot become helium.",
        "p + n → D + γ. The first step. Two millisieverts per fusion event.",
        "D + D → ⁴He. The doubly magic nucleus. Twenty-three megaelectronvolts released.",
        "The target is 24.5 percent helium by mass. Nature achieves this precisely.",
        "You have less than three minutes. Fuse. Now.",
    ],
    6: [
        "For 380,000 years the universe was opaque. A wall of light and matter.",
        "Then, in a moment — silence.",
        "Electrons and protons combine. The universe becomes transparent.",
        "The photons that stream free now will travel for 13.8 billion years.",
        "They are still here. We call them the Cosmic Microwave Background.",
        "Every temperature anisotropy you see is a seed. A future galaxy.",
    ],
    7: [
        "Gravity sculpts the dark.",
        "Filaments form. Voids open. Dark matter halos collapse.",
        "Inside the halos, gas cools and falls inward.",
        "The first stars are enormous. A hundred solar masses. Zero metals.",
        "They live ten million years and die in supernovae that seed the universe with iron.",
        "When two halos merge, the gas ignites. A quasar lights up.",
        "We are stardust assembled by gravity, powered by fusion, watching ourselves form.",
    ],
}

MILESTONE_NARRATIONS: dict[str, str] = {
    "first_graviton":    "The first graviton. The first force. Gravity is born.",
    "crystallisation":   "Spacetime crystallises. The four dimensions we know lock in.",
    "first_fusion":      "p + n → D + γ. The first nuclear bond in history.",
    "first_helium4":     "Helium-4. The first complex nucleus. Twenty-eight megaelectronvolts of binding.",
    "bbn_victory":       "Y_p ≈ 0.245. You matched the universe's own nucleosynthesis.",
    "bbn_defeat":        "Time expired. The window closed. The universe will have what it has.",
    "recombination":     "The universe is transparent. You can see to the beginning of time.",
    "first_star":        "A star ignites. Nuclear fusion in the cosmos for the first time since BBN.",
    "first_halo_merge":  "Two halos merge. A starburst. A galaxy is born.",
    "t_duality":         "T-duality applied. R → α′/R. The physics is unchanged. Distance is an illusion.",
    "s_duality":         "gₛ → 1/gₛ. Weak and strong coupling are the same theory. Duality is real.",
    "forbidden_gso":     "GSO mismatch. The tachyon would return. Supersymmetry forbids this.",
    "forbidden_confinement": "Colour confinement. You cannot isolate a quark. The flux tube snaps first.",
    "forbidden_diproton":"No bound diproton. The nuclear force is strong — but not quite strong enough.",
}


@dataclass
class NarrationState:
    text:        str
    display_len: int = 0       # characters revealed so far (typewriter)
    hold_t:      float = 0.0   # time held at full display
    fade_t:      float = 0.0   # time spent fading
    state:       str = "typing"  # "typing" | "holding" | "fading" | "done"

    TYPE_SPEED  = 35.0   # chars per second
    HOLD_TIME   = 4.5    # seconds to hold full text
    FADE_TIME   = 1.2    # seconds to fade out

    @property
    def alpha(self) -> float:
        if self.state == "typing":
            return min(1.0, self.display_len / max(1, len(self.text)) * 2)
        elif self.state == "holding":
            return 1.0
        elif self.state == "fading":
            return max(0.0, 1.0 - self.fade_t / self.FADE_TIME)
        return 0.0

    @property
    def visible_text(self) -> str:
        return self.text[:self.display_len]

    @property
    def done(self) -> bool:
        return self.state == "done"

    def update(self, dt: float):
        if self.state == "typing":
            self.display_len = min(len(self.text),
                                    self.display_len + int(self.TYPE_SPEED * dt) + 1)
            if self.display_len >= len(self.text):
                self.state = "holding"
        elif self.state == "holding":
            self.hold_t += dt
            if self.hold_t >= self.HOLD_TIME:
                self.state = "fading"
        elif self.state == "fading":
            self.fade_t += dt
            if self.fade_t >= self.FADE_TIME:
                self.state = "done"


class Narrator:
    """
    Manages the sequence of narrator subtitles.
    Each epoch gets rotating lines; milestones interrupt immediately.
    """
    def __init__(self):
        self._epoch       = 0
        self._line_idx    = 0
        self._current:    Optional[NarrationState] = None
        self._queue:      list[str] = []
        self._t_since_last= 0.0
        self._INTERVAL    = 7.0   # seconds between auto-advance

    def set_epoch(self, epoch_id: int):
        self._epoch     = epoch_id
        self._line_idx  = 0
        self._queue.clear()
        self._t_since_last = 0.0
        # Trigger entry narration
        lines = NARRATIONS.get(epoch_id, [])
        if lines:
            self._show(lines[0])
            self._line_idx = 1

    def milestone(self, key: str):
        """Interrupt with a milestone narration."""
        text = MILESTONE_NARRATIONS.get(key)
        if text:
            self._queue.insert(0, text)

    def update(self, dt: float):
        # Advance current
        if self._current and not self._current.done:
            self._current.update(dt)
            return

        # Pop from queue
        if self._queue:
            self._show(self._queue.pop(0))
            return

        # Auto-advance after interval
        self._t_since_last += dt
        if self._t_since_last >= self._INTERVAL:
            self._t_since_last = 0.0
            lines = NARRATIONS.get(self._epoch, [])
            if lines:
                text = lines[self._line_idx % len(lines)]
                self._line_idx += 1
                self._show(text)

    def _show(self, text: str):
        self._current       = NarrationState(text=text)
        self._t_since_last  = 0.0

    def draw(self, surface, W: int, H: int, font):
        """Draw current narration onto pygame surface."""
        if not self._current or self._current.alpha < 0.01:
            return
        try:
            import pygame
        except ImportError:
            return

        text  = self._current.visible_text
        alpha = int(self._current.alpha * 220)
        if not text:
            return

        # Background pill
        rendered = font.render(text, True, (210, 200, 240))
        tw, th   = rendered.get_width(), rendered.get_height()
        pad_x, pad_y = 20, 8
        pill_w = tw + pad_x * 2
        pill_h = th + pad_y * 2
        x0 = (W - pill_w) // 2
        y0 = H - 90

        bg = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
        bg.fill((4, 3, 18, min(200, alpha)))
        pygame.draw.rect(bg, (127, 119, 221, min(80, alpha)),
                         (0, 0, pill_w, pill_h), 1, border_radius=6)
        surface.blit(bg, (x0, y0))

        rendered.set_alpha(alpha)
        surface.blit(rendered, (x0 + pad_x, y0 + pad_y))

        # Cursor blink while typing
        if self._current.state == "typing" and int(self._current.hold_t * 2) % 2 == 0:
            cw = 2; ch = th
            cx = x0 + pad_x + tw + 3
            pygame.draw.rect(surface, (180, 160, 240, alpha), (cx, y0 + pad_y, cw, ch))
