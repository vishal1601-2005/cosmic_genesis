"""
render/interaction_fx.py — Visual effect manager for particle interactions.

Manages three categories of visual events:

1. VERTEX FLASH
   A radial burst of light when two strings join or a particle forms.
   Driven by the equation_dissolve shader in mode 0→1.

2. EQUATION DISSOLVE
   Physics equation text materialises as glowing glyphs, then dissolves
   into the particle itself. Uses equation_dissolve.frag.
   The equation is rendered to a texture atlas by PIL/Pillow.

3. FORBIDDEN BURST
   Red radial shockwave + ⊗ symbol + law text.
   Uses forbidden.frag + HUD text overlay.

4. EPOCH TRANSITION
   Full-screen cinematic wipe: old epoch fades, new epoch rises.
   Driven by a separate cinematic shader (simple alpha-over).

All effects are time-based and fire-and-forget:
  fx.add_vertex_flash(x, y, color)
  fx.add_equation_dissolve(x, y, equation_text, product_name, product_color)
  fx.add_forbidden(x, y, reason, equation, law, reason_type)
  fx.update(dt)           # advance all active effects
  fx.render(renderer)     # draw all active effects
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import moderngl
    MGL_OK = True
except ImportError:
    MGL_OK = False


# ── Effect data classes ────────────────────────────────────────

@dataclass
class VertexFlash:
    """Radial burst at a world position."""
    x:      float
    y:      float
    color:  tuple       # RGB float
    radius: float = 0.0
    alpha:  float = 1.0
    t:      float = 0.0
    duration: float = 0.5

    @property
    def alive(self) -> bool:
        return self.t < self.duration

    def update(self, dt: float):
        self.t      += dt
        self.radius  = self.t / self.duration * 3.5
        self.alpha   = max(0.0, 1.0 - (self.t / self.duration) ** 1.5)


@dataclass
class EquationDissolve:
    """
    A physics equation materialising then dissolving into a particle.

    States:
      phase 0 (0→0.4s): equation text appears (glow-in)
      phase 1 (0.4→1.2s): text dissolves → particle at centre
      phase 2 (done): alpha=0, remove
    """
    x:            float
    y:            float
    equation:     str
    product_name: str
    product_color: tuple
    mode:         int   = 0    # 0=appear, 1=dissolve
    progress:     float = 0.0
    alpha:        float = 0.0
    t:            float = 0.0
    duration:     float = 1.8
    # Texture (PIL Image → GL Texture)
    eq_tex_data:  Optional[np.ndarray] = field(default=None, repr=False)

    APPEAR_END   = 0.35   # fraction of duration for appear phase
    DISSOLVE_END = 1.0

    @property
    def alive(self) -> bool:
        return self.t < self.duration

    @property
    def phase_progress(self) -> float:
        """Progress within the current phase [0, 1]."""
        frac = self.t / self.duration
        if frac < self.APPEAR_END:
            return frac / self.APPEAR_END
        else:
            return (frac - self.APPEAR_END) / (self.DISSOLVE_END - self.APPEAR_END)

    def update(self, dt: float):
        self.t += dt
        frac = self.t / self.duration
        if frac < self.APPEAR_END:
            self.mode     = 0
            self.progress = frac / self.APPEAR_END
            self.alpha    = min(1.0, self.progress * 3.0)
        elif frac < self.DISSOLVE_END:
            self.mode     = 1
            self.progress = (frac - self.APPEAR_END) / (self.DISSOLVE_END - self.APPEAR_END)
            self.alpha    = max(0.0, 1.0 - (self.progress - 0.5) * 2.5)
        else:
            self.alpha = 0.0


@dataclass
class ForbiddenBurst:
    """Red shockwave for a forbidden interaction."""
    x:             float
    y:             float
    reason:        str
    equation:      str
    law:           str
    reason_type:   int     # 0=charge,1=colour,2=energy,3=epoch,4=baryon,5=gso
    radius:        float = 0.0
    alpha:         float = 1.0
    t:             float = 0.0
    duration:      float = 1.5
    eq_alpha:      float = 0.0   # equation text fade-in

    @property
    def alive(self) -> bool:
        return self.t < self.duration

    def update(self, dt: float):
        self.t       += dt
        frac          = self.t / self.duration
        self.radius   = frac * 2.8
        self.alpha    = max(0.0, 1.0 - frac ** 0.8)
        # Equation text: appears quickly, stays, then fades
        if frac < 0.15:
            self.eq_alpha = frac / 0.15
        elif frac < 0.7:
            self.eq_alpha = 1.0
        else:
            self.eq_alpha = max(0.0, 1.0 - (frac - 0.7) / 0.3)


@dataclass
class EpochTransition:
    """Full-screen cinematic wipe between epochs."""
    from_epoch: int
    to_epoch:   int
    t:          float = 0.0
    duration:   float = 3.0
    alpha:      float = 0.0

    @property
    def alive(self) -> bool:
        return self.t < self.duration

    @property
    def progress(self) -> float:
        return self.t / self.duration

    def update(self, dt: float):
        self.t += dt
        frac = self.t / self.duration
        # Fade to black at midpoint, then reveal new epoch
        if frac < 0.5:
            self.alpha = frac * 2.0
        else:
            self.alpha = (1.0 - frac) * 2.0


# ── Equation texture baking ────────────────────────────────────

def bake_equation_texture(text: str, width: int = 512, height: int = 64,
                            font_size: int = 18) -> np.ndarray:
    """
    Render equation text to a (height, width, 4) RGBA numpy array.
    The alpha channel contains the glyph mask (white=text, black=empty).
    Uses PIL/Pillow for font rendering.
    """
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if not PIL_OK:
        return np.array(img, dtype=np.float32) / 255.0

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    # Draw white text centred
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw   = bbox[2] - bbox[0]
        th   = bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(text, font=font)

    x0 = max(0, (width  - tw) // 2)
    y0 = max(0, (height - th) // 2)
    draw.text((x0, y0), text, fill=(255, 255, 255, 255), font=font)

    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 4)
    return arr


# ── FX Manager ─────────────────────────────────────────────────

class FXManager:
    """
    Central manager for all visual interaction effects.
    Stores, updates, and draws all active effects each frame.
    """

    def __init__(self):
        self.vertex_flashes:  list[VertexFlash]       = []
        self.eq_dissolves:    list[EquationDissolve]   = []
        self.forbidden_bursts:list[ForbiddenBurst]     = []
        self.epoch_transitions:list[EpochTransition]   = []
        self._eq_texture_cache: dict[str, np.ndarray] = {}

    # ── Public API ─────────────────────────────────────────
    def add_vertex_flash(self, x: float, y: float, color: tuple):
        self.vertex_flashes.append(VertexFlash(x=x, y=y, color=color))

    def add_equation_dissolve(self, x: float, y: float,
                               equation: str, product_name: str,
                               product_color: tuple):
        tex_data = self._get_or_bake(equation)
        self.eq_dissolves.append(EquationDissolve(
            x=x, y=y, equation=equation,
            product_name=product_name, product_color=product_color,
            eq_tex_data=tex_data,
        ))

    def add_forbidden(self, x: float, y: float,
                      reason: str, equation: str,
                      law: str, reason_type: int = 1):
        self.forbidden_bursts.append(ForbiddenBurst(
            x=x, y=y, reason=reason, equation=equation,
            law=law, reason_type=reason_type,
        ))

    def add_epoch_transition(self, from_epoch: int, to_epoch: int):
        self.epoch_transitions.append(EpochTransition(from_epoch, to_epoch))

    # ── Update ─────────────────────────────────────────────
    def update(self, dt: float):
        for fx in self.vertex_flashes:   fx.update(dt)
        for fx in self.eq_dissolves:     fx.update(dt)
        for fx in self.forbidden_bursts: fx.update(dt)
        for fx in self.epoch_transitions:fx.update(dt)
        # Remove dead effects
        self.vertex_flashes   = [f for f in self.vertex_flashes   if f.alive]
        self.eq_dissolves     = [f for f in self.eq_dissolves     if f.alive]
        self.forbidden_bursts = [f for f in self.forbidden_bursts if f.alive]
        self.epoch_transitions= [f for f in self.epoch_transitions if f.alive]

    # ── Pygame software fallback draw ──────────────────────
    def draw_pygame(self, surface, cam_x: float, cam_y: float,
                    W: int, H: int, font_sm=None, font_md=None):
        """
        Draw all FX onto a pygame surface (software fallback).
        Full GL rendering handled by renderer.py.
        """
        try:
            import pygame
        except ImportError:
            return

        def world_to_screen(wx, wy):
            sx = int(W//2 + (wx - cam_x) * W * 0.04)
            sy = int(H//2 - (wy - cam_y) * H * 0.04)
            return sx, sy

        # Vertex flashes
        for fx in self.vertex_flashes:
            sx, sy = world_to_screen(fx.x, fx.y)
            r  = int(fx.radius * W * 0.04)
            al = int(fx.alpha * 200)
            col = tuple(int(c * 255) for c in fx.color)
            if r > 0 and 0 <= sx < W and 0 <= sy < H:
                pygame.draw.circle(surface, (*col, al), (sx, sy), r, 1)
                pygame.draw.circle(surface, (*col, al // 3), (sx, sy), r // 2)

        # Forbidden bursts
        for fx in self.forbidden_bursts:
            sx, sy = world_to_screen(fx.x, fx.y)
            r   = int(fx.radius * W * 0.04)
            al  = int(fx.alpha * 200)
            red = (220, 50, 50)
            if r > 0 and 0 <= sx < W and 0 <= sy < H:
                pygame.draw.circle(surface, (*red, al), (sx, sy), r, 2)
                # X mark
                pygame.draw.line(surface, (*red, al),
                                 (sx-14, sy-14), (sx+14, sy+14), 2)
                pygame.draw.line(surface, (*red, al),
                                 (sx+14, sy-14), (sx-14, sy+14), 2)
            # Law text
            if font_sm and fx.eq_alpha > 0.1:
                text_al = int(fx.eq_alpha * 220)
                law_surf = font_sm.render(fx.law[:60], True, (220, 80, 80))
                eq_surf  = font_sm.render(fx.equation[:70], True, (200, 150, 150))
                reason_surf = font_sm.render(fx.reason[:70], True, (240, 200, 200))
                law_surf.set_alpha(text_al)
                eq_surf.set_alpha(text_al)
                reason_surf.set_alpha(text_al)
                surface.blit(law_surf,    (sx - law_surf.get_width()//2, sy + r + 8))
                surface.blit(eq_surf,     (sx - eq_surf.get_width()//2,  sy + r + 22))
                surface.blit(reason_surf, (sx - reason_surf.get_width()//2, sy + r + 36))

        # Equation dissolves
        for fx in self.eq_dissolves:
            sx, sy = world_to_screen(fx.x, fx.y)
            if font_md and fx.alpha > 0.05:
                col = tuple(int(c*255) for c in fx.product_color)
                al  = int(fx.alpha * 230)
                # Equation text
                eq_surf = font_md.render(fx.equation[:55], True, col)
                eq_surf.set_alpha(al)
                surface.blit(eq_surf, (sx - eq_surf.get_width()//2, sy - 30))
                # Product name
                pname = font_md.render(f"→ {fx.product_name}", True, col)
                pname.set_alpha(al)
                surface.blit(pname, (sx - pname.get_width()//2, sy - 12))
                # Dissolve particle core
                if fx.mode == 1:
                    core_r = int(fx.progress * 20)
                    if core_r > 0:
                        pygame.draw.circle(surface, (*col, al//2), (sx, sy), core_r)

        # Epoch transitions
        for fx in self.epoch_transitions:
            al = int(fx.alpha * 255)
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((2, 1, 10, al))
            surface.blit(overlay, (0, 0))
            if font_md and 0.3 < fx.progress < 0.7:
                msg = f"Epoch {fx.to_epoch}: {['String landscape','Inflation','Baryogenesis','QCD confinement','Axion condensation','Big Bang nucleosynthesis','Recombination','Structure formation'][fx.to_epoch]}"
                text_surf = font_md.render(msg, True, (180, 160, 240))
                text_al   = int(min(1.0, (fx.progress - 0.3)/0.2) * 200)
                text_surf.set_alpha(text_al)
                surface.blit(text_surf, (W//2 - text_surf.get_width()//2, H//2 - 10))

    # ── Private helpers ────────────────────────────────────
    def _get_or_bake(self, text: str) -> np.ndarray:
        if text not in self._eq_texture_cache:
            self._eq_texture_cache[text] = bake_equation_texture(text)
        return self._eq_texture_cache[text]

    @property
    def has_active_transition(self) -> bool:
        return len(self.epoch_transitions) > 0

    @property
    def transition_alpha(self) -> float:
        if not self.epoch_transitions:
            return 0.0
        return self.epoch_transitions[-1].alpha
