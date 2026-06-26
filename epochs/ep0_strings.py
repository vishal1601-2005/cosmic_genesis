"""
epochs/ep0_strings.py — Planck epoch: String landscape (t = 0 → 10⁻⁴³ s).

GAMEPLAY:
  You are in pre-geometric 10D space. Spacetime has not condensed —
  only vibrating strings exist. The Calabi-Yau manifold is choosing its vacuum.

  MECHANICS:
    - Open strings drift through the field
    - Click one open string + click another → join them → closed string
    - The vibrational mode that fires when they join determines which
      particle will exist when spacetime crystallises:
        n=1 closed → graviton (massless, spin-2) — ALWAYS appears
        n=2 open   → massive gauge boson
        tachyon (bosonic only) → signal the vacuum is wrong
    - Double-click a string to split it
    - Press T to trigger T-duality (R → α′/R)
    - Press S to trigger S-duality (gₛ → 1/gₛ)
    - Spacetime crystallises when enough gravitons form (n_gravitons ≥ 3)
    - The shape of the Calabi-Yau moduli when crystallisation happens
      determines the low-energy physics

  VISUAL:
    - All strings rendered via worldsheet.vert/frag
    - 6 compact dimensions shown as tiny CY blooms at every grid point
    - Tachyons have a red shimmer + instability ripples (bosonic only)
    - String joining: vertex operator flash + equation overlay
    - String splitting: snap sound + two daughters emerge
    - Calabi-Yau moduli: shown in sidebar, morphing continuously
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
    JAX_OK = True
except ImportError:
    JAX_OK = False

from config import THEORIES, N_SIGMA, N_MODES, G_S_DEFAULT, ALPHA_PRIME
from physics.worldsheet import (
    StringState, init_state, make_step_fn,
    split_string, join_strings, apply_t_duality, apply_s_duality,
)
from physics.calabi_yau import CalabiYauMetric


# ── String visual metadata (per-string, for renderer) ─────────
@dataclass
class StringVisual:
    """Per-string data the renderer needs (separate from physics state)."""
    id:          int
    closed:      bool
    x:           float = 0.0   # screen-space centre (projected from X[0,2])
    y:           float = 0.0
    z:           float = 0.0
    vx:          float = 0.0
    vy:          float = 0.0
    color_rgb:   tuple = (0.6, 0.5, 1.0)
    radius:      float = 0.04
    emission_rgb:tuple = (0.5, 0.4, 0.9)
    emission_str:float = 2.5
    age_s:       float = 0.0
    metallic:    float = 0.0
    roughness:   float = 0.4
    particle_type_int: int = 5   # type 5 = string
    selected:    bool = False
    mode_amps:   list = field(default_factory=list)
    mass_sq:     float = 0.0
    is_tachyon:  bool = False
    winding:     int  = 0
    fermion:     bool = False
    mode_n:      int  = 1        # dominant mode


# ── Vertex operator event ─────────────────────────────────────
@dataclass
class VertexEvent:
    x: float; y: float
    eq_text: str          # equation to display
    product_name: str     # particle name
    product_color: tuple
    alpha: float = 1.0
    progress: float = 0.0  # dissolve progress 0→1
    is_forbidden: bool = False
    forbidden_reason: str = ""


# ── Calabi-Yau state ──────────────────────────────────────────
@dataclass
class CYState:
    h11: int = 1
    h21: int = 101
    euler: int = -200
    vacua_exp: int = 274
    morph: float = 0.0
    label: str = "Quintic CY₃"


class PlanckEpoch:
    """
    Full state of the Planck epoch simulation.
    """

    def __init__(self, theory: str = "super", n_strings: int = 48):
        self.theory      = theory
        self.n_strings   = n_strings
        self.g_s         = G_S_DEFAULT
        self.compact_r   = 1.0
        self.t_world     = 0.0   # worldsheet time elapsed

        # JAX physics state
        key = jax.random.PRNGKey(0) if JAX_OK else None
        self.phys_state  = init_state(n_strings, theory=theory, g_s=self.g_s, key=key)
        self.step_fn     = make_step_fn(theory=theory)

        # Visual metadata (one per string)
        self.visuals: list[StringVisual] = self._init_visuals()

        # Calabi-Yau
        self.cy = CYState()
        try:
            self.cy_metric = CalabiYauMetric(preset="quintic")
        except Exception:
            self.cy_metric = None

        # Events
        self.vertex_events: list[VertexEvent] = []

        # Graviton count (crystallisation trigger)
        self.n_gravitons = 0
        self.crystallised = False

        # Selected strings for interaction
        self.selection: list[int] = []   # indices into visuals

        # Statistics
        self.n_splits = 0
        self.n_joins  = 0
        self.n_dualities = 0

    def _init_visuals(self) -> list[StringVisual]:
        th    = THEORIES[self.theory]
        cols  = th["stringColors"]
        visuals = []
        N = self.phys_state.X.shape[0] if JAX_OK else self.n_strings
        for i in range(N):
            closed = bool(i % 3 != 0)   # 2/3 closed, 1/3 open
            ci     = i % len(cols)
            col    = self._hex_to_rgb(cols[ci])
            visuals.append(StringVisual(
                id=i, closed=closed,
                x=random.uniform(-8, 8),
                y=random.uniform(-5, 5),
                z=random.uniform(-4, 4),
                vx=random.gauss(0, 0.08),
                vy=random.gauss(0, 0.08),
                color_rgb=col,
                emission_rgb=tuple(min(1.0, c * 1.3) for c in col),
                emission_str=2.5 + random.uniform(0, 1.5),
                radius=0.025 + random.uniform(0, 0.02),
                fermion=self.theory == "super" and random.random() > 0.5,
                winding=1 if random.random() > 0.85 else 0,
                mode_n=1,
            ))
        return visuals

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple:
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def update(self, dt: float, physics_steps: int = 4):
        """Advance physics and visuals by dt real seconds."""
        if self.crystallised:
            return

        # ── Physics: advance worldsheet PDE ──────────────
        for _ in range(physics_steps):
            if JAX_OK:
                self.phys_state = self.step_fn(self.phys_state)
            self.t_world += 0.005

        # ── Sync visuals from physics state ──────────────
        N = len(self.visuals)
        if JAX_OK and self.phys_state.X.shape[0] == N:
            X_np = np.array(self.phys_state.X)          # (N, D, S)
            m_np = np.array(self.phys_state.mass_sq)    # (N,)
            modes_np = np.array(self.phys_state.modes)  # (N, M)

            for i, v in enumerate(self.visuals):
                if i >= X_np.shape[0]:
                    break
                # Project 10D → 3D (use dims 0,1,2)
                xi = X_np[i]    # (D, S)
                v.x = float(xi[0, S//2] if (S := xi.shape[1]) > 0 else 0)
                v.y = float(xi[1, S//2] if xi.shape[1] > 1 else 0)
                v.z = float(xi[2, S//2] if xi.shape[0] > 2 else 0)
                v.mass_sq   = float(m_np[i])
                v.is_tachyon= float(m_np[i]) < -0.1
                # Dominant mode
                mode_amps = np.abs(modes_np[i]) if i < modes_np.shape[0] else np.zeros(N_MODES)
                v.mode_n    = int(np.argmax(mode_amps) + 1)
                v.mode_amps = mode_amps[:8].tolist()
                # Colour by mode
                v.color_rgb = self._mode_color(v.mode_n, v.is_tachyon)
                v.emission_str = 2.0 + float(np.sum(mode_amps[:4])) * 0.5

        # ── Move visuals ──────────────────────────────────
        for v in self.visuals:
            v.x += v.vx * dt
            v.y += v.vy * dt
            v.age_s += dt
            # Wrap at boundaries
            if abs(v.x) > 10: v.vx *= -1
            if abs(v.y) >  6: v.vy *= -1
            # Thermal jitter (Planck-scale foam)
            v.vx += random.gauss(0, 0.004) * dt
            v.vy += random.gauss(0, 0.004) * dt

        # ── CY moduli slow drift ──────────────────────────
        self.cy.morph = 0.3 * math.sin(self.t_world * 0.2)

        # ── Age vertex events ─────────────────────────────
        for ev in self.vertex_events:
            ev.progress += dt * 0.5
            ev.alpha = max(0, 1.0 - (ev.progress - 0.5) * 2.0)
        self.vertex_events = [e for e in self.vertex_events if e.alpha > 0]

        # ── Check crystallisation ─────────────────────────
        self.n_gravitons = sum(
            1 for v in self.visuals if v.closed and v.mode_n == 1 and not v.is_tachyon
        )
        if self.n_gravitons >= 3 and not self.crystallised:
            self.crystallised = True

    def select_string(self, idx: int):
        """Player clicks on string idx."""
        if idx in self.selection:
            return
        if len(self.selection) == 0:
            self.selection.append(idx)
            self.visuals[idx].selected = True
        elif len(self.selection) == 1:
            # Second click → try to join
            a, b = self.selection[0], idx
            self.selection.clear()
            self.visuals[a].selected = False
            result = self.attempt_join(a, b)
            return result

    def attempt_join(self, idx_a: int, idx_b: int) -> VertexEvent:
        """
        Attempt to join two strings via a vertex operator.
        Probability ∝ gₛ.  GSO check for superstring.
        """
        va, vb = self.visuals[idx_a], self.visuals[idx_b]

        # GSO check (superstring only)
        if self.theory == "super":
            if va.fermion == vb.fermion:
                # Same GSO sector → can join
                pass
            else:
                # Mixed GSO parity → FORBIDDEN
                ev = VertexEvent(
                    x=(va.x+vb.x)/2, y=(va.y+vb.y)/2,
                    eq_text="GSO: (−1)^F|phys⟩ = +|phys⟩ required",
                    product_name="FORBIDDEN",
                    product_color=(0.9, 0.1, 0.05),
                    is_forbidden=True,
                    forbidden_reason="GSO chirality mismatch — opposite parity states cannot couple",
                )
                self.vertex_events.append(ev)
                return ev

        # Probability check
        if random.random() > self.g_s * 5.0:
            ev = VertexEvent(
                x=(va.x+vb.x)/2, y=(va.y+vb.y)/2,
                eq_text=f"V = gₛ · :e^{{ik·X}}: , gₛ = {self.g_s:.2f}",
                product_name="No interaction (gₛ suppressed)",
                product_color=(0.5, 0.5, 0.7),
            )
            self.vertex_events.append(ev)
            return ev

        # Join!
        if JAX_OK and idx_a < self.phys_state.X.shape[0] and idx_b < self.phys_state.X.shape[0]:
            try:
                self.phys_state = join_strings(self.phys_state, idx_a, idx_b)
            except Exception:
                pass

        # Determine product
        mode_n = max(va.mode_n, vb.mode_n)
        if mode_n == 1:
            product = "graviton"
            eq = "n=1 closed, m²=0, spin=2  →  graviton"
            col = (0.686, 0.663, 0.925)
            self.n_gravitons += 1
        elif mode_n == 2:
            product = "B-field"
            eq = "n=1 closed antisymmetric  →  B_μν"
            col = (0.522, 0.718, 0.922)
        else:
            product = f"massive state (n={mode_n})"
            eq = f"m² = {4*(mode_n-1):.0f}/α′  →  massive closed string"
            col = (0.95, 0.62, 0.12)

        # Replace two visuals with one closed string
        mx = (va.x + vb.x) / 2
        my = (va.y + vb.y) / 2
        new_visual = StringVisual(
            id=len(self.visuals), closed=True,
            x=mx, y=my, vx=(va.vx+vb.vx)/2, vy=(va.vy+vb.vy)/2,
            color_rgb=col, emission_rgb=tuple(min(1,c*1.4) for c in col),
            emission_str=4.0, radius=0.035, mode_n=mode_n,
        )
        # Remove old, add new
        for rem in sorted([idx_a, idx_b], reverse=True):
            if rem < len(self.visuals):
                self.visuals.pop(rem)
        self.visuals.append(new_visual)

        ev = VertexEvent(
            x=mx, y=my,
            eq_text=eq,
            product_name=product,
            product_color=col,
            progress=0.0,
        )
        self.vertex_events.append(ev)
        self.n_joins += 1
        return ev

    def split_string_at(self, idx: int) -> VertexEvent:
        """Split string idx into two daughters."""
        if idx >= len(self.visuals):
            return None
        v = self.visuals[idx]

        if JAX_OK and idx < self.phys_state.X.shape[0]:
            key = jax.random.PRNGKey(self.n_splits)
            try:
                self.phys_state, c1, c2 = split_string(self.phys_state, idx, key)
            except Exception:
                c1, c2 = idx, len(self.visuals)

        # Spawn two visual daughters
        th    = THEORIES[self.theory]
        col1  = self._mode_color(max(1, v.mode_n - 1), False)
        col2  = self._mode_color(v.mode_n, False)
        d1 = StringVisual(id=len(self.visuals), closed=False,
                           x=v.x-0.3, y=v.y, vx=v.vx-0.1, vy=v.vy,
                           color_rgb=col1, radius=v.radius*0.8,
                           emission_str=v.emission_str*0.7, mode_n=max(1,v.mode_n-1))
        d2 = StringVisual(id=len(self.visuals)+1, closed=v.closed,
                           x=v.x+0.3, y=v.y, vx=v.vx+0.1, vy=v.vy,
                           color_rgb=col2, radius=v.radius*0.8,
                           emission_str=v.emission_str*0.7, mode_n=v.mode_n)
        if idx < len(self.visuals):
            self.visuals.pop(idx)
        self.visuals += [d1, d2]
        self.n_splits += 1

        ev = VertexEvent(
            x=v.x, y=v.y,
            eq_text="V = gₛ · :e^{ik·X}:  →  string splits at midpoint σ=π/2",
            product_name="2 open strings",
            product_color=col1,
        )
        self.vertex_events.append(ev)
        return ev

    def apply_t_duality(self):
        """T-duality: R → α′/R."""
        self.compact_r = ALPHA_PRIME / self.compact_r
        if JAX_OK:
            self.phys_state, _ = apply_t_duality(self.phys_state, self.compact_r)
        # Visuals: winding strings change length
        for v in self.visuals:
            if v.winding > 0:
                v.radius *= 1.4
                v.winding = 0
            elif random.random() > 0.6:
                v.winding = 1
                v.radius *= 0.75
        self.n_dualities += 1

    def apply_s_duality(self):
        """S-duality: gₛ → 1/gₛ."""
        self.g_s = 1.0 / self.g_s
        if JAX_OK:
            self.phys_state = apply_s_duality(self.phys_state)
        # At strong coupling, strings become heavy, D-branes become light
        for v in self.visuals:
            if self.g_s > 1.0:
                v.emission_str *= 0.7
                v.radius *= 1.1
            else:
                v.emission_str *= 1.3
                v.radius *= 0.9
        self.n_dualities += 1

    def shift_cy_moduli(self, h11: int = None, h21: int = None):
        """Random CY moduli deformation."""
        if h11 is None:
            h11 = random.randint(1, 200)
        if h21 is None:
            h21 = random.randint(1, 200)
        self.cy = CYState(
            h11=h11, h21=h21,
            euler=2*(h11 - h21),
            vacua_exp=min(500, abs(h11 - h21) * 2 + 50),
        )
        if self.cy_metric:
            self.cy_metric.shift_moduli(h11, h21)

    @staticmethod
    def _mode_color(n: int, tachyon: bool) -> tuple:
        if tachyon:
            return (0.88, 0.08, 0.04)
        palette = [
            (0.55, 0.42, 1.00),  # n=1 violet (graviton)
            (0.18, 0.52, 1.00),  # n=2 blue
            (0.10, 0.85, 0.80),  # n=3 cyan
            (0.18, 0.80, 0.35),  # n=4 green
            (0.95, 0.80, 0.12),  # n=5 yellow
            (0.95, 0.50, 0.08),  # n=6 orange
        ]
        return palette[min(n-1, len(palette)-1)]

    def get_render_particles(self) -> list:
        """Return visuals list for the renderer."""
        return self.visuals

    def narrator_text(self) -> str:
        """Current narrator subtitle."""
        if self.crystallised:
            return (f"Spacetime crystallises. {self.n_gravitons} gravitons fix the geometry. "
                    f"Calabi-Yau: χ={self.cy.euler}, h¹¹={self.cy.h11}. "
                    f"Inflation begins...")
        if self.n_gravitons == 0:
            return "Before time, before space — only vibration. Click two strings to join them."
        if self.n_gravitons == 1:
            return f"A graviton forms. Gravity stirs. {3 - self.n_gravitons} more needed to crystallise spacetime."
        return f"{self.n_gravitons}/3 gravitons. Spacetime is almost stable..."
