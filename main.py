"""
main.py — Cosmic Genesis: complete game loop.

Wires together:
  - CosmicClock      (Friedmann equations, cosmic time)
  - FieldManager     (JAX inflaton / axion / Higgs fields)
  - All epoch classes (ep0 strings → ep7 structure)
  - SoundEngine      (procedural physics-motivated audio)
  - FXManager        (equation dissolve, forbidden burst, vertex flash)
  - Narrator         (typewriter subtitles)
  - ParticleBuffer   (GPU instance upload)
  - Renderer         (full deferred PBR pipeline)
  - HUD              (pygame overlay)

Run:
    conda activate cosmic_genesis
    python main.py [--epoch N] [--theory super|bosonic|het]
                   [--no-sound] [--no-gpu] [--width W] [--height H]
"""

from __future__ import annotations
import sys, math, random, argparse, threading, time
from pathlib import Path
import numpy as np

# ── pygame ──────────────────────────────────────────────────────
try:
    import pygame
    from pygame.locals import *
except ImportError:
    print("ERROR: pip install pygame"); sys.exit(1)

# ── ModernGL ────────────────────────────────────────────────────
try:
    import moderngl
    MGL_OK = True
except ImportError:
    MGL_OK = False

# ── Project imports ──────────────────────────────────────────────
from config import WINDOW_W, WINDOW_H, TARGET_FPS, EPOCHS, THEORIES
from physics.cosmology import CosmicClock
from physics.fields import FieldManager
from physics.interactions import InteractionEngine, InteractionResult
from audio.sound_engine import SoundEngine
from render.interaction_fx import FXManager
from render.narrator    import Narrator
from render.particles_render import ParticleBuffer, collect_particles

# Epoch classes
from epochs.ep0_strings import PlanckEpoch
from epochs.ep1_inflation import InflationEpoch
from epochs.ep2_baryogenesis import BaryogenesisEpoch
from epochs.ep3_qcd import QCDEpoch
from epochs.ep4_ep6_ep7 import AxionEpoch, RecombinationEpoch, StructureEpoch
from epochs.ep5_bbn import BBNEpoch


# ══════════════════════════════════════════════════════════════
#  ARGS
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Cosmic Genesis")
    p.add_argument("--epoch",    type=int,  default=0)
    p.add_argument("--theory",   type=str,  default="super",
                   choices=["super","bosonic","het"])
    p.add_argument("--no-sound", action="store_true")
    p.add_argument("--no-gpu",   action="store_true")
    p.add_argument("--width",    type=int,  default=WINDOW_W)
    p.add_argument("--height",   type=int,  default=WINDOW_H)
    p.add_argument("--fps",      type=int,  default=TARGET_FPS)
    p.add_argument("--quality",  type=int,  default=1,
                   help="Volume render quality 0-3 (0=fast, 3=cinematic)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  HUD (pygame surface drawn over GL frame)
# ══════════════════════════════════════════════════════════════
class HUD:
    def __init__(self, W: int, H: int):
        self.W, self.H = W, H
        pygame.font.init()
        self.f_mono_sm = pygame.font.SysFont("Courier New", 11)
        self.f_mono_md = pygame.font.SysFont("Courier New", 13)
        self.f_sans_md = pygame.font.SysFont("Arial", 14)
        self.f_sans_lg = pygame.font.SysFont("Arial", 20, bold=True)
        # Colours
        self.C  = (220, 215, 240)
        self.CS = (157, 145, 200)
        self.CA = (127, 119, 221)
        self.CT = (29,  158, 117)
        self.CW = (239, 159, 39)
        self.CR = (226, 75,  74)

    def _panel(self, surf, x, y, w, h, border=None):
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((4, 3, 18, 185))
        if border:
            pygame.draw.rect(bg, (*border, 70), (0,0,w,h), 1, border_radius=6)
        surf.blit(bg, (x, y))
        return x, y

    def draw(self, surf, state, narrator: Narrator, fx: FXManager):
        W, H = self.W, self.H

        # ── Top bar ──────────────────────────────────────
        self._panel(surf, 0, 0, W, 42)
        logo = self.f_sans_lg.render("COSMIC GENESIS", True, self.C)
        surf.blit(logo, (18, 11))
        T_str = state.clock.display_temperature()
        t_txt = self.f_mono_md.render(f"T = {T_str}", True, self.CW)
        surf.blit(t_txt, (300, 14))
        t_elapsed = self.f_mono_md.render(f"t = {state.clock.display_time()}", True, self.CS)
        surf.blit(t_elapsed, (520, 14))
        ep = EPOCHS[state.epoch_id]
        ep_txt = self.f_mono_md.render(f"Epoch {state.epoch_id}: {ep['name']}", True, self.CA)
        surf.blit(ep_txt, (W - ep_txt.get_width() - 20, 14))

        # ── Top-left info ─────────────────────────────────
        self._panel(surf, 18, 52, 230, 90, border=self.CA)
        lines = [
            (f"zoom: {state.zoom*100:.0f}%  Planck", self.CS),
            (f"gₛ = {state.g_s:.3f}  R = {state.radius:.2f} ℓₛ", self.CS),
            (f"frame: {state.frame:,}", self.CS),
            (f"particles: {state.n_particles}", self.CT),
        ]
        for i, (txt, col) in enumerate(lines):
            r = self.f_mono_sm.render(txt, True, col)
            surf.blit(r, (26, 58 + i*18))

        # ── BBN HUD ───────────────────────────────────────
        ep_obj = state.active_epoch
        if state.epoch_id == 5 and ep_obj:
            self._draw_bbn(surf, ep_obj, W, H)

        # ── Last interaction panel ────────────────────────
        if state.last_ir and state.ir_timer > 0:
            self._draw_interaction(surf, state.last_ir, W, H)

        # ── Bottom hint bar ───────────────────────────────
        self._panel(surf, 0, H-26, W, 26)
        hints = [("WASD","move"),("scroll","zoom"),("click","select"),
                 ("2 clicks","interact"),("T","T-dual"),("S","S-dual"),
                 ("M","moduli"),("C","CY shift"),("+/-","time speed"),("SPACE","pause")]
        x = 16
        for k, lab in hints:
            ks = self.f_mono_sm.render(f"[{k}]", True, self.CA)
            ls = self.f_mono_sm.render(f" {lab}  ", True, self.CS)
            surf.blit(ks, (x, H-19)); x += ks.get_width()
            surf.blit(ls, (x, H-19)); x += ls.get_width()

        # ── Narrator ─────────────────────────────────────
        narrator.draw(surf, W, H, self.f_mono_md)

        # ── FX overlay ───────────────────────────────────
        fx.draw_pygame(surf, state.cam_x, state.cam_y, W, H,
                       self.f_mono_sm, self.f_mono_md)

    def _draw_bbn(self, surf, bbn, W, H):
        self._panel(surf, W-290, 52, 272, 168, border=self.CT)
        def row(y, label, val, col):
            l = self.f_mono_sm.render(label, True, self.CS)
            v = self.f_mono_sm.render(val, True, col)
            surf.blit(l, (W-284, y)); surf.blit(v, (W-140, y))
        t = bbn.time_remaining
        tc = self.CR if t < 30 else self.CW
        row(58,  "Time remaining", f"{int(t//60)}:{int(t%60):02d}", tc)
        row(76,  "T (MeV)",        f"{bbn.T_MeV:.3f}", self.CW)
        row(94,  "n/p",            f"1:{1/max(bbn.np_ratio_current,0.001):.1f}", self.C)
        row(112, "Protons",         str(bbn.n_protons),   self.CW)
        row(130, "Neutrons",        str(bbn.n_neutrons),  self.CS)
        row(148, "Deuterium",       str(bbn.n_deuterium), (64,190,255))
        row(166, "He-4",            str(bbn.n_helium4),   self.CT)
        Yp = bbn.Y_p_current
        bw = int(min(1.0, Yp/0.30) * 240)
        pygame.draw.rect(surf, (*self.CS, 60), (W-284, 192, 240, 8), border_radius=3)
        if bw > 0:
            pygame.draw.rect(surf, (*self.CT, 200), (W-284, 192, bw, 8), border_radius=3)
        yp_r = self.f_mono_sm.render(f"Y_p = {Yp:.3f}  target 0.245", True, self.CT)
        surf.blit(yp_r, (W-284, 204))
        if bbn.deuterium_bottleneck_active:
            warn = self.f_mono_sm.render("⚠ D bottleneck: T > 70 keV", True, self.CR)
            surf.blit(warn, (W-284, 54))

    def _draw_interaction(self, surf, ir: InteractionResult, W, H):
        lines = []
        border = self.CT if getattr(ir,"allowed",True) else self.CR
        if getattr(ir,"allowed",True):
            prods = getattr(ir,"products",[]) or []
            pname = getattr(ir,"product_name",prods[0] if prods else "interaction")
            lines.append((f"✓  {pname}", self.CT))
            eq = getattr(ir,"equation","") or getattr(ir,"eq_text","")
            if eq: lines.append((eq[:65], self.CW))
            desc = getattr(ir,"description","")
            if desc: lines.append((desc[:70], self.CS))
        else:
            law = getattr(ir,"forbidden_law","") or ""
            lines.append((f"⊗  {law[:55]}", self.CR))
            feq = getattr(ir,"forbidden_equation","")
            if feq: lines.append((feq[:65], self.CW))
            desc = getattr(ir,"description","")
            if desc: lines.append((desc[:70], self.CS))
        pw, ph = 520, 20 + len(lines)*18
        px, py = (W-pw)//2, H//2 + 60
        self._panel(surf, px, py, pw, ph, border=border)
        for i, (txt, col) in enumerate(lines):
            r = self.f_mono_sm.render(txt, True, col)
            surf.blit(r, (px+12, py+8+i*18))


# ══════════════════════════════════════════════════════════════
#  GAME STATE
# ══════════════════════════════════════════════════════════════
class GameState:
    def __init__(self, args):
        self.epoch_id  = args.epoch
        self.theory    = args.theory
        self.paused    = False
        self.frame     = 0
        self.t_total   = 0.0
        self.dt        = 1.0 / args.fps
        # Camera
        self.cam_x, self.cam_y, self.cam_z = 0.0, 0.0, -18.0
        self.cam_yaw,  self.cam_pitch = 0.0, 0.0
        self.zoom      = 0.15
        # Physics
        self.g_s    = 0.1
        self.radius = 1.0
        self.clock  = CosmicClock()
        self.clock.set_epoch(args.epoch)
        self.fields = FieldManager()
        self.engine = InteractionEngine(current_epoch=args.epoch)
        # Active epoch object
        self.active_epoch = None
        # Selection
        self.selection: list = []
        self.last_ir:   InteractionResult | None = None
        self.ir_timer:  float = 0.0
        # Stats
        self.n_particles = 0
        # Epoch objects (lazy-init)
        self._ep: dict = {}

    def get_epoch(self, epoch_id: int):
        """Lazy-init and return the epoch object for epoch_id."""
        if epoch_id not in self._ep:
            if   epoch_id == 0: self._ep[0] = PlanckEpoch(self.theory)
            elif epoch_id == 1: self._ep[1] = InflationEpoch(self.clock, self.fields)
            elif epoch_id == 2: self._ep[2] = BaryogenesisEpoch(self.clock)
            elif epoch_id == 3: self._ep[3] = QCDEpoch(self.clock)
            elif epoch_id == 4: self._ep[4] = AxionEpoch(self.clock, self.fields)
            elif epoch_id == 5: self._ep[5] = BBNEpoch(self.clock)
            elif epoch_id == 6: self._ep[6] = RecombinationEpoch(self.clock)
            elif epoch_id == 7: self._ep[7] = StructureEpoch(self.clock)
        return self._ep.get(epoch_id)

    def set_epoch(self, epoch_id: int, fx: FXManager, sound: SoundEngine,
                  narrator: Narrator):
        old = self.epoch_id
        self.epoch_id   = epoch_id
        self.engine.epoch = epoch_id
        self.active_epoch = self.get_epoch(epoch_id)
        self.ep_strings=self._ep.get(0)
        self.ep_inflation=self._ep.get(1)
        self.ep_baryogenesis=self._ep.get(2)
        self.ep_qcd=self._ep.get(3)
        self.ep_axion=self._ep.get(4)
        self.ep_bbn=self._ep.get(5)
        self.ep_recombination=self._ep.get(6)
        self.ep_structure=self._ep.get(7)
        fx.add_epoch_transition(old, epoch_id)
        sound.play_epoch_transition(old, epoch_id)
        narrator.set_epoch(epoch_id)

    def update(self, dt: float):
        self.dt      = dt
        self.t_total += dt
        self.frame   += 1
        if not self.paused:
            self.clock.step(dt)
            ep = self.active_epoch
            if ep and hasattr(ep, "update"):
                ep.update(dt)
            self.ir_timer = max(0.0, self.ir_timer - dt)
            # Auto-transition epoch
            new_ep = self.clock.epoch_id
            if new_ep != self.epoch_id and new_ep <= 7:
                return new_ep  # signal transition
        return None


# ══════════════════════════════════════════════════════════════
#  PARTICLE DRAWING (pygame software fallback)
# ══════════════════════════════════════════════════════════════
def draw_particles_sw(surf, particles, state, W, H, t):
    """Software-rendered particle loop (used when no GPU / for HUD layer)."""
    mx_x = max((abs(getattr(p,"x",1)) for p in particles), default=1.0)
    sc = max(1.0, mx_x)
    half_w = W * 0.42 / sc
    half_h = H * 0.42 / sc
    cx = W//2
    cy = H//2

    for p in particles:
        sx = int(cx + getattr(p,"x",0) * half_w)
        sy = int(cy - getattr(p,"y",0) * half_h)
        if sx < -50 or sx > W+50 or sy < -50 or sy > H+50:
            continue
        r = max(4, int(getattr(p,'radius',0.025) * 60))
        col = getattr(p,'color_rgb',(0.75,0.75,0.85))
        ci  = [min(255, int(c*255)) for c in col]
        estr= getattr(p,'emission_str', 2.0)
        # glow + core
        pygame.draw.circle(surf, tuple(min(255,c+40) for c in ci), (sx,sy), r+4)
        pygame.draw.circle(surf, tuple(ci), (sx,sy), r)
        pygame.draw.circle(surf, (255,255,255), (sx,sy), max(2,r//3))
        # Selection
        if getattr(p,'selected',False):
            pulse = int(160 + 80*math.sin(t*7))
            pygame.draw.circle(surf, (200,190,255,pulse), (sx,sy), r+6, 1)

    # Fusion / forbidden event flashes
    ep = state.active_epoch
    if ep:
        for ev in getattr(ep,'annihilations',[]) + getattr(ep,'forbidden_events',[]):
            x  = getattr(ev,'x',0); y = getattr(ev,'y',0)
            sx = int(cx + x*half_w); sy = int(cy - y*half_h)
            al = int(getattr(ev,'alpha',0)*180)
            if al < 5: continue
            r2 = max(8, int(getattr(ev,"radius",0)*60 + 8))
            if hasattr(ev,'color'):
                cc = [min(255,int(c*255)) for c in ev.color]
                pygame.draw.circle(surf,(*cc,al),(sx,sy),r2,1)
            else:
                pygame.draw.line(surf,(220,50,50,al),(sx-12,sy-12),(sx+12,sy+12),2)
                pygame.draw.line(surf,(220,50,50,al),(sx+12,sy-12),(sx-12,sy+12),2)


# ══════════════════════════════════════════════════════════════
#  INTERACTION HANDLING
# ══════════════════════════════════════════════════════════════
def pick_particle(pos, state, W, H):
    ep = state.active_epoch
    if ep and hasattr(ep, 'field'):
        particles = ep.field
    elif ep and hasattr(ep, 'get_render_particles'):
        particles = ep.get_render_particles()
    else:
        return None
    if not particles:
        return
    mx_x = max((abs(getattr(p,'x',1)) for p in particles), default=1.0)
    sc    = max(1.0, mx_x)
    half_w = W * 0.42 / sc
    half_h = H * 0.42 / sc
    cx = W // 2
    cy = H // 2
    mx, my = pos
    best, bd = None, 9999
    for p in particles:
        sx = cx + getattr(p,'x',0) * half_w
        sy = cy - getattr(p,'y',0) * half_h
        d  = math.hypot(mx-sx, my-sy)
        if d < bd:
            bd = d; best = p
    return best
def handle_click(pos, state, sound, fx, narrator, W, H):
    p = pick_particle(pos, state, W, H)
    print("clicked")
    if p is None:
        for s in state.selection:
            s.selected = False
        state.selection.clear()
        return

    if len(state.selection) == 0:
        p.selected = True
        state.selection.append(p)
        sound.play_string_vibration(
            mode_n=getattr(p,'mode_n',1),
            closed=getattr(p,'closed',False),
            tachyon=getattr(p,'is_tachyon',False),
        )
    elif p not in state.selection:
        b = state.selection[0]
        b.selected = False
        state.selection.clear()
        _interact(b, p, state, sound, fx, narrator)
    else:
        p.selected = False
        state.selection.clear()


def handle_double_click(pos, state, sound, fx, W, H):
    """Double-click: split string (epoch 0) or drag (others)."""
    if state.epoch_id == 0:
        p = pick_particle(pos, state, W, H)
        ep = state.active_epoch
        if p and ep and hasattr(ep,'split_string_at'):
            idx = ep.visuals.index(p) if hasattr(ep,'visuals') and p in ep.visuals else -1
            if idx >= 0:
                ev = ep.split_string_at(idx)
                sound.play_string_split(state.g_s)
                if ev:
                    fx.add_vertex_flash(p.x, p.y, p.color_rgb)
                    fx.add_equation_dissolve(p.x, p.y, ev.eq_text,
                                              ev.product_name, ev.product_color)


def _interact(a, b, state, sound, fx, narrator):
    """Route interaction between two particles through the active epoch."""
    ep = state.active_epoch
    result = None

    if state.epoch_id == 0 and ep and hasattr(ep,'attempt_join'):
        ia = ep.visuals.index(a) if hasattr(ep,'visuals') and a in ep.visuals else -1
        ib = ep.visuals.index(b) if hasattr(ep,'visuals') and b in ep.visuals else -1
        if ia >= 0 and ib >= 0:
            result = ep.attempt_join(ia, ib)
            if result and not getattr(result,'is_forbidden',False):
                sound.play_string_join(state.g_s)
                fx.add_equation_dissolve(
                    (a.x+b.x)/2, (a.y+b.y)/2,
                    result.eq_text, result.product_name, result.product_color,
                )
                if "graviton" in result.product_name:
                    narrator.milestone("first_graviton")
                    sound.play_string_vibration(1, True)
                return
            return
        return
        result = ep.attempt_interaction([a, b])
        if not isinstance(result, InteractionResult):
            result = InteractionResult(
                allowed=result.get("allowed",False) if hasattr(result,"get") else getattr(result,"allowed",False),
                products=result.get("products",[]) if hasattr(result,"get") else getattr(result,"products",[]),
                equation=result.get("equation","") if hasattr(result,"get") else getattr(result,"equation",""),
                forbidden_reason=result.get("reason","") if hasattr(result,"get") else getattr(result,"forbidden_reason",""),
                forbidden_law=result.get("law","") if hasattr(result,"get") else getattr(result,"forbidden_law",""),
                forbidden_equation=result.get("equation","") if hasattr(result,"get") else getattr(result,"forbidden_equation",""),

        )

    elif state.epoch_id == 5 and ep and hasattr(ep,'attempt_interaction'):
        result = ep.attempt_interaction(a, b)

    elif state.epoch_id == 6 and ep and hasattr(ep,'attempt_recombination'):
        result_dict = ep.attempt_recombination(a, b)
        result = InteractionResult(
            allowed=result_dict.get("allowed", False),
            equation=result_dict.get("equation",""),
            description=result_dict.get("description",""),
            forbidden_reason=result_dict.get("reason",""),
            forbidden_equation=result_dict.get("equation",""),
        )

    else:
        from physics.interactions import Particle
        wrapped = [Particle(species=getattr(p,'species','proton'),
                            x=getattr(p,'x',0), y=getattr(p,'y',0),
                            energy_GeV=getattr(p,'mass_GeV',1.0))
                   for p in [a,b]]
        result = state.engine.attempt(wrapped)

    if result is None:
        return

    state.last_ir  = result
    state.ir_timer = 4.5
    mx = (getattr(a,'x',0)+getattr(b,'x',0))/2
    my = (getattr(a,'y',0)+getattr(b,'y',0))/2

    if getattr(result, "allowed", False):
        Q = max(0.1, getattr(result,'energy_released_GeV',0.1)*1000)
        Q = max(0.1, getattr(result,"energy_released_GeV",0.1)*1000)
        prods = getattr(result,"products",[]) or []
        if True:
            eq = getattr(result,"equation","") or getattr(result,"eq_text","")
            if eq:
                prod = getattr(result,"products",[None])[0] if getattr(result,"products",[]) else getattr(result,"product_name","product")
                col = getattr(b,"color_rgb",(0.5,0.8,0.5))
                fx.add_equation_dissolve(mx, my, eq[:55], prod, col)
            fx.add_equation_dissolve(mx, my, eq[:55], prod, col)
            narrator.milestone("first_helium4")
        if "proton" in str(result.products):
            narrator.milestone("first_fusion")
    else:
        rt_map = {"charge":0,"colour":1,"confinement":1,"energy":2,
                  "epoch":3,"baryon":4,"gso":5,"diproton":1}
        law = ""
        rt  = next((v for k,v in rt_map.items() if k in law), 1)
        sound.play_forbidden({0:"charge",1:"confinement",2:"energy",
                               3:"epoch",4:"generic",5:"gso"}.get(rt,"generic"))
        fx.add_forbidden(mx, my, getattr(result,"forbidden_reason","forbidden"),
                         getattr(result,"forbidden_equation",""), getattr(result,"forbidden_law",""), rt)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # ── pygame + OpenGL window ────────────────────────────
    pygame.init()
    W, H = args.width, args.height
    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.HWSURFACE if (MGL_OK and not args.no_gpu) else 0
    try:
        if flags:
            pygame.display.set_mode((0,0), flags|pygame.FULLSCREEN, 24)
            ctx = moderngl.create_context()
            use_gpu = True
        else:
            raise Exception("no gpu")
    except Exception:
        pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        ctx = None
        use_gpu = False

    pygame.display.set_caption("Cosmic Genesis — Interactive Universe Simulator")
    W, H = pygame.display.get_surface().get_size()
    print(f'Screen size: {W}x{H}')
    clock_pg = pygame.time.Clock()

    # ── Systems ───────────────────────────────────────────
    state    = GameState(args)
    sound    = SoundEngine(enabled=not args.no_sound)
    fx       = FXManager()
    narrator = Narrator()
    hud      = HUD(W, H)

    # GPU renderer
    renderer=None
    pbuf=None
    simple_gpu=None
    if use_gpu and ctx:
        try:
            from render.simple_gpu import SimpleGPURenderer
            simple_gpu=SimpleGPURenderer(ctx,W,H)
            print("GPU OK")
        except Exception as e:
            print("GPU failed:",e)
    if use_gpu and ctx:
        try:
            pass #
            pass #renderer disabled
            pbuf     = ParticleBuffer(ctx)
        except Exception as e:
            print(f"[renderer disabled]: {e}")
            use_gpu = False

    # Init first epoch
    state.active_epoch=state.get_epoch(args.epoch)
    state.ep_strings=state._ep.get(0)
    state.ep_inflation=state._ep.get(1)
    state.ep_baryogenesis=state._ep.get(2)
    state.ep_qcd=state._ep.get(3)
    state.ep_axion=state._ep.get(4)
    state.ep_bbn=state._ep.get(5)
    state.ep_recombination=state._ep.get(6)
    state.ep_structure=state._ep.get(7)
    narrator.set_epoch(args.epoch)
    sound.set_epoch(args.epoch)

    # HUD surface (pygame SRCALPHA composited over GL)
    hud_surf = pygame.Surface((W, H), pygame.SRCALPHA)

    # ── Mouse state ───────────────────────────────────────
    mouse_btn   = False
    last_click  = 0.0
    last_cpos   = (0, 0)
    DCLICK_MS   = 280
    mouse_rdown = False
    last_mouse  = (W//2, H//2)

    # ── Main loop ─────────────────────────────────────────
    running = True
    while running:
        dt = clock_pg.tick(args.fps) / 1000.0
        dt = min(dt, 0.05)

        # ── EVENTS ───────────────────────────────────────
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            elif event.type == KEYDOWN:
                k = event.key
                if   k == K_ESCAPE: running = False
                elif k == K_SPACE:  state.paused = not state.paused
                elif k == K_EQUALS or k == K_PLUS:
                    state.clock.time_scale = min(state.clock.time_scale * 10, 1e50)
                elif k == K_MINUS:
                    state.clock.time_scale = max(state.clock.time_scale / 10, 0.01)
                elif k == K_t:      _duality_t(state, sound, fx, narrator)
                elif k == K_s:      _duality_s(state, sound, fx, narrator)
                elif k == K_c:      _shift_cy(state, fx, narrator)
                elif k in (K_0,K_1,K_2,K_3,K_4,K_5,K_6,K_7):
                    ep = k - K_0
                    state.set_epoch(ep, fx, sound, narrator)
                elif k == K_F1:
                    narrator.milestone("first_graviton")  # demo

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    now = pygame.time.get_ticks()
                    dx  = abs(event.pos[0]-last_cpos[0])
                    dy  = abs(event.pos[1]-last_cpos[1])
                    if now - last_click < DCLICK_MS and dx < 20 and dy < 20:
                        handle_double_click(event.pos, state, sound, fx, W, H)
                    else:
                        W2,H2 = pygame.display.get_surface().get_size()
                    handle_click(event.pos, state, sound, fx, narrator, W2, H2)
                    last_click = now; last_cpos = event.pos
                    mouse_btn  = True
                elif event.button == 3:
                    mouse_rdown = True; last_mouse = event.pos
                elif event.button == 4:
                    state.zoom = min(1.0, state.zoom + 0.04)
                elif event.button == 5:
                    state.zoom = max(0.0, state.zoom - 0.04)

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1: mouse_btn = False
                if event.button == 3: mouse_rdown = False

            elif event.type == MOUSEMOTION:
                if mouse_rdown:
                    dx, dy = event.pos[0]-last_mouse[0], event.pos[1]-last_mouse[1]
                    state.cam_yaw   += dx * 0.003
                    state.cam_pitch += dy * 0.002
                    last_mouse = event.pos

        # ── CAMERA (WASD) ─────────────────────────────────
        if not state.paused:
            keys = pygame.key.get_pressed()
            spd  = 0.14
            if keys[K_w] or keys[K_UP]:    state.cam_y += spd
            if keys[K_s] or keys[K_DOWN]:  state.cam_y -= spd
            if keys[K_a] or keys[K_LEFT]:  state.cam_x -= spd
            if keys[K_d] or keys[K_RIGHT]: state.cam_x += spd

        # ── PHYSICS UPDATE ────────────────────────────────
        new_ep = state.update(dt)
        # Only auto-transition if clock moves naturally AND player hasnt set epoch
        if new_ep is not None and new_ep != state.epoch_id and new_ep == state.epoch_id + 1:
            pass  # disabled auto-transition for now

        fx.update(dt)
        narrator.update(dt)

        # Collect particles
        particles = collect_particles(state)
        state.n_particles = len(particles)

        # ── GPU RENDER ────────────────────────────────────
        ep_cfg  = EPOCHS[state.epoch_id]
        bg      = ep_cfg.get("bg_color", (0.01, 0.01, 0.04))

        if simple_gpu:
            try:
                simple_gpu.render(particles,ep_cfg.get("bg_color",(0.01,0.01,0.04)),state.t_total)
            except Exception as e:
                print("GPU render:",e)
        screen = pygame.display.get_surface()
        W, H = screen.get_size()
        if not simple_gpu:
            screen.fill([int(c*255) for c in bg])
        if not simple_gpu:
            draw_particles_sw(screen, particles, state, W, H, state.t_total)

        # ── HUD (pygame over GL) ──────────────────────────
        hud_surf.fill((0,0,0,0))
        hud.draw(hud_surf, state, narrator, fx)

        screen.blit(hud_surf, (0,0))
        pygame.display.flip()

        # Ambient sound update (once per second)
        if state.frame % max(1, args.fps) == 0:
            T_MeV = state.clock.T_GeV * 1000
            sound.update_ambient(state.epoch_id, T_MeV)

    sound.stop_all()
    pygame.quit()


# ── Duality helpers ──────────────────────────────────────────
def _duality_t(state, sound, fx, narrator):
    if state.epoch_id not in (0,1): return
    state.radius = max(0.01, 1.0 / state.radius)
    ep = state.active_epoch
    if ep and hasattr(ep,'apply_t_duality'): ep.apply_t_duality()
    sound.play_string_vibration(1, True)
    fx.add_equation_dissolve(0, 0, f"T-duality: R → α′/R = {state.radius:.2f} ℓₛ",
                              "T-duality", (0.2, 0.85, 0.6))
    narrator.milestone("t_duality")
    state.last_ir = InteractionResult(allowed=True,
        equation=f"R → α′/R = {state.radius:.2f} ℓₛ",
        description="Winding ↔ momentum modes exchanged. Physics invariant.")
    state.ir_timer = 3.5

def _duality_s(state, sound, fx, narrator):
    if state.epoch_id not in (0,1): return
    state.g_s = 1.0 / state.g_s
    ep = state.active_epoch
    if ep and hasattr(ep,'apply_s_duality'): ep.apply_s_duality()
    sound.play_string_vibration(2, True)
    fx.add_equation_dissolve(0, 0, f"S-duality: gₛ → 1/gₛ = {state.g_s:.3f}",
                              "S-duality", (0.95, 0.62, 0.12))
    narrator.milestone("s_duality")
    state.last_ir = InteractionResult(allowed=True,
        equation=f"gₛ → 1/gₛ = {state.g_s:.3f}",
        description="Strong ↔ weak coupling. D-branes become light.")
    state.ir_timer = 3.5

def _shift_cy(state, fx, narrator):
    import random
    ep = state.active_epoch
    if ep and hasattr(ep,'shift_cy_moduli'):
        ep.shift_cy_moduli()
        cy = ep.cy
        state.last_ir = InteractionResult(allowed=True,
            equation=f"CY shift: χ={cy.euler}, h¹¹={cy.h11}, h²¹={cy.h21}",
            description="Calabi-Yau moduli deformed. New particle spectrum in 4D.")
        state.ir_timer = 3.0
        fx.add_equation_dissolve(0, 0,
            f"χ={cy.euler}, h¹¹={cy.h11}, h²¹={cy.h21}",
            "CY vacuum shift", (0.95, 0.80, 0.20))


if __name__ == "__main__":
    main()
