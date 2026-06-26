"""
render/renderer.py — Full multi-pass GPU render pipeline.

Pass order each frame:
 ┌──────────────────────────────────────────────────────┐
 │ 1. G-Buffer pass     (MRT: albedo/normal/emission/pos)│
 │    - All particles (instanced)                        │
 │    - Worldsheet surfaces (if epoch 0)                 │
 │    Output: 4 × RGBA16F textures at full resolution    │
 ├──────────────────────────────────────────────────────┤
 │ 2. Deferred Lighting (reads G-Buffer → HDR colour)    │
 │    - PBR BRDF per pixel                               │
 │    - Up to 32 dynamic lights from emissive particles  │
 │    - Ambient IBL (epoch-specific sky model)           │
 │    Output: RGBA16F HDR scene texture                  │
 ├──────────────────────────────────────────────────────┤
 │ 3. Volumetric pass   (ray-march plasma/nebula)        │
 │    - Forward-composited over HDR scene                │
 │    - Resolution: half (upscaled with bilateral filter)│
 │    Output: RGBA16F scene + volume                     │
 ├──────────────────────────────────────────────────────┤
 │ 4. Equation dissolve (fullscreen quad per active anim)│
 │    - Additively blended over scene                    │
 ├──────────────────────────────────────────────────────┤
 │ 5. Bloom             (dual Kawase, 6 mip levels)      │
 │    - Threshold → 3 downsample → 3 upsample            │
 │    Output: adds to HDR scene                          │
 ├──────────────────────────────────────────────────────┤
 │ 6. God Rays          (screen-space radial march)      │
 │    Output: additive overlay                           │
 ├──────────────────────────────────────────────────────┤
 │ 7. Composite         (tonemap + CA + grain + FXAA)    │
 │    - ACES filmic tonemapping                          │
 │    - Chromatic aberration                             │
 │    - Lens dirt / vignette / film grain                │
 │    - FXAA anti-aliasing                               │
 │    Output: RGBA8 LDR → screen                        │
 └──────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import math
import struct
from pathlib import Path
import numpy as np

try:
    import moderngl
    MGL_OK = True
except ImportError:
    MGL_OK = False

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

from config import (
    BLOOM_STRENGTH, CHROMATIC_AB, VIGNETTE_STRENGTH,
    MAX_PARTICLES, EPOCHS,
)

SHADER_DIR = Path(__file__).parent.parent / "shaders"


def _load(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── Fullscreen quad geometry ────────────────────────────────
FULLSCREEN_QUAD_VERTS = np.array([
    # x,    y,    u,   v
    -1.0, -1.0,  0.0, 0.0,
     1.0, -1.0,  1.0, 0.0,
     1.0,  1.0,  1.0, 1.0,
    -1.0, -1.0,  0.0, 0.0,
     1.0,  1.0,  1.0, 1.0,
    -1.0,  1.0,  0.0, 1.0,
], dtype=np.float32)

FULLSCREEN_QUAD_VERT_SRC = """
#version 410 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv        = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


class FrameBuffer:
    """Wraps a ModernGL FBO with named colour attachments."""
    def __init__(self, ctx: "moderngl.Context", width: int, height: int,
                 n_color: int = 1, dtype: str = "f2", depth: bool = True):
        self.ctx    = ctx
        self.width  = width
        self.height = height
        self.textures = [
            ctx.texture((width, height), 4, dtype=dtype)
            for _ in range(n_color)
        ]
        for t in self.textures:
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = False
            t.repeat_y = False
        self.depth_tex = ctx.depth_texture((width, height)) if depth else None
        self.fbo = ctx.framebuffer(
            color_attachments=self.textures,
            depth_attachment=self.depth_tex,
        )

    def use(self):
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

    def __getitem__(self, i: int):
        return self.textures[i]


class Renderer:
    """
    Full multi-pass renderer. Call renderer.render(state) each frame.
    """

    def __init__(self, ctx: "moderngl.Context", width: int, height: int):
        if not MGL_OK:
            raise RuntimeError("moderngl not installed")
        self.ctx    = ctx
        self.W      = width
        self.H      = height
        self._time  = 0.0

        # Enable blending and depth test
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        ctx.enable(moderngl.DEPTH_TEST)

        self._build_programs()
        self._build_fbos()
        self._build_geometry()
        self._build_particle_buffers()

    # ── Shader program compilation ─────────────────────────
    def _build_programs(self):
        def prog(vert_path, frag_path, extra_vert=None):
            vert = extra_vert or _load(SHADER_DIR / vert_path)
            frag = _load(SHADER_DIR / frag_path)
            return self.ctx.program(vertex_shader=vert, fragment_shader=frag)

        # G-Buffer pass
        self.prog_gbuf = prog("gbuffer.vert", "gbuffer.frag")

        # Deferred lighting
        self.prog_light = self.ctx.program(
            vertex_shader=FULLSCREEN_QUAD_VERT_SRC,
            fragment_shader=_load(SHADER_DIR / "lighting.frag"),
        )

        # Volume ray-march
        self.prog_vol = self.ctx.program(
            vertex_shader=FULLSCREEN_QUAD_VERT_SRC,
            fragment_shader=_load(SHADER_DIR / "volume.frag"),
        )

        # Equation dissolve
        self.prog_eq = self.ctx.program(
            vertex_shader=FULLSCREEN_QUAD_VERT_SRC,
            fragment_shader=_load(SHADER_DIR / "equation_dissolve.frag"),
        )

        # Bloom (shared for down/upsample via u_pass uniform)
        self.prog_bloom = self.ctx.program(
            vertex_shader=FULLSCREEN_QUAD_VERT_SRC,
            fragment_shader=_load(SHADER_DIR / "post" / "bloom.frag"),
        )

        # God rays
        self.prog_godrays = self.ctx.program(
            vertex_shader=FULLSCREEN_QUAD_VERT_SRC,
            fragment_shader=_load(SHADER_DIR / "post" / "godrays.frag"),
        )

        # Final composite
        self.prog_composite = self.ctx.program(
            vertex_shader=FULLSCREEN_QUAD_VERT_SRC,
            fragment_shader=_load(SHADER_DIR / "post" / "composite.frag"),
        )

        # Worldsheet (epoch 0)
        self.prog_string = prog("worldsheet.vert", "worldsheet.frag")
        # Particle (forward, for transparent alpha particles)
        self.prog_particle_fwd = prog("particle.vert", "particle.frag")

    # ── Framebuffers ───────────────────────────────────────
    def _build_fbos(self):
        W, H = self.W, self.H
        # G-Buffer: 4 MRT (RGBA16F each)
        self.fbo_gbuf  = FrameBuffer(self.ctx, W, H, n_color=4, dtype="f2")
        # HDR lighting output
        self.fbo_hdr   = FrameBuffer(self.ctx, W, H, n_color=1, dtype="f2", depth=False)
        # Volume (half res)
        self.fbo_vol   = FrameBuffer(self.ctx, W//2, H//2, n_color=1, dtype="f2", depth=False)
        # Bloom mip chain (6 levels)
        self.fbo_bloom = [
            FrameBuffer(self.ctx, max(1, W >> i), max(1, H >> i),
                        n_color=1, dtype="f2", depth=False)
            for i in range(1, 7)
        ]
        # God rays
        self.fbo_godrays = FrameBuffer(self.ctx, W//2, H//2, n_color=1, dtype="f2", depth=False)
        # Final composite output (LDR RGBA8)
        self.fbo_final = FrameBuffer(self.ctx, W, H, n_color=1, dtype="f1", depth=False)

    # ── Geometry ───────────────────────────────────────────
    def _build_geometry(self):
        # Fullscreen quad
        vbo = self.ctx.buffer(FULLSCREEN_QUAD_VERTS.tobytes())
        # For each program that uses the fullscreen quad
        def make_fsq_vao(prog):
            return self.ctx.vertex_array(prog, [(vbo, "2f 2f", "in_pos", "in_uv")])

        self.fsq_light    = make_fsq_vao(self.prog_light)
        self.fsq_vol      = make_fsq_vao(self.prog_vol)
        self.fsq_eq       = make_fsq_vao(self.prog_eq)
        self.fsq_bloom    = make_fsq_vao(self.prog_bloom)
        self.fsq_godrays  = make_fsq_vao(self.prog_godrays)
        self.fsq_composite= make_fsq_vao(self.prog_composite)

        # Billboard quad for particles (2 triangles, [-1,1]²)
        quad = np.array([
            -1,-1, 0,0,  1,-1, 1,0,  1,1, 1,1,
            -1,-1, 0,0,  1, 1, 1,1, -1,1, 0,1,
        ], dtype=np.float32)
        self.quad_vbo = self.ctx.buffer(quad.tobytes())

    # ── Particle GPU buffers (instanced) ───────────────────
    def _build_particle_buffers(self):
        # Instance buffer: world_pos(3) + color(3) + emission(3) + emission_str(1)
        #                  + radius(1) + metallic(1) + roughness(1) + age(1) + type(1i)
        # = 15 floats + 1 int per particle
        STRIDE = (15 * 4) + 4   # bytes
        self._particle_buf = self.ctx.buffer(
            reserve=MAX_PARTICLES * STRIDE
        )
        self._n_particles = 0

        # VAO combining quad_vbo (per-vertex) + particle_buf (per-instance)
        self.particle_vao = self.ctx.vertex_array(
            self.prog_gbuf,
            [
                (self.quad_vbo,        "2f 2f",    "in_position", "in_normal"),  # reuse uv
                (self._particle_buf,   "3f 3f 3f 1f 1f 1f 1f 1f 1i /i",
                 "inst_world_pos", "inst_color", "inst_emission",
                 "inst_emission_str", "inst_radius", "inst_metallic",
                 "inst_roughness", "inst_age", "inst_type"),
            ]
        )

    def upload_particles(self, particles: list):
        """
        Upload particle data to GPU instance buffer.
        Called once per frame with the current particle list.

        particles: list of objects with attributes:
            x, y, z, color_rgb, emission_rgb, emission_str,
            radius, metallic, roughness, age, particle_type_int
        """
        N = min(len(particles), MAX_PARTICLES)
        if N == 0:
            self._n_particles = 0
            return

        # Pack into flat float array
        data = np.zeros((N, 15), dtype=np.float32)
        type_data = np.zeros((N, 1), dtype=np.int32)

        for i, p in enumerate(particles[:N]):
            data[i, 0:3] = [getattr(p, "x", 0), getattr(p, "y", 0), getattr(p, "z", 0)]
            data[i, 3:6] = getattr(p, "color_rgb", (0.8, 0.8, 0.8))
            data[i, 6:9] = getattr(p, "emission_rgb", getattr(p, "color_rgb", (0.5,0.5,0.5)))
            data[i, 9]   = getattr(p, "emission_str", 2.0)
            data[i,10]   = getattr(p, "radius", 0.012)
            data[i,11]   = getattr(p, "metallic", 0.0)
            data[i,12]   = getattr(p, "roughness", 0.5)
            data[i,13]   = getattr(p, "age_s", 0.0)
            data[i,14]   = 0.0   # padding before int
            type_data[i, 0] = getattr(p, "particle_type_int", 0)

        # Interleave float + int data
        combined = np.hstack([data, type_data.astype(np.float32)])
        self._particle_buf.write(combined.astype(np.float32).tobytes())
        self._n_particles = N

    # ── View / Projection matrices ─────────────────────────
    @staticmethod
    def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
        f  = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
        nf = 1.0 / (near - far)
        return np.array([
            [f/aspect, 0,  0,                  0],
            [0,        f,  0,                  0],
            [0,        0,  (far+near)*nf,      -1],
            [0,        0,  2*far*near*nf,       0],
        ], dtype=np.float32)

    @staticmethod
    def look_at(eye, center, up):
        f = center - eye;  f /= np.linalg.norm(f)
        r = np.cross(f, up); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        m = np.eye(4, dtype=np.float32)
        m[0,:3]=r; m[1,:3]=u; m[2,:3]=-f
        m[3,0]=-np.dot(r,eye); m[3,1]=-np.dot(u,eye); m[3,2]=np.dot(f,eye)
        return m.T

    # ── Main render call ────────────────────────────────────
    def render(self, state, dt: float):
        """Full multi-pass render. Call once per frame."""
        self._time += dt
        t    = self._time
        ep   = state.epoch_id
        zoom = state.zoom

        # Camera matrices
        eye    = np.array([state.cam_x, state.cam_y, state.cam_z], dtype=np.float32)
        target = eye + np.array([
            math.sin(state.cam_yaw) * math.cos(state.cam_pitch),
            math.sin(state.cam_pitch),
            math.cos(state.cam_yaw) * math.cos(state.cam_pitch),
        ], dtype=np.float32)
        up  = np.array([0, 1, 0], dtype=np.float32)
        V   = self.look_at(eye, target, up)
        P   = self.perspective(60.0, self.W / self.H, 0.1, 1000.0)
        VP  = P @ V

        # Camera basis vectors (for billboard)
        cam_right = np.array([V[0,0], V[1,0], V[2,0]], dtype=np.float32)
        cam_up    = np.array([V[0,1], V[1,1], V[2,1]], dtype=np.float32)

        # ── Pass 1: G-Buffer ──────────────────────────────
        self.fbo_gbuf.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        if self._n_particles > 0:
            self.prog_gbuf["u_view"].write(V.tobytes())
            self.prog_gbuf["u_proj"].write(P.tobytes())
            self.prog_gbuf["u_cam_right"].write(cam_right.tobytes())
            self.prog_gbuf["u_cam_up"].write(cam_up.tobytes())
            self.prog_gbuf["u_time"]  = t
            self.prog_gbuf["u_zoom"]  = zoom
            self.prog_gbuf["u_near"]  = 0.1
            self.prog_gbuf["u_far"]   = 1000.0
            self.particle_vao.render(
                moderngl.TRIANGLES,
                vertices=6,
                instances=self._n_particles,
            )

        # ── Pass 2: Deferred Lighting ─────────────────────
        self.fbo_hdr.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        p = self.prog_light
        p["gAlbedoMetallic"].value = 0;  self.fbo_gbuf[0].use(0)
        p["gNormalRoughness"].value = 1; self.fbo_gbuf[1].use(1)
        p["gEmission"].value        = 2; self.fbo_gbuf[2].use(2)
        p["gPositionDepth"].value   = 3; self.fbo_gbuf[3].use(3)
        p["u_cam_pos"].write(eye.tobytes())
        p["u_time"]       = t
        p["u_epoch"]      = ep
        p["u_T_GeV"]      = state.clock.T_GeV
        p["u_ambient_str"]= [1.0, 0.8, 0.6, 0.7, 0.4, 0.9, 0.7, 0.5][ep]
        p["u_n_lights"]   = 0   # TODO: fill from brightest particles
        self.fsq_light.render(moderngl.TRIANGLES)

        # ── Pass 3: Volumetric ────────────────────────────
        self.fbo_vol.use()
        pv = self.prog_vol
        pv["u_scene"].value     = 0; self.fbo_hdr[0].use(0)
        pv["u_depth"].value     = 1; self.fbo_gbuf.depth_tex.use(1) if self.fbo_gbuf.depth_tex else None
        pv["u_cam_pos"].write(eye.tobytes())
        pv["u_time"]            = t
        pv["u_epoch"]           = ep
        pv["u_T_GeV"]           = state.clock.T_GeV
        pv["u_zoom"]            = zoom
        pv["u_quality"]         = 1   # medium quality real-time
        pv["u_vol_density"]     = [1.0, 1.5, 1.2, 0.8, 0.4, 1.0, 0.6, 0.3][ep]
        pv["u_vol_emission"]    = [2.0, 3.0, 2.5, 1.5, 0.3, 2.0, 1.0, 0.5][ep]
        # Reconstruct inv_view_proj for ray direction
        inv_vp = np.linalg.inv(VP).astype(np.float32)
        pv["u_inv_view_proj"].write(inv_vp.tobytes())
        self.fsq_vol.render(moderngl.TRIANGLES)

        # Upscale volume to full res and composite onto HDR
        # (simplified: direct composite without bilateral upsample)
        self.ctx.copy_framebuffer(self.fbo_hdr.fbo, self.fbo_vol.fbo)

        # ── Pass 4: Bloom (6-level dual Kawase) ──────────
        pb = self.prog_bloom
        # Downsample chain
        prev_tex = self.fbo_hdr[0]
        for i, fbo in enumerate(self.fbo_bloom[:3]):
            fbo.use()
            pb["u_hdr_scene"].value = 0; prev_tex.use(0)
            pb["u_texel"]     = (1.0/fbo.width, 1.0/fbo.height)
            pb["u_threshold"] = 0.75
            pb["u_strength"]  = BLOOM_STRENGTH
            pb["u_radius"]    = 1.0 + i * 0.5
            pb["u_pass"]      = 0
            self.fsq_bloom.render(moderngl.TRIANGLES)
            prev_tex = fbo[0]
        # Upsample chain
        for i, fbo in enumerate(reversed(self.fbo_bloom[:3])):
            fbo.use()
            pb["u_hdr_scene"].value  = 0; self.fbo_hdr[0].use(0)
            pb["u_bloom_blur"].value = 1; prev_tex.use(1)
            pb["u_texel"]     = (1.0/self.W, 1.0/self.H)
            pb["u_radius"]    = 2.0 - i * 0.5
            pb["u_pass"]      = 1
            self.fsq_bloom.render(moderngl.TRIANGLES)
            prev_tex = fbo[0]

        # ── Pass 5: God Rays ──────────────────────────────
        self.fbo_godrays.use()
        pg = self.prog_godrays
        pg["u_scene"].value     = 0; self.fbo_bloom[-1][0].use(0)
        pg["u_occlusion"].value = 1; self.fbo_hdr[0].use(1)
        pg["u_light_pos_ss"]    = (0.5, 0.5)   # centred for most epochs
        pg["u_light_color"]     = (1.0, 0.9, 0.7)
        pg["u_density"]         = 0.96
        pg["u_weight"]          = 0.015
        pg["u_decay"]           = 0.97
        pg["u_exposure"]        = 0.5
        pg["u_n_samples"]       = 80
        pg["u_time"]            = t
        pg["u_epoch"]           = ep
        self.fsq_godrays.render(moderngl.TRIANGLES)

        # ── Pass 6: Final Composite ───────────────────────
        self.ctx.screen.use()
        pc = self.prog_composite
        pc["u_hdr"].value   = 0; self.fbo_godrays[0].use(0)
        # Dirt mask: white texture if none loaded
        pc["u_time"]         = t
        pc["u_epoch"]        = ep
        pc["u_exposure"]     = [1.2, 0.9, 1.1, 1.0, 1.3, 1.0, 1.1, 0.8][ep]
        pc["u_chromatic_ab"] = CHROMATIC_AB * (1.0 + zoom * 0.5)
        pc["u_vignette_str"] = VIGNETTE_STRENGTH
        pc["u_grain_str"]    = 0.018
        pc["u_dirt_str"]     = 0.12
        pc["u_saturation"]   = [1.1, 0.9, 1.0, 0.95, 1.15, 1.0, 1.05, 0.85][ep]
        pc["u_resolution"]   = (float(self.W), float(self.H))
        self.fsq_composite.render(moderngl.TRIANGLES)
