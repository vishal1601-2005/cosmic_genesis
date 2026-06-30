"""
render/particles_render.py — GPU particle instance buffer manager.

Manages the upload of all active particles to the GPU instance buffer
each frame. Handles:

  - Collecting particles from the active epoch
  - Sorting by depth (back-to-front for alpha blending)
  - Packing into a flat float32 buffer
  - Uploading to the ModernGL VBO
  - LOD: distant particles get smaller radius

Also manages dynamic lights: extracts the N_LIGHTS brightest emissive
particles and uploads their positions + colours as uniforms for the
deferred lighting pass.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional

try:
    import moderngl
    MGL_OK = True
except ImportError:
    MGL_OK = False

from config import MAX_PARTICLES

# Per-instance layout (must match gbuffer.vert):
# inst_world_pos (3f) + inst_color (3f) + inst_emission (3f) +
# inst_emission_str (1f) + inst_radius (1f) + inst_metallic (1f) +
# inst_roughness (1f) + inst_age (1f) + inst_type (1i) = 15f + 1i
FLOATS_PER_INSTANCE = 15
BYTES_PER_INSTANCE  = FLOATS_PER_INSTANCE * 4 + 4   # 15 floats + 1 int

N_LIGHTS = 32   # max dynamic lights for deferred pass


class ParticleBuffer:
    """
    Manages a pre-allocated GPU buffer for instanced particle rendering.
    Upload once per frame with the current particle list.
    """

    def __init__(self, ctx: "moderngl.Context"):
        self.ctx = ctx
        self._buf = ctx.buffer(reserve=MAX_PARTICLES * BYTES_PER_INSTANCE)
        self._n   = 0
        # Light extraction output
        self.light_positions: list[tuple] = []
        self.light_colors:    list[tuple] = []
        self.light_strengths: list[float] = []

    def upload(self, particles: list, cam_x: float = 0, cam_y: float = 0,
               cam_z: float = -20, zoom: float = 0.2):
        """
        Pack particle list into GPU buffer and upload.

        particles: any list of objects with attributes:
            x, y, z, color_rgb, emission_rgb, emission_str,
            radius, metallic, roughness, age_s, particle_type_int
        """
        N = min(len(particles), MAX_PARTICLES)
        if N == 0:
            self._n = 0
            return

        # Sort back-to-front by depth from camera (for alpha blending)
        def depth(p):
            dx = getattr(p,'x',0) - cam_x
            dy = getattr(p,'y',0) - cam_y
            dz = getattr(p,'z',0) - cam_z
            return -(dx*dx + dy*dy + dz*dz)
        try:
            particles_sorted = sorted(particles[:N], key=depth)
        except Exception:
            particles_sorted = particles[:N]

        # Pack into numpy array
        data      = np.zeros((N, FLOATS_PER_INSTANCE), dtype=np.float32)
        type_data = np.zeros((N, 1), dtype=np.int32)

        # LOD scale: particles far from camera get smaller radius
        lod_near = 0.8
        lod_far  = 3.0

        # Light extraction: collect brightest
        light_candidates = []

        for i, p in enumerate(particles_sorted):
            x   = float(getattr(p, 'x',   0.0))
            y   = float(getattr(p, 'y',   0.0))
            z   = float(getattr(p, 'z',   0.0))
            col = getattr(p, 'color_rgb',   (0.8, 0.8, 0.8))
            emi = getattr(p, 'emission_rgb', col)
            estr= float(getattr(p, 'emission_str', 2.0))
            r   = float(getattr(p, 'radius',        0.012))
            met = float(getattr(p, 'metallic',       0.0))
            rou = float(getattr(p, 'roughness',      0.5))
            age = float(getattr(p, 'age_s',          0.0))
            ptype = int(getattr(p, 'particle_type_int', 0))

            # LOD: scale radius by zoom
            r_lod = r * (lod_near + zoom * (lod_far - lod_near))

            # Clamp colour values
            col  = tuple(max(0.0, min(1.0, float(c))) for c in col[:3])
            emi  = tuple(max(0.0, float(c)) for c in emi[:3])  # HDR: no clamp on emission

            data[i, 0:3]  = [x, y, z]
            data[i, 3:6]  = col
            data[i, 6:9]  = emi
            data[i, 9]    = estr
            data[i, 10]   = r_lod
            data[i, 11]   = met
            data[i, 12]   = rou
            data[i, 13]   = age
            data[i, 14]   = 0.0  # padding
            type_data[i, 0] = ptype

            # Collect for light extraction
            if estr > 2.5:
                dx = x - cam_x; dy = y - cam_y; dz = z - cam_z
                dist2 = dx*dx + dy*dy + dz*dz
                light_candidates.append((estr / (dist2 + 0.1), x, y, z, emi, estr))

        # Interleave float + int (reinterpret int as float for upload)
        type_as_float = type_data.view(np.float32)
        combined = np.hstack([data, type_as_float])

        self._buf.write(combined.astype(np.float32).tobytes())
        self._n = N

        # Extract top N_LIGHTS lights
        light_candidates.sort(reverse=True, key=lambda c: c[0])
        self.light_positions = [(c[1], c[2], c[3]) for c in light_candidates[:N_LIGHTS]]
        self.light_colors    = [c[4]               for c in light_candidates[:N_LIGHTS]]
        self.light_strengths = [c[5]               for c in light_candidates[:N_LIGHTS]]

    @property
    def buffer(self) -> "moderngl.Buffer":
        return self._buf

    @property
    def n_particles(self) -> int:
        return self._n

    def upload_lights_to_program(self, prog: "moderngl.Program"):
        """Upload extracted lights to a shader program's uniforms."""
        n = min(len(self.light_positions), N_LIGHTS)
        try:
            prog["u_n_lights"] = n
            if n > 0:
                flat_pos = []
                for lp in self.light_positions[:n]:
                    flat_pos.extend([float(v) for v in lp])
                prog["u_light_pos"].write(np.array(flat_pos, dtype=np.float32).tobytes())

                flat_col = []
                for lc in self.light_colors[:n]:
                    flat_col.extend([float(v) for v in lc[:3]])
                prog["u_light_col"].write(np.array(flat_col, dtype=np.float32).tobytes())

                prog["u_light_str"].write(
                    np.array(self.light_strengths[:n], dtype=np.float32).tobytes()
                )
        except Exception:
            pass  # uniform may not exist in all programs


def collect_particles(state) -> list:
    """
    Collect all renderable particles from the current game state.
    Dispatches to the active epoch's get_render_particles().
    """
    epoch_id = getattr(state, "epoch_id", 0)

    # Try active epoch object
    epoch_attr = {
        0: "ep_strings",
        1: "ep_inflation",
        2: "ep_baryogenesis",
        3: "ep_qcd",
        4: "ep_axion",
        5: "ep_bbn",
        6: "ep_recombination",
        7: "ep_structure",
    }.get(epoch_id)

    if epoch_attr:
        ep = getattr(state, epoch_attr, None)
        if ep is not None:
            if hasattr(ep, "get_render_particles"):
                return ep.get_render_particles()
            if hasattr(ep, "field"):
                return ep.field

    # Fallback: check bbn directly
    if hasattr(state, "bbn") and state.bbn:
        return state.bbn.field

    return []
