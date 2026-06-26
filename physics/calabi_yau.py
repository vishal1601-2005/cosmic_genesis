from __future__ import annotations
import math, numpy as np
try:
    from config import CY_PRESETS, CURRENT_CY
except ImportError:
    CY_PRESETS = {"quintic": dict(h11=1,h21=101,euler=-200,label="Quintic",vacua_exp=274)}
    CURRENT_CY = "quintic"
class CalabiYauMetric:
    def __init__(self, preset=CURRENT_CY, device=None, grid_res=64):
        self.preset = CY_PRESETS.get(preset, list(CY_PRESETS.values())[0])
        self.grid_res = grid_res
        self._morph = 0.0
        self._target_morph = 0.0
    def texture(self):
        H = W = self.grid_res
        tex = np.zeros((H,W,4), dtype=np.float32)
        u = np.linspace(-0.8,0.8,W)
        v = np.linspace(-0.8,0.8,H)
        gv,gu = np.meshgrid(v,u,indexing="ij")
        z1 = gu+1j*gv; z2 = gv*0.7+1j*gu*0.7
        rhs = -(z1**5+z2**5+1.0)
        z3 = np.abs(rhs).clip(1e-8)**0.2*np.exp(1j*np.angle(rhs)/5.0)
        K = (np.abs(z1)**2+np.abs(z2)**2+np.abs(z3)**2).astype(np.float32)
        Kn = (K-K.min())/(K.max()-K.min()+1e-8)
        tex[:,:,0]=Kn; tex[:,:,3]=Kn
        return tex
    def shift_moduli(self,h11,h21): self._morph=float(np.random.uniform(-1,1))
    def set_morph_target(self,t): self._target_morph=float(t)
    def update_morph(self,dt=0.016): self._morph+=(self._target_morph-self._morph)*min(1.0,dt*2.0)
    def hodge_numbers(self): return self.preset["h11"],self.preset["h21"]
    def euler_characteristic(self):
        h11,h21=self.hodge_numbers(); return 2*(h11-h21)
