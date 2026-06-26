from __future__ import annotations
import math
from typing import NamedTuple
import numpy as np
try:
    import jax; JAX_OK = True
except ImportError:
    JAX_OK = False
from config import N_SIGMA, N_MODES, DT_WORLDSHEET, D_TARGET, D_LARGE, G_S_DEFAULT, ALPHA_PRIME

class StringState(NamedTuple):
    X: object
    X_dot: object
    psi_p: object
    psi_m: object
    modes: object
    mass_sq: object
    phase: object
    g_s: float

def init_state(n_strings, theory="super", g_s=G_S_DEFAULT, key=None, alpha_prime=ALPHA_PRIME):
    S, M, D, N = N_SIGMA, N_MODES, D_TARGET, n_strings
    rng = np.random.default_rng(42)
    sigma = np.linspace(0, math.pi, S)
    x0 = rng.uniform(-10, 10, (N, D)).astype(np.float32)
    x0[:, D_LARGE:] *= 0.01
    p = rng.normal(0, 0.05, (N, D)).astype(np.float32)
    p[:, D_LARGE:] = 0
    n_vec = np.arange(1, M+1, dtype=np.float32)
    alpha_n = rng.normal(0, 0.3, (N, D, M)).astype(np.float32) / n_vec[None,None,:]
    cos_ns = np.cos(np.outer(n_vec, sigma)).astype(np.float32)
    X = x0[:,:,None] + math.sqrt(2*alpha_prime)*np.einsum("ndm,ms->nds", alpha_n, cos_ns)
    sin_ns = (n_vec[:,None]*np.sin(np.outer(n_vec, sigma))).astype(np.float32)
    X_dot = 2*alpha_prime*p[:,:,None] - math.sqrt(2*alpha_prime)*np.einsum("ndm,ms->nds", alpha_n, sin_ns)
    psi_z = np.zeros((N,D,S), dtype=np.float32)
    psi_p = psi_z if theory=="bosonic" else rng.normal(0,0.1,(N,D,S)).astype(np.float32)
    psi_m = psi_z if theory=="bosonic" else rng.normal(0,0.1,(N,D,S)).astype(np.float32)
    modes = (alpha_n[:,0,:]+1j*alpha_n[:,1,:]).astype(np.complex64)
    mass_sq = np.zeros(N, dtype=np.float32)
    phase = rng.uniform(0, 2*math.pi, N).astype(np.float32)
    return StringState(X=X, X_dot=X_dot, psi_p=psi_p, psi_m=psi_m,
                       modes=modes, mass_sq=mass_sq, phase=phase, g_s=g_s)

def make_step_fn(theory="super", alpha_prime=ALPHA_PRIME):
    dt = DT_WORLDSHEET
    def step_fn(state):
        X, Xd = state.X, state.X_dot
        ds = math.pi / (X.shape[-1]-1)
        Xpad = np.concatenate([X[:,:,1:2], X, X[:,:,-2:-1]], axis=-1)
        Xpp = (Xpad[:,:,2:] - 2*Xpad[:,:,1:-1] + Xpad[:,:,:-2]) / ds**2
        Xd2 = Xd + dt*Xpp
        X2  = X  + dt*Xd2
        shift = max(1, int(dt/ds))
        pp, pm = state.psi_p, state.psi_m
        if theory != "bosonic":
            pp2 = np.roll(pp, -shift, axis=-1)
            pm2 = np.roll(pm,  shift, axis=-1)
            pp2[:,:,0]  = pm2[:,:,0]
            pp2[:,:,-1] = -pm2[:,:,-1]
        else:
            pp2, pm2 = pp, pm
        n_vec  = np.arange(1, N_MODES+1, dtype=np.float32)
        X_fft  = np.fft.rfft(X2, axis=-1)
        coeffs = X_fft[:,:,1:N_MODES+1]
        modes2 = (coeffs[:,0,:]+1j*coeffs[:,1,:])*n_vec/math.sqrt(2*alpha_prime)
        N_osc  = np.sum(n_vec*np.abs(modes2)**2, axis=-1)
        a      = 0.0 if theory=="super" else 1.0
        msq2   = (N_osc - a) / alpha_prime
        return StringState(X=X2, X_dot=Xd2, psi_p=pp2, psi_m=pm2,
                           modes=modes2.astype(np.complex64),
                           mass_sq=msq2.astype(np.float32),
                           phase=state.phase, g_s=state.g_s)
    return step_fn

def split_string(state, idx, key=None):
    N, D, S = state.X.shape
    half = S//2
    def rs(arr, s, e):
        seg = arr[:,s:e]
        return np.array([np.interp(np.linspace(0,seg.shape[1]-1,S),
                         np.arange(seg.shape[1]),seg[d]) for d in range(D)])
    c1 = rs(state.X[idx],0,half)
    c2 = rs(state.X[idx],half,S)
    X2  = np.concatenate([state.X[:idx],  c1[None], state.X[idx+1:],  c2[None]],  axis=0)
    Xd2 = np.concatenate([state.X_dot[:idx], rs(state.X_dot[idx],0,half)[None],
                           state.X_dot[idx+1:], rs(state.X_dot[idx],half,S)[None]], axis=0)
    pp2 = np.concatenate([state.psi_p[:idx], rs(state.psi_p[idx],0,half)[None],
                           state.psi_p[idx+1:], rs(state.psi_p[idx],half,S)[None]], axis=0)
    pm2 = np.concatenate([state.psi_m[:idx], rs(state.psi_m[idx],0,half)[None],
                           state.psi_m[idx+1:], rs(state.psi_m[idx],half,S)[None]], axis=0)
    ph2 = np.concatenate([state.phase[:idx],[state.phase[idx]+.5],
                           state.phase[idx+1:],[state.phase[idx]-.5]])
    nN  = X2.shape[0]
    return (StringState(X=X2,X_dot=Xd2,psi_p=pp2,psi_m=pm2,
            modes=np.zeros((nN,N_MODES),dtype=np.complex64),
            mass_sq=np.zeros(nN,dtype=np.float32),phase=ph2,g_s=state.g_s), idx, nN-1)

def join_strings(state, idx_a, idx_b):
    D, S = state.X.shape[1], state.X.shape[2]
    half = S//2
    def rj(a,b):
        seg = np.concatenate([a[:,:half],b[:,half:]], axis=-1)
        return np.array([np.interp(np.linspace(0,seg.shape[1]-1,S),
                         np.arange(seg.shape[1]),seg[d]) for d in range(D)])
    Xj  = rj(state.X[idx_a],     state.X[idx_b])
    Xdj = rj(state.X_dot[idx_a], state.X_dot[idx_b])
    ppj = rj(state.psi_p[idx_a], state.psi_p[idx_b])
    pmj = rj(state.psi_m[idx_a], state.psi_m[idx_b])
    keep = [i for i in range(state.X.shape[0]) if i not in (idx_a,idx_b)]
    X2  = np.concatenate([state.X[keep],     Xj[None]],  axis=0)
    Xd2 = np.concatenate([state.X_dot[keep], Xdj[None]], axis=0)
    pp2 = np.concatenate([state.psi_p[keep], ppj[None]], axis=0)
    pm2 = np.concatenate([state.psi_m[keep], pmj[None]], axis=0)
    ph2 = np.concatenate([state.phase[keep],
                           [(state.phase[idx_a]+state.phase[idx_b])/2]])
    nN  = X2.shape[0]
    return StringState(X=X2,X_dot=Xd2,psi_p=pp2,psi_m=pm2,
                       modes=np.zeros((nN,N_MODES),dtype=np.complex64),
                       mass_sq=np.zeros(nN,dtype=np.float32),phase=ph2,g_s=state.g_s)

def apply_t_duality(state, compact_radius):
    nr = ALPHA_PRIME/compact_radius
    X2 = state.X.copy(); X2[:,D_LARGE:,:] *= nr/compact_radius
    pp2 = state.psi_p.copy(); pm2 = state.psi_m.copy()
    pp2[:,D_LARGE:,:] = state.psi_m[:,D_LARGE:,:]
    pm2[:,D_LARGE:,:] = state.psi_p[:,D_LARGE:,:]
    return (StringState(X=X2,X_dot=state.X_dot,psi_p=pp2,psi_m=pm2,
            modes=state.modes,mass_sq=state.mass_sq,phase=state.phase,g_s=state.g_s), nr)

def apply_s_duality(state):
    ng  = 1.0/state.g_s
    Xd2 = state.X_dot*(state.g_s/ng)
    return StringState(X=state.X,X_dot=Xd2,psi_p=state.psi_p,psi_m=state.psi_m,
                       modes=state.modes,mass_sq=state.mass_sq,phase=state.phase,g_s=ng)

def split_string(state, idx, key=None):
    N, D, S = state.X.shape
    half = S//2
    def rs(arr, s, e):
        seg = arr[:,s:e]
        return np.array([np.interp(np.linspace(0,seg.shape[1]-1,S),
                         np.arange(seg.shape[1]),seg[d]) for d in range(D)])
    X2  = np.concatenate([state.X[:idx], rs(state.X[idx],0,half)[None],
                           state.X[idx+1:], rs(state.X[idx],half,S)[None]], axis=0)
    Xd2 = np.concatenate([state.X_dot[:idx], rs(state.X_dot[idx],0,half)[None],
                           state.X_dot[idx+1:], rs(state.X_dot[idx],half,S)[None]], axis=0)
    pp2 = np.concatenate([state.psi_p[:idx], rs(state.psi_p[idx],0,half)[None],
                           state.psi_p[idx+1:], rs(state.psi_p[idx],half,S)[None]], axis=0)
    pm2 = np.concatenate([state.psi_m[:idx], rs(state.psi_m[idx],0,half)[None],
                           state.psi_m[idx+1:], rs(state.psi_m[idx],half,S)[None]], axis=0)
    ph2 = np.concatenate([state.phase[:idx],[state.phase[idx]+.5],
                           state.phase[idx+1:],[state.phase[idx]-.5]])
    nN  = X2.shape[0]
    return (StringState(X=X2,X_dot=Xd2,psi_p=pp2,psi_m=pm2,
            modes=np.zeros((nN,N_MODES),dtype=np.complex64),
            mass_sq=np.zeros(nN,dtype=np.float32),phase=ph2,g_s=state.g_s),idx,nN-1)

def join_strings(state, idx_a, idx_b):
    D, S = state.X.shape[1], state.X.shape[2]
    half = S//2
    def rj(a,b):
        seg = np.concatenate([a[:,:half],b[:,half:]], axis=-1)
        return np.array([np.interp(np.linspace(0,seg.shape[1]-1,S),
                         np.arange(seg.shape[1]),seg[d]) for d in range(D)])
    Xj  = rj(state.X[idx_a],     state.X[idx_b])
    Xdj = rj(state.X_dot[idx_a], state.X_dot[idx_b])
    ppj = rj(state.psi_p[idx_a], state.psi_p[idx_b])
    pmj = rj(state.psi_m[idx_a], state.psi_m[idx_b])
    keep = [i for i in range(state.X.shape[0]) if i not in (idx_a,idx_b)]
    X2  = np.concatenate([state.X[keep],     Xj[None]],  axis=0)
    Xd2 = np.concatenate([state.X_dot[keep], Xdj[None]], axis=0)
    pp2 = np.concatenate([state.psi_p[keep], ppj[None]], axis=0)
    pm2 = np.concatenate([state.psi_m[keep], pmj[None]], axis=0)
    ph2 = np.concatenate([state.phase[keep],
                           [(state.phase[idx_a]+state.phase[idx_b])/2]])
    nN  = X2.shape[0]
    return StringState(X=X2,X_dot=Xd2,psi_p=pp2,psi_m=pm2,
                       modes=np.zeros((nN,N_MODES),dtype=np.complex64),
                       mass_sq=np.zeros(nN,dtype=np.float32),phase=ph2,g_s=state.g_s)

def apply_t_duality(state, compact_radius):
    nr = ALPHA_PRIME/compact_radius
    X2 = state.X.copy()
    X2[:,D_LARGE:,:] *= nr/compact_radius
    pp2 = state.psi_p.copy()
    pm2 = state.psi_m.copy()
    pp2[:,D_LARGE:,:] = state.psi_m[:,D_LARGE:,:]
    pm2[:,D_LARGE:,:] = state.psi_p[:,D_LARGE:,:]
    return (StringState(X=X2,X_dot=state.X_dot,psi_p=pp2,psi_m=pm2,
            modes=state.modes,mass_sq=state.mass_sq,
            phase=state.phase,g_s=state.g_s), nr)

def apply_s_duality(state):
    ng  = 1.0/state.g_s
    Xd2 = state.X_dot*(state.g_s/ng)
    return StringState(X=state.X,X_dot=Xd2,psi_p=state.psi_p,
                       psi_m=state.psi_m,modes=state.modes,
                       mass_sq=state.mass_sq,phase=state.phase,g_s=ng)
