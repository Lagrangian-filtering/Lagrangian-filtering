"""
Does the fit of the second-order filter term survive a Lorentz boost?  An analytic spacetime test, numpy only.

A smooth ideal-MHD-like field in (t, x, y, z) (random Fourier modes in space and time: velocity v, B = curl A + B0, E = -v x B, so F^{mu nu} u_nu = 0 exactly; the fields need not
solve the equations, the deficit is kinematic) is filtered with a box window and regressed against the gradient closure. Because the fields are analytic they can be evaluated at the
points of any frame, so a boost needs no interpolation: F'(x') = L F(L^-1 x') L^T, u'(x') = L u(L^-1 x').

Two frames enter and are chosen independently:
  - the frame of the analysis a (where the meso grid, the gradients and the library live): boost velocity beta_a relative to the frame S0 in which the fields are defined;
  - the frame of the window w (where the box of the filter is at rest and simultaneous): beta_w. Case 'lab' (A, the covariant case): the same physical window as in S0, so the
    filtered tensors are the boosts of the S0 ones; case 'frame' (B): the box is rebuilt in the frame of the analysis (tilted with respect to the S0 window).

The filtered tensor <F> and the filtered flux <u> (constant proper density: N = n u) are averages of components over the points of the window, which is frame covariant for a fixed
set of points. The observer is u~ = <u> / |<u>|, the target is the observer's electric 4-vector  e^mu = <F>^{mu nu} u~_nu  (the E_tilde of the real pipeline).

Two libraries for the filter covariance:
  - 'cov': e_cov^mu = eps^{mu nu a b} u~_nu Sigma^{rho sigma} d_rho u~_a d_sigma b~_b  (4-gradients of the 4-velocity and the comoving field, Sigma^{rho sigma} the covariance matrix of
    the window, Sigma = var sum_k a_k a_k^T over the window axes a_k): in the rest frame of u~ with a spatial window it is sigma^2 sum_k d_k v x d_k B, expected coefficient -1.
  - 'sp': the library of regress_sparse_library.py: lab-frame 3-vector velocity v = u~^i/u~^0 and B, spatial gradients in the analysis frame only, sigma^2 of the window's own
    proper lattice spacing (what the real-data pipeline uses; not covariant).
"""
import itertools

import numpy as np

import OhmsRegression as OR
import SparseLibrary as SL

ETA = np.array([-1.0, 1.0, 1.0, 1.0])            # diagonal of the metric


def levi4():
    """The Levi-Civita symbol with LC[0,1,2,3] = +1."""
    e = np.zeros((4,) * 4)
    for p in itertools.permutations(range(4)):
        q, s = list(p), 1
        for i in range(4):
            while q[i] != i:
                j = q[i]
                q[i], q[j] = q[j], q[i]
                s = -s
        e[p] = s
    return e


LC4 = levi4()


def lorentz_boost(beta):
    """L with X' = L X for the frame moving with velocity beta (3-vector) relative to the frame of X."""
    beta = np.asarray(beta, dtype=float)
    b2 = float(beta @ beta)
    L = np.eye(4)
    if b2 == 0.0:
        return L
    g = 1.0 / np.sqrt(1.0 - b2)
    L[0, 0] = g
    L[0, 1:] = -g * beta
    L[1:, 0] = -g * beta
    L[1:, 1:] += (g - 1.0) * np.outer(beta, beta) / b2
    return L


class Flow:
    """Random smooth fields in S0: velocity |v| < vmax, B = curl A + b0, E = -v x B, as functions of (t, x, y, z); amplitude falling as 1/(1 + k^2)."""

    def __init__(self, seed=0, kmax=2, vmax=0.4, bamp=0.6, b0=(0.0, 0.0, 1.0), wscale=0.5):
        rng = np.random.default_rng(seed)
        ks = [k for k in itertools.product(range(-kmax, kmax + 1), repeat=3) if k > (0, 0, 0) and 0 < sum(c * c for c in k) <= kmax * kmax]
        self.k = np.array(ks, dtype=float)                                       # (M, 3)
        k2 = (self.k ** 2).sum(axis=1)
        self.omega = wscale * np.sqrt(k2) * rng.uniform(-1.0, 1.0, len(ks))      # phase velocities below wscale
        amp = 1.0 / (1.0 + k2)
        self.av = rng.normal(size=(len(ks), 2, 3)) * amp[:, None, None]          # cos and sin coefficients of the three velocity components
        self.aa = rng.normal(size=(len(ks), 2, 3)) * amp[:, None, None] * bamp   # ... of the vector potential
        self.vmax, self.b0 = vmax, np.asarray(b0, dtype=float)

    def fields(self, X):
        """u^mu (..., 4) and F^{mu nu} (..., 4, 4) at the S0 points X (..., 4); F^{0i} = E_i, F^{ij} = eps_ijk B_k, so e^i = gamma (E + v x B)_i."""
        X = np.asarray(X, dtype=float)
        shape = X.shape[:-1]
        P = X.reshape(-1, 4)
        ph = P[:, 1:] @ self.k.T - np.outer(P[:, 0], self.omega)
        c, s = np.cos(ph), np.sin(ph)
        raw = c @ self.av[:, 0] + s @ self.av[:, 1]
        v = self.vmax * raw / np.sqrt(1.0 + np.sum(raw ** 2, axis=1, keepdims=True))
        dA = np.einsum('pm,mj,mk->pjk', -s, self.k, self.aa[:, 0]) + np.einsum('pm,mj,mk->pjk', c, self.k, self.aa[:, 1])      # d_j A_k
        B = np.einsum('ijk,pjk->pi', SL.EPS3, dA) + self.b0
        E = -np.cross(v, B)
        gam = 1.0 / np.sqrt(1.0 - np.sum(v ** 2, axis=1))
        u = np.concatenate([gam[:, None], gam[:, None] * v], axis=1)
        F = np.zeros((len(P), 4, 4))
        F[:, 0, 1:] = E
        F[:, 1:, 0] = -E
        F[:, 1:, 1:] = np.einsum('ijk,pk->pij', SL.EPS3, B)
        return u.reshape(shape + (4,)), F.reshape(shape + (4, 4))


def boost_fields(flow, Xp, L):
    """u and F of the boosted frame (X' = L X) at the points Xp (..., 4) of that frame."""
    u0, F0 = flow.fields(np.asarray(Xp) @ np.linalg.inv(L).T)
    return u0 @ L.T, np.einsum('ma,...ab,nb->...mn', L, F0, L)


def window_axes(L_w, h_axis):
    """The three lattice axes of a window at rest and simultaneous in the frame L_w, as 4-vectors in S0 (proper spacings h_axis)."""
    Linv = np.linalg.inv(L_w)
    return np.stack([h_axis[k] * Linv[:, k + 1] for k in range(3)])


def window_covariance(axes, width):
    """Sigma^{mu nu} (S0 components) of the discrete box with `width` points per axis."""
    return (width ** 2 - 1) / 12.0 * np.einsum('ka,kb->ab', axes, axes)


def box_mean(flow, centers, axes, width, chunk=4000):
    """Means of u and F over the lattice (width points per axis, spacing the axes) around each S0 point of centers (P, 4)."""
    offs = np.arange(width) - (width - 1) / 2.0
    shifts = np.array(list(itertools.product(offs, repeat=3))) @ axes            # (width^3, 4)
    ub = np.zeros((len(centers), 4))
    Fb = np.zeros((len(centers), 4, 4))
    for i in range(0, len(centers), chunk):
        c = centers[i:i + chunk]
        u, F = flow.fields(c[:, None, :] + shifts[None])
        ub[i:i + chunk] = u.mean(axis=1)
        Fb[i:i + chunk] = F.mean(axis=1)
    return ub, Fb


def meso_fields(flow, beta_a, beta_w, spacing, n=(3, 9, 9, 9), h=0.13, h_w=None, width=4):
    """
    The filtered fields on a meso grid n with `spacing` (dt, dx, dy, dz) in the frame of the analysis (boost beta_a), for a window at rest in the frame beta_w with proper lattice
    spacings h_w (default h). Returns u (the filtered flux, 4-vector), F (the filtered tensor), Sigma (in the analysis frame), sigma2 (the window's own (width^2 - 1)/12 h^2).
    """
    L_a, L_w = lorentz_boost(beta_a), lorentz_boost(beta_w)
    h_w = (h, h, h) if h_w is None else h_w
    axes = window_axes(L_w, h_w)
    grid = np.stack(np.meshgrid(*[(np.arange(m) - m // 2) * d for m, d in zip(n, spacing)], indexing='ij'), axis=-1)       # analysis-frame points
    centers = grid.reshape(-1, 4) @ np.linalg.inv(L_a).T
    ub, Fb = box_mean(flow, centers, axes, width)
    return {'u': (ub @ L_a.T).reshape(tuple(n) + (4,)),
            'F': np.einsum('ma,pab,nb->pmn', L_a, Fb, L_a).reshape(tuple(n) + (4, 4)),
            'Sigma': L_a @ window_covariance(axes, width) @ L_a.T,
            'sigma2': (width ** 2 - 1) / 12.0 * float(np.mean(np.asarray(h_w)) ** 2),
            'spacing': tuple(spacing)}


def observer_quantities(m):
    """u~ (4-vector), e^mu = <F>^{mu nu} u~_nu, the lab 3-velocity, B and the comoving field b~ in the analysis frame."""
    ub = m['u']
    ut = ub / np.sqrt(-np.sum(ETA * ub * ub, axis=-1, keepdims=True))
    F = m['F']
    e = np.einsum('...mn,...n->...m', F, ut * ETA)
    gam = ut[..., 0]
    vel = ut[..., 1:] / gam[..., None]
    B = 0.5 * np.einsum('kij,...ij->...k', SL.EPS3, F[..., 1:, 1:])
    vB = np.sum(vel * B, axis=-1)
    b = np.zeros_like(ut)
    b[..., 0] = gam * vB
    b[..., 1:] = B / gam[..., None] + (gam * vB)[..., None] * vel
    return {'u': ut, 'e': e, 'v': vel, 'B': B, 'b': b, 'gamma': gam}


def covariant_library(q, Sigma, spacing):
    """e_cov^mu = eps^{mu nu a b} u~_nu Sigma^{rho sigma} d_rho u~_a d_sigma b~_b on the whole grid (4-gradients, central differences)."""
    ul, bl = q['u'] * ETA, q['b'] * ETA
    dU = np.stack(np.gradient(ul, *spacing, axis=(0, 1, 2, 3)), axis=-2)         # [..., rho, a]
    dB = np.stack(np.gradient(bl, *spacing, axis=(0, 1, 2, 3)), axis=-2)
    return np.einsum('rs,mnab,...n,...ra,...sb->...m', Sigma, LC4, ul, dU, dB)


def spatial_library(q, spacing, sigma2):
    """The library of the real-data regression: 3-vector v and B, spatial gradients of the analysis frame, sigma2 of the window's own scale."""
    return SL.quadratic_library(q['v'], q['B'], spacing[1:], sigma2, q['gamma'], names=['dv_x_dB'])['dv_x_dB']


def fit_single(O, e):
    """Least-squares coefficient c of e ~ c O (Minkowski product) and R^2 over the points."""
    G, r, EE = float(OR.mink_dot(O, O).sum()), float(OR.mink_dot(O, e).sum()), float(OR.mink_dot(e, e).sum())
    c = r / G
    return c, 1.0 - (EE - 2.0 * c * r + c * c * G) / EE


def run_case(flow, beta_a, window, h=0.13, width=4, n=(3, 9, 9, 9), trim=2, mesh=1.0):
    """
    One analysis frame (beta_a, axis-aligned) and window case 'lab' (A: the S0 window) or 'frame' (B: a box at rest in the analysis frame, simultaneous there, whose lengths along the
    boost are h/gamma so that its extent in S0 is the same as that of the S0 window). The meso spacing is h * mesh, divided by gamma along the boost and in time.
    Returns {'cov': (c, R2), 'sp': (c, R2), 'gamma': gamma, 'rms': rms |e|}.
    """
    beta_a = np.asarray(beta_a, dtype=float)
    b = float(np.linalg.norm(beta_a))
    gam = 1.0 / np.sqrt(1.0 - b * b)
    axis = int(np.argmax(np.abs(beta_a))) if b > 0 else 0
    sp = [h * mesh] * 4
    h_w = [h] * 3
    if b > 0:
        sp[0] /= gam
        sp[1 + axis] /= gam
    if window == 'frame' and b > 0:
        h_w[axis] = h / gam
    m = meso_fields(flow, beta_a, beta_a if window == 'frame' else np.zeros(3), sp, n=n, h=h, h_w=h_w, width=width)
    q = observer_quantities(m)
    sl = (slice(n[0] // 2, n[0] // 2 + 1),) + (slice(trim, -trim),) * 3
    cov = covariant_library(q, m['Sigma'], sp)[sl]
    spl = spatial_library(q, sp, m['sigma2'])[sl]
    e = q['e'][sl]
    return {'cov': fit_single(cov, e), 'sp': fit_single(spl, e), 'gamma': gam, 'rms': float(np.sqrt(OR.mink_dot(e, e).mean()))}
