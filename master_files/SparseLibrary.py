"""
Sparse regression of E_tilde on a library of candidate operators (SINDy-like, sequentially thresholded least squares).

The question: does the data pick out the second-order filter term  E~ = -sigma^2 sum_k d_k v x d_k B  (+ density term) from a larger family of candidate
terms, without the form being imposed? The library has three kinds of columns, all 4-vectors in the observer frame (orthogonal to u~):
  - the Ohm's-law terms  J~, B~, Hall  (OhmsRegression.standard_operators);
  - the density term of the second-order model (OhmsRegression.gradient_model_operators, 'model_dens');
  - the quadratic gradient forms below, sigma^2 times a vector built from d_k v_i and d_l B_j (lab frame, spatial gradients of the filtered fields).

Quadratic gradient forms. A vector bilinear in the gradients S_ki = d_k v_i and T_lj = d_l B_j, linear in an odd number of epsilons (B is axial, E polar), is a contraction of the
rank-5 tensor S_ki T_lj with eps_{mab} delta_{cd} for a choice of 3 of the 5 indices on eps and the other 2 on delta: 10 forms, plus w x J (w = curl v, J = curl B), which is a
combination of them. The isotropic rank-5 tensors of three dimensions span only 6 dimensions (Schouten identity), so the 11 forms are linearly dependent: **6 independent
forms for general gradients, 5 when div B = 0, 4 when also div v = 0** (checked numerically in the tests). `independent_columns` keeps a maximal independent subset in the order
given, so the physically preferred forms come first.

The regression (`stlsq`) works on the Gram matrix G_ik = sum <O_i, O_k>, r_i = sum <O_i, E> (Minkowski product over the 4-vector, summed over the points of the region, or over
windows), exactly as OhmsRegression.fit_region, so it needs no change for 4-vectors. Columns are scaled to unit norm, so the threshold is on a dimensionless quantity: the
fraction of |E| carried by the term. Windows (`window_sums`) integrate each operator and E over a box of the (t,x,y,z) meso grid with a test function (box or Hann) before the
regression, which gives rows that are volume integrals and a natural unit for the bootstrap (`bootstrap_inclusion`). Note that this is not the integration by parts of the weak form
of the PDE-SINDy literature: the derivatives in the library are still finite differences of the filtered fields.

Pure numpy; all vectors are 4-vectors with the Minkowski product (-+++), which is positive on the vectors orthogonal to u~.
"""
import itertools

import numpy as np

import OhmsRegression as OR


def levi3():
    e = np.zeros((3, 3, 3))
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        e[i, j, k] = 1.0
        e[i, k, j] = -1.0
    return e


EPS3 = levi3()

# name -> einsum over (eps, S, T) with S[..., k, i] = d_k v_i, T[..., l, j] = d_l B_j; output [..., m]. 'w_x_J' is handled separately.
# The preferred order (the filter's own form first) decides which of the dependent forms is kept.
FORMS = {
    'dv_x_dB':        'mij,...ki,...kj->...m',    # sum_k d_k v x d_k B  (the covariance of the filter; expected coefficient -1)
    'w_x_J':          None,                       # (curl v) x (curl B)
    'divv_J':         'mlj,...kk,...lj->...m',    # (div v) (curl B)
    'J_grad_v':       'klj,...lj,...km->...m',    # (J . grad) v
    'w_grad_B':       'lki,...ki,...lm->...m',    # (w . grad) B
    'gradvT_J':       'ilj,...lj,...mi->...m',    # J_i d_m v_i
    'gradBT_w':       'jki,...ki,...mj->...m',    # w_j d_m B_j
    'gradv_i_x_gradB_i': 'mkl,...ki,...li->...m',  # sum_i grad v_i x grad B_i
    'mixed_vB':       'mkj,...ki,...ij->...m',    # eps_{mkj} d_k v_i d_i B_j
    'mixed_vB2':      'mil,...ki,...lk->...m',    # eps_{mil} d_k v_i d_l B_k
    'w_divB':         'mki,...ki,...jj->...m',    # (curl v)(div B): zero for solenoidal B
}


def gradient_tensor(f, spacings):
    """G[..., k, i] = d_k f_i for f of shape (Nt, Nx, Ny, Nz, 3) (central differences); k runs over the three spatial axes."""
    return np.stack([np.gradient(f, spacings[k], axis=k + 1) for k in range(3)], axis=-2)


def quadratic_forms(S, T, names=None):
    """The quadratic gradient forms (3-vectors, no sigma^2) of S[..., k, i] = d_k v_i and T[..., l, j] = d_l B_j, as a dict name -> (..., 3)."""
    names = list(FORMS) if names is None else list(names)
    out = {}
    for n in names:
        if n == 'w_x_J':
            w = np.einsum('lki,...ki->...l', EPS3, S)
            J = np.einsum('klj,...lj->...k', EPS3, T)
            out[n] = np.cross(w, J)
        else:
            out[n] = np.einsum(FORMS[n], EPS3, S, T)
    return out


def quadratic_library(vel, B, spacings, sigma2, gamma=None, names=None):
    """
    sigma^2 times the quadratic forms of the filtered lab-frame velocity vel (..., 3) and field B (..., 3), as 4-vectors in the observer frame
    gamma (1, vel) (OhmsRegression.observer_electric). gamma=None gives the spatial 4-vector (0, g), i.e. the static observer; pass gamma = u~^0 for the meso observer.
    """
    S, T = gradient_tensor(vel, spacings), gradient_tensor(B, spacings)
    forms = quadratic_forms(S, T, names)
    out = {}
    for n, g in forms.items():
        g = sigma2 * g
        if gamma is None:
            x = np.zeros(g.shape[:-1] + (4,))
            x[..., 1:] = g
            out[n] = x
        else:
            out[n] = OR.observer_electric(g, vel, gamma)
    return out


def density_form(vel, B, D, spacings, sigma2, gamma=None):
    """
    The density-weighting (Favre) term of the second-order model, g2 = sigma^2 (sum_k d_k D d_k v / D) x B (expected coefficient +1), as a 4-vector in the observer frame
    (static if gamma is None). Spatial gradients only, as the quadratic forms.
    """
    dD = np.stack([np.gradient(D, spacings[k], axis=k + 1) for k in range(3)], axis=-1)       # [..., k]
    S = gradient_tensor(vel, spacings)                                                        # [..., k, i]
    T = np.einsum('...k,...ki->...i', dD, S)
    g = sigma2 * np.cross(T / D[..., None], B)
    if gamma is None:
        x = np.zeros(g.shape[:-1] + (4,))
        x[..., 1:] = g
        return x
    return OR.observer_electric(g, vel, gamma)


def static_library(vel, B, D, spacings, sigma2, names=None, density=True, ohm=True):
    """
    The whole library for a static observer (4-vectors (0, g)): the quadratic forms, the density term (if `density`) and the Ohm's-law terms J = curl B, B, J x B
    (if `ohm`; named like OhmsRegression.TERMS). Used for synthetic data; for real data the Ohm terms come from the model (J~, B~, Hall in the meso observer frame).
    """
    lib = quadratic_library(vel, B, spacings, sigma2, None, names)
    if density:
        lib['model_dens'] = density_form(vel, B, D, spacings, sigma2)
    if ohm:
        J = np.einsum('klj,...lj->...k', EPS3, gradient_tensor(B, spacings))
        for n, x in zip(OR.TERMS, (J, B, np.cross(J, B))):
            a = np.zeros(x.shape[:-1] + (4,))
            a[..., 1:] = x
            lib[n] = a
    return lib


def spatial4(x):
    """3-vectors (..., 3) as 4-vectors (0, x)."""
    a = np.zeros(x.shape[:-1] + (4,))
    a[..., 1:] = x
    return a


# ---- fourth-order terms of the EMF ----------------------------------------------------------------------------------------------------------------------------------
#
# For a separable box filter along each axis with second and fourth moments m2 = sigma^2 and m4 (continuous box of width W: m4 = W^4/80 = 1.8 sigma^4; discrete uniform
# average of w points, spacing h: m4 = (w^2 - 1)(3 w^2 - 7) h^4 / 240), the covariance of two point fields a, b is, with F_k = 1 + (m2/2) d_k^2 + (m4/24) d_k^4,
#   cov(a, b) = m2 sum_k a_k b_k + sum_k [ (m4/6) (a_kkk b_k + a_k b_kkk) + (m4/4 - m2^2/4) a_kk b_kk ]
#               + (m2^2/4) sum_{k<l} [ 4 a_kl b_kl + 2 (a_kkl b_l + a_kll b_k + a_l b_kkl + a_k b_kll) ]  +  O(m2^3)
# (subscripts: partial derivatives along the axes; checked in one dimension against the exact filter of exponentials). The regression works with the FILTERED fields
# a~ = F a, b~ = F b, whose derivatives differ from those of a, b at this order (a_k = a~_k - (m2/2) sum_l a~_kll + ...), which changes the coefficients:
#   cov = m2 sum_k a~_k b~_k + sum_k [ (m4/6 - m2^2/2) (a~_kkk b~_k + a~_k b~_kkk) + (m4/4 - m2^2/4) a~_kk b~_kk ] + m2^2 sum_{k<l} a~_kl b~_kl  +  0 x (the 2+1 cross group).
# E = -cov(v~, B~) with the cross product. fourth_order_forms returns sigma^4 times the four groups (axis-aligned 3+1 and 2+2, cross 1+1 and 2+1) and
# fourth_order_coefficients the coefficients predicted in E~ (axis31: -(m4/(6 sigma^4) - 1/2), axis22: -(m4/(4 sigma^4) - 1/4), cross11: -1, cross21: 0). The box filter is
# not isotropic at this order (the axis-aligned groups); the real filter is a 27-point quadrature of a linear interpolation, whose m4 is not known: the data give it as a coefficient.

def fourth_order_forms(v, B, spacings, sigma2):
    """The four groups of fourth-order EMF terms (4-vectors in the static frame, times sigma^4): 'o4_axis31', 'o4_axis22', 'o4_cross11', 'o4_cross21'."""
    def d(f, *axes):
        for k in axes:
            f = np.gradient(f, spacings[k], axis=k + 1)
        return f
    s4 = sigma2 ** 2
    cx = np.cross
    axis31 = sum(cx(d(v, k, k, k), d(B, k)) + cx(d(v, k), d(B, k, k, k)) for k in range(3))
    axis22 = sum(cx(d(v, k, k), d(B, k, k)) for k in range(3))
    pairs = [(0, 1), (0, 2), (1, 2)]
    cross11 = sum(cx(d(v, k, l), d(B, k, l)) for k, l in pairs)
    cross21 = sum(cx(d(v, k, k, l), d(B, l)) + cx(d(v, k, l, l), d(B, k)) + cx(d(v, l), d(B, k, k, l)) + cx(d(v, k), d(B, k, l, l)) for k, l in pairs)
    return {n: spatial4(s4 * x) for n, x in (('o4_axis31', axis31), ('o4_axis22', axis22), ('o4_cross11', cross11), ('o4_cross21', cross21))}


def fourth_order_coefficients(sigma2, m4):
    """The coefficients predicted for E~ = -cov(v~, B~) on the groups of fourth_order_forms (m4: the fourth moment of the filter in the units of length^4, sigma2 = m2)."""
    return {'o4_axis31': -(m4 / (6 * sigma2 ** 2) - 0.5), 'o4_axis22': -(m4 / (4 * sigma2 ** 2) - 0.25), 'o4_cross11': -1.0, 'o4_cross21': 0.0}


# ---- sub-filter stress, Poynting and energy deficits: library --------------------------------------------------------------------------------------------------------
#
# The coarse run's momentum, energy and induction sources are -d_j Delta^{ij}, -d_j Delta^{0j}, -curl EMF with (lab frame, flat metric, E_r = -v x B, resolved fields v, B;
# OhmsRegression.em_flux_deficits for the derivation, to second order in the filter width, O(EMF^2) dropped)
#   Delta^{ij} = -(E_r^i EMF^j + EMF^i E_r^j) + delta^{ij} E_r.EMF + sigma^2 sum_l [-d_l E^i d_l E^j - d_l B^i d_l B^j + (1/2) delta^{ij} (|d_l E|^2 + |d_l B|^2)],
#   Delta^{0j} = EMF x B + sigma^2 sum_l d_l E x d_l B,           Delta^{00} = E_r.EMF + (sigma^2/2) sum_l (|d_l E|^2 + |d_l B|^2),
# with d_l E = a_l + b_l, a_l = -d_l v x B, b_l = -v x d_l B, and EMF = -sigma^2 sum_l d_l v x d_l B. The structure gives pieces with coefficient +1 each ('lin_*',
# 'quad_*'); the regression is given the pieces separately (the E and B parts of the gradient products, the v-weighted ones) and further gradient forms of the same
# order, with a coefficient per piece, to see which piece carries the factor (about 2) by which the quadratic part is under-predicted at the closure's own resolution.

DEFICIT_COMPONENTS = {'stress': (slice(0, 6), np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])), 'poynting': (slice(6, 9), np.ones(3)), 'energy': (slice(9, 10), np.ones(1))}
_IJ = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def maxwell_deficit_parts(E, B):
    """The 3+1 electromagnetic stress-energy of the fields E, B: stress T^ij = -E_i E_j - B_i B_j + (1/2) delta (E^2 + B^2), Poynting E x B, energy (E^2 + B^2)/2."""
    e2b2 = np.sum(E * E, axis=-1) + np.sum(B * B, axis=-1)
    stress = -E[..., :, None] * E[..., None, :] - B[..., :, None] * B[..., None, :] + 0.5 * np.eye(3) * e2b2[..., None, None]
    return {'stress': stress, 'poynting': np.cross(E, B), 'energy': 0.5 * e2b2}


def pack_deficit(stress, poynting, energy):
    """(..., 3, 3), (..., 3), (...) -> (..., 10): xx yy zz xy xz yz, Sx Sy Sz, e."""
    st = np.stack([stress[..., i, j] for i, j in _IJ], axis=-1)
    return np.concatenate([st, poynting, energy[..., None]], axis=-1)


def deficit_library(v, B, spacings, sigma2):
    """
    Operators for the deficits Delta^{ij} (6 independent components), Delta^{0j} (3) and Delta^{00}, each a (..., 10) packed array that is zero outside its own class.
    Returns {class: {name: array}} for class in 'stress', 'poynting', 'energy'. Spatial gradients, lab frame. The coefficient expected for each 'lin_' and 'quad_' piece is +1.
    """
    E = -np.cross(v, B)
    T = [gradient_tensor_k(B, spacings, l) for l in range(3)]            # T[l][..., i] = d_l B_i
    dv = [gradient_tensor_k(v, spacings, l) for l in range(3)]
    a = [-np.cross(dv[l], B) for l in range(3)]
    b = [-np.cross(v, T[l]) for l in range(3)]
    emf = -sigma2 * sum(np.cross(dv[l], T[l]) for l in range(3))
    eye = np.eye(3)
    out3 = lambda x: x[..., None, None] * eye

    def sym_sum(f, g):                                                   # sum_l f_l,i g_l,j  symmetrised
        return sum(0.5 * (f[l][..., :, None] * g[l][..., None, :] + g[l][..., :, None] * f[l][..., None, :]) for l in range(3))

    def dot_sum(f, g):
        return sum(np.sum(f[l] * g[l], axis=-1) for l in range(3))

    zero_s, zero_p, zero_e = np.zeros_like(E[..., :, None] * E[..., None, :]), np.zeros_like(E), np.zeros(E.shape[:-1])
    S = lambda x: pack_deficit(x, zero_p, zero_e)
    P = lambda x: pack_deficit(zero_s, x, zero_e)
    N = lambda x: pack_deficit(zero_s, zero_p, x)
    dE = [a[l] + b[l] for l in range(3)]
    stress = {
        'lin_stress': S(-(E[..., :, None] * emf[..., None, :] + emf[..., :, None] * E[..., None, :]) + out3(np.sum(E * emf, axis=-1))),
        'quad_stress': S(sigma2 * (-sym_sum(dE, dE) - sym_sum(T, T) + 0.5 * out3(dot_sum(dE, dE) + dot_sum(T, T)))),
        'BB_grad': S(-sigma2 * sym_sum(T, T)), 'BB_trace': S(0.5 * sigma2 * out3(dot_sum(T, T))),
        'EaEa_grad': S(-sigma2 * sym_sum(a, a)), 'EaEa_trace': S(0.5 * sigma2 * out3(dot_sum(a, a))),
        'EaEb_grad': S(-2 * sigma2 * sym_sum(a, b)), 'EaEb_trace': S(sigma2 * out3(dot_sum(a, b))),
        'EbEb_grad': S(-sigma2 * sym_sum(b, b)), 'EbEb_trace': S(0.5 * sigma2 * out3(dot_sum(b, b))),
    }
    M = np.stack(T, axis=-2)                                             # M[..., k, l] = d_k B_l
    J = np.einsum('klj,...lj->...k', EPS3, M)
    extra = {                                                            # further gradient-squared forms of B (the symmetric forms of two traceless M)
        'BB_dBi_dBj_t': np.einsum('...il,...jl->...ij', M, M),           # d_i B_l d_j B_l
        'BB_cross': np.einsum('...li,...jl->...ij', M, M), 'BB_cross2': np.einsum('...il,...lj->...ij', M, M),
        'BB_trace_MM': out3(np.einsum('...kl,...lk->...', M, M)), 'BB_JJ': J[..., :, None] * J[..., None, :], 'BB_J2': out3(np.sum(J * J, axis=-1)),
    }
    for k, x in extra.items():
        if k in ('BB_cross', 'BB_cross2'):
            x = 0.5 * (x + np.swapaxes(x, -1, -2))
        stress[k] = S(sigma2 * x)
    pb = {'lin_poynting': P(np.cross(emf, B)),
          'quad_poynting': P(sigma2 * sum(np.cross(dE[l], T[l]) for l in range(3))),
          'B_dvdB': P(sigma2 * B * dot_sum(dv, T)[..., None]),
          'dv_BdB': P(sigma2 * sum(dv[l] * np.sum(B * T[l], axis=-1)[..., None] for l in range(3))),
          'dB_Bdv': P(sigma2 * sum(T[l] * np.sum(B * dv[l], axis=-1)[..., None] for l in range(3))),
          'v_dBdB': P(sigma2 * v * dot_sum(T, T)[..., None]),
          'dB_vdB': P(sigma2 * sum(T[l] * np.sum(v * T[l], axis=-1)[..., None] for l in range(3)))}
    en = {'lin_energy': N(np.sum(E * emf, axis=-1)),
          'quad_energy': N(0.5 * sigma2 * (dot_sum(dE, dE) + dot_sum(T, T))),
          'BB_energy': N(0.5 * sigma2 * dot_sum(T, T)), 'EaEa_energy': N(0.5 * sigma2 * dot_sum(a, a)),
          'EaEb_energy': N(sigma2 * dot_sum(a, b)), 'EbEb_energy': N(0.5 * sigma2 * dot_sum(b, b)),
          'BB_MM_energy': N(sigma2 * np.einsum('...kl,...lk->...', M, M)), 'BB_J2_energy': N(sigma2 * np.sum(J * J, axis=-1))}
    return {'stress': stress, 'poynting': pb, 'energy': en}


def gradient_tensor_k(f, spacings, l):
    """d_l f for f of shape (Nt, Nx, Ny, Nz, ...) along the spatial axis l."""
    return np.gradient(f, spacings[l], axis=l + 1)


def gram_system_k(ops, y, names, comps, weights):
    """Gram system over the components `comps` (a slice of the packed last axis) with weights per component (Euclidean, not Minkowski)."""
    O = [np.asarray(ops[n], dtype=float)[..., comps].reshape(-1, len(weights)) for n in names]
    y = np.asarray(y, dtype=float)[..., comps].reshape(-1, len(weights))
    G = np.array([[np.sum(weights * a * b) for b in O] for a in O])
    r = np.array([np.sum(weights * a * y) for a in O])
    return G, r, float(np.sum(weights * y * y))


# ---- windows and the Gram system ---------------------------------------------------------------------------------------------------------------------------------------

def _test_function(shape, kind):
    if kind == 'box':
        return np.ones(shape)
    if kind == 'hann':
        w = [np.hanning(n + 2)[1:-1] if n > 1 else np.ones(1) for n in shape]
        return np.einsum('a,b,c,d->abcd', *w)
    raise ValueError("test function must be 'box' or 'hann'")


def window_sums(arrays, window, stride=None, kind='box'):
    """
    Integrals of each array (Nt, Nx, Ny, Nz, ...) over boxes of the meso grid with the test function `kind`: (wt, wx, wy, wz) = window, stride (default the window: tiles)
    in each direction. Returns {name: (Nw, ...)}. The sum over the box is the spacetime volume integral up to the constant cell volume.
    """
    shape = next(iter(arrays.values())).shape[:4]
    window = tuple(min(int(w), n) for w, n in zip(window, shape))
    stride = window if stride is None else tuple(int(s) for s in stride)
    phi = _test_function(window, kind)
    starts = [range(0, n - w + 1, s) for n, w, s in zip(shape, window, stride)]
    out = {k: [] for k in arrays}
    for i0 in itertools.product(*starts):
        sl = tuple(slice(i, i + w) for i, w in zip(i0, window))
        for k, a in arrays.items():
            x = a[sl]
            out[k].append(np.tensordot(phi, x, axes=4))
    return {k: np.array(v) for k, v in out.items()}


def gram_system(ops, E, names):
    """(G, r, EE): G_ik = sum <O_i, O_k>, r_i = sum <O_i, E>, EE = sum <E, E> over all leading indices (Minkowski product on the last axis of 4 components)."""
    O = [np.asarray(ops[n], dtype=float).reshape(-1, 4) for n in names]
    E = np.asarray(E, dtype=float).reshape(-1, 4)
    G = np.array([[OR.mink_dot(a, b).sum() for b in O] for a in O])
    r = np.array([OR.mink_dot(a, E).sum() for a in O])
    return G, r, float(OR.mink_dot(E, E).sum())


def independent_columns(G, tol=1e-3, order=None, min_norm=1e-6):
    """
    Indices of a maximal linearly independent subset of the columns, taken in `order` (default 0, 1, 2, ...): a column is dropped when the part of it not in the span of
    the columns kept so far has a norm below tol times its own (Gram-Schmidt in the Gram metric), or when its own norm is below min_norm times the largest (a column that is
    zero to round-off, such as (div v) J in an incompressible flow, would otherwise be an arbitrary unit vector after scaling). Returns (kept, dropped).
    """
    order = list(range(len(G))) if order is None else list(order)
    kept, dropped = [], []
    gmax = float(np.max(np.diag(G))) if len(G) else 0.0
    for i in order:
        if G[i, i] <= min_norm ** 2 * gmax:
            dropped.append(i)
            continue
        if kept:
            Gk = G[np.ix_(kept, kept)]
            c = np.linalg.solve(Gk + 1e-14 * np.trace(Gk) * np.eye(len(kept)), G[kept, i])
            resid = G[i, i] - c @ G[kept, i]
        else:
            resid = G[i, i]
        (kept if resid > tol ** 2 * G[i, i] else dropped).append(i)
    return kept, dropped


def stlsq(G, r, EE, threshold=0.05, ridge=1e-10, max_iter=20):
    """
    Sequentially thresholded least squares on the Gram system. Columns are scaled to unit norm, c_i is the coefficient of the unit-norm column, and `threshold` is on
    |c_i| / sqrt(EE): the fraction of the rms of E carried by the term. Terms below it are removed, the rest refitted, until the support does not change.
    Returns (coefficients in the units of the original columns, 0 for removed terms; the boolean support; R2 = 1 - rss/EE).
    """
    n = len(r)
    d = np.sqrt(np.maximum(np.diag(G), 1e-300))
    Gn, rn = G / np.outer(d, d), r / d
    active = np.ones(n, dtype=bool)
    c = np.zeros(n)
    for _ in range(max_iter):
        idx = np.where(active)[0]
        if len(idx) == 0:
            c = np.zeros(n)
            break
        A = Gn[np.ix_(idx, idx)] + ridge * np.eye(len(idx))
        c = np.zeros(n)
        c[idx] = np.linalg.solve(A, rn[idx])
        small = active & (np.abs(c) < threshold * np.sqrt(EE))
        if not small.any():
            break
        active &= ~small
        c[small] = 0.0
    rss = EE - 2.0 * c @ rn + c @ Gn @ c
    return c / d, active, float(1.0 - rss / EE) if EE > 0 else float('nan')


def stlsq_path(G, r, EE, thresholds):
    """The STLSQ solution at each threshold: list of (threshold, coefficients, support, R2)."""
    return [(t,) + stlsq(G, r, EE, t) for t in thresholds]


def bootstrap_inclusion(ops_w, E_w, names, threshold=0.05, n_boot=200, seed=0):
    """
    Stability of the selection: STLSQ on resamples (with replacement) of the windows. ops_w: {name: (Nw, 4)}, E_w: (Nw, 4).
    Returns {name: (inclusion frequency, median coefficient, 16th, 84th percentile of the coefficient over the resamples in which it is included)}.
    """
    rng = np.random.default_rng(seed)
    Nw = len(E_w)
    coefs = np.zeros((n_boot, len(names)))
    incl = np.zeros((n_boot, len(names)), dtype=bool)
    for b in range(n_boot):
        ix = rng.integers(0, Nw, Nw)
        G, r, EE = gram_system({n: ops_w[n][ix] for n in names}, E_w[ix], names)
        c, act, _ = stlsq(G, r, EE, threshold)
        coefs[b], incl[b] = c, act
    out = {}
    for j, n in enumerate(names):
        sel = coefs[incl[:, j], j]
        out[n] = (float(incl[:, j].mean()),
                  float(np.median(sel)) if len(sel) else float('nan'),
                  float(np.percentile(sel, 16)) if len(sel) else float('nan'),
                  float(np.percentile(sel, 84)) if len(sel) else float('nan'))
    return out


# ---- synthetic fields ----------------------------------------------------------------------------------------------------------------------------------------------

def _random_periodic(shape, rng, kmax=3, amp=1.0):
    """A smooth periodic scalar field on a cube: random Fourier modes with |k_i| <= kmax, amplitude falling as 1/(1+|k|^2)."""
    n = shape[0]
    x = np.arange(n) * 2 * np.pi / n
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    f = np.zeros(shape)
    for kx, ky, kz in itertools.product(range(-kmax, kmax + 1), repeat=3):
        if (kx, ky, kz) <= (0, 0, 0):                       # one of each +-k pair
            continue
        k2 = kx * kx + ky * ky + kz * kz
        if k2 == 0 or k2 > kmax * kmax:
            continue
        a, b = rng.normal(size=2) / (1.0 + k2)
        ph = kx * X + ky * Y + kz * Z
        f += amp * (a * np.cos(ph) + b * np.sin(ph))
    return f


def _curl_periodic(A, h):
    """Spectral curl of a periodic vector field (n, n, n, 3), grid spacing h."""
    n = A.shape[0]
    k = np.fft.fftfreq(n, d=h) * 2 * np.pi
    K = np.meshgrid(k, k, k, indexing='ij')
    Ah = [np.fft.fftn(A[..., i]) for i in range(3)]
    c = [1j * (K[(i + 1) % 3] * Ah[(i + 2) % 3] - K[(i + 2) % 3] * Ah[(i + 1) % 3]) for i in range(3)]
    return np.stack([np.fft.ifftn(x).real for x in c], axis=-1)


def box_average(f, w):
    """Periodic running mean over w cells along each of the three spatial axes (axes 0, 1, 2 of f)."""
    out = f
    for ax in range(3):
        out = sum(np.roll(out, s - (w - 1) // 2, axis=ax) for s in range(w)) / w
    return out


def synthetic_meso(n_fine=48, width=4, stride=2, seed=0, kmax=3, vamp=0.3, compressible=True, density_amp=0.2, deficits=False):
    """
    A fine periodic grid with smooth random v (|v| ~ vamp), solenoidal B = curl A and density D; box-averaged over `width` fine cells and sampled every `stride` cells.
    Returns a dict with the meso fields {'v', 'B', 'D'} (each with a leading time axis of length 1: shape (1, N, N, N, ...)), the meso spacing, sigma^2 of the discrete box
    filter ((width^2 - 1)/12 h^2, h = fine grid spacing), 'E_exact': the exact EMF deficit  E~ = -(<v x B> - <v>x<B>)  (plain average velocity, 'v_plain'), and 'E_favre':
    E~ = -<v x B> + vf x <B> with the density-weighted velocity vf (= 'v'), the quantity of the real model (u~ is the Favre velocity), which the second-order model
    with its density term describes: coefficients -1 (dv_x_dB) and +1 (density).
    """
    rng = np.random.default_rng(seed)
    shp = (n_fine,) * 3
    h = 2 * np.pi / n_fine
    v = np.stack([_random_periodic(shp, rng, kmax, vamp) for _ in range(3)], axis=-1)
    if not compressible:                                    # remove the compressive part spectrally
        k = np.fft.fftfreq(n_fine, d=h) * 2 * np.pi
        K = np.stack(np.meshgrid(k, k, k, indexing='ij'), axis=-1)
        vh = np.stack([np.fft.fftn(v[..., i]) for i in range(3)], axis=-1)
        k2 = np.sum(K ** 2, axis=-1)
        k2[0, 0, 0] = 1.0
        proj = vh - K * (np.sum(K * vh, axis=-1) / k2)[..., None]
        v = np.stack([np.fft.ifftn(proj[..., i]).real for i in range(3)], axis=-1)
    A = np.stack([_random_periodic(shp, rng, kmax, 1.0) for _ in range(3)], axis=-1)
    B = _curl_periodic(A, h) + np.array([0.0, 0.0, 1.0])
    D = 1.0 + density_amp * _random_periodic(shp, rng, kmax, 1.0) if compressible else np.ones(shp)
    avg = lambda f: box_average(f, width)
    sub = (slice(0, None, stride),) * 3
    vm, Bm, Dm = avg(v), avg(B), avg(D)
    E_exact = -(avg(np.cross(v, B)) - np.cross(vm, Bm))
    Dv = avg(D[..., None] * v)
    vf = Dv / Dm[..., None]                                 # Favre velocity
    E_favre = -avg(np.cross(v, B)) + np.cross(vf, Bm)       # the filtered E minus the ideal field of the Favre velocity: -<dv x dB> - (<v> - vf) x <B>
    cell = lambda f: f[sub][None]
    extra = {}
    if deficits:                                           # exact Delta = <T(E, B)> - T(E_r, B~) with E_r = -<v> x <B>
        Tf = maxwell_deficit_parts(-np.cross(v, B), B)
        Tr = maxwell_deficit_parts(-np.cross(vm, Bm), Bm)
        avgk = lambda x: avg(x.reshape(x.shape[:3] + (-1,))).reshape(x.shape)
        extra['deficit'] = cell(pack_deficit(avgk(Tf['stress']) - Tr['stress'], avgk(Tf['poynting']) - Tr['poynting'], avg(Tf['energy'][..., None])[..., 0] - Tr['energy']))
    return {**extra, 'v': cell(vf), 'B': cell(Bm), 'D': cell(Dm), 'v_plain': cell(vm), 'E_exact': cell(E_exact), 'E_favre': cell(E_favre),
            'spacing': (stride * h,) * 3, 'sigma2': (width ** 2 - 1) / 12.0 * h * h}
