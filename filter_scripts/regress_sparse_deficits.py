#!/usr/bin/env python
"""
Sparse regression of the sub-filter Maxwell-stress, Poynting-flux and energy-density deficits of the filtered electromagnetic field on a library of gradient forms.

The deficits Delta^{ab} = <T_EM^{ab}> - T_EM(E_r, B~) (E_r = -v x B~, lab frame, flat metric; <T_EM> = SET_EM of the pickle, which must contain it: pickle made after 2026-10-21)
are the sources a coarse run needs in its momentum and energy equations (-d_j Delta^{ij}, -d_j Delta^{0j}); their second-order model (OhmsRegression.em_flux_deficits) is
    Delta^{ij} = lin + quad,  lin = -(E_r^i EMF^j + EMF^i E_r^j) + delta^{ij} E_r.EMF,  quad = sigma^2 sum_l [-d_l E^i d_l E^j - d_l B^i d_l B^j + (1/2) delta^{ij} (|d_l E|^2 + |d_l B|^2)],
    Delta^{0j} = EMF x B + sigma^2 sum_l d_l E x d_l B,    Delta^{00} = E_r.EMF + (sigma^2/2) sum_l (|d_l E|^2 + |d_l B|^2),
each piece with coefficient +1. The planning notes (section 47) found the quadratic part needs about twice the nominal variance at the closure's own resolution (the stress and Poynting
terms of the closure are about 1.9 times too small compared with the EMF). This script asks which piece carries the factor: the library (SparseLibrary.deficit_library) has, per class,
  * the structural pieces lin_*, quad_* (expected +1);
  * the pieces of quad split by the field they come from: B x B gradients ('BB_*'), (d v x B)^2 ('EaEa_*'), the cross term ('EaEb_*'), (v x d B)^2 ('EbEb_*');
  * for the stress, further gradient-squared forms of B (d_i B_l d_j B_l, ..., J_i J_j); for the Poynting flux the five forms B (dv.dB), dv (B.dB), dB (B.dv), v |dB|^2, dB (v.dB);
    for the energy, |dB|^2 and the other scalars.
Sequentially thresholded least squares per class (columns scaled to unit norm; the threshold is the fraction of |Delta| carried by a term), and the structural fit with free coefficients.

    python regress_sparse_deficits.py /path/to/resMHD3d_..._setem.pickle --sigma2-factor 1.12
    python regress_sparse_deficits.py --synthetic        # box-filtered synthetic fields with the exact deficits

The central time slice, three cells dropped at each edge. Sign convention of the Poynting flux: T^{0j} = +(E x B)_j, flip with --poynting-sign -1 if the parameter-free R2 of the Poynting
class is negative. Does not modify the pickle.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'master_files'))
import OhmsRegression as OR
import SparseLibrary as SL


def real_data(a):
    import MesoModels                   # the pickled classes live here
    globals().update({k: getattr(MesoModels, k) for k in dir(MesoModels) if not k.startswith('_')})
    from system.BaseFunctionality import Base
    with open(a.pickle_file, 'rb') as fh:
        m = pickle.load(fh)
    st, dv, mv = m.meso_structures, m.domain_vars, m.meso_vars
    if 'SET_EM' not in st or not np.any(np.asarray(st['SET_EM']) != 0):
        sys.exit("this pickle has no filtered SET_EM: make it again with the current MesoModels")
    u = np.asarray(mv['u_tilde'], dtype=float)
    Nt = u.shape[0]
    _, B_lab = Base.observer_frame_fields(st['FaradayTensor'], np.array([1.0, 0.0, 0.0, 0.0]), m.metric)
    vel, B = u[..., 1:] / u[..., 0:1], np.asarray(B_lab[..., 1:], dtype=float)
    T = np.asarray(st['SET_EM'], dtype=float)
    spacings = (float(dv['Dx']), float(dv['Dy']), float(dv['Dz']))
    sigma2 = a.sigma2_factor * float(m.filter.filter_width) ** 2 / 12.0
    Tr = SL.maxwell_deficit_parts(-np.cross(vel, B), B)
    stress = T[..., 1:, 1:] - Tr['stress']
    poy = a.poynting_sign * T[..., 0, 1:] - Tr['poynting']
    en = T[..., 0, 0] - Tr['energy']
    y = SL.pack_deficit(stress, poy, en)
    info = "meso grid %s, filter width %.4g (sigma^2 = %.3g), meso spacing %.4g" % (u.shape[:4], m.filter.filter_width, sigma2, spacings[0])
    t_sl = slice(Nt // 2, Nt // 2 + 1) if Nt >= 3 else slice(None)
    return vel, B, y, spacings, sigma2, (t_sl,) + (slice(3, -3),) * 3, info


def synthetic_data(a):
    m = SL.synthetic_meso(n_fine=a.n_fine, width=a.width, stride=1, seed=a.seed, kmax=3, compressible=False, deficits=True)
    info = "synthetic: %d^3 fine cells, box width %d: the exact deficits of box-filtered fields; the structural coefficients are +1" % (a.n_fine, a.width)
    return m['v_plain'], m['B'], m['deficit'], m['spacing'], m['sigma2'], (slice(None),) + (slice(3, -3),) * 3, info


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pickle_file", nargs='?')
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n-fine", type=int, default=40)
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--sigma2-factor", type=float, default=1.0)
    p.add_argument("--poynting-sign", type=float, default=1.0)
    p.add_argument("--thresholds", type=float, nargs='+', default=[0.02, 0.05, 0.1, 0.2])
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--outdir", default=".")
    a = p.parse_args()
    if not a.synthetic and not a.pickle_file:
        p.error("give a pickle or --synthetic")
    v, B, y, spacings, sigma2, sl, info = synthetic_data(a) if a.synthetic else real_data(a)
    print(info)
    lib = SL.deficit_library(v, B, spacings, sigma2)
    y = y[sl]
    print("points used: %d\n" % int(np.prod(y.shape[:4])))
    out = {}
    for cls, (comps, w) in SL.DEFICIT_COMPONENTS.items():
        ops = {k: x[sl] for k, x in lib[cls].items()}
        names = list(ops)
        G, r, EE = SL.gram_system_k(ops, y, names, comps, w)
        struct = [n for n in names if n.startswith(('lin_', 'quad_'))]
        ps = sum(ops[n] for n in struct)[..., comps]
        d = y[..., comps]
        print("=" * 100 + "\n%s deficit: |Delta|^2 summed %.3g; parameter-free structural model (all coefficients 1) R2 %.3f" % (
            cls, EE, 1.0 - np.sum(w * (d - ps) ** 2) / EE))
        Gs, rs, _ = SL.gram_system_k(ops, y, struct, comps, w)
        c, _, R2 = SL.stlsq(Gs, rs, EE, 0.0)
        print("  structural pieces with free coefficients: R2 %.3f   %s" % (R2, "  ".join("%s %.3g" % (n, x) for n, x in zip(struct, c))))
        kept, dropped = SL.independent_columns(G, a.tol)
        nk = [names[i] for i in kept]
        print("  library: %d columns, %d independent kept; dropped as dependent or zero: %s" % (len(names), len(kept), ", ".join(names[i] for i in dropped) or "none"))
        Gk, rk = G[np.ix_(kept, kept)], r[kept]
        norms = np.sqrt(np.diag(Gk) / EE)
        for t, c, act, R2 in SL.stlsq_path(Gk, rk, EE, a.thresholds):
            print("  threshold %-5g R2 %.3f %2d terms:  %s" % (t, R2, int(act.sum()), "  ".join("%s %.3g (%.2f)" % (n, c[j], abs(c[j]) * norms[j]) for j, n in enumerate(nk) if act[j]) or "none"))
        print("  single terms (R2 alone):  " + "  ".join("%s %.2f" % (n, rk[j] ** 2 / (Gk[j, j] * EE)) for j, n in enumerate(nk)))
        out[cls] = (struct, c)
    os.makedirs(a.outdir, exist_ok=True)
    np.savez(os.path.join(a.outdir, "sparse_deficits.npz"), info=info)
    print("\nWrote sparse_deficits.npz to %s" % a.outdir)


if __name__ == '__main__':
    main()
