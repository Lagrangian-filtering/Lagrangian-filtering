#!/usr/bin/env python
"""
Sparse (SINDy-like) regression of E_tilde on a library of candidate operators: does the data select the second-order filter term from a larger family?

Library (master_files/SparseLibrary.py): the quadratic gradient forms sigma^2 (d v)(d B) (the filter's own covariance sum_k d_k v x d_k B, whose expected coefficient is -1, and the other
forms with one epsilon: (curl v) x (curl B), (div v) J, (J.grad) v, (w.grad) B, ... only 6 of the 11 forms are independent, 5 for div B = 0 (the case here), 4 if also div v = 0; dependent
columns are dropped in the order of the preferred list and reported), the density-weighting term (expected +1), and the Ohm's-law terms J~, B~, Hall of the meso model. Sequentially
thresholded least squares on the Gram system (columns scaled to unit norm, the threshold is the fraction of |E| carried by a term); the support at each threshold; the R^2 of the
selection against the correlation-preserving shift null; the stability of the selection under a bootstrap over windows (boxes of the meso grid, tiles or overlapping, box or Hann
test function).

    python regress_sparse_library.py /path/to/resMHD3d_khi384_w4_3sl.pickle [--window 1 6 6 3] [--thresholds 0.02 0.05 0.1 0.2]
    python regress_sparse_library.py --synthetic [--extra divv_J=0.5] [--noise 0.1]        # synthetic box-filtered compressible fields with a known answer (--incompressible: div v = 0)

The fields are the central time slice of the pickle with the edge cells dropped (spatial gradients only: the time pieces of the covariance are < 1 % of |E|, planning notes
section 8.5); the quadratic forms are mapped to the meso observer with observer_electric as the existing model. Does not modify the pickle.
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
    import MesoModels                   # the pickled classes live here; imported only for real data (the synthetic mode needs numpy alone)
    globals().update({k: getattr(MesoModels, k) for k in dir(MesoModels) if not k.startswith('_')})
    from system.BaseFunctionality import Base
    with open(a.pickle_file, 'rb') as fh:
        m = pickle.load(fh)
    v, st, dv = m.meso_vars, m.meso_structures, m.domain_vars
    u, E, J, B = v['u_tilde'], v['E_tilde'], v['J_tilde'], v['B_tilde']
    Nt = E.shape[0]
    _, B_lab = Base.observer_frame_fields(st['FaradayTensor'], np.array([1.0, 0.0, 0.0, 0.0]), m.metric)
    D = st['BC'][..., 0]
    vel = u[..., 1:] / u[..., 0:1]
    spacings = (float(dv['Dx']), float(dv['Dy']), float(dv['Dz']))
    sigma2 = a.sigma2_factor * float(m.filter.filter_width) ** 2 / 12.0
    lib = SL.quadratic_library(vel, B_lab[..., 1:], spacings, sigma2, u[..., 0])
    lib['model_dens'] = SL.density_form(vel, B_lab[..., 1:], D, spacings, sigma2, u[..., 0])
    if not a.no_ohm:
        lib.update(OR.standard_operators(J, B, u, Base.raise_LeviCivita4D(m.metric), m.metric))
    t_sl = slice(Nt // 2, Nt // 2 + 1) if Nt >= 3 else slice(None)
    s_sl = tuple(slice(1, -1) if n >= 3 else slice(None) for n in E.shape[1:4])
    sl = (t_sl,) + s_sl
    info = "meso grid %s, filter width %.4g (sigma^2 = %.3g), meso spacing %.4g" % (E.shape[:4], m.filter.filter_width, sigma2, spacings[0])
    return {k: x[sl] for k, x in lib.items()}, E[sl], info


def synthetic_data(a):
    comp = not a.incompressible
    m = SL.synthetic_meso(n_fine=a.n_fine, width=a.width, stride=1, seed=a.seed, kmax=2, compressible=comp)
    vel = m['v'] if comp else m['v_plain']
    lib = SL.static_library(vel, m['B'], m['D'], m['spacing'], m['sigma2'], density=comp, ohm=not a.no_ohm)
    E = SL.spatial4(m['E_favre'] if comp else m['E_exact'])
    for item in a.extra:
        name, c = item.split('=')
        E = E + float(c) * lib[name]
    sl = (slice(None),) + (slice(2, -2),) * 3
    if a.noise > 0:
        rng = np.random.default_rng(a.seed + 1)
        rms = np.sqrt(OR.mink_dot(E[sl], E[sl]).mean())
        E = E + a.noise * rms * SL.spatial4(rng.normal(size=E.shape[:-1] + (3,)))
    info = "synthetic: %d^3 fine cells, box width %d, %s, noise %.2g, extra terms %s; the exact answer is dv_x_dB -1%s" % (
        a.n_fine, a.width, "compressible" if comp else "incompressible", a.noise, a.extra or "none", ", model_dens +1" if comp else "")
    return {k: x[sl] for k, x in lib.items()}, E[sl], info


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pickle_file", nargs='?')
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--incompressible", action="store_true", help="synthetic: div v = 0 and a uniform density (default: compressible, a density field and the Favre-averaged velocity)")
    p.add_argument("--extra", nargs='*', default=[], help="synthetic: NAME=c, add c times a library term to E (e.g. w_x_J=0.5)")
    p.add_argument("--noise", type=float, default=0.0, help="synthetic: white noise in E, as a fraction of its rms")
    p.add_argument("--n-fine", type=int, default=40)
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--no-ohm", action="store_true", help="leave the Ohm's-law terms out of the library")
    p.add_argument("--sigma2-factor", type=float, default=1.0, help="multiply the nominal sigma^2 = W^2/12 by this (see regress_gradient_model.py)")
    p.add_argument("--window", type=int, nargs=4, default=[1, 6, 6, 3], help="window of the bootstrap rows (t, x, y, z meso cells)")
    p.add_argument("--stride", type=int, nargs=4, default=None, help="window stride (default: the window, i.e. tiles)")
    p.add_argument("--test-function", choices=['box', 'hann'], default='box')
    p.add_argument("--rows", choices=['points', 'windows'], default='points', help="fit to the points or to the window integrals")
    p.add_argument("--thresholds", type=float, nargs='+', default=[0.01, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--boot-threshold", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=1e-3, help="a column is dropped as dependent when its part outside the span of the kept ones is below tol of its norm")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--n-null", type=int, default=30)
    p.add_argument("--outdir", default=".")
    a = p.parse_args()
    if not a.synthetic and not a.pickle_file:
        p.error("give a pickle or --synthetic")

    lib, E, info = synthetic_data(a) if a.synthetic else real_data(a)
    print(info)
    names = list(lib)
    print("points used: %d; library of %d columns" % (int(np.prod(E.shape[:4])), len(names)))

    rows_ops, rows_E = lib, E
    win = SL.window_sums({**lib, '__E': E}, a.window, a.stride, a.test_function)
    if a.rows == 'windows':
        rows_ops, rows_E = {k: win[k] for k in names}, win['__E']
    G, r, EE = SL.gram_system(rows_ops, rows_E, names)
    kept, dropped = SL.independent_columns(G, a.tol)
    nk = [names[i] for i in kept]
    print("\nindependent columns kept (%d): %s" % (len(kept), ", ".join(nk)))
    print("dropped as linearly dependent or identically zero (%d): %s" % (len(dropped), ", ".join(names[i] for i in dropped) or "none"))
    Gk, rk = G[np.ix_(kept, kept)], r[kept]
    norms = np.sqrt(np.diag(Gk) / EE)

    print("\nSTLSQ PATH (support at each threshold; coefficient and the fraction of |E| the term carries, c |O| / |E|):")
    path = SL.stlsq_path(Gk, rk, EE, a.thresholds)
    for t, c, act, R2 in path:
        terms = "  ".join("%s %.3g (%.2f)" % (n, c[j], abs(c[j]) * norms[j]) for j, n in enumerate(nk) if act[j])
        print("  threshold %-5g R2 %.3f  %2d terms:  %s" % (t, R2, int(act.sum()), terms or "none"))
    full = SL.stlsq(Gk, rk, EE, 0.0)
    print("  all kept columns, no threshold: R2 %.3f" % full[2])

    print("\nSINGLE TERMS (R2 of each kept column alone):")
    for j, n in enumerate(nk):
        print("  %-20s R2 %7.3f   coefficient %9.3g" % (n, rk[j] ** 2 / (Gk[j, j] * EE), rk[j] / Gk[j, j]))

    t_sel = a.boot_threshold
    c_sel, act_sel, R2_sel = SL.stlsq(Gk, rk, EE, t_sel)
    sel = [n for n, x in zip(nk, act_sel) if x]
    if sel:
        ops_sel = {n: lib[n] for n in sel}
        null = OR.shift_null(E, ops_sel, tuple(sel), n_perm=a.n_null, min_shift=3)
        print("\nSELECTION at threshold %g: %s   R2 %.3f; shift-null 95th percentile of the R2 of the same terms %.3f" % (t_sel, ", ".join(sel), R2_sel, np.percentile(null, 95)))

    print("\nBOOTSTRAP over %d windows %s (%s, %s): inclusion frequency and coefficient median [16th, 84th pct] at threshold %g" % (
        len(win['__E']), tuple(a.window), a.test_function, "tiles" if a.stride is None else "stride %s" % tuple(a.stride), t_sel))
    bs = SL.bootstrap_inclusion({n: win[n] for n in nk}, win['__E'], nk, t_sel, a.n_boot)
    for n in nk:
        f, med, lo, hi = bs[n]
        print("  %-20s %5.2f   %9.3g [%9.3g, %9.3g]" % (n, f, med, lo, hi))

    os.makedirs(a.outdir, exist_ok=True)
    np.savez(os.path.join(a.outdir, "sparse_library.npz"), names=np.array(nk), dropped=np.array([names[i] for i in dropped]), thresholds=np.array(a.thresholds),
             path_R2=np.array([x[3] for x in path]), path_coef=np.array([x[1] for x in path]), path_support=np.array([x[2] for x in path]),
             boot_names=np.array(nk), boot_inclusion=np.array([bs[n][0] for n in nk]), boot_median=np.array([bs[n][1] for n in nk]),
             column_norm_over_E=norms, info=info)
    print("\nWrote sparse_library.npz to %s" % a.outdir)


if __name__ == '__main__':
    main()
