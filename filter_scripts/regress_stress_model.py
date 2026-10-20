#!/usr/bin/env python
"""
Test of the second-order (gradient) model of the sub-filter stress-energy tensor on a pickled resMHD3d meso model.

Relative to the Favre observer u~, the filtered fluid stress-energy tensor <T^ab> has an anisotropic stress pi^ab and a
heat-flux-like vector q^a that vanish for an ideal fluid. To second order in the filter width the filter's own covariance
predicts both with no free constant (derivation in master_files/OhmsRegression.py, "Stress-energy tensor"):
  pi_pred = s~ TF[h X h],   q_pred = q1 + q2  (q1 from the covariance of n^a, q2 from that of the enthalpy per n^2 with n^a).
This script scores those predictions (R2 = 1 - |target - pred|^2/|target|^2, can be negative), regresses the
targets on the model terms (expected coefficient +1 each), and gives the correlation-preserving shift null and the
spread of the coefficients across tiles.

    python regress_stress_model.py /path/to/resMHD3d_khi576_w4_3sl.pickle

Use a pickle with 3 meso time slices for the time derivatives; only the central slice is scored and one cell at the
grid edges is dropped. Norms are the invariant ones (pi_ab pi^ab, q_a q^a). Does not modify the pickle.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'master_files'))
import OhmsRegression as OR
from MesoModels import *            # noqa: F401  (needed so the pickled classes can be loaded)


def report(name, target, ops, weights, expected, block, min_points, n_null, rows):
    """Prediction score, best scale, regression and tile spread for one target; ops = model terms (expected coeff 1)."""
    k = target.shape[-1]
    pred = sum(ops.values())
    ratio = np.sqrt(OR._wdot(pred.reshape(-1, k), pred.reshape(-1, k), weights).sum()
                    / OR._wdot(target.reshape(-1, k), target.reshape(-1, k), weights).sum())
    r2 = OR.prediction_r2(target, pred, weights)
    null = OR.prediction_shift_null(target, pred, n_perm=n_null, weights=weights)
    sc = OR.fit_region(target, {'pred': pred}, ('pred',), weights)
    print("%s: PARAMETER-FREE PREDICTION" % name)
    print("  global R2 = %.3f  (shift null: median %.3f, 95th pct %.3f)   |pred| / |target| = %.3f" % (
        r2, np.median(null), np.percentile(null, 95), ratio))
    print("  best overall scale %.3g (1 = as predicted), R2 at that scale %.3f" % (sc['pred'], sc['R2']))
    names = tuple(ops)
    fit = OR.fit_region(target, ops, names, weights)
    fnull = OR.shift_null(target, ops, names, n_perm=n_null, weights=weights)
    print("  regression on %s: R2 %.3f (null95 %.3f), coefficients %s (expected %s)" % (
        "+".join(names), fit['R2'], np.percentile(fnull, 95),
        "  ".join("%s %.3g" % (n, fit[n]) for n in names), expected))
    tiles = OR.tile_fits(target, ops, names, block, min_points=min_points, weights=weights)
    for n in names:
        s_ = OR.summarize([t[n] for t in tiles])
        print("  tile spread of %-3s %9.3g [%9.3g, %9.3g]  %3.0f%% positive over %d tiles" % (
            n, s_['median'], s_['p16'], s_['p84'], 100 * s_['frac_pos'], s_['n']))
    rows[name] = dict(pred_R2=r2, pred_null95=float(np.percentile(null, 95)), amplitude_ratio=float(ratio),
                      best_scale=sc['pred'], best_scale_R2=sc['R2'], fit_R2=fit['R2'],
                      fit_null95=float(np.percentile(fnull, 95)),
                      coeffs=np.array([fit[n] for n in names]), tile_coeffs=np.array([[t[n] for n in names] for t in tiles]))
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pickle_file")
    p.add_argument("--block", type=int, nargs=3, default=[6, 6, 3])
    p.add_argument("--min-points", type=int, default=100)
    p.add_argument("--n-null", type=int, default=50)
    p.add_argument("--label", default="", help="stored in the npz, e.g. the filter width of a scan")
    p.add_argument("--sigma2-factor", type=float, default=1.0,
                   help="multiply the nominal sigma^2 = W^2/12 by this (e.g. 1.12 at W=4 cells: linear interpolation of "
                        "the micro data adds ~0.14 dx^2 of variance)")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()

    with open(a.pickle_file, 'rb') as fh:
        m = pickle.load(fh)
    st, dv = m.meso_structures, m.domain_vars
    u, T, N = m.meso_vars['u_tilde'], st['SET'], st['BC']
    metric = np.asarray(m.metric, dtype=float)
    Nt = u.shape[0]
    spacings = (float(dv['Dx']), float(dv['Dy']), float(dv['Dz']))
    dt = float(dv['Dt']) if Nt >= 3 else None
    sigma2 = a.sigma2_factor * float(m.filter.filter_width) ** 2 / 12.0
    print("meso grid %s, filter width %.4g (sigma^2 = %.3g, factor %.3g), meso spacing %.4g" % (u.shape[:4], m.filter.filter_width, sigma2, a.sigma2_factor, spacings[0]))
    if dt is None:
        print("WARNING: only %d time slice(s): time derivatives set to zero, the model is INCOMPLETE" % Nt)

    d = OR.decompose_set(T, u, metric)
    ntil = np.sqrt(-np.einsum('...a,ab,...b->...', N, metric, N))
    s = (d['eps'] + d['p']) / ntil ** 2
    vel = u[..., 1:] / u[..., 0:1]
    mod = OR.stress_model(N, s, u, vel, spacings, sigma2, metric, dt)

    t_sl = slice(Nt // 2, Nt // 2 + 1) if Nt >= 3 else slice(None)
    s_sl = tuple(slice(1, -1) if n >= 3 else slice(None) for n in u.shape[1:4])
    sl = (t_sl,) + s_sl
    flat = lambda x: x[sl].reshape(x[sl].shape[:4] + (16,))
    pi_t, pi_p = flat(d['pi']), flat(mod['pi'])
    q_t, q1, q2 = d['q'][sl], mod['q1'][sl], mod['q2'][sl]
    w16 = OR.set_weights(metric)
    wq = np.array([-1.0, 1.0, 1.0, 1.0]) * np.sign(np.diag(metric))
    npts = int(np.prod(pi_t.shape[:4]))
    print("points used: %d (central time slice, edge cells dropped)" % npts)
    pn = np.sqrt(np.sum(w16 * pi_t ** 2, axis=-1)); qn = np.sqrt(np.abs(OR._wdot(q_t, q_t, wq)))
    pt = d['p'][sl]; enth = (d['eps'] + d['p'])[sl]
    print("sizes: median |pi|/p~ = %.3g, median |q|/(eps~+p~) = %.3g\n" % (np.median(pn / pt), np.median(qn / enth)))

    rows = {}
    block = tuple(a.block)
    report("pi", pi_t, {'pi': pi_p}, w16, "+1", block, a.min_points, a.n_null, rows)
    report("q", q_t, {'q1': q1, 'q2': q2}, wq, "+1, +1", block, a.min_points, a.n_null, rows)

    os.makedirs(a.outdir, exist_ok=True)
    out = dict(label=a.label, sigma2=sigma2, filter_width=float(m.filter.filter_width), n_slices=Nt,
               pi_over_p=float(np.median(pn / pt)), q_over_enthalpy=float(np.median(qn / enth)))
    for name, r in rows.items():
        out.update({name + '_' + k: v for k, v in r.items()})
    np.savez(os.path.join(a.outdir, "stress_model.npz"), **out)
    print("Wrote stress_model.npz to %s" % a.outdir)


if __name__ == "__main__":
    main()
