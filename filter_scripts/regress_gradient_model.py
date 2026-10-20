#!/usr/bin/env python
"""
Test of the second-order (gradient) model of E_tilde on a pickled resMHD3d meso model.

The filter's own covariance predicts, to second order in the filter width and with no free
constant, E~ ~= -O(g1) + O(g2) (derivation in master_files/OhmsRegression.py): g1 comes from
the cross product of velocity and field gradients, g2 from the density weighting of the Favre
average. This script
  1. scores that parameter-free prediction (R2 = 1 - |E - pred|^2/|E|^2, can be negative),
  2. regresses E_tilde on the two model operators (expected coefficients -1 and +1),
  3. compares with the resistive / Hall terms and with both sets together,
with the correlation-preserving shift null for every fit and the spread of coefficients across tiles.

    python regress_gradient_model.py /path/to/resMHD3d_khi576_w4_3sl.pickle

Use a pickle with 3 meso time slices (--num-t-slices 3) for the complete model: the time
derivatives need them. With one slice they are set to zero and the model is incomplete (a warning
is printed). With 3+ slices only the central time slice is used. One cell at the grid edges is
dropped because derivatives are one-sided there. Does not modify the pickle.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'master_files'))
import OhmsRegression as OR
from MesoModels import *            # noqa: F401  (needed so the pickled classes can be loaded)
from system.BaseFunctionality import Base


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
    v, st, dv = m.meso_vars, m.meso_structures, m.domain_vars
    u, E, J, B = v['u_tilde'], v['E_tilde'], v['J_tilde'], v['B_tilde']
    Nt = E.shape[0]
    u_lab = np.array([1.0, 0.0, 0.0, 0.0])
    _, B_lab = Base.observer_frame_fields(st['FaradayTensor'], u_lab, m.metric)
    D = st['BC'][..., 0]
    vel = u[..., 1:] / u[..., 0:1]
    spacings = (float(dv['Dx']), float(dv['Dy']), float(dv['Dz']))
    dt = float(dv['Dt']) if Nt >= 3 else None
    sigma2 = a.sigma2_factor * float(m.filter.filter_width) ** 2 / 12.0
    print("meso grid %s, filter width %.4g (sigma^2 = %.3g, factor %.3g), meso spacing %.4g" % (E.shape[:4], m.filter.filter_width, sigma2, a.sigma2_factor, spacings[0]))
    if dt is None:
        print("WARNING: only %d time slice(s): time derivatives set to zero, the model is INCOMPLETE" % Nt)

    g = OR.gradient_model_operators(vel, B_lab[..., 1:], D, u, spacings, sigma2, dt)
    ops = OR.standard_operators(J, B, u, Base.raise_LeviCivita4D(m.metric), m.metric)
    ops.update({'model_vB': g['model_vB'], 'model_dens': g['model_dens']})

    t_sl = slice(Nt // 2, Nt // 2 + 1) if Nt >= 3 else slice(None)
    s_sl = tuple(slice(1, -1) if n >= 3 else slice(None) for n in E.shape[1:4])
    sl = (t_sl,) + s_sl
    E_, pred = E[sl], g['prediction'][sl]
    ops = {k: x[sl] for k, x in ops.items()}
    print("points used: %d (central time slice, edge cells dropped)\n" % int(np.prod(E_.shape[:4])))

    ratio = np.sqrt(OR.mink_dot(pred, pred).sum() / OR.mink_dot(E_, E_).sum())
    print("PARAMETER-FREE PREDICTION  E~ ~= -O(g1) + O(g2):")
    print("  global R2 = %.3f   |pred| / |E| = %.3f" % (OR.prediction_r2(E_, pred), ratio))
    null = OR.prediction_shift_null(E_, pred, n_perm=a.n_null)
    print("  shift null for this R2: median %.3f, 95th percentile %.3f" % (np.median(null), np.percentile(null, 95)))
    pred_null95 = float(np.percentile(null, 95))
    sc = OR.fit_region(E_, {'pred': pred}, ('pred',))
    print("  best overall scale of the prediction: %.3g (1 = as predicted); R2 at that scale = %.3f = cos^2 of the angle between E and the prediction" % (sc['pred'], sc['R2']))
    tiles = OR.tile_fits(E_, {'pred': pred}, ('pred',), tuple(a.block), min_points=a.min_points)
    if tiles:
        s_ = OR.summarize([t['pred'] for t in tiles])
        print("  best scale per tile: median %.3g [p16 %.3g, p84 %.3g] over %d tiles" % (s_['median'], s_['p16'], s_['p84'], s_['n']))

    models = [('model_vB',), ('model_dens',), ('model_vB', 'model_dens'), ('R_tilde',), ('gamma_hall',),
              ('R_tilde', 'alpha_dynamo', 'gamma_hall'), ('R_tilde', 'model_vB', 'model_dens'),
              ('R_tilde', 'alpha_dynamo', 'gamma_hall', 'model_vB', 'model_dens')]
    print("\nCONSTANT-COEFFICIENT FITS (global; shift-null 95th percentile in brackets); expected: model_vB -1, model_dens +1")
    print("%-52s %7s %9s   %s" % ("terms", "R2", "(null95)", "coefficients"))
    rows = {}
    for names in models:
        fit = OR.fit_region(E_, ops, names)
        null = OR.shift_null(E_, ops, names, n_perm=a.n_null, min_shift=3)
        rows[names] = (fit['R2'], float(np.percentile(null, 95)), {n: fit[n] for n in names})
        print("%-52s %7.3f (%7.3f)   %s" % ("+".join(names), fit['R2'], np.percentile(null, 95),
              "  ".join("%s %.3g" % (n.replace('_tilde', '').replace('alpha_dynamo', 'alpha').replace('gamma_hall', 'gamma'), fit[n]) for n in names)))

    both = ('model_vB', 'model_dens')
    tiles = OR.tile_fits(E_, ops, both, tuple(a.block), min_points=a.min_points)
    print("\nMODEL COEFFICIENTS ACROSS %d TILES (median [16th, 84th pct], fraction > 0):" % len(tiles))
    for n in both:
        s_ = OR.summarize([t[n] for t in tiles])
        print("  %-11s %9.3g [%9.3g, %9.3g]  %3.0f%% positive" % (n, s_['median'], s_['p16'], s_['p84'], 100 * s_['frac_pos']))

    os.makedirs(a.outdir, exist_ok=True)
    np.savez(os.path.join(a.outdir, "gradient_model.npz"),
             label=a.label, sigma2=sigma2, filter_width=float(m.filter.filter_width), n_slices=Nt,
             pred_R2=OR.prediction_r2(E_, pred), pred_null95=pred_null95, amplitude_ratio=ratio,
             best_scale=sc['pred'], best_scale_R2=sc['R2'],
             fit_names=np.array(["+".join(k) for k in rows]), fit_R2=np.array([v[0] for v in rows.values()]),
             fit_null95=np.array([v[1] for v in rows.values()]),
             model_vB_coeff=rows[('model_vB',)][2]['model_vB'],
             joint_vB_coeff=rows[('model_vB', 'model_dens')][2]['model_vB'],
             joint_dens_coeff=rows[('model_vB', 'model_dens')][2]['model_dens'],
             tile_model_vB=np.array([t['model_vB'] for t in tiles]), tile_model_dens=np.array([t['model_dens'] for t in tiles]))
    print("\nWrote gradient_model.npz to %s" % a.outdir)


if __name__ == "__main__":
    main()
