"""Shared reporting for the parameter-free second-order model tests (regress_stress_model.py, regress_lorentz_closure.py)."""
import numpy as np

import OhmsRegression as OR


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
        print("  tile spread of %-6s %9.3g [%9.3g, %9.3g]  %3.0f%% positive over %d tiles" % (
            n, s_['median'], s_['p16'], s_['p84'], 100 * s_['frac_pos'], s_['n']))
    rows[name] = dict(pred_R2=r2, pred_null95=float(np.percentile(null, 95)), amplitude_ratio=float(ratio),
                      best_scale=sc['pred'], best_scale_R2=sc['R2'], fit_R2=fit['R2'],
                      fit_null95=float(np.percentile(fnull, 95)),
                      coeffs=np.array([fit[n] for n in names]), tile_coeffs=np.array([[t[n] for n in names] for t in tiles]))
    print()
