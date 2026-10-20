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


def central_slice(shape4):
    """(time, x, y, z) slices of the central time slice with one edge cell dropped on each spatial axis (as the model scripts)."""
    Nt = shape4[0]
    t_sl = slice(Nt // 2, Nt // 2 + 1) if Nt >= 3 else slice(None)
    return (t_sl,) + tuple(slice(1, -1) if n >= 3 else slice(None) for n in shape4[1:4])


def total_force_ratio(m):
    """|F_closure| / |d_a <T^{ai}>| (spatial components, centred differences on the meso grid) at every meso point."""
    st, dv, mv = m.meso_structures, m.domain_vars, m.meso_vars
    T = np.asarray(st['SET'], dtype=float)
    spacings = (float(dv['Dx']), float(dv['Dy']), float(dv['Dz']))
    div = np.zeros(T.shape[:4] + (3,))
    if T.shape[0] >= 3:
        div += np.gradient(T[..., 0, 1:], float(dv['Dt']), axis=0)
    for l in range(3):
        div += np.gradient(T[..., l + 1, 1:], spacings[l], axis=l + 1)
    Fc = np.sqrt(np.sum(np.asarray(mv['F_closure'], dtype=float)[..., 1:] ** 2, axis=-1))
    return Fc / np.maximum(np.sqrt(np.sum(div ** 2, axis=-1)), 1e-300)
