#!/usr/bin/env python
"""
Does the second-order closure return energy to the resolved scales (backscatter)? A-priori, on a pickled resMHD3d meso model.

Gradient-type closures are known to be unstable when they feed energy back from the sub-filter to the resolved scales. Two
energy exchanges are checked, each for the data (the pipeline's own E~ and F_closure) and for the parameter-free model:
  * sub-filter Joule heating  P_J = J~ . E~   (E~ is the electric field in the fluid frame; > 0 means the resolved current
    loses energy to the sub-filter scales, < 0 is backscatter);
  * sub-filter work on the resolved fluid  P_F = -u~_b F_closure^b   (the rate at which the sub-filter Lorentz force
    does work on the fluid in its own frame; for a static fluid it is the time component F_closure^0, so > 0 means the resolved
    fluid gains energy from the sub-filter electromagnetic field, < 0 that it loses it).
For each, over all points, in the tail (|F_closure| > --threshold x the total force, as map_closure_tail.py) and elsewhere, it
prints the fraction of points with the negative sign, the energy-weighted backscatter sum(max(-P,0))/sum(|P|), the net
sum(P)/sum(|P|), and how often the model has the same sign as the data. It also gives the R2 of the model's predictions of E~ and
F_closure restricted to the tail and to the rest, since the overall R2 is weighted by |.|^2 and says little about where it holds.

    python check_backscatter.py /path/to/resMHD3d_khi576_w4_3sl_c2.pickle --sigma2-factor 1.12

Central time slice, one edge cell dropped. Does not modify the pickle.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'master_files'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import OhmsRegression as OR
from MesoModels import *            # noqa: F401  (needed so the pickled classes can be loaded)
from model_report import central_slice, total_force_ratio
from system.BaseFunctionality import Base


def summarise(P, mask):
    P = P[mask]
    if P.size == 0:
        return (np.nan,) * 3
    s = np.abs(P).sum()
    return float(np.mean(P < 0)), float(np.maximum(-P, 0).sum() / s), float(P.sum() / s)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pickle_file")
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--sigma2-factor", type=float, default=1.0)
    p.add_argument("--outdir", default=".")
    a = p.parse_args()

    with open(a.pickle_file, 'rb') as fh:
        m = pickle.load(fh)
    st, dv, mv = m.meso_structures, m.domain_vars, m.meso_vars
    metric = np.asarray(m.metric, dtype=float)
    u, E, J, B = mv['u_tilde'], mv['E_tilde'], mv['J_tilde'], mv['B_tilde']
    Nt = u.shape[0]
    spacings = (float(dv['Dx']), float(dv['Dy']), float(dv['Dz']))
    dt = float(dv['Dt']) if Nt >= 3 else None
    sigma2 = a.sigma2_factor * float(m.filter.filter_width) ** 2 / 12.0
    vel = u[..., 1:] / u[..., 0:1]

    _, B_lab = Base.observer_frame_fields(st['FaradayTensor'], np.array([1.0, 0.0, 0.0, 0.0]), m.metric)
    g = OR.gradient_model_operators(vel, B_lab[..., 1:], st['BC'][..., 0], u, spacings, sigma2, dt)
    pred_E = g['prediction']
    lm = OR.lorentz_force_model(st['ChargeCurrent'], st['FaradayTensor'], u, vel, spacings, sigma2, metric, dt)
    pred_F = lm['cov'] + lm['charge']
    F_data = mv['F_closure']

    sl = central_slice(u.shape[:4])
    ratio = total_force_ratio(m)[sl]
    tail = ratio > a.threshold
    mink = lambda x, y: OR.mink_dot(x[sl], y[sl])
    ul = np.einsum('ab,...b->...a', metric, u)                       # u~_a
    PJ_data, PJ_model = mink(J, E), mink(J, pred_E)
    PF_data = -np.einsum('...a,...a->...', ul[sl], F_data[sl])
    PF_model = -np.einsum('...a,...a->...', ul[sl], pred_F[sl])
    print("points: %d; tail %.1f%% (|F_closure| > %.2g x total force)" % (tail.size, 100 * tail.mean(), a.threshold))
    masks = (("all", np.ones_like(tail)), ("tail", tail), ("rest", ~tail))
    rows = {}
    for name, d_, m_ in (("J.E (sub-filter Joule heating)", PJ_data, PJ_model), ("-u.F_closure (work on the fluid)", PF_data, PF_model)):
        print("\n%s" % name)
        print("  %-8s %-6s %14s %20s %12s %16s" % ("", "", "frac negative", "backscatter weighted", "net", "model sign agrees"))
        for who, P in (("data", d_), ("model", m_)):
            for lab, mk in masks:
                f, b, n = summarise(P, mk)
                agree = float(np.mean(np.sign(d_[mk]) == np.sign(m_[mk]))) if who == "model" else np.nan
                rows[(name, who, lab)] = (f, b, n, agree)
                print("  %-8s %-6s %14.3f %20.3f %12.3f %16s" % (who, lab, f, b, n, ("%.3f" % agree) if who == "model" else ""))

    print("\nR2 of the parameter-free prediction restricted to the tail / the rest (the overall R2 is |.|^2 weighted):")
    r2 = {}
    for name, tgt, prd, w in (("E~", E, pred_E, None), ("F_closure", F_data, pred_F, np.ones(4))):
        t_, p_ = tgt[sl], prd[sl]
        for lab, mk in masks:
            r2[(name, lab)] = OR.prediction_r2(t_[mk], p_[mk], w)
        share = float(OR._wdot(t_[tail], t_[tail], w).sum() / OR._wdot(t_.reshape(-1, 4), t_.reshape(-1, 4), w).sum())
        print("  %-10s all %.3f   tail %.3f   rest %.3f   (the tail holds %.0f%% of sum |.|^2)" % (
            name, r2[(name, 'all')], r2[(name, 'tail')], r2[(name, 'rest')], 100 * share))

    os.makedirs(a.outdir, exist_ok=True)
    keys = list(rows)
    np.savez(os.path.join(a.outdir, "backscatter.npz"), keys=np.array(["|".join(k) for k in keys]),
             values=np.array([rows[k] for k in keys]), r2_keys=np.array(["|".join(k) for k in r2]),
             r2_values=np.array(list(r2.values())), threshold=a.threshold, sigma2=sigma2)
    print("\nWrote backscatter.npz to %s" % a.outdir)


if __name__ == "__main__":
    main()
