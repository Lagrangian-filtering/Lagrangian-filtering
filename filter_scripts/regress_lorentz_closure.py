#!/usr/bin/env python
"""
A-priori test of the second-order model of the sub-filter Lorentz force on a pickled resMHD3d meso model.

The momentum and energy equations of the filtered fluid contain the filtered Lorentz force <f^b> = <-j_a F^ab>, not
the force built from filtered fields. The pipeline stores the difference F_closure^b = <f^b> + J~_a F~^ab. To second order
(derivation in master_files/OhmsRegression.py, "Sub-filter Lorentz force")
    F_closure = -C - sigma~ u~_a F~^ab,   C^b = sigma^2 covariance_bracket(j, F),
where the second term is the force on the net charge (exact, from the filtered fields) and -C is the prediction, with no
fitted constant. This script scores that prediction (R2 = 1 - |F_closure - pred|^2/|F_closure|^2, can be negative),
regresses F_closure on the two terms (expected coefficients +1, +1), and gives the correlation-preserving shift
null and the tile spread of the coefficients.

    python regress_lorentz_closure.py /path/to/resMHD3d_khi576_w4_3sl_c2.pickle

Use a pickle with 3 meso time slices and a fine meso spacing (the model contains the gradient of j~, itself a derivative
of F~). Only the central time slice is scored and one edge cell is dropped. Norms are Euclidean over the four components.
Does not modify the pickle.
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
from model_report import report


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pickle_file")
    p.add_argument("--block", type=int, nargs=3, default=[6, 6, 3])
    p.add_argument("--min-points", type=int, default=100)
    p.add_argument("--n-null", type=int, default=50)
    p.add_argument("--label", default="", help="stored in the npz, e.g. the filter width of a scan")
    p.add_argument("--sigma2-factor", type=float, default=1.0,
                   help="multiply the nominal sigma^2 = W^2/12 by this (see regress_gradient_model.py)")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()

    with open(a.pickle_file, 'rb') as fh:
        m = pickle.load(fh)
    st, dv, mv = m.meso_structures, m.domain_vars, m.meso_vars
    u, F_closure = mv['u_tilde'], mv['F_closure']
    j, F = st['ChargeCurrent'], st['FaradayTensor']
    metric = np.asarray(m.metric, dtype=float)
    Nt = u.shape[0]
    spacings = (float(dv['Dx']), float(dv['Dy']), float(dv['Dz']))
    dt = float(dv['Dt']) if Nt >= 3 else None
    sigma2 = a.sigma2_factor * float(m.filter.filter_width) ** 2 / 12.0
    print("meso grid %s, filter width %.4g (sigma^2 = %.3g, factor %.3g), meso spacing %.4g" % (
        u.shape[:4], m.filter.filter_width, sigma2, a.sigma2_factor, spacings[0]))
    if dt is None:
        print("WARNING: only %d time slice(s): time derivatives set to zero, the model is INCOMPLETE" % Nt)

    vel = u[..., 1:] / u[..., 0:1]
    mod = OR.lorentz_force_model(j, F, u, vel, spacings, sigma2, metric, dt)

    t_sl = slice(Nt // 2, Nt // 2 + 1) if Nt >= 3 else slice(None)
    s_sl = tuple(slice(1, -1) if n >= 3 else slice(None) for n in u.shape[1:4])
    sl = (t_sl,) + s_sl
    target = F_closure[sl]
    ops = {k: x[sl] for k, x in mod.items()}
    w = np.ones(4)
    # the mean-field force itself, for scale: -j~_a F~^ab
    f_mean = -np.einsum('ac,...c,...ab->...b', metric, j[sl], F[sl])
    nrm = lambda x: np.sqrt(np.sum(x ** 2, axis=-1))
    print("points used: %d (central time slice, edge cells dropped)" % int(np.prod(target.shape[:4])))
    print("sizes: median |F_closure| / |-j~_a F~^ab| = %.3g;  |charge term| / |F_closure| = %.3g;  |cov term| / |F_closure| = %.3g\n" % (
        np.median(nrm(target) / np.maximum(nrm(f_mean), 1e-300)), np.median(nrm(ops['charge']) / nrm(target)),
        np.median(nrm(ops['cov']) / nrm(target))))
    print("R2 of each term alone as a parameter-free prediction: charge %.3f, cov %.3f, both %.3f\n" % (
        OR.prediction_r2(target, ops['charge'], w), OR.prediction_r2(target, ops['cov'], w),
        OR.prediction_r2(target, ops['charge'] + ops['cov'], w)))

    rows = {}
    report("lorentz", target, ops, w, "+1, +1", tuple(a.block), a.min_points, a.n_null, rows)

    os.makedirs(a.outdir, exist_ok=True)
    out = dict(label=a.label, sigma2=sigma2, filter_width=float(m.filter.filter_width), n_slices=Nt,
               charge_R2=OR.prediction_r2(target, ops['charge'], w), cov_R2=OR.prediction_r2(target, ops['cov'], w))
    for name, r in rows.items():
        out.update({name + '_' + k: v for k, v in r.items()})
    np.savez(os.path.join(a.outdir, "lorentz_closure.npz"), **out)
    print("Wrote lorentz_closure.npz to %s" % a.outdir)


if __name__ == "__main__":
    main()
