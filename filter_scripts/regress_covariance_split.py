#!/usr/bin/env python
"""
Covariance split of the second-order model of E_tilde: a separate coefficient for each piece.

The filter averages over a 3D spatial box in the observer's rest space, so to second order the covariance
of two fields is  sigma^2 [ grad a.grad b - d_t a d_t b + (u.d a)(u.d b) ]  (see OhmsRegression.py). Split by
derivative type this gives four pieces, for each of the two operators (v x B and the density weighting):

    ss  spatial gradients          tt  time derivatives
    ts  time-advective cross terms vv  advective (v.grad) terms

If the model is right every coefficient is -1 (v x B pieces) or +1 (density pieces). This script fits each piece
alone, all v x B pieces together, all density pieces together, and all eight, with the shift-null chance level for
each, plus the size of each piece relative to |E_tilde| and the spread of the joint coefficients across tiles. A
coefficient that departs from its expected value while its piece is large locates the part of the model that is off.

    python regress_covariance_split.py /path/to/resMHD3d_khi576_w4_3sl_c2.pickle --block 12 12 6

Needs a pickle with 3 meso time slices for the 'tt' and 'ts' pieces (with one slice they are zero and omitted).
The central time slice is used; one cell at the grid edges is dropped. Does not modify the pickle.
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

PIECES = ('ss', 'tt', 'ts', 'vv')


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pickle_file")
    p.add_argument("--block", type=int, nargs=3, default=[6, 6, 3])
    p.add_argument("--min-points", type=int, default=100)
    p.add_argument("--n-null", type=int, default=30)
    p.add_argument("--label", default="")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()

    with open(a.pickle_file, 'rb') as fh:
        m = pickle.load(fh)
    v, st, dv = m.meso_vars, m.meso_structures, m.domain_vars
    u, E = v['u_tilde'], v['E_tilde']
    Nt = E.shape[0]
    _, B_lab = Base.observer_frame_fields(st['FaradayTensor'], np.array([1.0, 0.0, 0.0, 0.0]), m.metric)
    D = st['BC'][..., 0]
    vel = u[..., 1:] / u[..., 0:1]
    spacings = (float(dv['Dx']), float(dv['Dy']), float(dv['Dz']))
    dt = float(dv['Dt']) if Nt >= 3 else None
    sigma2 = float(m.filter.filter_width) ** 2 / 12.0
    pieces = PIECES if dt is not None else ('ss', 'vv')
    if dt is None:
        print("WARNING: only %d time slice(s): the time pieces 'tt' and 'ts' are zero and omitted" % Nt)

    ops = OR.gradient_model_parts(vel, B_lab[..., 1:], D, u, spacings, sigma2, dt)
    t_sl = slice(Nt // 2, Nt // 2 + 1) if Nt >= 3 else slice(None)
    s_sl = tuple(slice(1, -1) if n >= 3 else slice(None) for n in E.shape[1:4])
    sl = (t_sl,) + s_sl
    E_ = E[sl]
    ops = {k: x[sl] for k, x in ops.items() if k.split('_')[1] in pieces}
    vB = tuple('vB_' + k for k in pieces)
    dens = tuple('dens_' + k for k in pieces)
    print("meso grid %s, W = %.4g, sigma^2 = %.3g, meso spacing %.4g; %d points used\n" % (
        E.shape[:4], m.filter.filter_width, sigma2, spacings[0], int(np.prod(E_.shape[:4]))))

    EE = OR.mink_dot(E_, E_).sum()
    print("SIZE OF EACH PIECE relative to |E_tilde|  (rms ratio):")
    for k in vB + dens:
        print("  %-9s %7.3f" % (k, np.sqrt(OR.mink_dot(ops[k], ops[k]).sum() / EE)))

    fits = [(k,) for k in vB + dens] + [vB, dens, vB + dens]
    print("\nCONSTANT-COEFFICIENT FITS (global; shift-null 95th percentile in brackets); expected -1 for vB_*, +1 for dens_*")
    print("%-44s %7s %9s   %s" % ("terms", "R2", "(null95)", "coefficients"))
    rows = {}
    for names in fits:
        fit = OR.fit_region(E_, ops, names)
        null = OR.shift_null(E_, ops, names, n_perm=a.n_null, min_shift=3)
        rows[names] = (fit['R2'], float(np.percentile(null, 95)), {n: fit[n] for n in names})
        label = "+".join(n.replace('vB_', 'vB:').replace('dens_', 'd:') for n in names)
        print("%-44s %7.3f (%7.3f)   %s" % (label if len(label) < 44 else "all %d pieces" % len(names), fit['R2'],
              rows[names][1], "  ".join("%s %.3g" % (n, fit[n]) for n in names)))

    tiles = OR.tile_fits(E_, ops, vB, tuple(a.block), min_points=a.min_points)
    print("\nvB PIECE COEFFICIENTS ACROSS %d TILES (joint fit; median [16th, 84th pct]):" % len(tiles))
    for n in vB:
        s_ = OR.summarize([t[n] for t in tiles])
        print("  %-6s %8.3f [%8.3f, %8.3f]" % (n, s_['median'], s_['p16'], s_['p84']))

    os.makedirs(a.outdir, exist_ok=True)
    np.savez(os.path.join(a.outdir, "covariance_split.npz"), label=a.label, pieces=np.array(pieces), n_slices=Nt,
             filter_width=float(m.filter.filter_width),
             fit_names=np.array(["+".join(k) for k in rows]), fit_R2=np.array([r[0] for r in rows.values()]),
             fit_null95=np.array([r[1] for r in rows.values()]),
             vB_joint_coeffs=np.array([rows[vB][2][n] for n in vB]), dens_joint_coeffs=np.array([rows[dens][2][n] for n in dens]),
             tile_vB=np.array([[t[n] for n in vB] for t in tiles]))
    print("\nWrote covariance_split.npz to %s" % a.outdir)


if __name__ == "__main__":
    main()
