#!/usr/bin/env python
"""
Regional regression of the generic Ohm's law on a pickled resMHD_3D meso model.

Fits one set of constant coefficients (R_tilde, alpha, gamma) to all meso points in a
region, for every subset of the three terms, and compares R^2 with two nulls: a point permutation
(E paired with operators from random other points; too optimistic for correlated fields)
and a circular shift of E (keeps spatial correlation; the honest chance level). Run it in the filtering environment:

    python regress_ohms_law.py /path/to/resMHD3d_khi576.pickle --block 6 6 4

It does not modify the pickle. Writes <outdir>/ohms_regression.npz and ohms_regression.png.

How to read it: R^2 = 1 - |residual|^2 / |E_tilde|^2 summed over the region (no intercept).
The permutation null shows the chance level. The spread of coefficients across tiles is a
more honest uncertainty than any formal error, because neighbouring meso points are strongly
correlated. A term whose coefficient changes sign between tiles is not a constant closure.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'master_files'))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import OhmsRegression as OR
from MesoModels import *            # noqa: F401  (needed so the pickled classes can be loaded)
from system.BaseFunctionality import Base

SHORT = {'R_tilde': 'R', 'alpha_dynamo': 'alpha', 'gamma_hall': 'gamma'}


def label(names):
    return '+'.join(SHORT[n] for n in names)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pickle_file")
    p.add_argument("--block", type=int, nargs=3, default=[6, 6, 4], metavar=("BX", "BY", "BZ"),
                   help="tile size in meso cells (all time slices are included in a tile)")
    p.add_argument("--min-points", type=int, default=100, help="skip tiles with fewer points")
    p.add_argument("--n-null", type=int, default=100)
    p.add_argument("--label", default="", help="stored in the npz, e.g. the filter width of a scan")
    p.add_argument("--min-shift", type=int, default=3,
                   help="minimum circular shift (meso cells) for the correlation-preserving null")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()

    with open(a.pickle_file, 'rb') as fh:
        m = pickle.load(fh)
    v = m.meso_vars
    E, J, B, u = v['E_tilde'], v['J_tilde'], v['B_tilde'], v['u_tilde']
    # Same Hall operator as resMHD_3D.ohms_law_fit_gridpoint: eps^{abcd} J_b u_c B_d
    levi_upper = Base.raise_LeviCivita4D(m.metric)
    H = np.einsum('abcd,...b,...c,...d->...a', levi_upper, J, u, B)
    ops = {'R_tilde': J, 'alpha_dynamo': B, 'gamma_hall': H}
    shape = E.shape[:4]
    print("meso grid (Nt,Nx,Ny,Nz) = %s, %d points; tile %s" % (shape, int(np.prod(shape)), tuple(a.block)))

    table = OR.subset_table(E, ops)
    results = []
    print("\nGLOBAL constant-coefficient fit (one set of coefficients for the whole region)")
    print("%-18s %8s %8s %8s | %10s %10s %10s | tiles: R2 med [p16,p84]; coefficient medians" % (
        "terms", "R2", "ptnull95", "shnull95", "R_tilde", "alpha", "gamma"))
    for fit in table:
        names = fit['names']
        null = OR.permutation_null(E, ops, names, n_perm=a.n_null)
        shnull = OR.shift_null(E, ops, names, n_perm=a.n_null, min_shift=a.min_shift)
        tiles = OR.tile_fits(E, ops, names, tuple(a.block), min_points=a.min_points)
        r2 = OR.summarize([t['R2'] for t in tiles])
        coef = {t: OR.summarize([x[t] for x in tiles]) for t in names}
        results.append({'names': names, 'fit': fit, 'null': null, 'shnull': shnull, 'tiles': tiles})
        print("%-18s %8.3f %8.3f %8.3f | %10.3g %10.3g %10.3g | %d tiles: %.2f [%.2f,%.2f]; %s" % (
            label(names), fit['R2'], np.percentile(null, 95), np.percentile(shnull, 95),
            fit['R_tilde'], fit['alpha_dynamo'], fit['gamma_hall'],
            r2['n'], r2['median'], r2['p16'], r2['p84'],
            "  ".join("%s %.2g (%d%%>0)" % (SHORT[t], coef[t]['median'], round(100 * coef[t]['frac_pos']))
                      for t in names)))

    full = results[-1]
    print("\nFULL MODEL, coefficients across tiles (median [16th, 84th pct], fraction > 0):")
    for t in OR.TERMS:
        s = OR.summarize([x[t] for x in full['tiles']])
        print("  %-13s %10.3g [%10.3g, %10.3g]  %3d%% positive over %d tiles" % (
            t, s['median'], s['p16'], s['p84'], round(100 * s['frac_pos']), s['n']))
    print("\nA coefficient that changes sign between tiles, or an R2 within the null range, "
          "means that term is not a constant closure for this data.")

    os.makedirs(a.outdir, exist_ok=True)
    np.savez(os.path.join(a.outdir, "ohms_regression.npz"),
             subsets=np.array([label(r['names']) for r in results]),
             global_R2=np.array([r['fit']['R2'] for r in results]),
             null_R2=np.array([r['null'] for r in results]),
             shift_null_R2=np.array([r['shnull'] for r in results]),
             label=a.label,
             global_coeffs=np.array([[r['fit'][t] for t in OR.TERMS] for r in results]),
             gamma_only_tile_gamma=np.array([x['gamma_hall'] for x in results[2]['tiles']]),
             R_only_tile_R_tilde=np.array([x['R_tilde'] for x in results[0]['tiles']]),
             full_tile_R_tilde=np.array([x['R_tilde'] for x in full['tiles']]),
             full_tile_alpha=np.array([x['alpha_dynamo'] for x in full['tiles']]),
             full_tile_gamma=np.array([x['gamma_hall'] for x in full['tiles']]),
             full_tile_R2=np.array([x['R2'] for x in full['tiles']]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    labels = [label(r['names']) for r in results]
    x = np.arange(len(results))
    axes[0].bar(x, [r['fit']['R2'] for r in results], color="C0", label="data")
    axes[0].plot(x, [np.percentile(r['null'], 95) for r in results], "_", color="orange", ms=18, mew=3,
                 label="point-permutation null (95th pct)")
    axes[0].plot(x, [np.percentile(r['shnull'], 95) for r in results], "r_", ms=18, mew=3,
                 label="shift null, keeps correlation (95th pct)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=30)
    axes[0].set_ylabel("global R2"); axes[0].set_ylim(0, 1); axes[0].legend()
    axes[0].set_title("constant-coefficient fit, whole region")
    axes[1].boxplot([[t['R2'] for t in r['tiles']] for r in results])
    axes[1].set_xticks(np.arange(1, len(labels) + 1))   # 'labels=' was renamed/removed across matplotlib versions
    axes[1].set_xticklabels(labels, rotation=30)
    axes[1].set_ylabel("tile R2"); axes[1].set_ylim(0, 1)
    axes[1].set_title("R2 per tile (block %s)" % (tuple(a.block),))
    fig.savefig(os.path.join(a.outdir, "ohms_regression.png"), dpi=130)
    print("\nWrote ohms_regression.npz and ohms_regression.png to %s" % a.outdir)


if __name__ == "__main__":
    main()
