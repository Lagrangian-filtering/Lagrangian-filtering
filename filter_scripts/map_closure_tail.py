#!/usr/bin/env python
"""
Where is the sub-filter Lorentz force large? Map and characterise the "tail" of F_closure on a pickled resMHD3d meso model.

A point is in the tail when |F_closure| > --threshold (default 0.1) times the total force |d_a <T^{ai}>| on the filtered
fluid (spatial components; central time slice, edge cells dropped; see compare_force_scales.py). For each local quantity
(|J~|, |B~|, plasma beta, sub-filter B^2, density, pressure, speed, |E~|, gradients of |B~| and |v|) it prints the median inside and
outside the tail, the rank correlation with the ratio |F_closure| / |d_a<T^{ai}>|, and how much of the tail lies in the top
decile of that quantity (10 % would be no enrichment). It also reports how clustered the tail is (connected regions) and
draws maps of the ratio, the tail, and three of the quantities in an xy and an xz plane.

    python map_closure_tail.py /path/to/resMHD3d_khi576_w4_3sl_c2.pickle --outdir tail_w4_c2

Does not modify the pickle. Non-interactive (Agg).
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'master_files'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import OhmsRegression as OR
from MesoModels import *            # noqa: F401  (needed so the pickled classes can be loaded)


def ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind='stable')] = np.arange(len(x))
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float('nan')


def grad_norm(f, spacings):
    return np.sqrt(sum(np.gradient(f, spacings[l], axis=l + 1) ** 2 for l in range(3)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pickle_file")
    p.add_argument("--threshold", type=float, default=0.1, help="tail: |F_closure| > threshold * |d_a<T^{ai}>|")
    p.add_argument("--outdir", default=".")
    p.add_argument("--dpi", type=int, default=120)
    a = p.parse_args()

    with open(a.pickle_file, 'rb') as fh:
        m = pickle.load(fh)
    st, dv, mv = m.meso_structures, m.domain_vars, m.meso_vars
    Nt = mv['u_tilde'].shape[0]
    spacings = (float(dv['Dx']), float(dv['Dy']), float(dv['Dz']))
    dt = float(dv['Dt']) if Nt >= 3 else None
    T = np.asarray(st['SET'], dtype=float)
    div = np.zeros(T.shape[:4] + (3,))
    if dt:
        div += np.gradient(T[..., 0, 1:], dt, axis=0)
    for l in range(3):
        div += np.gradient(T[..., l + 1, 1:], spacings[l], axis=l + 1)

    u = np.asarray(mv['u_tilde'], dtype=float)
    vel = np.sqrt(np.sum(u[..., 1:] ** 2, axis=-1)) / u[..., 0]
    eucl = lambda x: np.sqrt(np.sum(np.asarray(x, dtype=float)[..., 1:] ** 2, axis=-1))
    J, B, E = eucl(mv['J_tilde']), eucl(mv['B_tilde']), eucl(mv['E_tilde'])
    p_t, n_t = np.asarray(mv['p_tilde'], dtype=float), np.asarray(mv['n_tilde'], dtype=float)
    q = {'|J~|': J, '|B~|': B, 'beta = p/(B^2/2)': p_t / np.maximum(0.5 * B ** 2, 1e-300), 'density n~': n_t,
         'pressure p~': p_t, 'speed |v~|': vel, '|E~|': E, '|grad |B~||': grad_norm(B, spacings), '|grad |v~||': grad_norm(vel, spacings)}
    if 'B2_sub' in mv and np.any(np.asarray(mv['B2_sub']) != 0):
        q['sub-filter B^2'] = np.asarray(mv['B2_sub'], dtype=float)

    Fc = np.sqrt(np.sum(np.asarray(mv['F_closure'], dtype=float)[..., 1:] ** 2, axis=-1))
    ratio = Fc / np.maximum(np.sqrt(np.sum(div ** 2, axis=-1)), 1e-300)
    t_sl = slice(Nt // 2, Nt // 2 + 1) if Nt >= 3 else slice(None)
    s_sl = tuple(slice(1, -1) if n >= 3 else slice(None) for n in T.shape[1:4])
    sl = (t_sl,) + s_sl
    tail3 = ratio[sl][0] > a.threshold                                     # (Nx, Ny, Nz) at the central slice
    r, tail = ratio[sl].ravel(), (ratio[sl] > a.threshold).ravel()
    print("points: %d; tail (|F_closure| > %.2g |d_a<T^{ai}>|): %d (%.1f%%)" % (r.size, a.threshold, tail.sum(), 100.0 * tail.mean()))
    if tail.sum() == 0 or tail.sum() == tail.size:
        sys.exit("tail is empty or everything: change --threshold")

    print("\n%-20s %12s %12s %10s %14s" % ("quantity", "median tail", "median rest", "rank corr", "tail in top 10%"))
    rows = {}
    for name, f in q.items():
        x = f[sl].ravel()
        top = x >= np.percentile(x, 90)
        rho = spearman(r, x)
        rows[name] = (float(np.median(x[tail])), float(np.median(x[~tail])), rho, float(top[tail].mean()))
        print("%-20s %12.3g %12.3g %10.2f %13.0f%%" % ((name,) + rows[name][:3] + (100 * rows[name][3],)))

    clusters = None
    try:
        from scipy import ndimage
        lab, nlab = ndimage.label(tail3)
        sizes = np.bincount(lab.ravel())[1:]
        clusters = (int(nlab), int(sizes.max()) if nlab else 0)
        print("\nconnected tail regions (face-adjacent): %d; the largest holds %.0f%% of the tail points; median region %d points" % (
            nlab, 100.0 * sizes.max() / tail.sum(), int(np.median(sizes))))
    except ImportError:
        print("\n(scipy not available: clustering skipped)")

    # tail fraction by z-plane and by xy: is it concentrated?
    fz = tail3.mean(axis=(0, 1))
    print("tail fraction by z-plane: min %.2f, median %.2f, max %.2f" % (fz.min(), np.median(fz), fz.max()))

    os.makedirs(a.outdir, exist_ok=True)
    np.savez(os.path.join(a.outdir, "closure_tail.npz"), ratio=ratio[sl], tail=ratio[sl] > a.threshold, threshold=a.threshold,
             names=np.array(list(rows)), median_tail=np.array([v[0] for v in rows.values()]),
             median_rest=np.array([v[1] for v in rows.values()]), rank_corr=np.array([v[2] for v in rows.values()]),
             top_decile_share=np.array([v[3] for v in rows.values()]),
             n_clusters=(clusters[0] if clusters else -1))

    X, Y, Z = (np.asarray(dv[k], dtype=float) for k in 'XYZ')
    ix = lambda f: f[sl][0]                                                # (Nx, Ny, Nz) at the central slice
    iz, iy = len(Z[s_sl[2]]) // 2, len(Y[s_sl[1]]) // 2
    Xs, Ys, Zs = X[s_sl[0]], Y[s_sl[1]], Z[s_sl[2]]
    panels = [("log10 |F_closure| / |d_a<T^{ai}>|", np.log10(np.maximum(ix(ratio), 1e-6)), "viridis"),
              ("tail", ix(ratio) > a.threshold, "gray_r"), ("|J~|", ix(J), "magma"), ("|B~|", ix(B), "magma")]
    if 'sub-filter B^2' in q:
        panels.append(("sub-filter B^2", ix(q['sub-filter B^2']), "magma"))
    fig, axes = plt.subplots(2, len(panels), figsize=(4.8 * len(panels), 8), constrained_layout=True)
    for row, (take, a1, a2, lab_) in enumerate(((lambda f: f[:, :, iz], Xs, Ys, "z = %.3f" % Zs[iz]),
                                                (lambda f: f[:, iy, :], Xs, Zs, "y = %.3f" % Ys[iy]))):
        d1 = (a1[1] - a1[0]) if len(a1) > 1 else 1.0
        d2 = (a2[1] - a2[0]) if len(a2) > 1 else 1.0
        ext = [a1[0] - d1 / 2, a1[-1] + d1 / 2, a2[0] - d2 / 2, a2[-1] + d2 / 2]
        for col, (title, f, cm) in enumerate(panels):
            im = axes[row, col].imshow(take(np.asarray(f, dtype=float)).T, origin='lower', extent=ext, cmap=cm, aspect='equal', interpolation='nearest')
            axes[row, col].set_title("%s (%s)" % (title, lab_), fontsize=9)
            fig.colorbar(im, ax=axes[row, col], shrink=0.8)
    fig.suptitle("%s   tail threshold %.2g" % (os.path.basename(a.pickle_file), a.threshold))
    fig.savefig(os.path.join(a.outdir, "closure_tail.png"), dpi=a.dpi)
    plt.close(fig)
    print("\nWrote closure_tail.png and closure_tail.npz to %s" % a.outdir)


if __name__ == "__main__":
    main()
