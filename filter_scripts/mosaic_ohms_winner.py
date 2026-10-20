#!/usr/bin/env python
"""
Winner map of the Ohm's-law terms over a large region, from many abutting crops.

Loads the meso pickles of crops submitted with mosaic_submit.sh, places their meso fields on one
global lattice (OhmsRegression.merge_meso_tiles), and then runs the local fits and the winner
map on the MERGED grid, so there are no seams at the crop boundaries. The chance threshold is the
shift-null of plot_ohms_winner.py. Missing crops leave holes (blank); windows less than half
covered are blank.

    python mosaic_ohms_winner.py '/scratch/.../pickled/resMHD3d_khi576_m_*.pickle' --outdir mosaic

Writes mosaic_winner.png and mosaic_maps.npz. Non-interactive (Agg). The figure shows the xy plane at
the chosen z index: winner | best local R2 | local R_tilde | local gamma | filtered density | |B_tilde|^2 |
sub-filter B^2 (<B^2> - |<B>|^2, when every crop pickle has it).
"""
import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'master_files'))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

import OhmsRegression as OR
from MesoModels import *            # noqa: F401  (needed so the pickled classes can be loaded)
from system.BaseFunctionality import Base

LABELS = ['none', 'R (resistive)', 'alpha (dynamo)', 'gamma (Hall)']
COLOURS = ['#d9d9d9', '#d62728', '#2ca02c', '#1f77b4']


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pickle_glob", help="quoted glob of the crop pickles")
    p.add_argument("--window", type=int, nargs=3, default=[7, 7, 3], metavar=("WX", "WY", "WZ"))
    p.add_argument("--n-null", type=int, default=20)
    p.add_argument("--min-r2", type=float, default=None)
    p.add_argument("--z-index", type=int, default=None, help="merged-grid z index for the plane (default middle)")
    p.add_argument("--outdir", default=".")
    p.add_argument("--dpi", type=int, default=110)
    a = p.parse_args()
    window = tuple(a.window)
    if any(w % 2 == 0 for w in window):
        sys.exit("window sizes must be odd")

    files = sorted(glob.glob(a.pickle_glob))
    if not files:
        sys.exit("no pickles match %s" % a.pickle_glob)
    tiles, spacing, metric = [], None, None
    for f in files:
        t0 = time.time()
        with open(f, 'rb') as fh:
            m = pickle.load(fh)
        v = m.meso_vars
        tiles.append({'X': np.asarray(m.domain_vars['X'], dtype=float), 'Y': np.asarray(m.domain_vars['Y'], dtype=float),
                      'Z': np.asarray(m.domain_vars['Z'], dtype=float),
                      'fields': {k: np.asarray(v[n], dtype=float) for k, n in
                                 (('E', 'E_tilde'), ('J', 'J_tilde'), ('B', 'B_tilde'), ('u', 'u_tilde'), ('n', 'n_tilde'))}})
        if 'B2_sub' in v and np.any(np.asarray(v['B2_sub']) != 0):
            tiles[-1]['fields']['S'] = np.asarray(v['B2_sub'], dtype=float)
        if spacing is None:
            spacing = tuple(float(m.domain_vars[k]) for k in ('Dx', 'Dy', 'Dz'))
            metric = np.array(m.metric)
        del m, v
        print("loaded %s in %.1f s" % (os.path.basename(f), time.time() - t0), flush=True)

    if not all('S' in t['fields'] for t in tiles):       # sub-filter energy only if every crop has it
        for t in tiles:
            t['fields'].pop('S', None)
        print("note: not every pickle has B2_sub; sub-filter energy panel omitted")
    mer = OR.merge_meso_tiles(tiles, spacing)
    del tiles
    F = mer['fields']
    ops = OR.standard_operators(F['J'], F['B'], F['u'], Base.raise_LeviCivita4D(metric), metric)
    present = mer['present']
    X, Y, Z = mer['X'], mer['Y'], mer['Z']
    print("mosaic grid (Nx,Ny,Nz) = %s from %d crops; %.0f%% of lattice points covered" % (
        present.shape, len(files), 100.0 * present.mean()))

    maps = OR.local_term_maps(F['E'], ops, window)
    thr = a.min_r2 if a.min_r2 is not None else OR.local_null_threshold(F['E'], ops, window, n_null=a.n_null)
    covered = OR.box_sum(present.astype(float), window) >= 0.5 * np.prod(window)
    valid = maps['valid'] & covered
    win = np.where(valid, OR.winner_map(maps['R2'], thr), -2)
    best = np.nanmax(np.where(np.isnan(maps['R2']), -np.inf, maps['R2']), axis=0)
    best = np.where(valid & np.isfinite(best), best, np.nan)
    print("window %s, chance threshold for best local R2 = %.3f" % (window, thr))
    for code, name in zip((-1, 0, 1, 2), LABELS):
        print("  %-16s %5.1f%% of valid cells" % (name, 100.0 * np.sum(win == code) / max(1, valid.sum())))

    n_map = np.where(present, F['n'].mean(axis=0), np.nan)
    B2_map = np.where(present, OR.mink_dot(F['B'], F['B']).mean(axis=0), np.nan)
    S_map = np.where(present, F['S'].mean(axis=0), np.nan) if 'S' in F else None
    os.makedirs(a.outdir, exist_ok=True)
    np.savez(os.path.join(a.outdir, "mosaic_maps.npz"), winner=win, best_R2=best, R2=maps['R2'], coeff=maps['coeff'],
             n_tilde=n_map, B2_tilde=B2_map, B2_sub=(S_map if S_map is not None else np.nan), present=present, threshold=thr, X=X, Y=Y, Z=Z)

    iz = a.z_index if a.z_index is not None else len(Z) // 2
    dX, dY = spacing[0], spacing[1]
    ext = [X[0] - dX / 2, X[-1] + dX / 2, Y[0] - dY / 2, Y[-1] + dY / 2]
    take = lambda f: f[:, :, iz].T
    cmap = ListedColormap(['white'] + COLOURS)
    norm = BoundaryNorm(np.arange(-2.5, 3.5, 1.0), cmap.N)
    fig, axes = plt.subplots(2, 4, figsize=(32, 16), constrained_layout=True)
    ax = axes.ravel()
    ax[0].imshow(take(win), origin='lower', extent=ext, cmap=cmap, norm=norm, interpolation='nearest')
    ax[0].set_title("winner, xy-plane at z = %.3f" % Z[iz])
    ax[0].legend(handles=[Patch(color=c, label=l) for c, l in zip(COLOURS, LABELS)], loc='upper right', fontsize=8)
    panels = [(best, "best local R2 (chance %.2f)" % thr, "viridis", False),
              (np.where(valid, maps['coeff'][0], np.nan), "local R_tilde (single-term)", "RdBu_r", True),
              (np.where(valid, maps['coeff'][2], np.nan), "local gamma (single-term)", "RdBu_r", True),
              (n_map, "filtered density n_tilde", "viridis", False),
              (B2_map, "|B_tilde|^2", "magma", False)]
    if S_map is not None:
        panels.append((S_map, "sub-filter B^2 = <B^2> - |<B>|^2", "magma", False))
    for k, (f, title, cm, sym) in enumerate(panels, start=1):
        d = take(f)
        kw = {}
        if sym:
            fin = np.abs(d[np.isfinite(d)])               # inf as well as NaN must not reach the colour limits
            L = (np.percentile(fin, 98) if fin.size else 0.0) or 1.0
            kw = dict(vmin=-L, vmax=L)
        im = ax[k].imshow(d, origin='lower', extent=ext, cmap=cm, interpolation='nearest', **kw)
        ax[k].set_title(title)
        fig.colorbar(im, ax=ax[k], shrink=0.8)
    for a_ in ax[len(panels) + 1:]:
        a_.axis('off')
    for a_ in ax[:len(panels) + 1]:
        a_.set_xlabel("x"); a_.set_ylabel("y"); a_.set_aspect('equal')
    fig.suptitle("Ohm's-law winner mosaic: %d crops, window %s meso cells" % (len(files), window))
    fig.savefig(os.path.join(a.outdir, "mosaic_winner.png"), dpi=a.dpi)
    plt.close(fig)
    print("Wrote mosaic_winner.png and mosaic_maps.npz to %s" % a.outdir)


if __name__ == "__main__":
    main()
