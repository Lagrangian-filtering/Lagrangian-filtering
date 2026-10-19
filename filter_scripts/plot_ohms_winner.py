#!/usr/bin/env python
"""
Maps of which Ohm's-law term describes E_tilde where, on a pickled resMHD3d meso model.

For every meso point the three terms (resistive R_tilde J, dynamo alpha B, Hall gamma J x B)
are fitted one at a time over a small window around it (all time slices included), giving a
local R2 for each. The "winner" is the term with the highest local R2, or "none" if no term
beats chance. Chance is the 95th percentile of the best local R2 when E_tilde is circularly
shifted relative to the operators (spatial structure kept, alignment broken), so a coloured
cell means "better than chance", not just "best of three". Override it with --min-r2.

    python plot_ohms_winner.py /path/to/resMHD3d_khi576_w4.pickle --window 5 5 3

Writes ohms_winner.png and ohms_winner_maps.npz to --outdir. Non-interactive (Agg).
Rows: xy plane at the chosen z index, xz plane at the chosen y index.
Columns: winner | best local R2 | local R_tilde (single-term fit) | local gamma (single-term fit).
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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

import OhmsRegression as OR
from MesoModels import *            # noqa: F401  (needed so the pickled classes can be loaded)
from system.BaseFunctionality import Base

LABELS = ['none', 'R (resistive)', 'alpha (dynamo)', 'gamma (Hall)']
COLOURS = ['#d9d9d9', '#d62728', '#2ca02c', '#1f77b4']


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pickle_file")
    p.add_argument("--window", type=int, nargs=3, default=[5, 5, 3], metavar=("WX", "WY", "WZ"),
                   help="odd window size in meso cells for the local fits")
    p.add_argument("--n-null", type=int, default=20)
    p.add_argument("--min-r2", type=float, default=None, help="override the chance threshold")
    p.add_argument("--z-index", type=int, default=None, help="meso z index for the xy plane (default middle)")
    p.add_argument("--y-index", type=int, default=None, help="meso y index for the xz plane (default middle)")
    p.add_argument("--outdir", default=".")
    p.add_argument("--dpi", type=int, default=130)
    a = p.parse_args()
    window = tuple(a.window)
    if any(w % 2 == 0 for w in window):
        sys.exit("window sizes must be odd")

    with open(a.pickle_file, 'rb') as fh:
        m = pickle.load(fh)
    v = m.meso_vars
    E, J, B, u = v['E_tilde'], v['J_tilde'], v['B_tilde'], v['u_tilde']
    levi_upper = Base.raise_LeviCivita4D(m.metric)
    H = np.einsum('abcd,...b,...c,...d->...a', levi_upper, J, u, B)   # same Hall operator as the fit
    ops = {'R_tilde': J, 'alpha_dynamo': B, 'gamma_hall': H}
    X, Y, Z = (np.asarray(m.domain_vars[k], dtype=float) for k in 'XYZ')

    maps = OR.local_term_maps(E, ops, window)
    thr = a.min_r2 if a.min_r2 is not None else OR.local_null_threshold(E, ops, window, n_null=a.n_null)
    win = OR.winner_map(maps['R2'], thr)
    win = np.where(maps['valid'], win, -2)          # -2: window mostly outside the grid
    best = np.nanmax(np.where(np.isnan(maps['R2']), -np.inf, maps['R2']), axis=0)
    best = np.where(maps['valid'] & np.isfinite(best), best, np.nan)

    ok = win >= -1
    print("meso grid %s, window %s, chance threshold for best local R2 = %.3f" % (E.shape[:4], window, thr))
    for code, name in zip((-1, 0, 1, 2), LABELS):
        print("  %-16s %5.1f%% of valid cells" % (name, 100.0 * np.sum(win == code) / max(1, ok.sum())))

    os.makedirs(a.outdir, exist_ok=True)
    np.savez(os.path.join(a.outdir, "ohms_winner_maps.npz"), winner=win, best_R2=best, R2=maps['R2'],
             coeff=maps['coeff'], R2_full=maps['R2_full'], threshold=thr, X=X, Y=Y, Z=Z)

    iz = a.z_index if a.z_index is not None else len(Z) // 2
    iy = a.y_index if a.y_index is not None else len(Y) // 2
    planes = [("xy", (X, Y), lambda f: f[:, :, iz], "z = %.3f" % Z[iz]),
              ("xz", (X, Z), lambda f: f[:, iy, :], "y = %.3f" % Y[iy])]

    def extent(ax1, ax2):
        d1 = (ax1[1] - ax1[0]) if len(ax1) > 1 else 1.0
        d2 = (ax2[1] - ax2[0]) if len(ax2) > 1 else 1.0
        return [ax1[0] - d1 / 2, ax1[-1] + d1 / 2, ax2[0] - d2 / 2, ax2[-1] + d2 / 2]

    cmap = ListedColormap(['white'] + COLOURS)       # -2 white, -1 grey, 0 red, 1 green, 2 blue
    norm = BoundaryNorm(np.arange(-2.5, 3.5, 1.0), cmap.N)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
    cR, cG = maps['coeff'][0], maps['coeff'][2]
    lim = lambda f: np.nanpercentile(np.abs(f[maps['valid']]), 98) or 1.0
    for row, (plane, (a1, a2), take, where) in enumerate(planes):
        ext = extent(a1, a2)
        axes[row, 0].imshow(take(win).T, origin='lower', extent=ext, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')
        axes[row, 0].set_title("winner, %s-plane at %s" % (plane, where))
        im = axes[row, 1].imshow(take(best).T, origin='lower', extent=ext, cmap='viridis', vmin=0, aspect='equal', interpolation='nearest')
        axes[row, 1].set_title("best local R2 (chance %.2f)" % thr)
        fig.colorbar(im, ax=axes[row, 1], shrink=0.8)
        for col, (f, name) in enumerate(((cR, "local R_tilde"), (cG, "local gamma")), start=2):
            fm = np.where(maps['valid'], f, np.nan)
            L = lim(f)
            im = axes[row, col].imshow(take(fm).T, origin='lower', extent=ext, cmap='RdBu_r', vmin=-L, vmax=L,
                                       aspect='equal', interpolation='nearest')
            axes[row, col].set_title("%s (single-term fit)" % name)
            fig.colorbar(im, ax=axes[row, col], shrink=0.8)
        for col in range(4):
            axes[row, col].set_xlabel(plane[0]); axes[row, col].set_ylabel(plane[1])
    axes[0, 0].legend(handles=[Patch(color=c, label=l) for c, l in zip(COLOURS, LABELS)], loc='upper right', fontsize=8)
    fig.suptitle("%s   window %s meso cells   chance threshold %.3f" % (os.path.basename(a.pickle_file), window, thr))
    fig.savefig(os.path.join(a.outdir, "ohms_winner.png"), dpi=a.dpi)
    plt.close(fig)
    print("Wrote ohms_winner.png and ohms_winner_maps.npz to %s" % a.outdir)


if __name__ == "__main__":
    main()
