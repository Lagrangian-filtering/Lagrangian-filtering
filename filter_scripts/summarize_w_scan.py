#!/usr/bin/env python
"""Tabulate the regional regression against filter width W from scan_results/w*/ohms_regression.npz."""
import glob
import os
import sys

import numpy as np

def _label(path):
    with np.load(path) as z:
        return float(z["label"])


root = sys.argv[1] if len(sys.argv) > 1 else "scan_results"
rows = []
for f in glob.glob(os.path.join(root, "w*", "ohms_regression.npz")):
    with np.load(f) as _z:
        d = {k: _z[k] for k in _z.files}
    subs = list(d["subsets"])
    iR, iF = subs.index("R"), subs.index("R+alpha+gamma")
    R2, sh, c = d["global_R2"], d["shift_null_R2"], d["global_coeffs"]
    tile = d["R_only_tile_R_tilde"]
    p16, med, p84 = np.percentile(tile, [16, 50, 84]) if len(tile) else (np.nan,) * 3
    rows.append((float(d["label"]), R2[iR], np.percentile(sh[iR], 95), c[iR, 0], med, p16, p84,
                 len(tile), R2[iF], np.percentile(sh[iF], 95), R2[subs.index("alpha")], R2[subs.index("gamma")]))
if not rows:
    sys.exit("no results under %s" % root)
print("%5s | %-30s | %-33s | %-14s | %s" % ("W/dx", "R-only: R2  shift95  R_tilde", "R_tilde over tiles: med [p16, p84] n",
      "full: R2 shift95", "alpha-only R2, gamma-only R2"))
for r in sorted(rows):
    print("%5g | %7.3f %8.3f %12.3g | %11.3g [%9.3g, %9.3g] %2d | %7.3f %7.3f | %6.3f %6.3f" % r)
print("\nHALL-ONLY fit (gamma): is it a stable constant?")
print("%5s | %12s | %11s [%9s, %9s] | %s" % ("W/dx", "global gamma", "tile median", "p16", "p84", "tiles with gamma > 0"))
for f in sorted(glob.glob(os.path.join(root, "w*", "ohms_regression.npz")), key=lambda x: _label(x)):
    with np.load(f) as _z:
        d = {k: _z[k] for k in _z.files}
    if "gamma_only_tile_gamma" not in d:
        print("%5g | (re-run regress_ohms_law.py: this npz predates the gamma columns)" % float(d["label"]))
        continue
    g = d["gamma_only_tile_gamma"]
    p16, med, p84 = np.percentile(g, [16, 50, 84])
    print("%5g | %12.3g | %11.3g [%9.3g, %9.3g] | %d%% of %d" % (
        float(d["label"]), d["global_coeffs"][2, 2], med, p16, p84, round(100 * np.mean(g > 0)), len(g)))
print("\nA closure for sub-filter turbulence should show R_tilde varying smoothly with W. Compare R2 "
      "with shift95 (the correlation-preserving null), not with a point permutation.")
