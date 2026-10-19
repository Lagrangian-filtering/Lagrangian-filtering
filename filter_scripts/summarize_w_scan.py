#!/usr/bin/env python
"""Tabulate the regional regression against filter width W from scan_results/w*/ohms_regression.npz."""
import glob
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "scan_results"
rows = []
for f in glob.glob(os.path.join(root, "w*", "ohms_regression.npz")):
    d = np.load(f)
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
print("\nA closure for sub-filter turbulence should show R_tilde varying smoothly with W. Compare R2 "
      "with shift95 (the correlation-preserving null), not with a point permutation.")
