#!/usr/bin/env python
"""Tabulate the regional regression across crop regions from region_results/*/ohms_regression.npz."""
import glob
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "region_results"
files = sorted(glob.glob(os.path.join(root, "*", "ohms_regression.npz")))
if not files:
    sys.exit("no results under %s" % root)


def spread(a):
    if a is None or len(a) == 0:
        return (np.nan,) * 3 + (np.nan,)
    p16, med, p84 = np.percentile(a, [16, 50, 84])
    return med, p16, p84, 100.0 * np.mean(a > 0)


print("%-12s | %-26s | %-47s | %-47s | %-15s | %s" % (
    "region", "R-only: R2 (null95) ", "R_tilde: global  tile med [p16, p84]  %>0", "gamma: global  tile med [p16, p84]  %>0",
    "full: R2 (n95)", "winner  mean|J| |B| |E|"))
for f in files:
    with np.load(f) as _z:
        d = {k: _z[k] for k in _z.files}
    subs = list(d["subsets"])
    iR, iG, iF = subs.index("R"), subs.index("gamma"), subs.index("R+alpha+gamma")
    R2, sh, c = d["global_R2"], d["shift_null_R2"], d["global_coeffs"]
    nR, nG = np.percentile(sh[iR], 95), np.percentile(sh[iG], 95)
    iA = subs.index("alpha")
    nA = np.percentile(sh[iA], 95)
    excess = {"R": R2[iR] - nR, "alpha": R2[iA] - nA, "gamma": R2[iG] - nG}
    best = max(excess, key=excess.get)
    win = "none" if excess[best] <= 0 else best
    mR, loR, hiR, fR = spread(d["R_only_tile_R_tilde"])
    mG, loG, hiG, fG = spread(d["gamma_only_tile_gamma"] if "gamma_only_tile_gamma" in d else None)
    st = d["region_stats"] if "region_stats" in d else [np.nan] * 3
    print("%-12s | %6.3f (%5.3f)            | %9.3g %9.3g [%8.2g,%8.2g] %3.0f%% | %9.3g %9.3g [%8.2g,%8.2g] %3.0f%% | %5.3f (%5.3f)   | %-6s %5.1f %5.2f %.1e" % (
        str(d["label"]), R2[iR], nR, c[iR, 0], mR, loR, hiR, fR, c[iG, 2], mG, loG, hiG, fG,
        R2[iF], np.percentile(sh[iF], 95), win, *st))
print("\nwinner = the single term whose R2 exceeds its shift-null 95th percentile by the most ('none' if neither does).")
print("Reference, centre region (scan): gamma wins, R2 0.24; shifted region (80,300,120): R wins, R2 0.08, R_tilde 6e-6.")
