#!/usr/bin/env python
"""Tabulate the gradient-model regression against filter width W from <root>/w*/gradient_model.npz."""
import glob
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "gradient_scan"
files = glob.glob(os.path.join(root, "w*", "gradient_model.npz"))
if not files:
    sys.exit("no results under %s" % root)
rows = []
for f in files:
    d = np.load(f)
    if "fit_names" not in d.files:
        print("%s predates the scan columns: re-run regress_gradient_model.py with --label" % f)
        continue
    names = list(d["fit_names"])
    i_vB, i_g, i_R = names.index("model_vB"), names.index("gamma_hall"), names.index("R_tilde")
    tile = d["tile_model_vB"]
    p16, med, p84 = np.percentile(tile, [16, 50, 84]) if len(tile) else (np.nan,) * 3
    R2, n95 = d["fit_R2"], d["fit_null95"]
    rows.append((float(d["label"]), float(d["pred_R2"]), float(d["pred_null95"]), float(d["best_scale"]),
                 R2[i_vB], n95[i_vB], float(d["model_vB_coeff"]), med, p16, p84, len(tile), R2[i_g], R2[i_R],
                 float(d["joint_dens_coeff"])))
print("%5s | %-26s | %-27s | %-30s | %-11s | %s" % ("W/dx", "parameter-free: R2 (null95)  scale", "model_vB alone: R2 (null95)  coeff",
      "model_vB coefficient over tiles", "gamma, R only", "model_dens (joint)"))
print("%5s | %-26s | %-27s | %-30s | %-11s | %s" % ("", "", "(expected -1)", "median [p16, p84]  n", "R2   R2", "(expected +1)"))
for r in sorted(rows):
    print("%5g | %7.3f (%7.3f)  %6.2f   | %7.3f (%6.3f)  %8.3f   | %7.3f [%6.3f, %6.3f] %3d | %5.3f %5.3f | %8.3f" % r)
print("\nIf the second-order model describes E_tilde, the model_vB coefficient should stay roughly constant in W "
      "(the prediction already contains sigma^2 = W^2/12), with R2 high until the Taylor expansion fails for W near "
      "the structure size. A coefficient that drifts with W, or an R2 that falls steadily, shows where it stops working.")
