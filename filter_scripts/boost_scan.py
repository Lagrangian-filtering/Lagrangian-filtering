#!/usr/bin/env python
"""
The fit of the filter covariance in boosted frames (analytic spacetime test, master_files/BoostTest.py).

For each boost speed and direction and for each window case ('lab': the S0 window, the covariant case A; 'frame': a box rebuilt in the boosted frame, case B), the
coefficient c and R^2 of  e = c * O  for the covariant library (e^mu = eps u~ Sigma d u~ d b~; expected -1) and for the spatial library of the real-data regression
(3-vector v and B, spatial gradients; not covariant).

    python boost_scan.py [--betas 0 0.3 0.5 0.9 0.99] [--axes x z] [--seeds 0 1 2] [--h 0.13] [--width 4]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'master_files'))
import BoostTest as BT


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--betas", nargs='+', type=float, default=[0.0, 0.3, 0.5, 0.9, 0.99])
    p.add_argument("--axes", nargs='+', default=['x', 'z'])
    p.add_argument("--seeds", nargs='+', type=int, default=[0, 1, 2])
    p.add_argument("--h", type=float, default=0.13, help="lattice spacing of the window (the fields vary on scales ~ 1/k, k <= 2)")
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--mesh", type=float, default=1.0, help="meso spacing in units of h (divided by gamma along the boost and in time)")
    a = p.parse_args()
    flows = [BT.Flow(seed=s) for s in a.seeds]
    print("h %.3g, window width %d, meso spacing %.3g h, seeds %s; mean (min..max) over seeds" % (a.h, a.width, a.mesh, a.seeds))
    print("%-5s %-3s %-6s %6s | %-24s %-8s | %-24s %-8s" % ("beta", "ax", "window", "gamma", "cov c", "R2", "sp c", "R2"))
    for ax in a.axes:
        for beta in a.betas:
            b = np.zeros(3)
            b['xyz'.index(ax)] = beta
            for window in ('lab', 'frame'):
                rs = [BT.run_case(f, b, window, h=a.h, width=a.width, mesh=a.mesh) for f in flows]
                col = lambda key, i: np.array([r[key][i] for r in rs])
                fmt = lambda x: "%.3f (%.3f..%.3f)" % (x.mean(), x.min(), x.max())
                print("%-5g %-3s %-6s %6.2f | %-24s %-8.4f | %-24s %-8.3f" % (beta, ax, window, rs[0]['gamma'], fmt(col('cov', 0)), col('cov', 1).mean(), fmt(col('sp', 0)), col('sp', 1).mean()))
                if beta == 0.0:
                    break


if __name__ == '__main__':
    main()
