#!/usr/bin/env python
"""
Lorentz-boost METHOD primitive frames: the same flow in another frame, for the test of the covariant fits (planning doc, closure_stage2_design.md sections 84 and 85).

    python boost_frames.py RESTDIR --beta 0.8 --center-rest 8.5 --n-side 4 --spacing 0.0125 --crop-x 68 188 --crop-y 68 188 --nz 120 --outdir DIR
    python boost_frames.py RESTDIR --beta 0.8 --times 4.2 4.5 4.8 --crop-x 68 188 --crop-y 68 188 --nz 120 --outdir DIR

RESTDIR holds frame.checkpoint.*.hdf5 of the rest frame S (primitives rho, p, vx, vy, vz, Bx, By, Bz on a periodic grid; the fine control run has a frame every 0.1). The frames written
are those of the frame S' that moves with velocity beta along +z relative to S, on the cells crop-x, crop-y (indices of the rest-frame grid) and nz cells in z' centred on z' = 0, at the
times T' of S' given by --times, or T'_c + k spacing, k = -n..n, with T'_c = center-rest / gamma (the event of the central cell z' = 0 at T'_c is at the rest time center-rest).
Details and the formulas: master_files/FrameBoost.py. The boosted slice T' = const contains rest times t = gamma (T' + beta z'), so RESTDIR needs frames from
gamma (T' + beta z'_min) - 2 dt to gamma (T' + beta z'_max) + 2 dt (cubic interpolation in time): the script checks this before it starts.

Why many closely spaced frames: the pipeline interpolates the stored snapshots linearly in time, and the filter's quadrature points lie off the snapshot time by gamma_obs v W/2 (observer
velocity v). With snapshots far apart in the boosted frame (a pattern moving at v = 0.8 shifts by tens of cells between them) a few per cent of a displaced copy of the flow enters the filtered
fields; a cluster of frames a few hundredths apart around the central time does not have this error. A cluster with the spacing of the earlier boosted runs shows the effect: --spacing 0.8333, --n-side 1.
Rule of thumb: the linear interpolation in time leaves an error of order (v dt / l)^2 / 8 of the fields (l the scale of the structure), so choose the spacing so that the pattern moves by at most about a cell
between the frames, dt ~ dx / v (0.005 for beta 0.8 at 256^3), and let the cluster cover the reach of the fastest observers (the window check of make_mhd_config.py refuses a cluster that is too short).
Does not modify RESTDIR.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'master_files'))
import FrameBoost as FB


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("rest_dir")
    p.add_argument("--beta", type=float, required=True, help="velocity of the new frame along +z (0 gives the rest frame on the same cells)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--times", type=float, nargs='+', help="times T' of the boosted frames")
    g.add_argument("--center-rest", type=float, help="rest time of the central cell z' = 0 of the central frame; with --n-side and --spacing")
    p.add_argument("--n-side", type=int, default=4)
    p.add_argument("--spacing", type=float, default=0.0125)
    p.add_argument("--crop-x", type=int, nargs=2, required=True)
    p.add_argument("--crop-y", type=int, nargs=2, required=True)
    p.add_argument("--nz", type=int, required=True, help="number of z' cells (centred on z' = 0)")
    p.add_argument("--no-repair", action="store_true")
    p.add_argument("--outdir", required=True)
    a = p.parse_args()

    g = 1.0 / np.sqrt(1.0 - a.beta ** 2)
    if a.times is not None:
        times = list(a.times)
    else:
        times = [a.center_rest / g + k * a.spacing for k in range(-a.n_side, a.n_side + 1)]
    xs, ys = slice(*a.crop_x), slice(*a.crop_y)
    paths = glob.glob(os.path.join(a.rest_dir, 'frame.checkpoint.*.hdf5'))
    if not paths:
        p.error("no frame.checkpoint.*.hdf5 in %s" % a.rest_dir)
    rest = FB.RestFrames(paths, xs, ys, repair=not a.no_repair)
    dz = rest.dz
    zp = np.array([(0 - a.nz // 2) * dz, (a.nz - 1 - a.nz // 2) * dz])
    need = (g * (min(times) + a.beta * zp[0]), g * (max(times) + a.beta * zp[1]))
    have = (rest.times[1], rest.times[-2])
    print("gamma %.4f; %d boosted frames at T' = %.5f .. %.5f; rest times needed %.4f .. %.4f, available (cubic stencil) %.4f .. %.4f" % (
        g, len(times), min(times), max(times), need[0], need[1], have[0], have[1]))
    if need[0] < have[0] or need[1] > have[1]:
        p.error("the rest frames do not cover the rest times the boosted frames need")
    os.makedirs(a.outdir, exist_ok=True)
    for T in times:
        t_lo, t_hi = g * (T + a.beta * zp[0]), g * (T + a.beta * zp[1])
        keep = set(k for k in range(len(rest.times)) if rest.times[k] >= t_lo - 3 * 0.5 * (rest.times[2] - rest.times[1]) and rest.times[k] <= t_hi + 3 * 0.5 * (rest.times[2] - rest.times[1]))
        rest.drop_except(keep)
        state = FB.boosted_frame(rest, a.beta, T, a.nz)
        out = os.path.join(a.outdir, 'frame.checkpoint.%.6f.hdf5' % T)
        FB.write_frame(out, state, T, rest.attrs, xs, ys, a.nz, dz)
        print("  T' = %.6f (rest times %.4f .. %.4f): %s" % (T, t_lo, t_hi, os.path.basename(out)))
    bad = {rest.times[k]: n for k, n in rest.repaired.items() if n}
    print("cells repaired in the rest frames read (time: cells): %s" % (bad if bad else "none"))


if __name__ == '__main__':
    main()
