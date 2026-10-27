#!/usr/bin/env python
"""
Generate a config file and Slurm script for pickling_meso_mhd.py from the actual METHOD
output, checking that the filter fits inside the time window of the snapshots.

Reads only HDF5 attributes and a strided subsample of one snapshot. Run it on the
cluster, in the filtering conda environment, once the simulation has finished:

    python make_mhd_config.py /scratch/.../khi_window_576x576x288_8x8x3_seed1/ --name khi576 \\
        --crop-size 96 96 64 --filter-width 4 --box-len 6

Then:  sbatch submit_<name>.sh

Why the check: the filter averages over a 3D box in the observer's rest space (27 Gauss
points on the three spatial legs of the tetrad orthogonal to u; the time-like leg is not averaged), and the
observer search uses a box of side L. The boosted legs have time components, so the samples still reach in
lab time by up to ~ f * sqrt(3/5) * max(L, W)/2, with f = gamma (1 + sqrt(3) v) for the fastest flow speed v.
(The '1' stands for a time-like leg that does not exist, so f is a conservative over-estimate; the sharper
bound is gamma * sqrt(3) * v.) Samples outside the snapshot window crash interpn. The meso time slices (num_T_slices) also have to fit inside the window.
All widths below are in units of the micro grid spacing dx.
"""
import argparse
import json
import math
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'master_files'))
from SnapshotOrder import snapshot_files

# Measured once: 12.3 GB peak for 10 snapshots x 64*64*32 cells at 16 processes.
BYTES_PER_CELL_SNAPSHOT = 9.4e3
NODE_MEM_GB = 620


def attr(group, name):
    return float(np.ravel(group.attrs[name])[0])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("hdf5_dir")
    p.add_argument("--name", required=True, help="short run name used in output file names")
    p.add_argument("--crop-size", type=int, nargs=3, default=[96, 96, 64], metavar=("NX", "NY", "NZ"))
    p.add_argument("--crop-start", type=int, nargs=3, default=None, metavar=("I", "J", "K"),
                   help="first cell of the crop; default: centred. Pick it from plot_slices.py output.")
    p.add_argument("--filter-width", type=float, default=4.0, help="filter width W in dx")
    p.add_argument("--box-len", type=float, default=6.0, help="observer-search box side L in dx")
    p.add_argument("--num-t-slices", type=int, default=3, help="odd number of meso time slices")
    p.add_argument("--coarse", type=int, default=4, help="meso grid spacing in micro cells")
    p.add_argument("--pad-cells", type=int, default=0,
                   help="minimum margin (cells) between the crop faces and the meso grid; use the same "
                        "value for every run of a filter-width scan so the meso grids are identical")
    p.add_argument("--adiabatic-index", type=float, default=4.0 / 3.0,
                   help="Gamma of the equation of state (METHOD's ClosureScan uses 4/3); used only if the snapshots have no Auxiliary group (OUTPUT_MINIMAL frames)")
    p.add_argument("--n-cpus", type=int, default=16)
    p.add_argument("--partition", default="batch")
    p.add_argument("--pickled-dir", default=None, help="default: <hdf5_dir>/pickled")
    p.add_argument("--outdir", default=".")
    p.add_argument("--force", action="store_true", help="write the files even if the window check fails")
    a = p.parse_args()

    hdf5_dir = os.path.join(os.path.abspath(a.hdf5_dir), "")        # absolute: the job runs in another directory (filter_scripts)
    files = snapshot_files(hdf5_dir)
    if len(files) < 3:
        sys.exit("Need at least 3 snapshots in %s (found %d)" % (hdf5_dir, len(files)))

    times = []
    for fn in files:
        with h5py.File(fn, "r") as f:
            times.append(attr(f["Domain"], "endTime"))
    times = np.array(times)
    with h5py.File(files[len(files) // 2], "r") as f:
        dom = f["Domain"]
        dx = attr(dom, "dx")
        shape = f["Primitive/rho"].shape
        mins = [attr(dom, c + "min") for c in "xyz"]
        sub = (slice(None, None, 8),) * 3
        v2 = sum(f["Primitive/v" + c][sub] ** 2 for c in "xyz")
    vmax = min(float(np.sqrt(v2.max())), 0.999)
    gamma = 1.0 / math.sqrt(1.0 - vmax ** 2)
    f_reach = gamma * (1.0 + math.sqrt(3.0) * vmax)

    n = len(times)
    spacing = np.diff(times)
    c = int((n - 1) / 2)
    margin = min(times[c] - times[0], times[-1] - times[c]) / dx  # in dx
    W, L = a.filter_width, a.box_len
    # Gauss sample points sit at +-sqrt(3/5) * W/2 (Filters.py), not the full +-W/2
    reach = f_reach * math.sqrt(3.0 / 5.0) * max(W, L) / 2.0
    half = a.num_t_slices // 2
    t_off = half * a.coarse
    cond_a = (a.num_t_slices / 2.0) * a.coarse <= margin
    cond_b = t_off + reach <= margin

    print("snapshots: %d, spacing %.4g (%.2f dx), uniform: %s" % (
        n, spacing.mean(), spacing.mean() / dx, bool(np.allclose(spacing, spacing.mean(), rtol=1e-4))))
    print("window span %.4g = %.1f dx; usable half-window about the central snapshot: %.1f dx" % (
        times[-1] - times[0], (times[-1] - times[0]) / dx, margin))
    print("max |v| (sampled) %.3f -> gamma %.2f, reach factor %.2f" % (vmax, gamma, f_reach))
    print("filter W=%.1f dx, box L=%.1f dx -> reach %.1f dx; time slices offset up to %d dx" % (W, L, reach, t_off))

    ok = cond_a and cond_b
    if not ok:
        allowed = max(0.0, 2.0 * (margin - t_off) / (f_reach * math.sqrt(3.0 / 5.0)))
        print("\nWINDOW CHECK FAILED: %s" % ("meso time slices do not fit (reduce --num-t-slices or --coarse)"
              if not cond_a else "filter reaches outside the window"))
        print("With these slices, max(W, L) must be <= %.1f dx. Reduce widths, or re-run the simulation "
              "with a larger snapStride." % allowed)
        if not a.force:
            sys.exit(1)

    # Crop and the interior region the meso grid may occupy
    size = a.crop_size
    start = a.crop_start or [(N - s) // 2 for N, s in zip(shape, size)]
    pad = max(int(math.ceil(reach)) + 2, a.pad_cells)
    for ax, (N, s0, sz) in enumerate(zip(shape, start, size)):
        if s0 < 0 or s0 + sz > N:
            sys.exit("Crop axis %d [%d, %d) outside the domain 0..%d" % (ax, s0, s0 + sz, N))
        if sz < 2 * pad + 2 * a.coarse:
            sys.exit("Crop axis %d too small (%d cells) for margin %d cells each side" % (ax, sz, pad))
    ranges = []
    for mn, s0, sz in zip(mins, start, size):
        lo = mn + (s0 + pad) * dx
        hi = mn + (s0 + sz - 1 - pad) * dx
        ranges.append([math.ceil(lo * 1e6) / 1e6, math.floor(hi * 1e6) / 1e6])

    cells = int(np.prod(size))
    mem_gb = BYTES_PER_CELL_SNAPSHOT * cells * n / 1e9
    request = int(math.ceil(mem_gb * 1.3))
    print("\ncrop start %s size %s (%d cells); margin %d cells each side" % (start, size, cells, pad))
    print("meso ranges (physical): %s" % ranges)
    print("memory estimate %.0f GB (from one measurement at 16 processes), requesting %d GB" % (mem_gb, request))
    if request > NODE_MEM_GB:
        print("WARNING: request exceeds the ~%d GB node; reduce --crop-size." % NODE_MEM_GB)
        if not a.force:
            sys.exit(1)

    pickled = os.path.abspath(a.pickled_dir) if a.pickled_dir else os.path.join(hdf5_dir, "pickled")
    cfg = """# Generated by make_mhd_config.py for %s
# window: %d snapshots, %.2f dx apart; W=%g dx, L=%g dx; crop %s from %s
# Run from filter_scripts:  python pickling_meso_mhd.py %s

[Directories]
hdf5_dir = %s
pickled_files_dir = %s
figures_dir = .

[Filenames]
meso_pickled_filename = /resMHD3d_%s.pickle
micro_pickled_filename = /

[Micro_model_settings]
snapshots_opts = {"fewer_snaps_required": false, "smaller_list": []}
read_region = %s
adiabatic_index = %r

[Meso_model_settings]
meso_grid_smart = %s
filtering_options = %s
n_cpus = %d

[Ohms_law_settings]
terms = {"resistive": true, "dynamo": true, "hall": true}
""" % (a.name, n, spacing.mean() / dx, W, L, size, start, "config_%s.txt" % a.name,
       hdf5_dir, pickled, a.name,
       json.dumps({"crop": [[s0, s0 + sz] for s0, sz in zip(start, size)], "stride": 1}), a.adiabatic_index,
       json.dumps({"x_range": ranges[0], "y_range": ranges[1], "z_range": ranges[2],
                   "num_T_slices": a.num_t_slices, "coarse_grain_factor": a.coarse}),
       json.dumps({"box_len_ratio": L, "filter_width_ratio": W}), a.n_cpus)

    sub_sh = """#!/bin/bash
#SBATCH --job-name=lagfilt-%s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=%d
#SBATCH --mem=%dG
#SBATCH --time=02:00:00
#SBATCH --partition=%s
#SBATCH --output=%%x-%%j.out
#SBATCH --error=%%x-%%j.err

set -euo pipefail

module load conda                  # use the module name you used to create the env
eval "$(conda shell.bash hook)"
conda activate lagfilt

export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

mkdir -p %s
cd ~/Lagrangian-filtering/filter_scripts
python pickling_meso_mhd.py config_%s.txt
""" % (a.name, a.n_cpus, request, a.partition, pickled, a.name)

    os.makedirs(a.outdir, exist_ok=True)
    cfg_path = os.path.join(a.outdir, "config_%s.txt" % a.name)
    sh_path = os.path.join(a.outdir, "submit_%s.sh" % a.name)
    with open(cfg_path, "w") as fh:
        fh.write(cfg)
    with open(sh_path, "w") as fh:
        fh.write(sub_sh)
    os.chmod(sh_path, 0o755)
    print("\nWrote %s and %s" % (cfg_path, sh_path))


if __name__ == "__main__":
    main()
