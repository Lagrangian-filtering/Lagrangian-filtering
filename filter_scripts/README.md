# filter_scripts: MHD filtering pipeline and analysis scripts

All scripts are run from this directory in the `lagfilt` conda environment (they import `../master_files`). Design, derivations and results are in
`action_filtering/planning/` (`mhd_filtering_extension.md` §8, `closure_stage2_design.md`); each script's docstring explains its method and output. Data paths below are the
iridis ones used so far. Set `KHI_DATA=/path/to/run` (no trailing slash) to override the default data directory of the scan/mosaic shell scripts.

## 1. From METHOD output to a meso pickle

| step | command |
|---|---|
| config and Slurm script (checks that the filter fits in the snapshot window and estimates memory) | `python make_mhd_config.py <hdf5_dir>/ --name NAME --crop-size 96 96 48 --pad-cells 12 --filter-width 4 --box-len 6 --num-t-slices 3 --coarse 2 [--crop-start I J K]` |
| run the filter (writes `<hdf5_dir>/pickled/resMHD3d_NAME.pickle`) | `sbatch submit_NAME.sh`, which runs `python pickling_meso_mhd.py config_NAME.txt` |
| a first test on a small crop | `config_mhd_crop_test.txt`, `submit_mhd_crop_test.sh` |

Options that matter: `--num-t-slices 3` is needed for time derivatives (the gradient, stress, Lorentz and deficit models use the central slice); `--coarse 2` (meso spacing 2 dx) resolves gradients
much better than 4 (grid refinement, planning §8.6); `--filter-width` is in micro cells; the generator refuses a width the snapshot window cannot support unless `--force`. Pickles made before 2026-10-21 lack
`meso_structures['SET_EM']` (needed by `regress_em_deficits.py`); before the `BSq` structure was added they lack `B2_sub`.

Batch scans (names `khi576_[r<i>_<j>_<k>_]w<W>[_<n>sl][_c<C>]`): `scan_w_submit.sh` / `scan_w_analyse.sh` (a filter-width scan; `REGION=i,j,k NSLICES=n COARSE=c MODE=ohms|gradient`),
`scan_regions_submit.sh` / `scan_regions_analyse.sh` (a survey of crops), `mosaic_submit.sh` (abutting crops for a mosaic), and the summarizers `summarize_w_scan.py`, `summarize_gradient_scan.py`, `summarize_regions.py`.

**Which windows to analyse.** The closure these scripts test is a second-order expansion for the transfer between resolved and sub-filter scales, expected to work (a hypothesis supported by the a-priori trend, planning/closure_stage2_design.md section 15) in the *developed* phase of the flow: after the onset of field amplification, when the run's kinetic energy is below about a quarter of its initial value and the growth rate of the magnetic energy has fallen below half its peak, and for filter widths of 2-3 resolved cells. Choose the window time from the `diag` lines of the run (METHOD `energyDiagnostics.h`), not a fixed time; results from the shear-layer-formation or the kinematic-growth phases do not test the closure.

## 2. Analysis scripts (each takes a pickle, does not modify it, writes an `.npz` and a log/figure)

| script | question it answers | example (centre crop, `W = 4`, 3 slices, spacing 2) | output |
|---|---|---|---|
| `regress_ohms_law.py` | constant-coefficient fit of `Ẽ` to the resistive, dynamo and Hall terms over regions and tiles, with shift nulls | `python regress_ohms_law.py P --block 6 6 4` | `ohms_regression.npz/png` |
| `plot_ohms_winner.py`, `mosaic_ohms_winner.py` | which term describes `Ẽ` where (local fits; winner maps, density, `|B̃|²`, sub-filter `B²` panels) | `python plot_ohms_winner.py P --window 5 5 3` | `ohms_winner.png/npz` |
| `regress_gradient_model.py` | second-order covariance model of `Ẽ` (the EMF), parameter-free, and regression on its two operators | `python regress_gradient_model.py P --block 12 12 6 --sigma2-factor 1.12` | `gradient_model.npz` |
| `regress_sparse_library.py` | SINDy-like sparse regression (STLSQ on the Gram system) of `Ẽ` on a library: quadratic gradient forms `σ²(∂v)(∂B)` (5 independent for solenoidal `B`; the filter covariance, `(curl v)×(curl B)`, `(div v)J`, `(J·∇)v`, ...), the density term, J~/B~/Hall; stability by bootstrap over windows | `python regress_sparse_library.py P --window 1 6 6 3` or `--synthetic [--extra divv_J=0.5 --noise 0.1]` | `sparse_library.npz` |
| `regress_covariance_split.py` | which derivative pieces (spatial, time, advective) of the covariance carry `Ẽ` | `python regress_covariance_split.py P --block 12 12 6` | `covariance_split.npz` |
| `regress_covariant_emf.py` | covariant second-order model of `Ẽ` (4-velocity, comoving field, 4-gradients with `d_t`, window = rest space of the observer) fitted on one or several pickles; compares the coefficients across frames (boost test) and with the lab-frame 3-vector model | `python regress_covariant_emf.py P0 P5 --labels beta0 beta0.5 --block 12 12 6 --sigma2-factor 1.12` | `covariant_emf.npz`  (`--common-region` fits all pickles on the common physical box) |
| `regress_stress_model.py` | sub-filter fluid stress `π̃` and heat flux `q̃` (found negligible, `~1e-4`) | `python regress_stress_model.py P --block 12 12 6 --sigma2-factor 1.12` | `stress_model.npz` |
| `regress_lorentz_closure.py` | sub-filter Lorentz force `F_closure` | `python regress_lorentz_closure.py P --block 12 12 6 --sigma2-factor 1.12` | `lorentz_closure.npz` |
| `compare_force_scales.py` | size of `F_closure` against the other forces on the filtered fluid | `python compare_force_scales.py P` | `force_scales.npz` |
| `map_closure_tail.py` | where `F_closure` is large (correlations with `|J̃|`, sub-filter `B²`, ...; maps) | `python map_closure_tail.py P` | `closure_tail.png/npz` |
| `check_backscatter.py` | sign of the sub-filter energy exchange (`J̃·Ẽ`, work on the fluid), data against model; model `R²` in the tail | `python check_backscatter.py P --sigma2-factor 1.12` | `backscatter.npz` |
| `regress_em_deficits.py` | covariant Maxwell-stress / Poynting / energy-density deficit `⟨T_EM⟩ − T_EM(F_r)` and its sources; **needs `SET_EM`** | `python regress_em_deficits.py P_setem --sigma2-factor 1.12` | `em_deficits.npz` |

`--sigma2-factor` multiplies the nominal filter variance `W²/12`: linear interpolation of the micro data inflates the effective variance (`~1.12` at `W = 4` cells, `1.51` at `W = 2`, `1.01` at `W = 12`; planning §8.7).
Shared code: `model_report.py` (reporting of the parameter-free model tests), `winner_plots.py` (colours for the winner maps). The numerical methods are in `../master_files/OhmsRegression.py`
(index in its header).

Where the results are interpreted: EMF and Ohm's-law terms, planning `mhd_filtering_extension.md` §8.4-8.6; stress and interpolation variance, §8.7-8.9; the closure, `closure_stage2_design.md`.
