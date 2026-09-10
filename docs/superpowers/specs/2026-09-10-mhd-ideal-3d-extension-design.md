# Covariant MHD filtering extension — 3D ideal MHD only

Status: approved by user 2026-09-10, implementing on `feature/mhd-ideal-3d-filtering`.

## Background

This extends `Lagrangian-filtering` to filter electromagnetism (Sec. 6D of the
Higher-Level filtering paper, arXiv:2407.18012v2), following the design worked out in
`~/Documents/github/action_filtering/planning/mhd_filtering_extension.md` and
empirically checked against real METHOD `SRMHD` output in that same repo's
`method_srmhd_interop.md`. Those documents are the physics source of truth; this doc
records the concrete, repo-specific implementation decisions and file-by-file plan.

Two design decisions that were previously open are now settled by explicit user
instruction:
1. **3D only.** No 2+1D EM model — 2+1D electromagnetism is degenerate (no vector `B`,
   only a pseudoscalar) and the Hall term needs a genuinely 4D `ε^{abcd}`, so `IdealMHD_2D`/
   `resMHD_2D` are not built.
2. **Generic Ohm's law, each term independently switchable.** The resistive (`R̃`),
   alpha-dynamo (`α`), and Hall (`γ`) terms of eq. 711-729 must each be toggleable by a
   parameter, not hardcoded together.

## Facts established by reading the current code (2026-09-10)

- `IdealHD_3D` (`master_files/MicroModels.py:249`) and `resHD_3D`
  (`master_files/MesoModels.py:1696`) already exist and are the classes to mirror.
- `resHD_3D` is **missing** `closure_ingredients`/`EL_style_closure`/
  `modelling_coefficients` — present in `resHD2D` but never ported to 3D. This is a
  pre-existing gap, unrelated to MHD, and is **not** being fixed here — the EM closure
  this extension adds does not depend on those fluid-closure methods (it only needs
  `ũ^a`, which `decompose_structures_parallel` already produces in `resHD_3D`).
- `FileReaders.METHOD_HDF5.read_in_data3D` (`master_files/FileReaders.py:233`) reads
  `Domain/*` metadata as HDF5 *datasets* (`self.hdf5_files[0]['Domain/'+var][:]`). Real
  METHOD output writes this metadata as HDF5 *attributes* only (confirmed via `h5dump`
  against actual output in `method_srmhd_interop.md` §3 and again in
  `mhd_filtering_extension.md` §7). This means `read_in_data3D` cannot currently load
  any real METHOD 3D file, MHD or hydro. Fixing this is a precondition for this work,
  not an optional nicety.
- No automated test framework exists (`3Dscripts/test_meso3D.py` is a manual
  smoke-test script that exercises the pipeline against real data, not a unit test
  suite).
- Config files (`filter_scripts/config_filter.txt`) are `configparser` sections whose
  values are JSON blobs, parsed with `json.loads` in the driver scripts
  (`filter_scripts/pickling_meso.py`).

## Component design

### 1. `master_files/system/BaseFunctionality.py`

Add to `Base`:
- `LeviCivita4D()` — returns the `(4,4,4,4)` totally antisymmetric symbol as an
  `ndarray` (`ε_{0123}=+1` convention), computed once from index parity, not
  hand-written.
- `dual_tensor(F_upper, metric)` — given a fully-contravariant rank-2 tensor `F^{ab}`
  and the spacetime metric, returns `*F^{ab} = ½ ε^{abcd} F_{cd}` (lowers indices
  internally via `metric` before contracting with `LeviCivita4D()`).

Both are pure array-algebra, unit-testable against a hand-computed boosted uniform
field with no HDF5/grid machinery involved.

### 2. `master_files/MicroModels.py` — new `IdealMHD_3D`

Mirrors `IdealHD_3D` (`__init__`, `get_*`, `get_interpol_var`, `get_var_gridpoint`)
with:
- `prim_strs` = `("vx","vy","vz","n","p","Bx","By","Bz")`.
- `structures_strs` = `("BC","SET","bar_vel","FaradayTensor","ChargeCurrent",
  "LorentzForceDensity","SET_EM")`. `SET` stays fluid-only (unchanged from
  `IdealHD_3D`); EM stress-energy is `SET_EM`, per `mhd_filtering_extension.md` §3.2/
  §4.3 (filter tensors separately, not a combined-then-split `SET`).
- `setup_structures()`: after the existing fluid `bar_vel`/`BC`/`SET` loop (unchanged),
  adds, per gridpoint:
  1. `E^i = -ε_{ijk} v^j B^k` (3D spatial Levi-Civita; ideal Ohm's law — this is
     definitional for `IdealMHD_3D`, not a fallback, since METHOD's `SRMHD` never
     stores `E`, confirmed empirically).
  2. `FaradayTensor` `F^{ab}` built **directly from the lab-frame `E,B`** via the flat
     3+1 formula `F^{0i}=-E^i`, `F^{ij}=-ε^{ijk}B_k`. **This is deliberately not eq.
     619 evaluated with the fluid four-velocity `u^a`** — it is that same formula
     specialized to the lab/Eulerian observer `n^a=(1,0,0,0)`, because METHOD's `B^i`
     (primitive) and the `E^i` reconstructed above are lab-frame quantities, not
     fluid-frame ones. Using the fluid `u^a` here would silently build the wrong
     tensor. A code comment states this explicitly.
  3. `ChargeCurrent` `j^b = ∂_a F^{ba}` (`μ₀=1`), computed via `np.gradient` on the
     already-built `FaradayTensor` array, using `domain_vars['dt'/'dx'/'dy'/'dz']` as
     spacings. Vectorized over the whole micro grid, not a per-gridpoint Python loop.
  4. `LorentzForceDensity` = `j_a F^{ab}` (lower `j`'s index with the metric first) —
     stored as its own structure so it gets filtered as one micro-scale quantity
     (needed for eq. 692's `𝓕^b`, see `mhd_filtering_extension.md` §3.5.3).
  5. `SET_EM^{ab} = F^{ac}F^b{}_c - ¼ g^{ab}F_{cd}F^{cd}`.

### 3. `master_files/FileReaders.py`

Fix `read_in_data3D`'s three `Domain/*` loops (`domain_int_strs`, `domain_float_strs`,
`domain_array_strs`) to read `.attrs[...]` instead of indexing as a dataset. Scoped to
`read_in_data3D` only — `read_in_data`/`read_in_data_HDF5_missing_xy` (2D readers) have
the same bug but are out of scope for this change.

### 4. `master_files/MesoModels.py` — new `resMHD_3D(resHD_3D)`

Subclasses `resHD_3D` (inherits grid setup, fluid filtering, fluid decomposition,
derivatives unchanged) rather than duplicating them, per
`mhd_filtering_extension.md` §3.5.8's preferred option.

- `__init__`: calls `super().__init__(...)`, then extends `meso_structures_strs` with
  `FaradayTensor`, `ChargeCurrent`, `LorentzForceDensity`, and `meso_vars_strs` with
  scalars `sigma_tilde`, `R_tilde`, `alpha_dynamo`, `gamma_hall`, and vectors
  `E_tilde`, `B_tilde`, `J_tilde`, `F_closure` (`𝓕^b`), `Ohm_res` (`𝓦^a`).
- `filter_micro_vars_parallel`: overridden (not calling super, since the base method's
  `vars` list is a local literal, not an overridable attribute) — copies the base
  implementation with `vars = ['BC','SET','p','FaradayTensor','ChargeCurrent',
  'LorentzForceDensity']`.
- `decompose_structures_parallel`: calls `super().decompose_structures_parallel(n_cpus)`
  first (fills fluid `meso_vars` including `u_tilde`), then a new
  `decompose_EM_parallel(n_cpus)` that runs a new staticmethod `decompose_EM_task`
  per gridpoint:
  - `Ẽ^a = ũ_b⟨F^{ab}⟩`, `B̃_a = -½ε_{abcd}ũ^c⟨F^{cd}⟩` (via `Base.dual_tensor`).
  - `σ̃ = -ũ_a⟨j^a⟩`, `J̃^a = ⊥̃^a_b⟨j^b⟩` (reusing the same projector construction
    already used in `decompose_structures_task`).
  - `𝓕^b = -⟨j_aF^{ab}⟩ + J̃_a⟨F^{ab}⟩` (first term is the filtered
    `LorentzForceDensity` structure, **not** a product of separately-filtered `j`
    and `F`).
- `fit_ohms_law_closure(self, use_resistive=True, use_dynamo=True, use_hall=True)`:
  new method, vectorized (no multiprocessing — the per-point linear system is at most
  3x3). Builds only the *enabled* candidate operators
  `{J̃^a, B̃^a, ε^{abcd}J̃_bũ_cB̃_d}`, forms the Minkowski-contracted Gram matrix and
  RHS at every meso gridpoint, solves via batched `np.linalg.solve`, writes the
  enabled coefficients into `R_tilde`/`alpha_dynamo`/`gamma_hall` (disabled ones set to
  `np.nan`, not `0`, so "not fitted" is distinguishable from "fitted to zero"), and
  writes the residual into `Ohm_res`.

### 5. Config

New `[Ohms_law_settings]` section in `filter_scripts/config_filter.txt`:
```
terms = {"resistive": true, "dynamo": true, "hall": true}
```
New driver script `filter_scripts/pickling_meso_mhd.py`, mirroring `pickling_meso.py`
but using `IdealMHD_3D`/`resMHD_3D` and reading the new section to call
`fit_ohms_law_closure(**terms)`.

## Testing

New `tests/` directory (stdlib `unittest`, no new dependency, no HDF5 fixtures):
- `test_base_tensor_algebra.py`: `LeviCivita4D` antisymmetry/values;
  `dual_tensor` against a hand-computed boosted uniform field (`E=0`, uniform `B`,
  boost by known `v`) checked against the textbook transformed `E,B`.
- `test_ideal_mhd_3d_micro.py`: on a small synthetic grid, check `E` reconstruction,
  `F^{ab}` antisymmetry and correct `E,B` recoverability from it, `SET_EM` matches the
  independent `S^{ij}` formula quoted in `method_srmhd_interop.md` §2 as a
  cross-check, `ChargeCurrent` recovers a known analytic `∂_aF^{ba}` for a synthetic
  field with a prescribed spatial gradient.
- `test_ohms_law_closure.py`: synthetic `Ẽ = R̃J̃ + αB̃ + γ·Hall` at each point for
  known `R̃,α,γ`; fit with all three terms on recovers them; fitting with one or more
  terms off (a) doesn't crash, (b) sets disabled coefficients to `NaN`, (c) absorbs
  the missing term into a larger residual `Ohm_res` rather than a wrong-but-plausible
  fitted value for the remaining terms (checked by comparing residual magnitude
  with/without the term artificially included in the synthetic data).

Not tested: the full HDF5-driven pipeline end-to-end (no fixture data checked into
this repo, and the real 916MB METHOD files live outside it) — this mirrors the
existing repo convention of exercising the full pipeline via a manual script
(`3Dscripts/test_meso3D.py`) rather than an automated integration test.

## Explicitly out of scope

- 2+1D EM (`IdealMHD_2D`/`resMHD_2D`) — retired per user decision above.
- Completing `resHD_3D`'s missing fluid closure suite — pre-existing gap, independent
  of MHD.
- `Analysis.py`/`Visualization.py`/calibration-script wiring beyond the one new config
  section and driver script — can follow once the core pipeline is validated against
  real data.
- HDF5 chunking, multi-snapshot single-file consolidation, fixing the attribute bug in
  the 2D readers — all noted in the source planning docs as independent, non-blocking
  improvements.
