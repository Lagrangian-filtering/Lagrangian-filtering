# Tests

    python -m pytest tests -q

Two environments are used:
  - **numpy only** (plus matplotlib and scipy for the script smoke tests): `test_sparse_library.py` (the sparse-regression library on box-filtered synthetic fields with a known answer, and the script), `test_ohms_regression.py` (the regression, covariance-model, stress-energy, Lorentz-force, EM-deficit and covariant-closure
  tests, including end-to-end checks on analytic fields filtered with the 27-point quadrature) and `test_analysis_scripts_smoke.py` (runs the analysis scripts on synthetic meso pickles with `MesoModels` stubbed).
  - **the full environment** (`h5py`, `multimethod`, `scipy`, `seaborn`, `scikit-learn`, `matplotlib`, `sympy`; the `lagfilt` environment, or a throwaway venv with those packages): the rest,
  including `test_ideal_mhd_3d_micro.py`, `test_res_mhd_3d.py` (including the `SET_EM` wiring test with a stub filter), `test_ohms_law_closure.py`, `test_file_readers_domain_attrs.py` and
  `test_make_mhd_config.py` (needs `h5py`; skipped without it).

All numbers in the planning notes come from real data on iridis; these tests check the code on synthetic or analytic data only.
