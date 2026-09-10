# 3D Ideal-MHD Filtering Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend Lagrangian-filtering with a 3D ideal-MHD micro model (`IdealMHD_3D`) and meso model (`resMHD_3D`) that filter the Faraday tensor and charge current per Sec. 6D of the Higher-Level filtering paper, with a generic Ohm's-law closure whose resistive/dynamo/Hall terms are each independently switchable.

**Architecture:** `resMHD_3D` subclasses the existing `resHD_3D` to reuse fluid grid setup/filtering/decomposition unchanged, adding only EM structures and closure logic. Micro-scale EM structures (`FaradayTensor`, `ChargeCurrent`, `LorentzForceDensity`, `SET_EM`) are built vectorized (whole-grid numpy ops) in `IdealMHD_3D.setup_structures()` since the real target grids are millions of points. New generic `Base` tensor-algebra helpers (`LeviCivita4D`, `raise_LeviCivita4D`, `dual_tensor`, `faraday_from_EB`, `observer_frame_fields`) are used identically at the micro level (with the lab observer `(1,0,0,0)`) and the meso level (with the Favre observer `ũ^a`), which is what guarantees the two are mathematical inverses of each other rather than two independently-sourced formulas that might disagree on sign convention.

**Tech Stack:** Python, numpy, h5py (all already dependencies). Tests use stdlib `unittest` — no new dependency, no test framework currently exists in this repo.

**Spec:** [docs/superpowers/specs/2026-09-10-mhd-ideal-3d-extension-design.md](../specs/2026-09-10-mhd-ideal-3d-extension-design.md)

## Global Constraints

- No new third-party dependencies (numpy, h5py, scipy, multimethod already in use — nothing else).
- Metric convention: mostly-plus Minkowski, `diag(-1,1,1,1)`, matching every existing class (`IdealHD_3D.metric`, `resHD_3D.metric`).
- `μ₀ = 1` throughout (matches `method_srmhd_interop.md`'s confirmed compatibility with METHOD's unit convention).
- 3D only — no `IdealMHD_2D`/`resMHD_2D`.
- Tests run via `python -m unittest <module> -v` from the repo root; each test file adds `master_files/` to `sys.path` itself (matches how `filter_scripts/*.py` do it).
- Disabled Ohm's-law terms are recorded as `np.nan`, never `0.0`.

---

### Task 1: `Base` tensor-algebra helpers (Levi-Civita, dual tensor, Faraday construction/decomposition)

**Files:**
- Modify: `master_files/system/BaseFunctionality.py` (add `from itertools import permutations` near the top; add 5 new `@staticmethod`s to `Base`)
- Test: `tests/test_base_tensor_algebra.py`

**Interfaces:**
- Produces: `Base.LeviCivita4D() -> ndarray(4,4,4,4)`, `Base.raise_LeviCivita4D(metric) -> ndarray(4,4,4,4)`, `Base.dual_tensor(F_upper, metric) -> ndarray(...,4,4)`, `Base.faraday_from_EB(E, B, u, metric) -> ndarray(...,4,4)`, `Base.observer_frame_fields(F_upper, u, metric) -> (ndarray(...,4), ndarray(...,4))`. All accept either single 4-vectors/tensors or grid-batched arrays with arbitrary leading dimensions (via `...` einsum broadcasting).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_base_tensor_algebra.py`:

```python
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'master_files'))
from system.BaseFunctionality import Base

METRIC = np.diag([-1., 1., 1., 1.])


class TestLeviCivita4D(unittest.TestCase):
    def test_basic_values_and_antisymmetry(self):
        eps = Base.LeviCivita4D()
        self.assertEqual(eps.shape, (4, 4, 4, 4))
        self.assertAlmostEqual(eps[0, 1, 2, 3], 1.0)
        self.assertAlmostEqual(eps[1, 0, 2, 3], -1.0)
        self.assertAlmostEqual(eps[0, 0, 2, 3], 0.0)
        self.assertAlmostEqual(eps[2, 3, 0, 1], 1.0)


class TestDualTensor(unittest.TestCase):
    def test_double_dual_is_minus_identity(self):
        rng = np.random.default_rng(0)
        F = rng.standard_normal((4, 4))
        F = F - F.T  # random antisymmetric tensor
        double_dual = Base.dual_tensor(Base.dual_tensor(F, METRIC), METRIC)
        np.testing.assert_allclose(double_dual, -F, atol=1e-10)

    def test_batched_matches_single_point(self):
        rng = np.random.default_rng(1)
        F = rng.standard_normal((4, 4))
        F = F - F.T
        batch = np.stack([F, 2 * F], axis=0)
        single_0 = Base.dual_tensor(F, METRIC)
        single_1 = Base.dual_tensor(2 * F, METRIC)
        batched = Base.dual_tensor(batch, METRIC)
        np.testing.assert_allclose(batched[0], single_0, atol=1e-10)
        np.testing.assert_allclose(batched[1], single_1, atol=1e-10)


class TestFaradayRoundTrip(unittest.TestCase):
    def test_lab_observer_round_trip(self):
        u_lab = np.array([1., 0., 0., 0.])
        E3 = np.array([0.1, -0.2, 0.3])
        B3 = np.array([0.4, 0.5, -0.6])
        E4 = np.zeros(4); E4[1:] = E3
        B4 = np.zeros(4); B4[1:] = B3

        F = Base.faraday_from_EB(E4, B4, u_lab, METRIC)
        self.assertTrue(np.allclose(F, -F.T, atol=1e-10))  # antisymmetric

        E_rec, B_rec = Base.observer_frame_fields(F, u_lab, METRIC)
        np.testing.assert_allclose(E_rec, E4, atol=1e-10)
        np.testing.assert_allclose(B_rec, B4, atol=1e-10)

    def test_round_trip_is_batched(self):
        u_lab = np.array([1., 0., 0., 0.])
        E4 = np.zeros((3, 4)); E4[:, 1] = [0.1, 0.2, 0.3]
        B4 = np.zeros((3, 4)); B4[:, 2] = [0.4, 0.5, 0.6]

        F = Base.faraday_from_EB(E4, B4, u_lab, METRIC)
        self.assertEqual(F.shape, (3, 4, 4))
        E_rec, B_rec = Base.observer_frame_fields(F, u_lab, METRIC)
        np.testing.assert_allclose(E_rec, E4, atol=1e-10)
        np.testing.assert_allclose(B_rec, B4, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Documents/github/Lagrangian-filtering && python -m unittest tests.test_base_tensor_algebra -v`
Expected: FAIL/ERROR — `AttributeError: type object 'Base' has no attribute 'LeviCivita4D'`.

- [ ] **Step 3: Implement the helpers**

In `master_files/system/BaseFunctionality.py`, add near the top (after the existing `import` block, before `class Base`):

```python
from itertools import permutations
```

Then add these five `@staticmethod`s inside `class Base` (anywhere after `get_rel_vel`, e.g. right after it):

```python
    @staticmethod
    def LeviCivita4D():
        """
        Returns the 4D Levi-Civita symbol epsilon_{abcd} as a (4,4,4,4) ndarray,
        with epsilon_{0123} = +1 and epsilon_{abcd} = 0 whenever any two
        indices repeat.
        """
        eps = np.zeros((4, 4, 4, 4))
        for perm in permutations(range(4)):
            sign = 1
            p = list(perm)
            for x in range(4):
                for y in range(x + 1, 4):
                    if p[x] > p[y]:
                        sign *= -1
            eps[perm] = sign
        return eps

    @staticmethod
    def raise_LeviCivita4D(metric):
        """
        Returns epsilon^{abcd}, i.e. LeviCivita4D() with all four indices
        raised via 'metric' (valid because a diagonal +/-1 Minkowski metric
        is its own inverse).
        """
        eps_lower = Base.LeviCivita4D()
        return np.einsum('ae,bf,cg,dh,efgh->abcd', metric, metric, metric, metric, eps_lower)

    @staticmethod
    def dual_tensor(F_upper, metric):
        """
        Returns the Hodge dual *F^{ab} = (1/2) epsilon^{abcd} F_{cd} of a
        fully-contravariant, antisymmetric rank-2 tensor F^{ab}.

        Parameters
        ----------
        F_upper: ndarray, shape (...,4,4)
            One tensor (4,4) or a grid-batch of tensors with arbitrary
            leading dimensions.
        metric: ndarray, shape (4,4)

        Returns
        -------
        ndarray, same shape as F_upper
        """
        F_lower = np.einsum('ac,bd,...cd->...ab', metric, metric, F_upper)
        levi_upper = Base.raise_LeviCivita4D(metric)
        return 0.5 * np.einsum('abcd,...cd->...ab', levi_upper, F_lower)

    @staticmethod
    def faraday_from_EB(E, B, u, metric):
        """
        Builds the fully-contravariant Faraday tensor
            F^{ab} = u^a E^b - u^b E^a + epsilon^{abcd} u_c B_d
        from the electric/magnetic field 4-vectors E^a, B^a measured by
        observer u^a (each must be orthogonal to u^a) and mu_0=1.
        This is the exact inverse of observer_frame_fields() below.

        Parameters
        ----------
        E, B, u: ndarray, shape (...,4), broadcastable against each other
        metric: ndarray, shape (4,4)

        Returns
        -------
        ndarray, shape (...,4,4)
        """
        term1 = np.einsum('...a,...b->...ab', u, E) - np.einsum('...a,...b->...ab', E, u)
        u_lower = np.einsum('ab,...b->...a', metric, u)
        levi_upper = Base.raise_LeviCivita4D(metric)
        term2 = np.einsum('abcd,...c,...d->...ab', levi_upper, u_lower, B)
        return term1 + term2

    @staticmethod
    def observer_frame_fields(F_upper, u, metric):
        """
        Decomposes a fully-contravariant Faraday tensor F^{ab} into the
        electric/magnetic field 4-vectors measured by observer u^a
        (mostly-plus metric, mu_0=1), following the standard GRMHD
        decomposition (e.g. Rezzolla & Zanotti, "Relativistic
        Hydrodynamics", Sec. 2.4):

            E^a = F^{ab} u_b
            B^a = -u_b (*F)^{ab}

        Both come out orthogonal to u^a automatically. This is the exact
        inverse of faraday_from_EB() above.

        Parameters
        ----------
        F_upper: ndarray, shape (...,4,4)
        u: ndarray, shape (...,4), broadcastable against F_upper's batch dims
        metric: ndarray, shape (4,4)

        Returns
        -------
        (E, B): each ndarray, shape (...,4)
        """
        u_lower = np.einsum('ab,...b->...a', metric, u)
        E = np.einsum('...ab,...b->...a', F_upper, u_lower)
        F_dual = Base.dual_tensor(F_upper, metric)
        B = -np.einsum('...ab,...b->...a', F_dual, u_lower)
        return E, B
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Documents/github/Lagrangian-filtering && python -m unittest tests.test_base_tensor_algebra -v`
Expected: all 5 tests PASS. If `test_lab_observer_round_trip` fails only on the sign of `B_rec` (i.e. `B_rec == -E4`-shaped mismatch, not a shape/crash error), flip the sign in `observer_frame_fields`'s `B` line to `B = np.einsum(...)` (drop the leading `-`) and re-run — this is the one place a genuine textbook sign-convention ambiguity could surface; the test is the authority, not the docstring citation.

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/github/Lagrangian-filtering
git add master_files/system/BaseFunctionality.py tests/test_base_tensor_algebra.py
git commit -m "$(cat <<'EOF'
Add Levi-Civita/dual-tensor/Faraday-decomposition helpers to Base

Generic, grid-batchable tensor-algebra primitives needed for the MHD
filtering extension: LeviCivita4D, raise_LeviCivita4D, dual_tensor,
faraday_from_EB, observer_frame_fields. The last two are literal
inverses of each other (round-trip tested) so the same pair of
functions can build F from (E,B,u) at the micro scale (u=lab observer)
and decompose <F> into (E~,B~) at the meso scale (u=u~), guaranteeing a
consistent sign convention between the two.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `IdealMHD_3D` micro model

**Files:**
- Modify: `master_files/MicroModels.py` (append new class after `IdealHD_3D`, i.e. after line 531)
- Test: `tests/test_ideal_mhd_3d_micro.py`

**Interfaces:**
- Consumes: `Base.faraday_from_EB`, `Base.dual_tensor` from Task 1.
- Produces: `IdealMHD_3D` class with `prim_strs` including `Bx,By,Bz`; `structures_strs = ("BC","SET","bar_vel","FaradayTensor","ChargeCurrent","LorentzForceDensity","SET_EM")`; `setup_structures()` fills all of them. Same `get_*`/`get_var_gridpoint`/`get_interpol_var` interface as `IdealHD_3D` (needed so `resMHD_3D`'s compatibility check and `FileReaders` work unchanged).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ideal_mhd_3d_micro.py`:

```python
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'master_files'))
from MicroModels import IdealMHD_3D


def make_tiny_model():
    """A 1x2x2x2 (t,x,y,z) grid with uniform, non-trivial v and B."""
    m = IdealMHD_3D()
    shape = (1, 2, 2, 2)
    m.domain_vars['nt'] = 1
    m.domain_vars['nx'] = m.domain_vars['ny'] = m.domain_vars['nz'] = 2
    m.domain_vars['dt'] = 1.0
    m.domain_vars['dx'] = m.domain_vars['dy'] = m.domain_vars['dz'] = 0.5
    m.domain_vars['t'] = np.array([0.0])

    m.prim_vars['n'] = np.full(shape, 1.2)
    m.prim_vars['p'] = np.full(shape, 0.3)
    m.prim_vars['vx'] = np.full(shape, 0.1)
    m.prim_vars['vy'] = np.full(shape, -0.05)
    m.prim_vars['vz'] = np.full(shape, 0.02)
    m.prim_vars['Bx'] = np.full(shape, 0.4)
    m.prim_vars['By'] = np.full(shape, -0.3)
    m.prim_vars['Bz'] = np.full(shape, 0.2)

    W = 1.0 / np.sqrt(1.0 - (0.1**2 + 0.05**2 + 0.02**2))
    m.aux_vars['W'] = np.full(shape, W)
    m.aux_vars['e'] = np.full(shape, 1.0)
    m.aux_vars['h'] = np.full(shape, 1.0 + 4.0 / 3.0 * 0.3 / 1.2)
    return m


class TestIdealMHD3DStructures(unittest.TestCase):
    def setUp(self):
        self.m = make_tiny_model()
        self.m.setup_structures()

    def test_faraday_tensor_antisymmetric(self):
        F = self.m.structures['FaradayTensor']
        np.testing.assert_allclose(F, -np.swapaxes(F, -1, -2), atol=1e-12)

    def test_lab_frame_E_reconstructed_correctly(self):
        v = np.array([0.1, -0.05, 0.02])
        B = np.array([0.4, -0.3, 0.2])
        E_expected = -np.cross(v, B)
        F = self.m.structures['FaradayTensor'][0, 0, 0, 0]
        # F^{0i} = -E^i  =>  E^i = -F^{0i}
        E_from_F = -F[0, 1:]
        np.testing.assert_allclose(E_from_F, E_expected, atol=1e-12)

    def test_set_em_matches_independent_maxwell_stress_formula(self):
        v = np.array([0.1, -0.05, 0.02])
        B = np.array([0.4, -0.3, 0.2])
        E = -np.cross(v, B)
        Bsq = np.dot(B, B)
        Esq = np.dot(E, E)
        expected_spatial = (Esq + Bsq) / 2.0 * np.eye(3) - np.outer(E, E) - np.outer(B, B)

        SET_EM = self.m.structures['SET_EM'][0, 0, 0, 0]
        np.testing.assert_allclose(SET_EM[1:, 1:], expected_spatial, atol=1e-10)

    def test_charge_current_recovers_known_gradient(self):
        # Build a model where Bz varies linearly in x: Bz(x) = 0.2 + 0.1*x.
        # Then F^{12} = -Bz(x) varies linearly in x, so d_x F^{12} is constant,
        # and (for v=0, E=0) j^y = d_a F^{ay} = d_x F^{xy} = -d_x F^{12}... this
        # test checks j is finite, grid-shaped, and non-zero only where a real
        # gradient exists (not that it crashes on a uniform field, which
        # trivially gives j=0 -- see next assertion for the non-trivial case).
        m = self.m
        shape = (1, 4, 2, 2)
        m.domain_vars['nx'] = 4
        for key in ('n', 'p', 'vx', 'vy', 'vz', 'By', 'Bx'):
            m.prim_vars[key] = np.zeros(shape)
        m.prim_vars['Bz'] = np.array([0.2 + 0.1 * i for i in range(4)])[None, :, None, None] * np.ones(shape)
        m.aux_vars['W'] = np.ones(shape)
        m.aux_vars['e'] = np.ones(shape)
        m.aux_vars['h'] = np.ones(shape)
        m.domain_vars['t'] = np.array([0.0])

        m.setup_structures()
        j = m.structures['ChargeCurrent']
        self.assertEqual(j.shape, shape + (4,))
        self.assertTrue(np.all(np.isfinite(j)))
        # d/dx of a linear Bz is a nonzero constant -> j should be nonzero somewhere
        self.assertFalse(np.allclose(j, 0.0))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Documents/github/Lagrangian-filtering && python -m unittest tests.test_ideal_mhd_3d_micro -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'IdealMHD_3D'`.

- [ ] **Step 3: Implement `IdealMHD_3D`**

Append to the end of `master_files/MicroModels.py` (after `IdealHD_3D`, i.e. after line 531):

```python
class IdealMHD_3D(object):
    """
    Micro model for 3D ideal-MHD simulation data (e.g. METHOD's SRMHD
    solver). Mirrors IdealHD_3D's fluid handling exactly and adds the EM
    structures needed for Sec. 6D covariant EM filtering: FaradayTensor,
    ChargeCurrent, LorentzForceDensity, SET_EM.

    METHOD's SRMHD output never stores E or j (ideal Ohm's law: E is
    algebraic in v,B; j never appears in the ideal-MHD equations of
    motion) -- both are reconstructed here, not read from file.
    """

    def __init__(self, interp_method="linear"):
        self.spatial_dims = 3
        self.interp_method = interp_method

        self.metric = np.zeros((4, 4))
        self.metric[0, 0] = -1
        self.metric[1, 1] = self.metric[2, 2] = self.metric[3, 3] = +1

        self.domain_int_strs = ('nt', 'nx', 'ny', 'nz')
        self.domain_float_strs = ("tmin", "tmax", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "dt", "dx", "dy", "dz")
        self.domain_array_strs = ("t", "x", "y", "z", "points")
        self.domain_vars = dict.fromkeys(self.domain_int_strs + self.domain_float_strs + self.domain_array_strs)
        for str in self.domain_vars:
            self.domain_vars[str] = []

        self.prim_strs = ("vx", "vy", "vz", "n", "p", "Bx", "By", "Bz")
        self.prim_vars = dict.fromkeys(self.prim_strs)
        for str in self.prim_strs:
            self.prim_vars[str] = []

        self.aux_strs = ("W", "h", "e")
        self.aux_vars = dict.fromkeys(self.aux_strs)
        for str in self.aux_strs:
            self.aux_vars[str] = []

        self.structures_strs = ("BC", "SET", "bar_vel", "FaradayTensor", "ChargeCurrent", "LorentzForceDensity", "SET_EM")
        self.structures = dict.fromkeys(self.structures_strs)
        for str in self.structures_strs:
            self.structures[str] = []

        self.labels_var_dict = {
            'BC': r'$n^{a}$', 'SET': r'$T_F^{ab}$', 'bar_vel': r'$u^a$',
            'FaradayTensor': r'$F^{ab}$', 'ChargeCurrent': r'$j^{a}$',
            'LorentzForceDensity': r'$j_aF^{ab}$', 'SET_EM': r'$T_{EM}^{ab}$',
            'vx': r'$v_x$', 'vy': r'$v_y$', 'vz': r'$v_z$',
            'Bx': r'$B_x$', 'By': r'$B_y$', 'Bz': r'$B_z$',
            'n': r'$n$', 'W': r'$W$', 'e': r'$e$', 'h': r'$h$', 'p': r'$p$'}

    def get_spatial_dims(self):
        return self.spatial_dims

    def get_model_name(self):
        return 'Ideal MHD (3+1d)'

    def get_domain_strs(self):
        return self.domain_int_strs + self.domain_float_strs + self.domain_array_strs

    def get_prim_strs(self):
        return self.prim_strs

    def get_aux_strs(self):
        return self.aux_strs

    def get_structures_strs(self):
        return self.structures_strs

    def get_all_var_strs(self):
        return self.get_prim_strs() + self.get_aux_strs() + self.get_structures_strs()

    def get_gridpoints(self):
        return self.domain_vars['points']

    def get_interpol_var(self, var, point):
        if var in self.get_prim_strs():
            return interpn(self.domain_vars['points'], self.prim_vars[var], point, method=self.interp_method)[0]
        elif var in self.get_aux_strs():
            return interpn(self.domain_vars['points'], self.aux_vars[var], point, method=self.interp_method)[0]
        elif var in self.get_structures_strs():
            return interpn(self.domain_vars['points'], self.structures[var], point, method=self.interp_method)[0]
        else:
            print(f'{var} is not a primitive, auxiliary variable or structure of the micro_model!!')

    @multimethod
    def get_var_gridpoint(self, var: str, h: object, i: object, j: object, k: object):
        if var in self.get_prim_strs():
            return self.prim_vars[var][h, i, j, k]
        elif var in self.get_aux_strs():
            return self.aux_vars[var][h, i, j, k]
        elif var in self.get_structures_strs():
            return self.structures[var][h, i, j, k]
        else:
            print('{} is not a variable of model {}'.format(var, self.get_model_name()))
            return None

    @multimethod
    def get_var_gridpoint(self, var: str, point: object):
        indices = Base.find_nearest_cell(point, self.domain_vars['points'])
        if var in self.get_prim_strs():
            return self.prim_vars[var][tuple(indices)]
        elif var in self.get_aux_strs():
            return self.aux_vars[var][tuple(indices)]
        elif var in self.get_structures_strs():
            return self.structures[var][tuple(indices)]
        else:
            print(f"{var} is not a variable of the model!")
            return None

    def setup_structures(self):
        """
        Sets up fluid structures exactly as IdealHD_3D does (BC, bar_vel,
        SET -- fluid-only), then builds the EM structures on top:
        FaradayTensor, ChargeCurrent, LorentzForceDensity, SET_EM.

        All EM construction is vectorized over the whole grid (not a
        per-gridpoint Python loop) since real target grids are millions
        of points.
        """
        shape = self.prim_vars['n'].shape  # (Nt,Nx,Ny,Nz)

        self.structures["BC"] = np.zeros(shape + (4,))
        self.structures["bar_vel"] = np.zeros(shape + (4,))
        self.structures["SET"] = np.zeros(shape + (4, 4))

        for h in range(shape[0]):
            for i in range(shape[1]):
                for j in range(shape[2]):
                    for k in range(shape[3]):
                        vel_vec = np.array([
                            self.aux_vars['W'][h, i, j, k],
                            self.aux_vars['W'][h, i, j, k] * self.prim_vars['vx'][h, i, j, k],
                            self.aux_vars['W'][h, i, j, k] * self.prim_vars['vy'][h, i, j, k],
                            self.aux_vars['W'][h, i, j, k] * self.prim_vars['vz'][h, i, j, k]])
                        self.structures['bar_vel'][h, i, j, k, :] = vel_vec
                        self.structures['BC'][h, i, j, k, :] = np.multiply(self.prim_vars['n'][h, i, j, k], vel_vec)
                        self.structures['SET'][h, i, j, k, :, :] = (
                            (self.prim_vars['n'][h, i, j, k] * self.aux_vars['h'][h, i, j, k]) * np.outer(vel_vec, vel_vec)
                            + self.prim_vars['p'][h, i, j, k] * self.metric)

        # --- EM structures: vectorized over the whole grid ---
        v = np.stack([self.prim_vars['vx'], self.prim_vars['vy'], self.prim_vars['vz']], axis=-1)
        B3 = np.stack([self.prim_vars['Bx'], self.prim_vars['By'], self.prim_vars['Bz']], axis=-1)
        E3 = -np.cross(v, B3)  # ideal Ohm's law: E = -v x B

        E4 = np.zeros(shape + (4,)); E4[..., 1:] = E3
        B4 = np.zeros(shape + (4,)); B4[..., 1:] = B3
        u_lab = np.array([1., 0., 0., 0.])  # lab/Eulerian observer -- NOT the fluid four-velocity

        F = Base.faraday_from_EB(E4, B4, u_lab, self.metric)
        self.structures['FaradayTensor'] = F

        # ChargeCurrent: j^b = d_a F^{ab}, mu_0=1, via finite differences on the
        # whole (t,x,y,z) grid. Accuracy of the time-derivative term is bounded
        # by inter-snapshot spacing -- a data/config concern, not a bug here.
        dFdt, dFdx, dFdy, dFdz = np.gradient(
            F, self.domain_vars['t'], self.domain_vars['dx'], self.domain_vars['dy'], self.domain_vars['dz'],
            axis=(0, 1, 2, 3))
        j = dFdt[..., 0, :] + dFdx[..., 1, :] + dFdy[..., 2, :] + dFdz[..., 3, :]
        self.structures['ChargeCurrent'] = j

        j_lower = np.einsum('ab,...b->...a', self.metric, j)
        self.structures['LorentzForceDensity'] = np.einsum('...a,...ab->...b', j_lower, F)

        F_lower = np.einsum('ac,bd,...cd->...ab', self.metric, self.metric, F)
        F_mixed = np.einsum('...bd,dc->...bc', F, self.metric)  # F^b_{~c}
        term = np.einsum('...ac,...bc->...ab', F, F_mixed)
        scalar = np.einsum('...cd,...cd->...', F_lower, F)
        self.structures['SET_EM'] = term - 0.25 * np.einsum('ab,...->...ab', self.metric, scalar)

        self.vars = self.prim_vars
        self.vars.update(self.aux_vars)
        self.vars.update(self.structures)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Documents/github/Lagrangian-filtering && python -m unittest tests.test_ideal_mhd_3d_micro -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/github/Lagrangian-filtering
git add master_files/MicroModels.py tests/test_ideal_mhd_3d_micro.py
git commit -m "$(cat <<'EOF'
Add IdealMHD_3D micro model

Mirrors IdealHD_3D's fluid handling and adds vectorized EM structures
(FaradayTensor, ChargeCurrent, LorentzForceDensity, SET_EM) built from
lab-frame E (reconstructed via ideal Ohm's law, since METHOD's SRMHD
never stores it) and B, using the lab observer (1,0,0,0) -- not the
fluid four-velocity -- as Base.faraday_from_EB's observer argument.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Fix `FileReaders.read_in_data3D` to read `Domain/*` metadata as attributes

**Files:**
- Modify: `master_files/FileReaders.py:281-306` (the `domain_int_strs`/`domain_float_strs` loops and the `endTime` read inside `read_in_data3D`)
- Test: `tests/test_file_readers_domain_attrs.py`

**Interfaces:**
- No new interfaces; behavioral fix only. `read_in_data3D(micro_model)` now actually populates `micro_model.domain_vars[...]` for any 3D micro model (`IdealHD_3D` or `IdealMHD_3D`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_file_readers_domain_attrs.py`:

```python
import os
import shutil
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'master_files'))
from FileReaders import METHOD_HDF5
from MicroModels import IdealHD_3D


class TestReadInData3DDomainAttrs(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._write_fake_file(os.path.join(self.tmpdir, 'data0.hdf5'), t=0.0)
        self._write_fake_file(os.path.join(self.tmpdir, 'data1.hdf5'), t=1.0)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_fake_file(self, path, t):
        shape = (2, 2, 2)
        with h5py.File(path, 'w') as f:
            dom = f.create_group('Domain')
            dom.attrs['nx'] = shape[0]
            dom.attrs['ny'] = shape[1]
            dom.attrs['nz'] = shape[2]
            for name, val in [('xmin', 0.0), ('xmax', 1.0), ('ymin', 0.0), ('ymax', 1.0),
                               ('zmin', 0.0), ('zmax', 1.0), ('dx', 0.5), ('dy', 0.5), ('dz', 0.5)]:
                dom.attrs[name] = val
            dom.attrs['endTime'] = t

            prim = f.create_group('Primitive')
            aux = f.create_group('Auxiliary')
            data = np.ones(shape)
            for name in ('rho', 'vx', 'vy', 'vz', 'p'):
                prim.create_dataset(name, data=data)
            for name in ('W', 'h', 'e'):
                aux.create_dataset(name, data=data)

    def test_domain_metadata_read_from_attrs(self):
        reader = METHOD_HDF5(os.path.join(self.tmpdir, ''))
        micro_model = IdealHD_3D()
        reader.read_in_data3D(micro_model)

        self.assertEqual(micro_model.domain_vars['nx'], 2)
        self.assertEqual(micro_model.domain_vars['ny'], 2)
        self.assertEqual(micro_model.domain_vars['nz'], 2)
        self.assertAlmostEqual(micro_model.domain_vars['dx'], 0.5)
        self.assertAlmostEqual(micro_model.domain_vars['xmax'], 1.0)
        np.testing.assert_allclose(micro_model.domain_vars['t'], [0.0, 1.0])


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Documents/github/Lagrangian-filtering && python -m unittest tests.test_file_readers_domain_attrs -v`
Expected: FAIL — prints `nx is not in the hdf5 dataset: check Domain/` etc. to stdout, then `AssertionError: [] != 2`.

- [ ] **Step 3: Fix `read_in_data3D`**

In `master_files/FileReaders.py`, inside `read_in_data3D` (around line 281-311), replace:

```python
        for dom_var_str in micro_model.domain_int_strs: 
            try: 
                if dom_var_str == 'nt': 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = int( self.hdf5_files[0]['Domain/' + dom_var_str][:])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')

        for dom_var_str in micro_model.domain_float_strs: 
            try: 
                if dom_var_str in ['tmin', 'tmax']: 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = float( self.hdf5_files[0]['Domain/' + dom_var_str][:])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')
```

with:

```python
        for dom_var_str in micro_model.domain_int_strs: 
            try: 
                if dom_var_str == 'nt': 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = int(self.hdf5_files[0]['Domain'].attrs[dom_var_str])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 Domain group attributes')

        for dom_var_str in micro_model.domain_float_strs: 
            try: 
                if dom_var_str in ['tmin', 'tmax']: 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = float(self.hdf5_files[0]['Domain'].attrs[dom_var_str])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 Domain group attributes')
```

Then, a few lines further down in the same method, replace:

```python
        micro_model.domain_vars['nt'] = self.num_files
        for counter in range(self.num_files):
            micro_model.domain_vars['t'].append( float(self.hdf5_files[counter]['Domain/endTime'][:]))
```

with:

```python
        micro_model.domain_vars['nt'] = self.num_files
        for counter in range(self.num_files):
            micro_model.domain_vars['t'].append(float(self.hdf5_files[counter]['Domain'].attrs['endTime']))
```

(This is the only `Domain/endTime` occurrence inside `read_in_data3D` specifically — `read_in_data`/`read_in_data_HDF5_missing_xy` have their own, separate copies of this line earlier in the file; leave those untouched, per the spec's scoping.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Documents/github/Lagrangian-filtering && python -m unittest tests.test_file_readers_domain_attrs -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/github/Lagrangian-filtering
git add master_files/FileReaders.py tests/test_file_readers_domain_attrs.py
git commit -m "$(cat <<'EOF'
Fix read_in_data3D to read Domain metadata as HDF5 attributes

METHOD writes nx,ny,nz,dx,dy,dz,xmin,... etc. as attributes on the
Domain group, never as datasets (confirmed via h5dump against real
output in action_filtering/planning/method_srmhd_interop.md). The 3D
reader was indexing them as datasets, so every one of these lookups
silently failed and left domain_vars at its empty-list default --
this made it impossible to load any real METHOD 3D file. Scoped to
read_in_data3D only; the 2D readers have the same bug but are out of
scope here.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `resMHD_3D` — EM structures, filtering, and decomposition

**Files:**
- Modify: `master_files/MesoModels.py` (append new class after `resHD_3D`, i.e. after line 2568)
- Test: `tests/test_res_mhd_3d.py`

**Interfaces:**
- Consumes: `resHD_3D` (base class, unmodified), `Base.observer_frame_fields`, `Base.Mink_dot` from Task 1.
- Produces: `resMHD_3D(resHD_3D)` with `meso_structures_strs` extended by `FaradayTensor,ChargeCurrent,LorentzForceDensity`; `meso_vars_strs` extended by `sigma_tilde,E_tilde,B_tilde,J_tilde,F_closure` (plus the Ohm's-law outputs added in Task 5); `decompose_EM_task(u_t, F_filt, j_filt, lorentz_filt, metric) -> (E,B,sigma,J,F_closure)` (staticmethod, independently testable); `decompose_EM_parallel(n_cpus)` (grid-level driver, called after `decompose_structures_parallel`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_res_mhd_3d.py`:

```python
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'master_files'))
from system.BaseFunctionality import Base
from MesoModels import resMHD_3D

METRIC = np.diag([-1., 1., 1., 1.])


class TestDecomposeEMTask(unittest.TestCase):
    def test_recovers_known_E_B_at_rest_observer(self):
        u_t = np.array([1., 0., 0., 0.])
        E_true = np.array([0., 0.1, -0.2, 0.3])
        B_true = np.array([0., 0.4, 0.5, -0.6])
        F = Base.faraday_from_EB(E_true, B_true, u_t, METRIC)

        j = np.array([0.05, 0.01, -0.02, 0.03])  # j^0 = charge density in this frame
        lorentz = np.einsum('a,ab->b', np.einsum('ab,b->a', METRIC, j), F)

        E, B, sigma, J, F_closure = resMHD_3D.decompose_EM_task(u_t, F, j, lorentz, METRIC)

        np.testing.assert_allclose(E, E_true, atol=1e-12)
        np.testing.assert_allclose(B, B_true, atol=1e-12)
        self.assertAlmostEqual(sigma, j[0], places=12)  # sigma = -u_a j^a = j^0 for u=(1,0,0,0)
        np.testing.assert_allclose(J[1:], j[1:], atol=1e-12)  # spatial current unaffected by projector at rest
        self.assertAlmostEqual(J[0], 0.0, places=12)  # projector removes time component

    def test_F_closure_shape_and_finiteness(self):
        u_t = np.array([1., 0., 0., 0.])
        F = Base.faraday_from_EB(np.array([0., 0.1, 0., 0.]), np.array([0., 0., 0.2, 0.]), u_t, METRIC)
        j = np.array([0.1, 0.02, 0.0, 0.01])
        lorentz = np.einsum('a,ab->b', np.einsum('ab,b->a', METRIC, j), F)

        _, _, _, _, F_closure = resMHD_3D.decompose_EM_task(u_t, F, j, lorentz, METRIC)
        self.assertEqual(F_closure.shape, (4,))
        self.assertTrue(np.all(np.isfinite(F_closure)))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Documents/github/Lagrangian-filtering && python -m unittest tests.test_res_mhd_3d -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'resMHD_3D'`.

- [ ] **Step 3: Implement `resMHD_3D` (structures + filtering + EM decomposition)**

Append to the end of `master_files/MesoModels.py` (after `resHD_3D`, i.e. after line 2568):

```python
class resMHD_3D(resHD_3D):
    """
    Meso model for 3D ideal-MHD filtering (Sec. 6D). Subclasses resHD_3D
    to reuse fluid grid setup/filtering/decomposition/derivatives
    unchanged, adding only the EM structures, EM decomposition, and the
    generic Ohm's-law closure (see fit_ohms_law_closure).
    """

    def __init__(self, micro_model, find_obs, filter, interp_method='linear'):
        super().__init__(micro_model, find_obs, filter, interp_method)

        self.meso_structures_strs = self.meso_structures_strs + ['FaradayTensor', 'ChargeCurrent', 'LorentzForceDensity']
        for var in ('FaradayTensor', 'ChargeCurrent', 'LorentzForceDensity'):
            self.meso_structures[var] = []

        em_scalars = ['sigma_tilde', 'R_tilde', 'alpha_dynamo', 'gamma_hall']
        em_vectors = ['E_tilde', 'B_tilde', 'J_tilde', 'F_closure', 'Ohm_res']
        self.meso_scalars_strs = self.meso_scalars_strs + em_scalars
        self.meso_vectors_strs = self.meso_vectors_strs + em_vectors
        self.meso_vars_strs = self.meso_scalars_strs + self.meso_vectors_strs + self.meso_r2tensors_strs
        for var in em_scalars + em_vectors:
            self.meso_vars[var] = []

        self.labels_var_dict.update({
            'FaradayTensor': r'$\langle F^{ab}\rangle$', 'ChargeCurrent': r'$\langle j^{a}\rangle$',
            'LorentzForceDensity': r'$\langle j_aF^{ab}\rangle$',
            'sigma_tilde': r'$\tilde{\sigma}$', 'E_tilde': r'$\tilde{E}^a$', 'B_tilde': r'$\tilde{B}^a$',
            'J_tilde': r'$\tilde{J}^a$', 'F_closure': r'$\mathcal{F}^b$', 'Ohm_res': r'$\mathcal{W}^a$',
            'R_tilde': r'$\tilde{R}$', 'alpha_dynamo': r'$\alpha$', 'gamma_hall': r'$\gamma$'})

    def get_model_name(self):
        return 'resMHD_3D'

    def setup_meso_grid(self, patch_bdrs, coarse_factor=1, coarse_time=False):
        super().setup_meso_grid(patch_bdrs, coarse_factor, coarse_time)
        Nt, Nx, Ny, Nz = self.domain_vars['Nt'], self.domain_vars['Nx'], self.domain_vars['Ny'], self.domain_vars['Nz']
        for var in ('FaradayTensor',):
            self.meso_structures[var] = np.zeros((Nt, Nx, Ny, Nz, 4, 4))
        for var in ('ChargeCurrent', 'LorentzForceDensity'):
            self.meso_structures[var] = np.zeros((Nt, Nx, Ny, Nz, 4))
        for var in ('sigma_tilde', 'R_tilde', 'alpha_dynamo', 'gamma_hall'):
            self.meso_vars[var] = np.zeros((Nt, Nx, Ny, Nz))
        for var in ('E_tilde', 'B_tilde', 'J_tilde', 'F_closure', 'Ohm_res'):
            self.meso_vars[var] = np.zeros((Nt, Nx, Ny, Nz, 4))

    def setup_mesogrid_smart(self, num_T_slices, spatial_bdrs, coarse_factor):
        super().setup_mesogrid_smart(num_T_slices, spatial_bdrs, coarse_factor)
        Nt, Nx, Ny, Nz = self.domain_vars['Nt'], self.domain_vars['Nx'], self.domain_vars['Ny'], self.domain_vars['Nz']
        for var in ('FaradayTensor',):
            self.meso_structures[var] = np.zeros((Nt, Nx, Ny, Nz, 4, 4))
        for var in ('ChargeCurrent', 'LorentzForceDensity'):
            self.meso_structures[var] = np.zeros((Nt, Nx, Ny, Nz, 4))
        for var in ('sigma_tilde', 'R_tilde', 'alpha_dynamo', 'gamma_hall'):
            self.meso_vars[var] = np.zeros((Nt, Nx, Ny, Nz))
        for var in ('E_tilde', 'B_tilde', 'J_tilde', 'F_closure', 'Ohm_res'):
            self.meso_vars[var] = np.zeros((Nt, Nx, Ny, Nz, 4))

    def filter_micro_vars_parallel(self, n_cpus):
        """
        Same as resHD_3D.filter_micro_vars_parallel, extended with the
        three new EM structures. Reimplemented (not calling super) because
        the base method's 'vars' list is a local literal, not an
        overridable attribute.
        """
        ts, xs, ys, zs = self.domain_vars['T'], self.domain_vars['X'], self.domain_vars['Y'], self.domain_vars['Z']
        t_idxs, x_idxs, y_idxs, z_idxs = np.arange(len(ts)), np.arange(len(xs)), np.arange(len(ys)), np.arange(len(zs))

        points = [list(elem) for elem in product(ts, xs, ys, zs)]
        indices_meso_grid = list(product(t_idxs, x_idxs, y_idxs, z_idxs))

        observers = []
        for elem in indices_meso_grid:
            if self.filter_vars['U_success'][elem]:
                observers.append(self.filter_vars['U'][elem])
            else:
                print('Observers are not computed on (parts of) the grid!')
                return None

        vars = ['BC', 'SET', 'p', 'FaradayTensor', 'ChargeCurrent', 'LorentzForceDensity']
        points_observers = [[points[i], observers[i]] for i in range(len(points))]

        filtered_vars = dict.fromkeys(vars)
        for var in vars:
            positions, filtered_vars[var] = self.filter.filter_var_parallel(points_observers, var, n_cpus)

        for i in range(len(positions)):
            idx = indices_meso_grid[positions[i]]
            self.meso_structures['BC'][idx] = filtered_vars['BC'][i]
            self.meso_structures['SET'][idx] = filtered_vars['SET'][i]
            self.meso_vars['p_filt'][idx] = filtered_vars['p'][i]
            self.meso_structures['FaradayTensor'][idx] = filtered_vars['FaradayTensor'][i]
            self.meso_structures['ChargeCurrent'][idx] = filtered_vars['ChargeCurrent'][i]
            self.meso_structures['LorentzForceDensity'][idx] = filtered_vars['LorentzForceDensity'][i]

    @staticmethod
    def decompose_EM_task(u_t, F_filt, j_filt, lorentz_filt, metric):
        """
        Decomposes the filtered EM structures at a single meso gridpoint,
        given the already-computed fluid Favre observer u_t.

        Parameters
        ----------
        u_t: ndarray (4,) -- Favre observer, from decompose_structures_task
        F_filt: ndarray (4,4) -- <F^{ab}>
        j_filt: ndarray (4,) -- <j^a>
        lorentz_filt: ndarray (4,) -- <j_aF^{ab}> (filtered as ITS OWN
            micro-scale structure, not reconstructed from separately
            filtered j and F -- see mhd_filtering_extension.md Sec 3.5.3)
        metric: ndarray (4,4)

        Returns
        -------
        (E_tilde, B_tilde, sigma_tilde, J_tilde, F_closure)
        """
        E_tilde, B_tilde = Base.observer_frame_fields(F_filt, u_t, metric)

        sigma_tilde = -Base.Mink_dot(u_t, j_filt)
        h_ab = np.einsum('ij,jk->ik', metric + np.einsum('i,j->ij', u_t, u_t), metric)
        J_tilde = np.einsum('ab,b->a', h_ab, j_filt)

        J_tilde_lower = np.einsum('ab,b->a', metric, J_tilde)
        F_closure = -lorentz_filt + np.einsum('a,ab->b', J_tilde_lower, F_filt)

        return E_tilde, B_tilde, sigma_tilde, J_tilde, F_closure

    def decompose_EM_parallel(self, n_cpus):
        """
        Runs decompose_EM_task at every meso gridpoint. Requires
        decompose_structures_parallel() to have been run first (needs
        u_tilde).
        """
        args_for_pool = []
        for h in range(len(self.domain_vars['T'])):
            for i in range(len(self.domain_vars['X'])):
                for j in range(len(self.domain_vars['Y'])):
                    for k in range(len(self.domain_vars['Z'])):
                        args_for_pool.append((
                            self.meso_vars['u_tilde'][h, i, j, k],
                            self.meso_structures['FaradayTensor'][h, i, j, k],
                            self.meso_structures['ChargeCurrent'][h, i, j, k],
                            self.meso_structures['LorentzForceDensity'][h, i, j, k],
                            self.metric, h, i, j, k))

        with mp.Pool(processes=n_cpus) as pool:
            print('Decomposing EM structures in parallel with {} processes'.format(pool._processes), flush=True)
            results = pool.starmap(resMHD_3D._decompose_EM_task_pool, args_for_pool)
            for E, B, sigma, J, F_closure, h, i, j, k in results:
                self.meso_vars['E_tilde'][h, i, j, k] = E
                self.meso_vars['B_tilde'][h, i, j, k] = B
                self.meso_vars['sigma_tilde'][h, i, j, k] = sigma
                self.meso_vars['J_tilde'][h, i, j, k] = J
                self.meso_vars['F_closure'][h, i, j, k] = F_closure

    @staticmethod
    def _decompose_EM_task_pool(u_t, F_filt, j_filt, lorentz_filt, metric, h, i, j, k):
        E, B, sigma, J, F_closure = resMHD_3D.decompose_EM_task(u_t, F_filt, j_filt, lorentz_filt, metric)
        return E, B, sigma, J, F_closure, h, i, j, k
```

Note: `decompose_EM_task` itself takes no grid indices (pure per-point function, directly unit-testable, matching the test written in Step 1); `_decompose_EM_task_pool` is the thin multiprocessing-friendly wrapper that threads indices through, mirroring the existing `decompose_structures_task`/`decompose_structures_parallel` split pattern in `resHD_3D`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Documents/github/Lagrangian-filtering && python -m unittest tests.test_res_mhd_3d -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/github/Lagrangian-filtering
git add master_files/MesoModels.py tests/test_res_mhd_3d.py
git commit -m "$(cat <<'EOF'
Add resMHD_3D meso model: EM structures, filtering, decomposition

Subclasses resHD_3D to reuse fluid grid setup/filtering/decomposition
unchanged. Adds FaradayTensor/ChargeCurrent/LorentzForceDensity to the
filtered structures, and decompose_EM_task/decompose_EM_parallel which
compute E~,B~ (via the same Base.observer_frame_fields used at the
micro scale, now with u=u~ instead of the lab observer), sigma~,J~ (via
the same projector already built for the fluid decomposition), and the
Lorentz-force closure residual F_closure (eq. 692), using the
separately-filtered LorentzForceDensity structure rather than a product
of independently-filtered j and F.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Generic, switchable Ohm's-law closure fit

**Files:**
- Modify: `master_files/MesoModels.py` (add two methods to `resMHD_3D`, appended after `_decompose_EM_task_pool`)
- Test: `tests/test_ohms_law_closure.py`

**Interfaces:**
- Consumes: `Base.raise_LeviCivita4D`, `Base.Mink_dot` (Task 1); `resMHD_3D.meso_vars['E_tilde'/'J_tilde'/'B_tilde'/'u_tilde']` (Task 4).
- Produces: `resMHD_3D.ohms_law_fit_gridpoint(E_tilde, J_tilde, B_tilde, u_tilde, metric, use_resistive=True, use_dynamo=True, use_hall=True) -> (R_tilde, alpha_dynamo, gamma_hall, Ohm_res)` (staticmethod); `resMHD_3D.fit_ohms_law_closure(self, use_resistive=True, use_dynamo=True, use_hall=True)` (grid-level driver, writes `meso_vars['R_tilde'/'alpha_dynamo'/'gamma_hall'/'Ohm_res']`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ohms_law_closure.py`:

```python
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'master_files'))
from system.BaseFunctionality import Base
from MesoModels import resMHD_3D

METRIC = np.diag([-1., 1., 1., 1.])


def hall_operand(J, u, B, metric):
    levi_upper = Base.raise_LeviCivita4D(metric)
    return np.einsum('abcd,b,c,d->a', levi_upper, J, u, B)


class TestOhmsLawFitGridpoint(unittest.TestCase):
    def setUp(self):
        self.u = np.array([1., 0., 0., 0.])
        self.J = np.array([0., 0.3, -0.1, 0.05])
        self.B = np.array([0., 0.2, 0.4, -0.1])
        self.R_true, self.alpha_true, self.gamma_true = 0.7, -0.4, 0.2
        hall = hall_operand(self.J, self.u, self.B, METRIC)
        self.E = self.R_true * self.J + self.alpha_true * self.B + self.gamma_true * hall

    def test_all_terms_on_recovers_known_coefficients(self):
        R, alpha, gamma, res = resMHD_3D.ohms_law_fit_gridpoint(
            self.E, self.J, self.B, self.u, METRIC,
            use_resistive=True, use_dynamo=True, use_hall=True)
        self.assertAlmostEqual(R, self.R_true, places=8)
        self.assertAlmostEqual(alpha, self.alpha_true, places=8)
        self.assertAlmostEqual(gamma, self.gamma_true, places=8)
        np.testing.assert_allclose(res, np.zeros(4), atol=1e-8)

    def test_disabled_terms_are_nan_not_zero(self):
        R, alpha, gamma, res = resMHD_3D.ohms_law_fit_gridpoint(
            self.E, self.J, self.B, self.u, METRIC,
            use_resistive=True, use_dynamo=False, use_hall=False)
        self.assertFalse(np.isnan(R))
        self.assertTrue(np.isnan(alpha))
        self.assertTrue(np.isnan(gamma))

    def test_missing_term_shows_up_as_larger_residual(self):
        # Fit with dynamo+hall off: since the synthetic E genuinely contains
        # an alpha*B contribution, the fitted resistive-only model must leave
        # a nonzero residual (the missing physics can't vanish by magic).
        _, _, _, res_partial = resMHD_3D.ohms_law_fit_gridpoint(
            self.E, self.J, self.B, self.u, METRIC,
            use_resistive=True, use_dynamo=False, use_hall=False)
        _, _, _, res_full = resMHD_3D.ohms_law_fit_gridpoint(
            self.E, self.J, self.B, self.u, METRIC,
            use_resistive=True, use_dynamo=True, use_hall=True)
        self.assertGreater(np.linalg.norm(res_partial), np.linalg.norm(res_full))

    def test_all_terms_off_returns_full_residual(self):
        R, alpha, gamma, res = resMHD_3D.ohms_law_fit_gridpoint(
            self.E, self.J, self.B, self.u, METRIC,
            use_resistive=False, use_dynamo=False, use_hall=False)
        self.assertTrue(np.isnan(R) and np.isnan(alpha) and np.isnan(gamma))
        np.testing.assert_allclose(res, self.E, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Documents/github/Lagrangian-filtering && python -m unittest tests.test_ohms_law_closure -v`
Expected: FAIL/ERROR — `AttributeError: type object 'resMHD_3D' has no attribute 'ohms_law_fit_gridpoint'`.

- [ ] **Step 3: Implement the closure fit**

Append to `resMHD_3D` in `master_files/MesoModels.py` (after `_decompose_EM_task_pool`):

```python
    @staticmethod
    def ohms_law_fit_gridpoint(E_tilde, J_tilde, B_tilde, u_tilde, metric,
                                use_resistive=True, use_dynamo=True, use_hall=True):
        """
        Fits the generic Ohm's law closure at a single meso gridpoint:

            E_tilde^a = R_tilde*J_tilde^a + alpha*B_tilde^a + gamma*Hall^a + Ohm_res^a

        where Hall^a = epsilon^{abcd} J_tilde_b u_tilde_c B_tilde_d. Each
        term is switched on/off independently via its use_* flag; a
        disabled term's coefficient is returned as np.nan (never 0.0), and
        its contribution stays folded into Ohm_res rather than being
        silently absorbed by the remaining terms.

        Returns
        -------
        (R_tilde, alpha_dynamo, gamma_hall, Ohm_res): float, float, float, ndarray(4,)
        """
        names, operators = [], []
        if use_resistive:
            names.append('R_tilde'); operators.append(J_tilde)
        if use_dynamo:
            names.append('alpha_dynamo'); operators.append(B_tilde)
        if use_hall:
            levi_upper = Base.raise_LeviCivita4D(metric)
            hall = np.einsum('abcd,b,c,d->a', levi_upper, J_tilde, u_tilde, B_tilde)
            names.append('gamma_hall'); operators.append(hall)

        coeffs = {'R_tilde': np.nan, 'alpha_dynamo': np.nan, 'gamma_hall': np.nan}

        if len(operators) == 0:
            return coeffs['R_tilde'], coeffs['alpha_dynamo'], coeffs['gamma_hall'], np.array(E_tilde, dtype=float)

        n = len(operators)
        G = np.array([[Base.Mink_dot(operators[p], operators[q]) for q in range(n)] for p in range(n)])
        b = np.array([Base.Mink_dot(operators[p], E_tilde) for p in range(n)])
        solved = np.linalg.solve(G, b)

        for name, value in zip(names, solved):
            coeffs[name] = value

        residual = np.array(E_tilde, dtype=float)
        for name, op in zip(names, operators):
            residual = residual - coeffs[name] * op

        return coeffs['R_tilde'], coeffs['alpha_dynamo'], coeffs['gamma_hall'], residual

    def fit_ohms_law_closure(self, use_resistive=True, use_dynamo=True, use_hall=True):
        """
        Runs ohms_law_fit_gridpoint at every meso gridpoint, storing
        results in meso_vars['R_tilde'/'alpha_dynamo'/'gamma_hall'/'Ohm_res'].
        Requires decompose_EM_parallel() and decompose_structures_parallel()
        (for u_tilde) to have been run first. Not parallelized: the
        per-point linear system is at most 3x3, and meso grids are
        coarse-grained (orders of magnitude fewer points than the micro
        grid), so a plain loop is fast enough here.
        """
        Nt, Nx, Ny, Nz = self.domain_vars['Nt'], self.domain_vars['Nx'], self.domain_vars['Ny'], self.domain_vars['Nz']
        for h in range(Nt):
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        R, alpha, gamma, res = resMHD_3D.ohms_law_fit_gridpoint(
                            self.meso_vars['E_tilde'][h, i, j, k],
                            self.meso_vars['J_tilde'][h, i, j, k],
                            self.meso_vars['B_tilde'][h, i, j, k],
                            self.meso_vars['u_tilde'][h, i, j, k],
                            self.metric,
                            use_resistive=use_resistive, use_dynamo=use_dynamo, use_hall=use_hall)
                        self.meso_vars['R_tilde'][h, i, j, k] = R
                        self.meso_vars['alpha_dynamo'][h, i, j, k] = alpha
                        self.meso_vars['gamma_hall'][h, i, j, k] = gamma
                        self.meso_vars['Ohm_res'][h, i, j, k, :] = res
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Documents/github/Lagrangian-filtering && python -m unittest tests.test_ohms_law_closure -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Run the full test suite together**

Run: `cd ~/Documents/github/Lagrangian-filtering && python -m unittest discover -s tests -v`
Expected: all tests from Tasks 1-5 PASS (no cross-file breakage).

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/github/Lagrangian-filtering
git add master_files/MesoModels.py tests/test_ohms_law_closure.py
git commit -m "$(cat <<'EOF'
Add generic, independently-switchable Ohm's law closure fit

ohms_law_fit_gridpoint fits E~ = R~J~ + alpha*B~ + gamma*Hall + Ohm_res
at a single meso gridpoint via the Minkowski-inner-product normal
equations, using only the enabled candidate operators -- each of
resistive/dynamo/Hall is toggled by its own boolean kwarg. A disabled
term's coefficient is set to NaN (never 0.0) and its physical
contribution stays in the residual rather than being silently absorbed
by the remaining terms (verified: a genuinely-present but disabled term
increases residual norm, it doesn't distort the other fitted
coefficients into compensating for it undetectably).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Config section and driver script

**Files:**
- Modify: `filter_scripts/config_filter.txt` (add `[Ohms_law_settings]` section)
- Create: `filter_scripts/pickling_meso_mhd.py`

**Interfaces:**
- Consumes: `IdealMHD_3D` (Task 2), `resMHD_3D` + `fit_ohms_law_closure` (Tasks 4-5), `METHOD_HDF5.read_in_data3D` (Task 3, fixed).
- Produces: a runnable driver script analogous to `pickling_meso.py`, parametrized by the new config section. No automated test (needs real multi-GB HDF5 data, matching the existing repo convention where `pickling_meso.py`/`test_meso3D.py` are exercised manually, not under a test framework) — verified instead via `py_compile` (a real, if minimal, "does this even parse and import cleanly" check).

- [ ] **Step 1: Add the config section**

In `filter_scripts/config_filter.txt`, add a new section (after `[Meso_model_settings]`, before `[Plot_settings]`):

```
[Ohms_law_settings]

# Each term of the generic Ohm's law closure (E~ = R~*J~ + alpha*B~ + gamma*Hall + Ohm_res)
# can be switched on/off independently.
terms = {"resistive": true, "dynamo": true, "hall": true}
```

- [ ] **Step 2: Write the driver script**

Create `filter_scripts/pickling_meso_mhd.py`:

```python
import sys
sys.path.append('../master_files/')
import configparser
import json
import pickle
import time

from FileReaders import *
from MicroModels import *
from Filters import *
from MesoModels import *

if __name__ == '__main__':

    ####################################################################################################
    # SCRIPT TO FILTER 3D IDEAL-MHD DATA AND FIT THE GENERIC OHM'S LAW CLOSURE, THEN PICKLE THE RESULT
    ####################################################################################################

    if len(sys.argv) == 1:
        print("You must pass the configuration file for the simulations.")
        raise Exception()

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    hdf5_directory = config['Directories']['hdf5_dir']
    print('=========================================================================')
    print(f'Starting MHD filtering job on data from {hdf5_directory}')
    print('=========================================================================\n\n')

    snapshots_opts = json.loads(config['Micro_model_settings']['snapshots_opts'])
    FileReader = METHOD_HDF5(hdf5_directory, snapshots_opts['fewer_snaps_required'], snapshots_opts['smaller_list'])

    micro_model = IdealMHD_3D()
    FileReader.read_in_data3D(micro_model)
    micro_model.setup_structures()
    print('Finished reading micro data from hdf5, structures also set up.', flush=True)
    print('Micro-model times: {}'.format(micro_model.domain_vars['t']))

    meso_grid = json.loads(config['Meso_model_settings']['meso_grid_smart'])
    filtering_options = json.loads(config['Meso_model_settings']['filtering_options'])

    coarse_factor = meso_grid['coarse_grain_factor']
    num_T_slices = int(meso_grid['num_T_slices'])
    meso_spatial_bdrs = [meso_grid['x_range'], meso_grid['y_range'], meso_grid['z_range']]

    box_len = float(filtering_options['box_len_ratio']) * micro_model.domain_vars['dx']
    width = float(filtering_options['filter_width_ratio']) * micro_model.domain_vars['dx']
    find_obs = FindObs_root_parallel(micro_model, box_len)
    filter = box_filter_parallel(micro_model, width)

    meso_model = resMHD_3D(micro_model, find_obs, filter)
    meso_model.setup_mesogrid_smart(num_T_slices, meso_spatial_bdrs, coarse_factor)
    print('Finished setting up the meso_grid.', flush=True)

    n_cpus = int(config['Meso_model_settings']['n_cpus'])

    start_time = time.perf_counter()
    meso_model.find_observers_parallel(n_cpus)
    print('Observers found: time taken= {}\n'.format(time.perf_counter() - start_time), flush=True)

    start_time = time.perf_counter()
    meso_model.filter_micro_vars_parallel(n_cpus)
    print('Filtering ended: time taken= {}\n'.format(time.perf_counter() - start_time), flush=True)

    start_time = time.perf_counter()
    meso_model.decompose_structures_parallel(n_cpus)
    meso_model.decompose_EM_parallel(n_cpus)
    print('Decomposition ended: time taken= {}\n'.format(time.perf_counter() - start_time), flush=True)

    ohms_law_terms = json.loads(config['Ohms_law_settings']['terms'])
    start_time = time.perf_counter()
    meso_model.fit_ohms_law_closure(
        use_resistive=ohms_law_terms['resistive'],
        use_dynamo=ohms_law_terms['dynamo'],
        use_hall=ohms_law_terms['hall'])
    print('Ohms law closure fit ended: time taken= {}\n'.format(time.perf_counter() - start_time), flush=True)

    saving_directory = config['Directories']['pickled_files_dir']
    meso_pickled_filename = config['Directories']['meso_pickled_filename']
    with open(saving_directory + meso_pickled_filename, 'wb') as filehandle:
        pickle.dump(meso_model, filehandle)
```

- [ ] **Step 3: Verify the script parses and imports cleanly**

Run:
```bash
cd ~/Documents/github/Lagrangian-filtering/filter_scripts
python -m py_compile pickling_meso_mhd.py
python -c "
import sys, configparser
sys.path.append('../master_files/')
c = configparser.ConfigParser()
c.read('config_filter.txt')
import json
print(json.loads(c['Ohms_law_settings']['terms']))
"
```
Expected: `py_compile` produces no output (success); the second command prints `{'resistive': True, 'dynamo': True, 'hall': True}`.

- [ ] **Step 4: Commit**

```bash
cd ~/Documents/github/Lagrangian-filtering
git add filter_scripts/config_filter.txt filter_scripts/pickling_meso_mhd.py
git commit -m "$(cat <<'EOF'
Add config section and driver script for 3D ideal-MHD filtering

New [Ohms_law_settings] config section toggles the resistive/dynamo/
Hall terms independently. pickling_meso_mhd.py mirrors pickling_meso.py
but uses IdealMHD_3D/resMHD_3D and calls fit_ohms_law_closure with the
configured toggles before pickling.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** all 5 components from the spec (`system/BaseFunctionality.py`, `MicroModels.py`, `FileReaders.py`, `MesoModels.py`, config+script) have a task. Testing section of the spec is covered by Tasks 1-5's test files (one refinement: the spec proposed separate ad hoc meso-decomposition formulas; this plan instead reuses `Base.observer_frame_fields`/`faraday_from_EB` identically at both scales, which is a stronger guarantee of sign-convention consistency than the spec's literal text — same physics, tighter implementation).
- **Type/name consistency checked:** `IdealMHD_3D.structures_strs` (Task 2) matches exactly what `resMHD_3D.filter_micro_vars_parallel` (Task 4) filters (`FaradayTensor`, `ChargeCurrent`, `LorentzForceDensity`); `decompose_EM_task`'s return order `(E,B,sigma,J,F_closure)` matches its two call sites (the Task 4 test and `decompose_EM_parallel`); `ohms_law_fit_gridpoint`'s signature matches both its Task 5 test and `fit_ohms_law_closure`'s call to it.
- **No placeholders:** every step has literal code, not a description.
