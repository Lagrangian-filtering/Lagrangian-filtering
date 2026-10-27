import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'master_files'))
import SparseLibrary as SL


def _tracefree(a):
    return a - np.einsum('...kk->...', a)[..., None, None] * np.eye(3) / 3.0


def _form_matrix(S, T):
    f = SL.quadratic_forms(S, T)
    return np.stack([x.reshape(-1) for x in f.values()], axis=1), list(f)


def _rank(M, tol=1e-8):
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol * s[0]))


class TestForms(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.S = rng.normal(size=(300, 3, 3))
        self.T = rng.normal(size=(300, 3, 3))

    def test_eleven_forms_span_six_five_or_four_dimensions(self):
        # isotropic rank-5 tensors of 3 dimensions: 6 independent; 5 with div B = 0 (trace of T), 4 with also div v = 0 (trace of S)
        self.assertEqual(_rank(_form_matrix(self.S, self.T)[0]), 6)
        self.assertEqual(_rank(_form_matrix(self.S, _tracefree(self.T))[0]), 5)
        self.assertEqual(_rank(_form_matrix(_tracefree(self.S), _tracefree(self.T))[0]), 4)

    def test_curl_v_cross_curl_J_is_a_combination_of_the_others(self):
        M, names = _form_matrix(self.S, self.T)
        i = names.index('w_x_J')
        others = np.delete(M, i, axis=1)
        c, *_ = np.linalg.lstsq(others, M[:, i], rcond=None)
        self.assertLess(np.linalg.norm(others @ c - M[:, i]) / np.linalg.norm(M[:, i]), 1e-10)

    def test_the_filter_form_is_the_sum_over_k_of_cross_products(self):
        f = SL.quadratic_forms(self.S, self.T, ['dv_x_dB'])['dv_x_dB']
        ref = sum(np.cross(self.S[:, k, :], self.T[:, k, :]) for k in range(3))
        np.testing.assert_allclose(f, ref, atol=1e-12)

    def test_observer_frame_vector_is_orthogonal_to_u(self):
        rng = np.random.default_rng(1)
        shape = (1, 6, 6, 6)
        vel = 0.3 * rng.normal(size=shape + (3,))
        B = rng.normal(size=shape + (3,))
        gamma = 1.0 / np.sqrt(1.0 - np.sum(vel ** 2, axis=-1))
        lib = SL.quadratic_library(vel, B, (0.1, 0.1, 0.1), 0.01, gamma)
        u = np.concatenate([gamma[..., None], gamma[..., None] * vel], axis=-1)
        for x in lib.values():
            dot = -x[..., 0] * u[..., 0] + np.sum(x[..., 1:] * u[..., 1:], axis=-1)
            np.testing.assert_allclose(dot, 0.0, atol=1e-10)


class TestSTLSQ(unittest.TestCase):
    def columns(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        O = {k: SL.spatial4(rng.normal(size=(n, 3))) for k in 'abcdef'}
        return O

    def test_recovers_sparse_truth_among_distractors(self):
        O = self.columns()
        E = 2.0 * O['b'] - 0.5 * O['e']
        names = list(O)
        G, r, EE = SL.gram_system(O, E, names)
        c, act, R2 = SL.stlsq(G, r, EE, 0.05)
        self.assertEqual([n for n, a in zip(names, act) if a], ['b', 'e'])
        self.assertAlmostEqual(c[1], 2.0, places=6)
        self.assertAlmostEqual(c[4], -0.5, places=6)
        self.assertGreater(R2, 1 - 1e-10)

    def test_threshold_removes_a_term_that_carries_little(self):
        O = self.columns()
        E = 2.0 * O['b'] + 0.02 * O['c']
        names = list(O)
        G, r, EE = SL.gram_system(O, E, names)
        _, act, _ = SL.stlsq(G, r, EE, 0.05)
        self.assertEqual([n for n, a in zip(names, act) if a], ['b'])
        _, act, _ = SL.stlsq(G, r, EE, 0.001)
        self.assertEqual([n for n, a in zip(names, act) if a], ['b', 'c'])

    def test_independent_columns_drops_a_linear_combination_and_respects_order(self):
        O = self.columns()
        O['sum'] = O['a'] + O['c']
        names = list(O)
        G, _, _ = SL.gram_system(O, O['a'], names)
        kept, dropped = SL.independent_columns(G)
        self.assertEqual([names[i] for i in dropped], ['sum'])
        names2 = ['sum'] + [n for n in names if n != 'sum']
        G2, _, _ = SL.gram_system(O, O['a'], names2)
        kept2, dropped2 = SL.independent_columns(G2)
        self.assertEqual(len(dropped2), 1)
        self.assertEqual(names2[dropped2[0]], 'c')            # the first of the dependent set to be reached last is dropped

    def test_all_zero_target_gives_no_terms(self):
        O = self.columns()
        names = list(O)
        G, r, _ = SL.gram_system(O, 0.0 * O['a'], names)
        c, act, _ = SL.stlsq(G, r, 1e-12, 0.05)
        self.assertFalse(act.any())


class TestWindows(unittest.TestCase):
    def test_tiles_conserve_the_total(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(2, 8, 8, 4, 4))
        w = SL.window_sums({'a': a}, (2, 4, 4, 2))['a']
        self.assertEqual(w.shape, (8, 4))
        np.testing.assert_allclose(w.sum(axis=0), a.sum(axis=(0, 1, 2, 3)), atol=1e-10)

    def test_overlapping_windows_and_hann_weights(self):
        a = np.ones((1, 6, 6, 6, 4))
        w = SL.window_sums({'a': a}, (1, 3, 3, 3), stride=(1, 1, 1, 1), kind='box')['a']
        self.assertEqual(w.shape[0], 4 * 4 * 4)
        np.testing.assert_allclose(w, 27.0)
        h = SL.window_sums({'a': a}, (1, 4, 4, 4), kind='hann')['a']
        phi = np.hanning(6)[1:-1]
        np.testing.assert_allclose(h[0, 0], phi.sum() ** 3)


def _library(m, compressible, **kw):
    vel = m['v'] if compressible else m['v_plain']
    lib = SL.static_library(vel, m['B'], m['D'], m['spacing'], m['sigma2'], density=compressible, **kw)
    sl = (slice(None),) + (slice(2, -2),) * 3                # drop the edge cells (one-sided derivatives)
    return {k: x[sl] for k, x in lib.items()}, sl


def _select(lib, E, thr=0.05, tol=1e-3):
    names = list(lib)
    G, r, EE = SL.gram_system(lib, E, names)
    kept, dropped = SL.independent_columns(G, tol)
    nk = [names[i] for i in kept]
    c, act, R2 = SL.stlsq(G[np.ix_(kept, kept)], r[kept], EE, thr)
    return {n: float(x) for n, x, a in zip(nk, c, act) if a}, R2, dropped, names


class TestSynthetic(unittest.TestCase):
    """Fields filtered with a box average on a fine periodic grid: the exact EMF deficit against the library."""

    @classmethod
    def setUpClass(cls):
        cls.mi = SL.synthetic_meso(n_fine=40, width=4, stride=1, seed=3, kmax=2, compressible=False)
        cls.mc = SL.synthetic_meso(n_fine=40, width=4, stride=1, seed=3, kmax=2, compressible=True)

    def test_incompressible_flow_selects_only_the_filter_covariance_with_coefficient_minus_one(self):
        lib, sl = _library(self.mi, False)
        E = SL.spatial4(self.mi['E_exact'])[sl]
        coef, R2, dropped, _ = _select(lib, E)
        self.assertEqual(list(coef), ['dv_x_dB'])
        self.assertAlmostEqual(coef['dv_x_dB'], -1.0, delta=0.06)
        self.assertGreater(R2, 0.99)

    def test_compressible_flow_selects_the_covariance_and_the_density_term(self):
        lib, sl = _library(self.mc, True)
        E = SL.spatial4(self.mc['E_favre'])[sl]
        coef, R2, _, _ = _select(lib, E)
        self.assertEqual(sorted(coef), ['dv_x_dB', 'model_dens'])
        self.assertAlmostEqual(coef['dv_x_dB'], -1.0, delta=0.08)
        self.assertAlmostEqual(coef['model_dens'], 1.0, delta=0.08)
        self.assertGreater(R2, 0.99)

    def test_compressible_flow_has_five_independent_quadratic_forms(self):
        lib, sl = _library(self.mc, True)
        G, _, _ = SL.gram_system(lib, SL.spatial4(self.mc['E_favre'])[sl], list(lib))
        names = list(lib)
        kept, dropped = SL.independent_columns(G)
        quad = [n for n in SL.FORMS if names.index(n) in kept]
        self.assertEqual(len(quad), 5)                                       # 6 for general gradients, minus one for div B = 0
        self.assertIn('divv_J', quad)                                        # (div v) J is a live term when div v != 0
        self.assertIn(names.index('w_divB'), dropped)

    def test_an_added_compressible_term_is_found(self):
        lib, sl = _library(self.mc, True)
        E = SL.spatial4(self.mc['E_favre'])[sl] + 0.5 * lib['divv_J']
        coef, R2, _, _ = _select(lib, E)
        self.assertEqual(sorted(coef), ['divv_J', 'dv_x_dB', 'model_dens'])
        self.assertAlmostEqual(coef['divv_J'], 0.5, delta=0.04)
        self.assertAlmostEqual(coef['dv_x_dB'], -1.0, delta=0.1)
        self.assertGreater(R2, 0.99)

    def test_an_added_curl_curl_term_is_found_in_the_incompressible_case(self):
        lib, sl = _library(self.mi, False)
        E = SL.spatial4(self.mi['E_exact'])[sl] + 0.6 * lib['w_x_J']
        coef, R2, _, _ = _select(lib, E)
        self.assertEqual(sorted(coef), ['dv_x_dB', 'w_x_J'])
        self.assertAlmostEqual(coef['w_x_J'], 0.6, delta=0.03)
        self.assertGreater(R2, 0.99)

    def test_a_term_that_is_a_combination_of_kept_forms_is_reproduced_not_named(self):
        lib, sl = _library(self.mc, True)
        E = 0.8 * lib['gradBT_w']                              # a dropped (dependent) form
        coef, R2, dropped, names = _select(lib, E)
        self.assertIn(names.index('gradBT_w'), dropped)
        self.assertGreater(R2, 0.999)                          # the span is the same: the fit reproduces it with kept forms
        self.assertNotIn('gradBT_w', coef)

    def test_forms_that_vanish_identically_are_dropped(self):
        lib, sl = _library(self.mi, False)                     # solenoidal flow and field: (div v) J and (curl v)(div B) are zero to round-off
        _, _, dropped, names = _select(lib, SL.spatial4(self.mi['E_exact'])[sl])
        self.assertIn(names.index('divv_J'), dropped)
        self.assertIn(names.index('w_divB'), dropped)

    def test_noise_does_not_change_the_selection_and_the_bootstrap_agrees(self):
        lib, sl = _library(self.mc, True)
        E0 = SL.spatial4(self.mc['E_favre'])[sl]
        rng = np.random.default_rng(5)
        E = E0 + 0.1 * np.sqrt(SL.OR.mink_dot(E0, E0).mean()) * SL.spatial4(rng.normal(size=E0.shape[:-1] + (3,)))
        coef, R2, _, _ = _select(lib, E)
        self.assertEqual(sorted(coef), ['dv_x_dB', 'model_dens'])
        names = list(lib)
        G, _, _ = SL.gram_system(lib, E, names)
        kept, _ = SL.independent_columns(G)
        nk = [names[i] for i in kept]
        win = SL.window_sums({**{n: lib[n] for n in nk}, 'E': E}, (1, 6, 6, 6))
        bs = SL.bootstrap_inclusion({n: win[n] for n in nk}, win['E'], nk, 0.05, n_boot=60)
        self.assertEqual(bs['dv_x_dB'][0], 1.0)
        self.assertAlmostEqual(bs['dv_x_dB'][1], -1.0, delta=0.15)
        for n in nk:
            if n not in ('dv_x_dB', 'model_dens'):
                self.assertLess(bs[n][0], 0.5, n)


class TestScript(unittest.TestCase):
    def test_synthetic_run_finds_the_filter_term_and_writes_the_npz(self):
        import subprocess
        import tempfile
        script = os.path.join(os.path.dirname(__file__), '..', 'filter_scripts', 'regress_sparse_library.py')
        with tempfile.TemporaryDirectory() as d:
            out = subprocess.run([sys.executable, script, '--synthetic', '--n-fine', '32', '--n-boot', '10', '--n-null', '5', '--stride', '1', '3', '3', '1', '--outdir', d],
                                 capture_output=True, text=True, check=True).stdout
            self.assertIn('SELECTION at threshold 0.05: dv_x_dB, model_dens', out)
            with np.load(os.path.join(d, 'sparse_library.npz')) as _z:
                z = {k: _z[k] for k in _z.files}
            j = list(z['names']).index('dv_x_dB')
            self.assertAlmostEqual(float(z['path_coef'][2][j]), -1.0, delta=0.1)


if __name__ == '__main__':
    unittest.main()


class TestDeficits(unittest.TestCase):
    """Stress, Poynting and energy deficits of box-filtered synthetic fields against the library."""

    @classmethod
    def setUpClass(cls):
        cls.m = SL.synthetic_meso(n_fine=40, width=4, stride=1, seed=3, kmax=2, compressible=False, deficits=True)
        cls.sl = (slice(None),) + (slice(3, -3),) * 3
        cls.lib = SL.deficit_library(cls.m['v_plain'], cls.m['B'], cls.m['spacing'], cls.m['sigma2'])
        cls.y = cls.m['deficit'][cls.sl]

    def fit(self, cls, names):
        comps, w = SL.DEFICIT_COMPONENTS[cls]
        ops = {k: x[self.sl] for k, x in self.lib[cls].items()}
        G, r, EE = SL.gram_system_k(ops, self.y, names, comps, w)
        c, act, R2 = SL.stlsq(G, r, EE, 0.0)
        pred = sum(ops[n] for n in names)[..., comps]
        d = self.y[..., comps]
        return dict(zip(names, c)), R2, 1.0 - np.sum(w * (d - pred) ** 2) / EE

    def test_structural_model_reproduces_the_exact_deficits_with_unit_coefficients(self):
        for cls in ('stress', 'poynting', 'energy'):
            coef, R2, R2_free = self.fit(cls, ['lin_' + cls, 'quad_' + cls])
            self.assertGreater(R2_free, 0.97, cls)                       # parameter-free
            for n, c in coef.items():
                self.assertAlmostEqual(c, 1.0, delta=0.2, msg=n)

    def test_pieces_of_the_quadratic_part_add_up(self):
        st = self.lib['stress']
        parts = st['BB_grad'] + st['BB_trace'] + st['EaEa_grad'] + st['EaEa_trace'] + st['EaEb_grad'] + st['EaEb_trace'] + st['EbEb_grad'] + st['EbEb_trace']
        np.testing.assert_allclose(parts, st['quad_stress'], atol=1e-12 * np.abs(st['quad_stress']).max())
        en = self.lib['energy']
        np.testing.assert_allclose(en['BB_energy'] + en['EaEa_energy'] + en['EaEb_energy'] + en['EbEb_energy'], en['quad_energy'], atol=1e-12 * np.abs(en['quad_energy']).max())

    def test_the_library_selects_the_structure_among_the_further_forms(self):
        comps, w = SL.DEFICIT_COMPONENTS['stress']
        ops = {k: x[self.sl] for k, x in self.lib['stress'].items()}
        names = list(ops)
        G, r, EE = SL.gram_system_k(ops, self.y, names, comps, w)
        kept, _ = SL.independent_columns(G)
        c, act, R2 = SL.stlsq(G[np.ix_(kept, kept)], r[kept], EE, 0.05)
        sel = [names[i] for i, a in zip(kept, act) if a]
        self.assertEqual(sorted(sel), ['lin_stress', 'quad_stress'])
        self.assertGreater(R2, 0.99)

    def test_a_class_has_zero_operators_outside_its_components(self):
        for cls, ops in self.lib.items():
            comps, _ = SL.DEFICIT_COMPONENTS[cls]
            for n, x in ops.items():
                other = np.delete(x, np.arange(10)[comps], axis=-1)
                self.assertEqual(float(np.abs(other).max()), 0.0, n)


class TestFourthOrder(unittest.TestCase):
    def test_the_predicted_fourth_order_terms_reduce_the_residual(self):
        n, width = 48, 6
        m = SL.synthetic_meso(n_fine=n, width=width, stride=1, seed=3, kmax=3, compressible=False)
        sl = (slice(None),) + (slice(3, -3),) * 3
        q = SL.quadratic_library(m['v_plain'], m['B'], m['spacing'], m['sigma2'], None, ['dv_x_dB'])['dv_x_dB'][sl]
        f4 = {k: x[sl] for k, x in SL.fourth_order_forms(m['v_plain'], m['B'], m['spacing'], m['sigma2']).items()}
        E = SL.spatial4(m['E_exact'])[sl]
        m4 = (width ** 2 - 1) * (3 * width ** 2 - 7) / 240 * (2 * np.pi / n) ** 4
        c = SL.fourth_order_coefficients(m['sigma2'], m4)
        resid = lambda P: SL.OR.mink_dot(E - P, E - P).sum()
        r2 = resid(-q)
        r4 = resid(-q + sum(c[k] * f4[k] for k in f4))
        self.assertLess(r4, 0.15 * r2)                                  # the residual of the second-order model falls by a factor of at least 7
        self.assertEqual(c['o4_cross21'], 0.0)
        self.assertAlmostEqual(c['o4_cross11'], -1.0)

    def test_continuous_box_moment_gives_the_continuous_coefficients(self):
        c = SL.fourth_order_coefficients(1.0, 1.8)
        self.assertAlmostEqual(c['o4_axis31'], 0.2)                    # -(0.3 - 0.5)
        self.assertAlmostEqual(c['o4_axis22'], -0.2)                   # -(0.45 - 0.25)
