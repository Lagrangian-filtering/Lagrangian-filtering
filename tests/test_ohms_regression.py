import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'master_files'))
import OhmsRegression as OR


def synthetic(shape=(1, 8, 8, 4), R=0.3, alpha=-0.2, gamma=0.5, noise=0.01, seed=1):
    """Spatial 4-vectors (time component 0, i.e. u = (1,0,0,0)) with E built from the three operators."""
    rng = np.random.default_rng(seed)
    J = np.zeros(shape + (4,)); B = np.zeros(shape + (4,))
    J[..., 1:] = rng.normal(size=shape + (3,))
    B[..., 1:] = rng.normal(size=shape + (3,))
    H = np.zeros(shape + (4,))
    H[..., 1:] = np.cross(J[..., 1:], B[..., 1:])
    E = R * J + alpha * B + gamma * H
    E[..., 1:] += noise * rng.normal(size=shape + (3,))
    return E, {'R_tilde': J, 'alpha_dynamo': B, 'gamma_hall': H}


class TestOhmsRegression(unittest.TestCase):
    def test_recovers_constant_coefficients(self):
        E, ops = synthetic()
        fit = OR.fit_region(E, ops, OR.TERMS)
        self.assertAlmostEqual(fit['R_tilde'], 0.3, delta=0.02)
        self.assertAlmostEqual(fit['alpha_dynamo'], -0.2, delta=0.02)
        self.assertAlmostEqual(fit['gamma_hall'], 0.5, delta=0.02)
        self.assertGreater(fit['R2'], 0.99)

    def test_dropping_a_term_lowers_r2_and_gives_nan(self):
        E, ops = synthetic()
        full = OR.fit_region(E, ops, OR.TERMS)
        no_hall = OR.fit_region(E, ops, ('R_tilde', 'alpha_dynamo'))
        self.assertLess(no_hall['R2'], full['R2'] - 0.1)
        self.assertTrue(np.isnan(no_hall['gamma_hall']))
        self.assertEqual(len(OR.subset_table(E, ops)), 7)

    def test_unrelated_data_gives_chance_level_r2(self):
        E, ops = synthetic()
        null = OR.permutation_null(E, ops, OR.TERMS, n_perm=20)
        self.assertLess(null.max(), 0.05)
        # pointwise identity does not hold once the coefficients are shared: random E fits badly
        rng = np.random.default_rng(5)
        E_rand = np.zeros_like(E)
        E_rand[..., 1:] = rng.normal(size=E[..., 1:].shape)
        self.assertLess(OR.fit_region(E_rand, ops, OR.TERMS)['R2'], 0.05)

    def test_tiles_recover_constant_coefficients(self):
        E, ops = synthetic()
        tiles = OR.tile_fits(E, ops, OR.TERMS, block=(4, 4, 2), min_points=20)
        self.assertEqual(len(tiles), 2 * 2 * 2)
        for t in tiles:
            self.assertAlmostEqual(t['gamma_hall'], 0.5, delta=0.1)
        spread = OR.summarize([t['gamma_hall'] for t in tiles])
        self.assertAlmostEqual(spread['median'], 0.5, delta=0.05)

    def test_tile_spread_reveals_varying_coefficient(self):
        # gamma differs between the x halves: a constant-coefficient closure is wrong
        E1, ops = synthetic(gamma=0.5, seed=2)
        E2, ops2 = synthetic(gamma=-0.5, seed=2)
        E = E1.copy(); E[:, 4:] = E2[:, 4:]
        for k in ops:
            ops[k][:, 4:] = ops2[k][:, 4:]
        whole = OR.fit_region(E, ops, OR.TERMS)
        self.assertLess(whole['R2'], 0.9)   # shared coefficient cannot explain both halves
        tiles = OR.tile_fits(E, ops, OR.TERMS, block=(4, 8, 4), min_points=20)
        vals = sorted(t['gamma_hall'] for t in tiles)
        self.assertGreater(vals[-1] - vals[0], 0.8)

    @staticmethod
    def smooth_field(shape, rng, sigma=2.0):
        """Spatially correlated random 4-vector field (time component 0)."""
        white = rng.normal(size=shape + (3,))
        k = [np.fft.fftfreq(n) * 2 * np.pi for n in shape[1:]]
        k2 = k[0][:, None, None] ** 2 + k[1][None, :, None] ** 2 + k[2][None, None, :] ** 2
        kern = np.exp(-0.5 * sigma ** 2 * k2)[None, ..., None]
        f = np.real(np.fft.ifftn(np.fft.fftn(white, axes=(1, 2, 3)) * kern, axes=(1, 2, 3)))
        out = np.zeros(shape + (4,)); out[..., 1:] = f / f.std()
        return out

    def test_point_null_gives_false_positive_that_shift_null_avoids(self):
        # Unrelated smooth fields still fit by chance (they share large-scale structure).
        # The point-permutation null destroys that structure and so calls the fit significant;
        # the shift null keeps it and correctly does not.
        rng = np.random.default_rng(3)
        shape = (1, 16, 16, 8)
        E = self.smooth_field(shape, rng)
        ops = {t: self.smooth_field(shape, rng) for t in OR.TERMS}
        real = OR.fit_region(E, ops, OR.TERMS)['R2']
        point = OR.permutation_null(E, ops, OR.TERMS, n_perm=30)
        shifted = OR.shift_null(E, ops, OR.TERMS, n_perm=30, min_shift=3)
        self.assertGreater(real, point.max())
        self.assertLessEqual(real, shifted.max())

    def test_shift_null_below_a_real_relation(self):
        rng = np.random.default_rng(4)
        shape = (1, 16, 16, 8)
        J = self.smooth_field(shape, rng); B = self.smooth_field(shape, rng)
        H = np.zeros_like(J); H[..., 1:] = np.cross(J[..., 1:], B[..., 1:])
        ops = {'R_tilde': J, 'alpha_dynamo': B, 'gamma_hall': H}
        E = 0.4 * J + 0.05 * self.smooth_field(shape, rng)
        real = OR.fit_region(E, ops, OR.TERMS)['R2']
        null = OR.shift_null(E, ops, OR.TERMS, n_perm=30, min_shift=3)
        self.assertGreater(real, 0.9)
        self.assertLess(null.max(), real - 0.3)

    def test_shift_null_rejects_grids_too_small_to_shift(self):
        E, ops = synthetic(shape=(1, 4, 4, 3))
        with self.assertRaises(ValueError):
            OR.shift_null(E, ops, OR.TERMS, n_perm=2, min_shift=3)

    def test_degenerate_operators_do_not_crash(self):
        E, ops = synthetic()
        ops['alpha_dynamo'] = ops['R_tilde'].copy()   # linearly dependent operators
        ops['gamma_hall'] = np.zeros_like(ops['gamma_hall'])
        fit = OR.fit_region(E, ops, OR.TERMS)
        self.assertTrue(np.isfinite(fit['R2']))

    def test_tile_edges_merge_short_remainder(self):
        self.assertEqual(OR._edges(10, 4), [(0, 4), (4, 8), (8, 10)])   # remainder 2 is not < b//2
        self.assertEqual(OR._edges(9, 4), [(0, 4), (4, 9)])             # remainder 1 < 2: merged
        self.assertEqual(OR._edges(8, 4), [(0, 4), (4, 8)])

    def test_box_sum_matches_brute_force_with_clipped_edges(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(6, 5, 4))
        got = OR.box_sum(a, (3, 3, 1))
        for i, j, k in [(0, 0, 0), (2, 3, 1), (5, 4, 3), (0, 4, 2)]:
            ref = a[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2, k:k + 1].sum()
            self.assertAlmostEqual(got[i, j, k], ref)

    def test_local_maps_pick_the_right_term_in_each_half(self):
        # left half: E = 0.3 J (resistive); right half: E = 0.5 J x B (Hall)
        EL, ops = synthetic(shape=(1, 16, 16, 8), R=0.3, alpha=0.0, gamma=0.0, noise=0.01, seed=7)
        EH, _ = synthetic(shape=(1, 16, 16, 8), R=0.0, alpha=0.0, gamma=0.5, noise=0.01, seed=7)
        E = EL.copy(); E[:, 8:] = EH[:, 8:]
        maps = OR.local_term_maps(E, ops, (5, 5, 3))
        win = OR.winner_map(maps['R2'], threshold=0.3)
        self.assertTrue(np.all(win[:5][maps['valid'][:5]] == 0))     # R (index 0) on the left
        self.assertTrue(np.all(win[11:][maps['valid'][11:]] == 2))   # gamma (index 2) on the right
        self.assertAlmostEqual(float(np.median(maps['coeff'][0][:5])), 0.3, delta=0.03)
        self.assertGreater(float(np.nanmedian(maps['R2_full'])), 0.9)

    def test_winner_none_below_threshold_and_null_threshold_is_small_for_unrelated_fields(self):
        rng = np.random.default_rng(8)
        shape = (1, 16, 16, 8)
        E = self.smooth_field(shape, rng)
        ops = {t: self.smooth_field(shape, rng) for t in OR.TERMS}
        thr = OR.local_null_threshold(E, ops, (5, 5, 3), n_null=10)
        self.assertTrue(0.0 < thr < 1.0)
        maps = OR.local_term_maps(E, ops, (5, 5, 3))
        self.assertTrue(np.all(OR.winner_map(maps['R2'], threshold=2.0) == -1))

    def test_merge_places_abutting_tiles_and_marks_missing(self):
        d = 0.5
        def tile(x0, y0, val):
            X = x0 + d * np.arange(3); Y = y0 + d * np.arange(2); Z = np.array([0.0])
            return {'X': X, 'Y': Y, 'Z': Z,
                    'fields': {'a': np.full((1, 3, 2, 1, 4), val), 's': np.full((1, 3, 2, 1), val)}}
        # two tiles side by side in x, a third offset in y leaving a hole at the far corner
        merged = OR.merge_meso_tiles([tile(0.0, 0.0, 1.0), tile(1.5, 0.0, 2.0), tile(0.0, 1.0, 3.0)], (d, d, d))
        self.assertEqual(merged['fields']['a'].shape, (1, 6, 4, 1, 4))
        self.assertTrue(np.allclose(merged['X'], 0.5 * np.arange(6)))
        self.assertEqual(merged['fields']['s'][0, 0, 0, 0], 1.0)
        self.assertEqual(merged['fields']['s'][0, 5, 1, 0], 2.0)
        self.assertEqual(merged['fields']['s'][0, 2, 3, 0], 3.0)
        self.assertFalse(merged['present'][5, 3, 0])        # uncovered corner
        self.assertEqual(int(merged['present'].sum()), 18)

    def test_merge_rejects_misaligned_tiles(self):
        d = 0.5
        a = {'X': d * np.arange(3), 'Y': d * np.arange(2), 'Z': np.array([0.0]), 'fields': {'s': np.ones((1, 3, 2, 1))}}
        b = {'X': 0.2 + d * np.arange(3), 'Y': d * np.arange(2), 'Z': np.array([0.0]), 'fields': {'s': np.ones((1, 3, 2, 1))}}
        with self.assertRaises(ValueError):
            OR.merge_meso_tiles([a, b], (d, d, d))

    @staticmethod
    def linear_fields(shape=(1, 6, 5, 4), a=2.0, b=3.0, vx=0.0, gamma=1.0, D=None):
        x = np.arange(shape[1], dtype=float)[None, :, None, None] * np.ones(shape)
        v = np.zeros(shape + (3,)); v[..., 0] = vx; v[..., 1] = a * x       # v = (vx, a x, 0)
        B = np.zeros(shape + (3,)); B[..., 2] = b * x                        # B = (0, 0, b x)
        Dd = np.ones(shape) if D is None else D
        u = np.zeros(shape + (4,)); u[..., 0] = gamma; u[..., 1:] = gamma * v
        return v, B, Dd, u

    def test_gradient_model_static_observer_matches_hand_calculation(self):
        # v = (0, a x, 0), B = (0, 0, b x): d_x v x d_x B = (a b, 0, 0); no time or advective part
        v, B, D, u = self.linear_fields()
        ops = OR.gradient_model_operators(v, B, D, u, (1.0, 1.0, 1.0), sigma2=0.5)
        g1 = 0.5 * 2.0 * 3.0
        np.testing.assert_allclose(ops['model_vB'][0, 2, 2, 2, 1:], [g1, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(ops['model_dens'], 0.0, atol=1e-12)    # uniform density: no weighting term

    def test_gradient_model_advective_term(self):
        # advection speed 0.5 along x adds 2 gamma^2 (Dt v x Dt B) with Dt = 0.5 d_x: extra 2 g^2 (0.25) a b
        gam = 2.0
        v, B, D, u = self.linear_fields(vx=0.5, gamma=gam)
        u[..., 0] = gam; u[..., 1:] = gam * v   # keep gamma as given (not 1/sqrt(1-v^2)); only the algebra is tested
        ops = OR.gradient_model_operators(v, B, D, u, (1.0, 1.0, 1.0), sigma2=1.0)
        a, b = 2.0, 3.0
        expect = a * b + 2 * gam ** 2 * 0.25 * a * b
        self.assertAlmostEqual(float(ops['model_vB'][0, 2, 2, 2, 1] / gam), expect, places=6)

    def test_observer_electric_is_orthogonal_to_the_observer(self):
        rng = np.random.default_rng(1)
        v = rng.uniform(-0.4, 0.4, size=(2, 3, 3, 3, 3)); gam = 1.0 / np.sqrt(1 - np.sum(v ** 2, axis=-1))
        u = np.zeros(v.shape[:-1] + (4,)); u[..., 0] = gam; u[..., 1:] = gam[..., None] * v
        O = OR.observer_electric(rng.normal(size=v.shape), v, gam)
        np.testing.assert_allclose(OR.mink_dot(u, O), 0.0, atol=1e-10)

    def test_density_term_is_zero_for_uniform_density_and_nonzero_otherwise(self):
        v, B, D, u = self.linear_fields()
        x = np.arange(6, dtype=float)[None, :, None, None] * np.ones(D.shape)
        ops = OR.gradient_model_operators(v, B, 1.0 + 0.1 * x, u, (1.0, 1.0, 1.0), sigma2=1.0)
        self.assertGreater(float(np.abs(ops['model_dens']).max()), 1e-3)

    def test_prediction_r2_and_subgrid_energy(self):
        E, _ = synthetic()
        self.assertAlmostEqual(OR.prediction_r2(E, E), 1.0)
        self.assertLess(OR.prediction_r2(E, -E), 0.0)
        rng = np.random.default_rng(2)
        b = rng.normal(size=(500, 3)) + np.array([1.0, 0.0, 2.0])
        sub = OR.subgrid_B_energy(np.mean(np.sum(b ** 2, axis=1)), np.mean(b, axis=0))
        self.assertAlmostEqual(float(sub), float(np.sum(np.var(b, axis=0))), places=10)
        self.assertGreaterEqual(float(sub), 0.0)

    def test_prediction_shift_null_is_lower_than_a_good_prediction(self):
        rng = np.random.default_rng(6)
        shape = (1, 16, 16, 8)
        E = self.smooth_field(shape, rng)
        pred = E + 0.2 * self.smooth_field(shape, rng)       # a good prediction
        real = OR.prediction_r2(E, pred)
        null = OR.prediction_shift_null(E, pred, n_perm=20, min_shift=3)
        self.assertGreater(real, 0.8)
        self.assertLess(null.max(), real - 0.5)


if __name__ == '__main__':
    unittest.main()
