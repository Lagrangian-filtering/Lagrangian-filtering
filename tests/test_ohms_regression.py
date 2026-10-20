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
        # advection speed 0.5 along x adds gamma^2 (Dt v x Dt B) with Dt = 0.5 d_x: extra g^2 (0.25) a b
        gam = 2.0
        v, B, D, u = self.linear_fields(vx=0.5, gamma=gam)
        u[..., 0] = gam; u[..., 1:] = gam * v   # keep gamma as given (not 1/sqrt(1-v^2)); only the algebra is tested
        ops = OR.gradient_model_operators(v, B, D, u, (1.0, 1.0, 1.0), sigma2=1.0)
        a, b = 2.0, 3.0
        expect = a * b + gam ** 2 * 0.25 * a * b
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

    @staticmethod
    def levi_upper_minkowski():
        """eps^{abcd} for metric diag(-1,1,1,1) with eps_{0123} = +1, built independently of Base."""
        import itertools
        eps = np.zeros((4, 4, 4, 4))
        for perm in itertools.permutations(range(4)):
            sign = np.linalg.det(np.eye(4)[list(perm)])
            eps[perm] = -sign                      # raising four indices with diag(-1,1,1,1) multiplies by -1
        return eps

    def test_hall_operator_is_orthogonal_to_observer_and_JxB_when_static(self):
        metric = np.diag([-1.0, 1.0, 1.0, 1.0]); eps = self.levi_upper_minkowski()
        rng = np.random.default_rng(3)
        v = rng.uniform(-0.5, 0.5, (2, 3, 3)); g = 1.0 / np.sqrt(1.0 - np.sum(v ** 2, axis=-1))
        u = np.zeros(v.shape[:-1] + (4,)); u[..., 0] = g; u[..., 1:] = g[..., None] * v
        def orth(x3):
            x = np.zeros(v.shape[:-1] + (4,)); x[..., 1:] = x3
            return x + u * OR.mink_dot(u, x)[..., None]
        J, B = orth(rng.normal(size=v.shape)), orth(rng.normal(size=v.shape))
        H = OR.hall_operator(J, u, B, eps, metric)
        np.testing.assert_allclose(OR.mink_dot(u, H), 0.0, atol=1e-10)
        # static observer: spatial part is J x B
        u0 = np.zeros_like(u); u0[..., 0] = 1.0
        J0 = np.zeros_like(u); J0[..., 1:] = rng.normal(size=v.shape)
        B0 = np.zeros_like(u); B0[..., 1:] = rng.normal(size=v.shape)
        np.testing.assert_allclose(OR.hall_operator(J0, u0, B0, eps, metric)[..., 1:],
                                   np.cross(J0[..., 1:], B0[..., 1:]), atol=1e-12)
        ops = OR.standard_operators(J, B, u, eps, metric)
        self.assertEqual(sorted(ops), sorted(OR.TERMS))

    def test_covariance_parts_sum_to_the_operators_and_time_parts_vanish_without_time(self):
        rng = np.random.default_rng(11)
        shape = (3, 8, 8, 6)
        v = rng.uniform(-0.4, 0.4, shape + (3,))
        gam = 1.0 / np.sqrt(1.0 - np.sum(v ** 2, axis=-1))
        u = np.zeros(shape + (4,)); u[..., 0] = gam; u[..., 1:] = gam[..., None] * v
        B = self.smooth_field(shape, rng)[..., 1:]
        D = 1.0 + 0.2 * self.smooth_field(shape, rng)[..., 1]
        sp = (0.1, 0.1, 0.1)
        for dt in (0.1, None):
            parts = OR.gradient_model_parts(v, B, D, u, sp, 0.3, dt)
            ops = OR.gradient_model_operators(v, B, D, u, sp, 0.3, dt)
            np.testing.assert_allclose(sum(parts['vB_' + k] for k in ('ss', 'tt', 'ts', 'vv')), ops['model_vB'], atol=1e-10)
            np.testing.assert_allclose(sum(parts['dens_' + k] for k in ('ss', 'tt', 'ts', 'vv')), ops['model_dens'], atol=1e-10)
        for k in ('vB_tt', 'vB_ts', 'dens_tt', 'dens_ts'):      # dt=None: no time derivatives
            np.testing.assert_allclose(parts[k], 0.0, atol=1e-14)
        self.assertGreater(float(np.abs(OR.gradient_model_parts(v, B, D, u, sp, 0.3, 0.1)['vB_tt']).max()), 1e-6)

    def test_covariance_parts_match_hand_calculation(self):
        gam = 2.0
        v, B, D, u = self.linear_fields(vx=0.5, gamma=gam)
        parts = OR.gradient_model_parts(v, B, D, u, (1.0, 1.0, 1.0), 1.0)
        a, b = 2.0, 3.0
        # ss = d_x v x d_x B = (a b, 0, 0); vv = gam^2 (0.5 d_x v) x (0.5 d_x B) = gam^2 0.25 a b
        self.assertAlmostEqual(float(parts['vB_ss'][0, 2, 2, 2, 1] / gam), a * b, places=6)
        self.assertAlmostEqual(float(parts['vB_vv'][0, 2, 2, 2, 1] / gam), gam ** 2 * 0.25 * a * b, places=6)

    def test_static_observer_covariance_is_purely_spatial(self):
        # advecting velocity 0 and gamma 1: the 3D spatial box has no time-derivative content,
        # even for fields that vary in time (tt = (gamma^2-1) ..., ts, vv all vanish)
        rng = np.random.default_rng(12)
        shape = (3, 8, 8, 6)
        a = self.smooth_field(shape, rng)[..., 1:]; b = self.smooth_field(shape, rng)[..., 1:]
        v = np.zeros(shape + (3,)); gam = np.ones(shape)
        with_time = OR.covariance_bracket_parts(a, b, v, gam, (0.1,) * 3, 0.1, np.cross)
        without = OR.covariance_bracket_parts(a, b, v, gam, (0.1,) * 3, None, np.cross)
        np.testing.assert_allclose(with_time['ss'], without['ss'], atol=1e-12)
        for k in ('tt', 'ts', 'vv'):
            np.testing.assert_allclose(with_time[k], 0.0, atol=1e-12)
        self.assertGreater(float(np.abs(with_time['ss']).max()), 1e-3)
    def test_local_coefficients_are_masked_like_r2_where_the_operator_is_not_spacelike(self):
        E, ops = synthetic(shape=(1, 10, 10, 6))
        timelike = ops['R_tilde'].copy()
        timelike[..., 0] = 100.0                      # |O|^2 = -1e4 + ... < 0: not a spacelike operator
        ops = dict(ops, R_tilde=timelike)
        maps = OR.local_term_maps(E, ops, (3, 3, 3))
        self.assertTrue(np.all(np.isnan(maps['R2'][0])))
        self.assertTrue(np.all(np.isnan(maps['coeff'][0])))         # was finite garbage before the mask
        self.assertTrue(np.all(np.isfinite(maps['coeff'][1])))      # the other operators are untouched


MINK = np.diag([-1.0, 1.0, 1.0, 1.0])


def _boost(v):
    """Lorentz boost matrix Lambda^a_b taking (1,0,0,0) to gamma (1, v)."""
    v = np.asarray(v, dtype=float)
    g = 1.0 / np.sqrt(1.0 - v @ v)
    L = np.eye(4)
    L[0, 0] = g
    L[0, 1:] = L[1:, 0] = g * v
    vv = v @ v
    L[1:, 1:] += ((g - 1.0) * np.outer(v, v) / vv) if vv > 0 else 0.0
    return L


class TestStressTensor(unittest.TestCase):
    def test_decompose_round_trip_boosted(self):
        rng = np.random.default_rng(3)
        for v in ([0, 0, 0], [0.3, -0.2, 0.4], [-0.6, 0.1, 0.2]):
            L = _boost(v)
            u = L[:, 0]
            eps, p = 2.3, 0.7
            q = L @ np.array([0, 0.1, -0.2, 0.05])                       # spatial in the rest frame
            A = rng.normal(size=(3, 3)); A = A + A.T; A -= np.trace(A) / 3 * np.eye(3)
            pi = L @ np.pad(A, ((1, 0), (1, 0))) @ L.T                   # trace-free, spatial in the rest frame
            h = MINK + np.outer(u, u)                                    # h^{ab} (inverse metric is MINK)
            T = eps * np.outer(u, u) + p * h + np.outer(q, u) + np.outer(u, q) + pi
            d = OR.decompose_set(T, u, MINK)
            self.assertAlmostEqual(float(d['eps']), eps, places=10)
            self.assertAlmostEqual(float(d['p']), p, places=10)
            np.testing.assert_allclose(d['q'], q, atol=1e-10)
            np.testing.assert_allclose(d['pi'], pi, atol=1e-10)

    def test_weights_give_the_invariant_norm(self):
        pi = np.zeros((4, 4)); pi[1, 2] = pi[2, 1] = 1.0; pi[0, 1] = pi[1, 0] = 0.5
        w = OR.set_weights(MINK)
        lowered = MINK @ pi @ MINK
        self.assertAlmostEqual(float(np.sum(w * pi.reshape(16) ** 2)), float(np.sum(lowered * pi)))

    def test_weighted_fit_recovers_coefficient(self):
        rng = np.random.default_rng(1)
        O = rng.normal(size=(4, 5, 5, 3, 16))
        w = OR.set_weights(MINK)
        fit = OR.fit_region(1.7 * O, {'a': O}, ('a',), weights=w)
        self.assertAlmostEqual(fit['a'], 1.7, places=10)
        self.assertAlmostEqual(OR.prediction_r2(O, O, weights=w), 1.0)

    def _filtered(self, W, n=36, nq=3):
        """Filter analytic smooth micro fields with the 27-point quadrature in each point's rest space (fields are
        time independent, so only the spatial part of the boosted legs matters). Returns meso fields on an n^3 grid."""
        h = 1.0 / n
        ax = (np.arange(n) + 0.5) * h
        X = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), axis=-1)

        def micro(x):
            k = 2 * np.pi
            sx, sy, sz = (np.sin(k * x[..., i] + ph) for i, ph in enumerate((0.3, 1.1, 2.0)))
            cx, cy, cz = (np.cos(k * x[..., i] + ph) for i, ph in enumerate((0.3, 1.1, 2.0)))
            v = 0.25 * np.stack([sy * cz, sz * cx, sx * cy], axis=-1)
            dens = 1.0 + 0.3 * sx * sy
            w = 2.0 + 0.4 * cy * sz
            pr = 0.5 + 0.1 * cz * sx
            gam = 1.0 / np.sqrt(1.0 - np.sum(v ** 2, axis=-1))
            u = np.concatenate([gam[..., None], gam[..., None] * v], axis=-1)
            nvec = dens[..., None] * u
            s = w / dens ** 2
            T = s[..., None, None] * nvec[..., :, None] * nvec[..., None, :] + pr[..., None, None] * MINK
            return nvec, s, T

        # pass 1: lab-box means give the observer; pass 2: boosted-leg quadrature around each point
        pts = np.array([0.0, np.sqrt(3.0 / 5.0), -np.sqrt(3.0 / 5.0)]) * W / 2.0
        wts = np.array([4.0 / 9.0, 5.0 / 18.0, 5.0 / 18.0])

        def average(U, field):
            N = None
            for i, (a, wa) in enumerate(zip(pts, wts)):
                for j, (b, wb) in enumerate(zip(pts, wts)):
                    for k, (c, wc) in enumerate(zip(pts, wts)):
                        off = np.einsum('...ij,j->...i', U, np.array([a, b, c]))
                        f = field(X + off)
                        N = wa * wb * wc * f if N is None else N + wa * wb * wc * f
            return N

        N0 = average(np.broadcast_to(np.eye(3), X.shape[:3] + (3, 3)), lambda x: micro(x)[0])
        u0 = N0 / np.sqrt(-np.einsum('...a,ab,...b->...', N0, MINK, N0))[..., None]
        v0 = u0[..., 1:] / u0[..., :1]
        vv = np.sum(v0 ** 2, axis=-1)[..., None, None]
        g0 = u0[..., 0][..., None, None]
        U = np.eye(3) + (g0 - 1.0) * v0[..., :, None] * v0[..., None, :] / np.maximum(vv, 1e-30)   # spatial boost block
        N = average(U, lambda x: micro(x)[0])
        T = average(U, lambda x: micro(x)[2])
        u = N / np.sqrt(-np.einsum('...a,ab,...b->...', N, MINK, N))[..., None]
        return h, N[None], T[None], u[None]

    def _errors(self, W):
        h, N, T, u = self._filtered(W)
        d = OR.decompose_set(T, u, MINK)
        ntil = np.sqrt(-np.einsum('...a,ab,...b->...', N, MINK, N))
        s = (d['eps'] + d['p']) / ntil ** 2
        vel = u[..., 1:] / u[..., :1]
        m = OR.stress_model(N, s, u, vel, (h, h, h), W ** 2 / 12.0, MINK, dt=None)
        sl = (slice(None),) + (slice(4, -4),) * 3
        wt = OR.set_weights(MINK)
        err = {}
        pi_t, pi_p = d['pi'][sl].reshape(-1, 16), m['pi'][sl].reshape(-1, 16)
        q_t, q_p = d['q'][sl], (m['q1'] + m['q2'])[sl]
        err['pi'] = 1.0 - OR.prediction_r2(pi_t, pi_p, weights=wt)
        err['q'] = 1.0 - OR.prediction_r2(q_t, q_p)
        err['pi_amp'] = float(np.sqrt(np.sum(wt * pi_t ** 2)))
        err['q_amp'] = float(np.sqrt(OR.mink_dot(q_t, q_t).sum()))
        return err

    def test_second_order_model_matches_filtered_stress(self):
        e1, e2 = self._errors(0.2), self._errors(0.1)
        # the sub-filter stress and heat flux are genuinely there ...
        self.assertGreater(e1['pi_amp'], 0.0); self.assertGreater(e1['q_amp'], 0.0)
        # ... the parameter-free model gets almost all of it ...
        self.assertLess(e1['pi'], 0.05); self.assertLess(e1['q'], 0.05)
        # ... and the residual (relative, squared) falls like W^4 when the filter is halved
        self.assertLess(e2['pi'], e1['pi'] / 8.0 + 1e-6)
        self.assertLess(e2['q'], e1['q'] / 8.0 + 1e-6)


class TestConditioning(unittest.TestCase):
    def test_operators_of_very_different_size(self):
        rng = np.random.default_rng(2)
        big, tiny = rng.normal(size=(3, 6, 6, 4, 4)), 1e-9 * rng.normal(size=(3, 6, 6, 4, 4))
        E = 2.0 * big + 3e7 * tiny
        fit = OR.fit_region(E, {'alpha_dynamo': big, 'R_tilde': tiny}, ('alpha_dynamo', 'R_tilde'))
        self.assertAlmostEqual(fit['alpha_dynamo'], 2.0, places=6)
        self.assertAlmostEqual(fit['R_tilde'] / 3e7, 1.0, places=6)
        self.assertGreater(fit['R2'], 1 - 1e-10)

    def test_zero_operator_gets_zero_coefficient(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(2, 4, 4, 4, 4))
        fit = OR.fit_region(a, {'alpha_dynamo': a, 'R_tilde': np.zeros_like(a)}, ('alpha_dynamo', 'R_tilde'))
        self.assertAlmostEqual(fit['alpha_dynamo'], 1.0)
        self.assertEqual(fit['R_tilde'], 0.0)

    def test_null_threshold_valid_mask_ignores_holes(self):
        rng = np.random.default_rng(4)
        sh = (1, 12, 12, 6, 4)
        ops = {t: rng.normal(size=sh) for t in OR.TERMS}
        E = rng.normal(size=sh)
        valid = np.zeros(sh[1:4], dtype=bool); valid[3:9, 3:9, 1:5] = True
        thr_masked = OR.local_null_threshold(E, ops, (3, 3, 3), n_null=4, valid=valid)
        thr_all = OR.local_null_threshold(E, ops, (3, 3, 3), n_null=4)
        self.assertTrue(np.isfinite(thr_masked) and np.isfinite(thr_all))
        self.assertNotEqual(thr_masked, thr_all)


class TestLorentzClosure(unittest.TestCase):
    """F_closure = <f> + J~_a F~^{ab} against -C - sigma~ u~_a F~^{ab}, with analytic fields and a static observer."""

    def _errors(self, W, n=36):
        h = 1.0 / n
        ax = (np.arange(n) + 0.5) * h
        X = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), axis=-1)
        k = 2 * np.pi
        pts = np.array([0.0, np.sqrt(3.0 / 5.0), -np.sqrt(3.0 / 5.0)]) * W / 2.0
        wts = np.array([4.0 / 9.0, 5.0 / 18.0, 5.0 / 18.0])

        def fields(x):
            s = [np.sin(k * x[..., i] + p) for i, p in enumerate((0.3, 1.1, 2.0))]
            c = [np.cos(k * x[..., i] + p) for i, p in enumerate((0.3, 1.1, 2.0))]
            j = np.stack([0.5 * s[0] * c[1], s[1] * c[2], c[0] * s[2], 0.7 * s[2] * s[1]], axis=-1)
            E = 0.6 * np.stack([s[1] * c[2], s[2] * c[0], s[0] * c[1]], axis=-1)
            B = np.stack([c[0] * s[1], c[1] * s[2], c[2] * s[0]], axis=-1)
            F = np.zeros(x.shape[:-1] + (4, 4))
            F[..., 0, 1:], F[..., 1:, 0] = E, -E
            F[..., 1, 2], F[..., 2, 1] = B[..., 2], -B[..., 2]
            F[..., 2, 3], F[..., 3, 2] = B[..., 0], -B[..., 0]
            F[..., 3, 1], F[..., 1, 3] = B[..., 1], -B[..., 1]
            f = -np.einsum('ac,...c,...ab->...b', MINK, j, F)
            return j, F, f

        acc = None
        for a, wa in zip(pts, wts):
            for b, wb in zip(pts, wts):
                for c, wc in zip(pts, wts):
                    out = fields(X + np.array([a, b, c]))
                    acc = [wa * wb * wc * o for o in out] if acc is None else [x + wa * wb * wc * o for x, o in zip(acc, out)]
        jt, Ft, ft = (a_[None] for a_ in acc)
        u = np.zeros(jt.shape); u[..., 0] = 1.0
        Jt = jt.copy(); Jt[..., 0] = 0.0                              # current orthogonal to the static observer
        target = ft + np.einsum('ac,...c,...ab->...b', MINK, Jt, Ft)       # F_closure
        m = OR.lorentz_force_model(jt, Ft, u, np.zeros(jt.shape[:-1] + (3,)), (h, h, h), W ** 2 / 12.0, MINK, dt=None)
        sl = (slice(None),) + (slice(4, -4),) * 3
        w = np.ones(4)
        return {'full': 1.0 - OR.prediction_r2(target[sl], (m['cov'] + m['charge'])[sl], w),
                'charge_only': 1.0 - OR.prediction_r2(target[sl], m['charge'][sl], w),
                'amp': float(np.sqrt(np.sum(target[sl] ** 2)))}

    def test_second_order_model_matches_filtered_force(self):
        e1, e2 = self._errors(0.2), self._errors(0.1)
        self.assertLess(e1['full'], 0.05)
        self.assertGreater(e1['charge_only'], 5 * e1['full'])           # the covariance term is what carries the signal
        self.assertLess(e2['full'], e1['full'] / 8.0 + 1e-6)             # residual falls ~ W^4


if __name__ == '__main__':
    unittest.main()
