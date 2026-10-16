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


if __name__ == '__main__':
    unittest.main()
