import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'master_files'))
import BoostTest as BT

ETA = np.diag(BT.ETA)


class TestBoost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flow = BT.Flow(seed=0)

    def test_boost_preserves_the_metric_and_composes_collinear_boosts(self):
        for beta in ([0, 0, 0.6], [0.3, -0.4, 0.5], [0.99, 0, 0]):
            L = BT.lorentz_boost(beta)
            self.assertLess(np.abs(L @ ETA @ L.T - ETA).max(), 1e-12)
        L = BT.lorentz_boost([0.5, 0, 0]) @ BT.lorentz_boost([0.5, 0, 0])
        self.assertAlmostEqual(L[1, 0] / L[0, 0], -0.8, places=12)                # (0.5 + 0.5) / (1 + 0.25)

    def test_fields_are_ideal_in_every_frame_and_the_invariants_agree(self):
        X = np.random.default_rng(1).uniform(-1, 1, size=(20, 4))
        u, F = self.flow.fields(X)
        self.assertLess(np.abs(np.einsum('pmn,pn->pm', F, u * BT.ETA)).max(), 1e-12)
        L = BT.lorentz_boost([0.2, 0.1, 0.9])
        up, Fp = BT.boost_fields(self.flow, X @ L.T, L)
        self.assertLess(np.abs(np.einsum('pmn,pn->pm', Fp, up * BT.ETA)).max(), 1e-11)
        inv = lambda F: np.einsum('pmn,pmn,m,n->p', F, F, BT.ETA, BT.ETA)
        self.assertLess(np.abs(inv(F) - inv(Fp)).max(), 1e-10)

    def test_filtered_electric_four_vector_transforms_as_a_vector(self):
        L = BT.lorentz_boost([0.0, 0.7, 0.0])
        X0 = np.array([[0.1, 0.2, -0.3, 0.4]])
        axes = BT.window_axes(np.eye(4), (0.1, 0.1, 0.1))
        ub, Fb = BT.box_mean(self.flow, X0, axes, 4)
        q0 = BT.observer_quantities({'u': ub, 'F': Fb})
        q1 = BT.observer_quantities({'u': ub @ L.T, 'F': np.einsum('ma,pab,nb->pmn', L, Fb, L)})
        self.assertLess(np.abs(q0['e'] @ L.T - q1['e']).max(), 1e-13)
        self.assertLess(np.abs(q0['u'] @ L.T - q1['u']).max(), 1e-13)

    def test_window_covariance_is_tilted_in_a_boosted_frame_for_the_lab_window(self):
        L = BT.lorentz_boost([0, 0, 0.8])
        g = 1 / 0.6
        ax = BT.window_axes(np.eye(4), (0.1, 0.1, 0.1))
        S = L @ BT.window_covariance(ax, 4) @ L.T
        s2 = 15 / 12 * 0.01
        self.assertAlmostEqual(S[3, 3] / s2, g * g, places=10)
        self.assertAlmostEqual(S[0, 0] / s2, (0.8 * g) ** 2, places=10)
        self.assertAlmostEqual(S[0, 3] / s2, -0.8 * g * g, places=10)
        ax_b = BT.window_axes(L, (0.1, 0.1, 0.1 / g))                             # a window rebuilt in the frame: purely spatial there
        S_b = L @ BT.window_covariance(ax_b, 4) @ L.T
        self.assertAlmostEqual(S_b[0, 0], 0.0, places=14)
        self.assertAlmostEqual(S_b[0, 3], 0.0, places=14)


class TestFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flow = BT.Flow(seed=0)
        cls.h = 0.05

    def run_case(self, beta, axis, window):
        b = np.zeros(3)
        b[axis] = beta
        return BT.run_case(self.flow, b, window, h=self.h)

    def test_the_covariant_library_recovers_minus_one_without_a_boost(self):
        r = self.run_case(0.0, 0, 'lab')
        self.assertAlmostEqual(r['cov'][0], -1.0, delta=0.02)
        self.assertGreater(r['cov'][1], 0.999)
        self.assertAlmostEqual(r['cov'][0], r['sp'][0], delta=1e-3)                  # the two libraries differ only by gamma factors at rest

    def test_the_covariant_coefficient_is_unchanged_by_the_boost_for_both_windows(self):
        for beta, axis in ((0.5, 2), (0.9, 0), (0.99, 2)):
            for window in ('lab', 'frame'):
                r = self.run_case(beta, axis, window)
                self.assertAlmostEqual(r['cov'][0], -1.0, delta=0.02, msg=(beta, axis, window))
                self.assertGreater(r['cov'][1], 0.999, msg=(beta, axis, window))

    def test_the_spatial_library_fails_in_a_boosted_frame(self):
        r = self.run_case(0.99, 2, 'lab')
        self.assertGreater(abs(r['sp'][0] + 1.0), 0.5)
        self.assertLess(r['sp'][1], 0.85)


if __name__ == '__main__':
    unittest.main()
