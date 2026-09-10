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
