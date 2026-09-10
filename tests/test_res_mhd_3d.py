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
