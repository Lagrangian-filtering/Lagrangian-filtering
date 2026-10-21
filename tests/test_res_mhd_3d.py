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


class TestFilterMicroVarsWiring(unittest.TestCase):
    """filter_micro_vars_parallel must request every EM structure, including SET_EM, and store it on the meso grid."""

    def test_set_em_is_filtered_and_stored(self):
        shapes = {'BC': (4,), 'SET': (4, 4), 'p': (), 'FaradayTensor': (4, 4), 'ChargeCurrent': (4,),
                  'LorentzForceDensity': (4,), 'SET_EM': (4, 4), 'BSq': ()}
        requested = []

        class StubFilter(object):
            def filter_var_parallel(self, points_observers, var, n_cpus):
                requested.append(var)
                n = len(points_observers)
                # value of point i is i + a tag per variable, so a mix-up between variables would be visible
                tag = 100.0 * len(requested)
                return list(range(n)), [np.full(shapes[var], tag + i) for i in range(n)]

        m = resMHD_3D.__new__(resMHD_3D)
        Nt, Nx, Ny, Nz = 1, 2, 2, 1
        m.domain_vars = {'T': [0.0], 'X': [0.0, 1.0], 'Y': [0.0, 1.0], 'Z': [0.0]}
        m.filter = StubFilter()
        m.metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        m.filter_vars = {'U_success': np.ones((Nt, Nx, Ny, Nz), dtype=bool),
                         'U': np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (Nt, Nx, Ny, Nz, 1))}
        m.meso_structures = {k: np.zeros((Nt, Nx, Ny, Nz) + shapes[k]) for k in shapes if k not in ('p', 'BSq')}
        m.meso_vars = {'p_filt': np.zeros((Nt, Nx, Ny, Nz)), 'BSq_filt': np.zeros((Nt, Nx, Ny, Nz)), 'B2_sub': np.zeros((Nt, Nx, Ny, Nz))}
        m.filter_micro_vars_parallel(1)
        self.assertIn('SET_EM', requested)
        k = requested.index('SET_EM') + 1
        # point index 3 = grid cell (0, 1, 1, 0); its SET_EM equals the stub's value for that variable and point
        np.testing.assert_allclose(m.meso_structures['SET_EM'][0, 1, 1, 0], np.full((4, 4), 100.0 * k + 3))
        self.assertEqual(m.meso_structures['SET_EM'].shape[-2:], (4, 4))


if __name__ == '__main__':
    unittest.main()
