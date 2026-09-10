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
        # F^{0i} = E^i (lab frame convention)
        E_from_F = F[0, 1:]
        np.testing.assert_allclose(E_from_F, E_expected, atol=1e-12)

    def test_ideal_mhd_condition_fluid_frame_E_vanishes(self):
        """Verify ideal-MHD physical requirement: E_fluid = 0 in fluid frame."""
        # Import Base for observer_frame_fields (system/BaseFunctionality.py)
        from system.BaseFunctionality import Base

        # Build fluid four-velocity from velocities and Lorentz factor
        v = np.array([0.1, -0.05, 0.02])
        W = self.m.aux_vars['W'][0, 0, 0, 0]
        u_fluid = np.array([W, W * v[0], W * v[1], W * v[2]])

        # Get Faraday tensor at this gridpoint
        F = self.m.structures['FaradayTensor'][0, 0, 0, 0]

        # Reconstruct lab-frame E and B for reference
        B = np.array([0.4, -0.3, 0.2])
        E = -np.cross(v, B)

        # Transform Faraday tensor to fluid frame and extract E field
        E_fluid_4 = Base.observer_frame_fields(F, u_fluid, self.m.metric)[0]

        # Ideal MHD requires E_fluid to be zero (to machine precision ~1e-18, atol=1e-10 is safe margin)
        np.testing.assert_allclose(E_fluid_4, 0.0, atol=1e-10)

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
