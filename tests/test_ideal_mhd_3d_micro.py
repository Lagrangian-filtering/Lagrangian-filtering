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
        """Quantitative ground-truth check for ChargeCurrent.

        Finding C2 of the final whole-branch review: the previous version of
        this test only asserted shape/finiteness/non-zero-ness, which is why
        Finding C1 (ChargeCurrent summing over the wrong index of F, giving
        exactly the negative of the physical (rho, J)) survived six prior
        review gates. This version builds a synthetic v(x,y,z), B(x,y,z)
        field with an analytically known div(E)/curl(B), computes the
        expected (rho, Jx, Jy, Jz) independently via np.gradient directly on
        the primitive field arrays (NOT by calling into IdealMHD_3D), and
        checks ChargeCurrent against that ground truth at an interior point.
        """
        m = self.m
        n_pts = 5
        d = 0.1
        shape = (1, n_pts, n_pts, n_pts)
        m.domain_vars['nx'] = m.domain_vars['ny'] = m.domain_vars['nz'] = n_pts
        m.domain_vars['dx'] = m.domain_vars['dy'] = m.domain_vars['dz'] = d
        m.domain_vars['dt'] = 1.0
        m.domain_vars['t'] = np.array([0.0])

        x = np.arange(n_pts) * d
        y = np.arange(n_pts) * d
        z = np.arange(n_pts) * d
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # v = (0.2*y, 0, 0), B = (0.1*z, 0, 0.3+0.5*x) -> E = -v x B.
        # Both v and B vary in space (so curl B and div E are non-zero), and
        # every partial derivative relevant below is of a component that is
        # affine in the differentiation variable, so central differencing is
        # exact at interior points (no finite-difference truncation error to
        # worry about here).
        vx, vy, vz = 0.2 * Y, np.zeros_like(X), np.zeros_like(X)
        Bx, By, Bz = 0.1 * Z, np.zeros_like(X), 0.3 + 0.5 * X

        m.prim_vars['n'] = np.full(shape, 1.2)
        m.prim_vars['p'] = np.full(shape, 0.3)
        m.prim_vars['vx'] = vx[None, ...]
        m.prim_vars['vy'] = vy[None, ...]
        m.prim_vars['vz'] = vz[None, ...]
        m.prim_vars['Bx'] = Bx[None, ...]
        m.prim_vars['By'] = By[None, ...]
        m.prim_vars['Bz'] = Bz[None, ...]
        m.aux_vars['W'] = np.ones(shape)
        m.aux_vars['e'] = np.ones(shape)
        m.aux_vars['h'] = np.ones(shape)

        m.setup_structures()
        j = m.structures['ChargeCurrent']
        self.assertEqual(j.shape, shape + (4,))
        self.assertTrue(np.all(np.isfinite(j)))

        # Independent ground truth: rho = div(E), J = curl(B) (dE/dt = 0
        # since the field is static -- single time snapshot), computed via
        # np.gradient directly on v/B, with no call into IdealMHD_3D.
        v = np.stack([vx, vy, vz], axis=-1)
        B = np.stack([Bx, By, Bz], axis=-1)
        E = -np.cross(v, B)

        dEx_dx, dEx_dy, dEx_dz = np.gradient(E[..., 0], x, y, z)
        dEy_dx, dEy_dy, dEy_dz = np.gradient(E[..., 1], x, y, z)
        dEz_dx, dEz_dy, dEz_dz = np.gradient(E[..., 2], x, y, z)

        dBx_dx, dBx_dy, dBx_dz = np.gradient(Bx, x, y, z)
        dBy_dx, dBy_dy, dBy_dz = np.gradient(By, x, y, z)
        dBz_dx, dBz_dy, dBz_dz = np.gradient(Bz, x, y, z)

        i0, i1, i2 = 2, 2, 2  # interior grid point, away from all boundaries
        rho_expected = dEx_dx[i0, i1, i2] + dEy_dy[i0, i1, i2] + dEz_dz[i0, i1, i2]
        Jx_expected = dBz_dy[i0, i1, i2] - dBy_dz[i0, i1, i2]
        Jy_expected = dBx_dz[i0, i1, i2] - dBz_dx[i0, i1, i2]
        Jz_expected = dBy_dx[i0, i1, i2] - dBx_dy[i0, i1, i2]

        j_at_point = j[0, i0, i1, i2]
        np.testing.assert_allclose(
            j_at_point, [rho_expected, Jx_expected, Jy_expected, Jz_expected], atol=0.05)


if __name__ == '__main__':
    unittest.main()
