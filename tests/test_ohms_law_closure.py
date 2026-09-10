import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'master_files'))
from system.BaseFunctionality import Base
from MesoModels import resMHD_3D

METRIC = np.diag([-1., 1., 1., 1.])


def hall_operand(J, u, B, metric):
    levi_upper = Base.raise_LeviCivita4D(metric)
    return np.einsum('abcd,b,c,d->a', levi_upper, J, u, B)


class TestOhmsLawFitGridpoint(unittest.TestCase):
    def setUp(self):
        self.u = np.array([1., 0., 0., 0.])
        self.J = np.array([0., 0.3, -0.1, 0.05])
        self.B = np.array([0., 0.2, 0.4, -0.1])
        self.R_true, self.alpha_true, self.gamma_true = 0.7, -0.4, 0.2
        hall = hall_operand(self.J, self.u, self.B, METRIC)
        self.E = self.R_true * self.J + self.alpha_true * self.B + self.gamma_true * hall

    def test_all_terms_on_recovers_known_coefficients(self):
        R, alpha, gamma, res = resMHD_3D.ohms_law_fit_gridpoint(
            self.E, self.J, self.B, self.u, METRIC,
            use_resistive=True, use_dynamo=True, use_hall=True)
        self.assertAlmostEqual(R, self.R_true, places=8)
        self.assertAlmostEqual(alpha, self.alpha_true, places=8)
        self.assertAlmostEqual(gamma, self.gamma_true, places=8)
        np.testing.assert_allclose(res, np.zeros(4), atol=1e-8)

    def test_disabled_terms_are_nan_not_zero(self):
        R, alpha, gamma, res = resMHD_3D.ohms_law_fit_gridpoint(
            self.E, self.J, self.B, self.u, METRIC,
            use_resistive=True, use_dynamo=False, use_hall=False)
        self.assertFalse(np.isnan(R))
        self.assertTrue(np.isnan(alpha))
        self.assertTrue(np.isnan(gamma))

    def test_missing_term_shows_up_as_larger_residual(self):
        # Fit with dynamo+hall off: since the synthetic E genuinely contains
        # an alpha*B contribution, the fitted resistive-only model must leave
        # a nonzero residual (the missing physics can't vanish by magic).
        _, _, _, res_partial = resMHD_3D.ohms_law_fit_gridpoint(
            self.E, self.J, self.B, self.u, METRIC,
            use_resistive=True, use_dynamo=False, use_hall=False)
        _, _, _, res_full = resMHD_3D.ohms_law_fit_gridpoint(
            self.E, self.J, self.B, self.u, METRIC,
            use_resistive=True, use_dynamo=True, use_hall=True)
        self.assertGreater(np.linalg.norm(res_partial), np.linalg.norm(res_full))

    def test_all_terms_off_returns_full_residual(self):
        R, alpha, gamma, res = resMHD_3D.ohms_law_fit_gridpoint(
            self.E, self.J, self.B, self.u, METRIC,
            use_resistive=False, use_dynamo=False, use_hall=False)
        self.assertTrue(np.isnan(R) and np.isnan(alpha) and np.isnan(gamma))
        np.testing.assert_allclose(res, self.E, atol=1e-12)


class TestOhmsLawFitGridpointSingularSystems(unittest.TestCase):
    """Finding I3 (final whole-branch review): np.linalg.solve raises
    LinAlgError on a singular Gram matrix, which happens for ordinary
    (non-pathological) MHD field configurations -- e.g. J parallel to B
    (the Hall operator epsilon^{abcd}J_b u_c B_d vanishes identically,
    which occurs in force-free/low-beta MHD regions) or B=0/J=0 at a single
    gridpoint. Since fit_ohms_law_closure loops the whole grid with no
    exception handling, one such point used to abort the entire fit. The
    fix (np.linalg.lstsq instead of np.linalg.solve) must degrade
    gracefully instead of raising."""

    def setUp(self):
        self.u = np.array([1., 0., 0., 0.])

    def test_J_parallel_to_B_does_not_raise_and_returns_finite_values(self):
        # J proportional to B -> Hall operand epsilon^{abcd}J_b u_c B_d = 0
        # identically (antisymmetric contraction with two parallel vectors),
        # and since R_tilde's operand (J) and alpha_dynamo's operand (B) are
        # then also parallel, the 3x3 Gram matrix G is rank-1 (singular).
        B = np.array([0., 0.2, 0.4, -0.1])
        J = 2.5 * B
        E = np.array([0.1, -0.2, 0.05, 0.3])  # arbitrary target field

        try:
            R, alpha, gamma, res = resMHD_3D.ohms_law_fit_gridpoint(
                E, J, B, self.u, METRIC,
                use_resistive=True, use_dynamo=True, use_hall=True)
        except np.linalg.LinAlgError:
            self.fail("ohms_law_fit_gridpoint raised LinAlgError on a "
                      "singular system (J parallel to B) -- Finding I3 regression")

        self.assertTrue(np.isfinite(R))
        self.assertTrue(np.isfinite(alpha))
        self.assertTrue(np.isfinite(gamma))
        self.assertTrue(np.all(np.isfinite(res)))
        # The fitted model (R*J + alpha*B + gamma*Hall) must still reproduce
        # a reasonable fraction of E along the one well-posed direction
        # (span{J,B}, since Hall=0 contributes nothing) -- i.e. it should be
        # a genuine least-squares fit, not garbage.
        hall = hall_operand(J, self.u, B, METRIC)
        reconstructed = R * J + alpha * B + gamma * hall
        np.testing.assert_allclose(reconstructed + res, E, atol=1e-8)

    def test_B_zero_does_not_raise_and_fits_resistive_term(self):
        # B=0 -> both alpha_dynamo's operand (B) and the Hall operand are
        # zero vectors, so two of the three columns of G are identically
        # zero (singular). The resistive direction (J) is still well-posed
        # and should be recovered correctly via lstsq.
        B = np.zeros(4)
        J = np.array([0., 0.3, -0.1, 0.05])
        R_true = 0.7
        E = R_true * J

        try:
            R, alpha, gamma, res = resMHD_3D.ohms_law_fit_gridpoint(
                E, J, B, self.u, METRIC,
                use_resistive=True, use_dynamo=True, use_hall=True)
        except np.linalg.LinAlgError:
            self.fail("ohms_law_fit_gridpoint raised LinAlgError on a "
                      "singular system (B=0) -- Finding I3 regression")

        self.assertAlmostEqual(R, R_true, places=6)
        self.assertTrue(np.isfinite(alpha))
        self.assertTrue(np.isfinite(gamma))
        np.testing.assert_allclose(res, np.zeros(4), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
