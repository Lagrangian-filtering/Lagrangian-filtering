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


if __name__ == '__main__':
    unittest.main()
