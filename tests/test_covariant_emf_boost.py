"""
The pipeline's covariant model of E_tilde (OhmsRegression.covariant_emf, window = the rest space of the observer, Sigma = sigma^2 h(u~)) is a Lorentz vector: the model evaluated on
the fields of a boosted frame, with 4-gradients taken on that frame's own grid, equals the boost of the model evaluated in the original frame. Analytic fields (BoostTest.Flow), numpy only.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'master_files'))
import BoostTest as BT
import OhmsRegression as OR

G = np.diag([-1.0, 1.0, 1.0, 1.0])
EPS = OR.levi_civita_up(G)
S = OR.faraday_ideal_sign(G)


def comoving_b(u, F):
    ul = np.einsum('ab,...b->...a', G, u)
    Fl = np.einsum('ac,bd,...cd->...ab', G, G, F)
    return -np.einsum('...ab,...b->...a', 0.5 * np.einsum('abcd,...cd->...ab', EPS, Fl), ul)


def model_at(flow, X0, L, h, sigma2):
    """covariant_emf (vB part, zeta = 0) at the point X0 of the frame L (X' = L X), from a 3^4 patch of that frame's grid with spacing h."""
    ax = np.arange(-1, 2) * h
    Xp = X0 + np.stack(np.meshgrid(ax, ax, ax, ax, indexing='ij'), axis=-1)
    u, F = BT.boost_fields(flow, Xp, L)
    b = comoving_b(u, F)
    out = OR.covariant_emf(u, b, np.ones(u.shape[:-1]), (h, h, h), sigma2, G, dt=h, zeta=0.0, s=S)
    return out[1, 1, 1, 1]


class TestCovariantEmfBoost(unittest.TestCase):
    def test_vector_under_boost(self):
        flow = BT.Flow(seed=2)
        X0 = np.array([0.3, 0.1, -0.2, 0.4])
        h, sigma2 = 0.005, 0.01
        e0 = model_at(flow, X0, np.eye(4), h, sigma2)
        self.assertGreater(np.linalg.norm(e0), 1e-4)                         # the test is not about zeros
        for beta in ((0.0, 0.0, 0.5), (0.9, 0.0, 0.0), (0.0, 0.6, 0.6)):
            L = BT.lorentz_boost(beta)
            ep = model_at(flow, L @ X0, L, h, sigma2)
            self.assertLess(np.linalg.norm(ep - L @ e0), 2e-3 * np.linalg.norm(e0), beta)


if __name__ == '__main__':
    unittest.main()
