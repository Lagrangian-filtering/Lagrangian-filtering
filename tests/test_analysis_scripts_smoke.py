"""
Smoke test of the Ohm's-law analysis scripts on small synthetic meso pickles.

The scripts only need a few attributes of a pickled resMHD_3D, so MesoModels (heavy imports) is
replaced by a stub class for the duration of the test. This checks that each script runs to completion
and writes its outputs, not that the numbers are physically meaningful.
"""
import os
import pickle
import runpy
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace

import numpy as np

try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SCRIPTS = os.path.join(ROOT, 'filter_scripts')
sys.path.insert(0, os.path.join(ROOT, 'master_files'))


def fake_model(stub, x0=0.4, seed=0, nt=3, shape=(14, 14, 8), spacing=0.01):
    rng = np.random.default_rng(seed)
    full = (nt,) + shape
    v = rng.uniform(-0.3, 0.3, full + (3,))
    gam = 1.0 / np.sqrt(1.0 - np.sum(v ** 2, axis=-1))
    u = np.zeros(full + (4,)); u[..., 0] = gam; u[..., 1:] = gam[..., None] * v
    F = rng.normal(size=full + (4, 4)); F = F - np.swapaxes(F, -1, -2)
    BC = np.abs(rng.normal(size=full + (4,))) + 1.0
    m = stub.FakeMeso()
    m.metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    m.filter = SimpleNamespace(filter_width=4 * spacing / 4.0)
    m.meso_vars = {'E_tilde': 0.01 * rng.normal(size=full + (4,)), 'J_tilde': rng.normal(size=full + (4,)),
                   'B_tilde': rng.normal(size=full + (4,)), 'u_tilde': u,
                   'n_tilde': 1.0 + 0.1 * rng.normal(size=full), 'B2_sub': np.abs(rng.normal(size=full)) * 1e-3}
    m.meso_structures = {'FaradayTensor': F, 'BC': BC}
    m.domain_vars = {'X': list(x0 + spacing * np.arange(shape[0])), 'Y': list(0.5 + spacing * np.arange(shape[1])),
                     'Z': list(0.2 + spacing * np.arange(shape[2])), 'Dx': spacing, 'Dy': spacing, 'Dz': spacing,
                     'Dt': spacing}
    return m


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class TestAnalysisScriptsSmoke(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in ('MesoModels',)}
        self.stub = types.ModuleType('MesoModels')

        class FakeMeso(object):
            pass
        FakeMeso.__module__ = 'MesoModels'
        FakeMeso.__qualname__ = 'FakeMeso'      # so pickle can find it as MesoModels.FakeMeso
        self.stub.FakeMeso = FakeMeso
        sys.modules['MesoModels'] = self.stub
        self.tmp = tempfile.mkdtemp()
        self._argv = sys.argv

    def tearDown(self):
        sys.argv = self._argv
        for k, mod in self._saved.items():
            if mod is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = mod

    def write_pickle(self, name, **kw):
        path = os.path.join(self.tmp, name)
        with open(path, 'wb') as fh:
            pickle.dump(fake_model(self.stub, **kw), fh)
        return path

    def run_script(self, script, *args):
        sys.argv = [script] + list(args)
        runpy.run_path(os.path.join(SCRIPTS, script), run_name='__main__')

    def test_regress_ohms_law(self):
        out = os.path.join(self.tmp, 'reg')
        self.run_script('regress_ohms_law.py', self.write_pickle('a.pickle'), '--block', '6', '6', '4',
                        '--min-points', '20', '--n-null', '4', '--outdir', out)
        self.assertTrue(os.path.exists(os.path.join(out, 'ohms_regression.npz')))
        self.assertTrue(os.path.exists(os.path.join(out, 'ohms_regression.png')))

    def test_plot_ohms_winner(self):
        out = os.path.join(self.tmp, 'win')
        self.run_script('plot_ohms_winner.py', self.write_pickle('a.pickle'), '--window', '5', '5', '3',
                        '--n-null', '3', '--outdir', out)
        self.assertTrue(os.path.exists(os.path.join(out, 'ohms_winner.png')))
        self.assertTrue(np.all(np.load(os.path.join(out, 'ohms_winner_maps.npz'))['B2_sub'] >= 0))

    def test_mosaic_of_two_abutting_tiles(self):
        out = os.path.join(self.tmp, 'mos')
        self.write_pickle('t1.pickle', x0=0.40, seed=1)
        self.write_pickle('t2.pickle', x0=0.40 + 14 * 0.01, seed=2)      # abuts the first in x
        self.run_script('mosaic_ohms_winner.py', os.path.join(self.tmp, 't*.pickle'), '--window', '5', '5', '3',
                        '--n-null', '3', '--outdir', out)
        d = np.load(os.path.join(out, 'mosaic_maps.npz'))
        self.assertEqual(d['winner'].shape[0], 28)
        self.assertTrue(bool(d['present'].all()))

    def test_regress_gradient_model(self):
        out = os.path.join(self.tmp, 'grad')
        self.run_script('regress_gradient_model.py', self.write_pickle('a.pickle'), '--block', '6', '6', '3',
                        '--min-points', '20', '--n-null', '4', '--outdir', out)
        self.assertTrue(os.path.exists(os.path.join(out, 'gradient_model.npz')))


if __name__ == '__main__':
    unittest.main()
