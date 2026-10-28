"""
Smoke test of the Ohm's-law analysis scripts on small synthetic meso pickles.

The scripts only need a few attributes of a pickled resMHD_3D, so MesoModels (heavy imports) is
replaced by a stub class for the duration of the test. This checks that each script runs to completion
and writes its outputs, not that the numbers are physically meaningful.
"""
import contextlib
import io
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
    Tm = rng.normal(size=full + (4, 4)); Tm = 0.1 * (Tm + np.swapaxes(Tm, -1, -2)) + np.eye(4) * 2.0
    m.meso_structures = {'FaradayTensor': F, 'BC': BC, 'SET': Tm}
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

    def test_summarize_regions_can_name_alpha_as_winner(self):
        root = os.path.join(self.tmp, 'regions')
        os.makedirs(os.path.join(root, 'r1'))
        subs = np.array(["R", "alpha", "gamma", "R+alpha", "R+gamma", "alpha+gamma", "R+alpha+gamma"])
        np.savez(os.path.join(root, 'r1', 'ohms_regression.npz'), label='1,2,3', subsets=subs,
                 global_R2=np.array([0.01, 0.30, 0.02, 0.31, 0.03, 0.31, 0.32]), shift_null_R2=np.full((7, 20), 0.01),
                 global_coeffs=np.full((7, 3), 1e-6), R_only_tile_R_tilde=np.array([1e-6, 2e-6, 3e-6]),
                 gamma_only_tile_gamma=np.array([1e-6, 2e-6]), region_stats=np.array([1.0, 1.0, 1.0]))
        buf = io.StringIO()
        sys.argv = ['summarize_regions.py', root]
        with contextlib.redirect_stdout(buf):
            runpy.run_path(os.path.join(SCRIPTS, 'summarize_regions.py'), run_name='__main__')
        row = [l for l in buf.getvalue().splitlines() if l.startswith('1,2,3')][0]
        self.assertIn(' alpha ', row)                       # alpha has by far the largest excess over its null

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
        with np.load(os.path.join(out, 'ohms_winner_maps.npz')) as m:
            self.assertTrue(np.all(m['B2_sub'] >= 0))

    def test_mosaic_of_two_abutting_tiles(self):
        out = os.path.join(self.tmp, 'mos')
        self.write_pickle('t1.pickle', x0=0.40, seed=1)
        self.write_pickle('t2.pickle', x0=0.40 + 14 * 0.01, seed=2)      # abuts the first in x
        self.run_script('mosaic_ohms_winner.py', os.path.join(self.tmp, 't*.pickle'), '--window', '5', '5', '3',
                        '--n-null', '3', '--outdir', out)
        with np.load(os.path.join(out, 'mosaic_maps.npz')) as _z:
            d = {k: _z[k] for k in _z.files}
        self.assertEqual(d['winner'].shape[0], 28)
        self.assertTrue(bool(d['present'].all()))

    def test_regress_gradient_model(self):
        out = os.path.join(self.tmp, 'grad')
        self.run_script('regress_gradient_model.py', self.write_pickle('a.pickle'), '--block', '6', '6', '3',
                        '--min-points', '20', '--n-null', '4', '--outdir', out)
        self.assertTrue(os.path.exists(os.path.join(out, 'gradient_model.npz')))

    def test_regress_covariance_split(self):
        out = os.path.join(self.tmp, 'split')
        self.run_script('regress_covariance_split.py', self.write_pickle('a.pickle'), '--block', '6', '6', '3',
                        '--min-points', '20', '--n-null', '3', '--outdir', out)
        with np.load(os.path.join(out, 'covariance_split.npz')) as _z:
            d = {k: _z[k] for k in _z.files}
        self.assertEqual(list(d['pieces']), ['ss', 'tt', 'ts', 'vv'])      # 3 slices: all four pieces
        self.assertEqual(d['vB_joint_coeffs'].shape, (4,))

    def test_regress_covariant_emf_two_pickles(self):
        out = os.path.join(self.tmp, 'covemf')
        a, b = self.write_pickle('a.pickle', seed=0), self.write_pickle('b.pickle', seed=1)
        self.run_script('regress_covariant_emf.py', a, b, '--labels', 'one', 'two', '--block', '6', '6', '3',
                        '--min-points', '20', '--n-null', '3', '--outdir', out)
        with np.load(os.path.join(out, 'covariant_emf.npz')) as _z:
            d = {k: _z[k] for k in _z.files}
        self.assertEqual(list(d['labels']), ['one', 'two'])
        for v in ('cov', 'cov_spatial', 'lab3'):
            self.assertEqual(d[v + '_R2'].shape, (2,))
            self.assertTrue(np.all(np.isfinite(d[v + '_coef_vB'])))

    def test_regress_covariant_emf_common_region(self):
        out = os.path.join(self.tmp, 'covemf_common')
        a, b = self.write_pickle('a.pickle', seed=0, x0=0.4), self.write_pickle('b.pickle', seed=1, x0=0.43)
        args = ('--labels', 'one', 'two', '--block', '6', '6', '3', '--min-points', '20', '--n-null', '3', '--outdir', out)
        self.run_script('regress_covariant_emf.py', a, b, *args)
        with np.load(os.path.join(out, 'covariant_emf.npz')) as _z:
            self.assertEqual(list(_z['n_points']), [12 * 12 * 6] * 2)             # the whole trimmed box without the option
        self.run_script('regress_covariant_emf.py', a, b, '--common-region', *args)
        with np.load(os.path.join(out, 'covariant_emf.npz')) as _z:
            self.assertEqual(list(_z['n_points']), [9 * 12 * 6] * 2)              # x overlap: 9 of the 12 cells, same for both
            self.assertTrue(np.all(np.isfinite(_z['cov_coef_vB'])))
        c = self.write_pickle('c.pickle', seed=2, x0=0.435)                       # half-cell offset: cells do not coincide
        with self.assertRaises(SystemExit):
            self.run_script('regress_covariant_emf.py', a, c, '--common-region', *args)
        self.run_script('regress_covariant_emf.py', a, c, '--common-region', '--allow-offset', '0.5', *args)
        with np.load(os.path.join(out, 'covariant_emf.npz')) as _z:
            self.assertEqual(list(_z['n_points']), [9 * 12 * 6] * 2)

    def test_regress_stress_model(self):
        out = os.path.join(self.tmp, 'stress')
        path = os.path.join(self.tmp, 's.pickle')
        m = fake_model(self.stub)
        m.meso_structures['BC'] = m.meso_vars['n_tilde'][..., None] * m.meso_vars['u_tilde']      # timelike, N = n u
        with open(path, 'wb') as fh:
            pickle.dump(m, fh)
        self.run_script('regress_stress_model.py', path, '--block', '6', '6', '3',
                        '--min-points', '20', '--n-null', '3', '--outdir', out)
        with np.load(os.path.join(out, 'stress_model.npz')) as _z:
            d = {k: _z[k] for k in _z.files}
        self.assertEqual(d['q_coeffs'].shape, (2,))
        self.assertEqual(d['pi_coeffs'].shape, (1,))

    def test_regress_lorentz_closure(self):
        out = os.path.join(self.tmp, 'lorentz')
        path = os.path.join(self.tmp, 'l.pickle')
        m = fake_model(self.stub)
        rng = np.random.default_rng(7)
        shape = m.meso_vars['u_tilde'].shape
        m.meso_structures['ChargeCurrent'] = rng.normal(size=shape)
        m.meso_vars['F_closure'] = 1e-3 * rng.normal(size=shape)
        with open(path, 'wb') as fh:
            pickle.dump(m, fh)
        self.run_script('regress_lorentz_closure.py', path, '--block', '6', '6', '3',
                        '--min-points', '20', '--n-null', '3', '--outdir', out)
        with np.load(os.path.join(out, 'lorentz_closure.npz')) as _z:
            d = {k: _z[k] for k in _z.files}
        self.assertEqual(d['lorentz_coeffs'].shape, (2,))

    def test_compare_force_scales(self):
        out = os.path.join(self.tmp, 'forces')
        path = os.path.join(self.tmp, 'f.pickle')
        m = fake_model(self.stub)
        rng = np.random.default_rng(8)
        shape = m.meso_vars['u_tilde'].shape
        m.meso_vars['F_closure'] = 1e-3 * rng.normal(size=shape)
        m.meso_vars['p_tilde'] = 1.0 + rng.normal(size=shape[:4])
        m.meso_structures['LorentzForceDensity'] = rng.normal(size=shape)
        with open(path, 'wb') as fh:
            pickle.dump(m, fh)
        self.run_script('compare_force_scales.py', path, '--outdir', out)
        with np.load(os.path.join(out, 'force_scales.npz')) as _z:
            d = {k: _z[k] for k in _z.files}
        self.assertEqual(d['div_T'].shape, d['F_closure'].shape)

    def test_map_closure_tail(self):
        out = os.path.join(self.tmp, 'tail')
        path = os.path.join(self.tmp, 't.pickle')
        m = fake_model(self.stub)
        rng = np.random.default_rng(9)
        shape = m.meso_vars['u_tilde'].shape
        m.meso_vars['F_closure'] = 1e-1 * rng.normal(size=shape)
        m.meso_vars['p_tilde'] = 1.0 + np.abs(rng.normal(size=shape[:4]))
        with open(path, 'wb') as fh:
            pickle.dump(m, fh)
        self.run_script('map_closure_tail.py', path, '--threshold', '0.05', '--outdir', out)
        self.assertTrue(os.path.exists(os.path.join(out, 'closure_tail.png')))
        with np.load(os.path.join(out, 'closure_tail.npz')) as _z:
            d = {k: _z[k] for k in _z.files}
        self.assertEqual(d['rank_corr'].shape, d['median_tail'].shape)

    def test_check_backscatter(self):
        out = os.path.join(self.tmp, 'back')
        path = os.path.join(self.tmp, 'b.pickle')
        m = fake_model(self.stub)
        rng = np.random.default_rng(10)
        shape = m.meso_vars['u_tilde'].shape
        m.meso_vars['F_closure'] = 1e-1 * rng.normal(size=shape)
        m.meso_structures['ChargeCurrent'] = rng.normal(size=shape)
        m.meso_structures['BC'] = m.meso_vars['n_tilde'][..., None] * m.meso_vars['u_tilde']
        with open(path, 'wb') as fh:
            pickle.dump(m, fh)
        self.run_script('check_backscatter.py', path, '--threshold', '0.05', '--outdir', out)
        with np.load(os.path.join(out, 'backscatter.npz')) as _z:
            d = {k: _z[k] for k in _z.files}
        self.assertEqual(d['values'].shape[1], 4)

    def test_regress_em_deficits(self):
        out = os.path.join(self.tmp, 'emdef')
        path = os.path.join(self.tmp, 'e.pickle')
        m = fake_model(self.stub)
        rng = np.random.default_rng(11)
        shape = m.meso_vars['u_tilde'].shape
        T = rng.normal(size=shape + (4,)); m.meso_structures['SET_EM'] = T + np.swapaxes(T, -1, -2)
        with open(path, 'wb') as fh:
            pickle.dump(m, fh)
        self.run_script('regress_em_deficits.py', path, '--block', '6', '6', '3', '--min-points', '20', '--n-null', '3', '--outdir', out)
        with np.load(os.path.join(out, 'em_deficits.npz')) as _z:
            d = {k: _z[k] for k in _z.files}
        self.assertEqual(d['sources'].shape[1], 2)

    def test_regress_em_deficits_refuses_old_pickle(self):
        path = os.path.join(self.tmp, 'old.pickle')
        with open(path, 'wb') as fh:
            pickle.dump(fake_model(self.stub), fh)
        with self.assertRaises(SystemExit):
            self.run_script('regress_em_deficits.py', path, '--outdir', os.path.join(self.tmp, 'x'))

    def test_regress_em_deficits_on_analytic_data(self):
        """Fed the exactly filtered analytic fields of TestCovariantClosure, the script's parameter-free model must score near 1."""
        from test_ohms_regression import TestCovariantClosure
        W = 0.1
        h, N, F, T, nt, ut = TestCovariantClosure()._setup(W, n=30)
        m = self.stub.FakeMeso()
        m.metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        m.filter = SimpleNamespace(filter_width=W)
        m.meso_vars = {'u_tilde': ut, 'n_tilde': nt}
        m.meso_structures = {'FaradayTensor': F, 'SET_EM': T, 'BC': N}
        ax = list(0.5 + h * np.arange(30))
        m.domain_vars = {'X': ax, 'Y': ax, 'Z': ax, 'Dx': h, 'Dy': h, 'Dz': h, 'Dt': h}
        path = os.path.join(self.tmp, 'a.pickle')
        with open(path, 'wb') as fh:
            pickle.dump(m, fh)
        buf = io.StringIO()
        sys.argv = ['regress_em_deficits.py', path, '--block', '12', '12', '12', '--min-points', '50', '--n-null', '3',
                    '--outdir', os.path.join(self.tmp, 'ana')]
        with contextlib.redirect_stdout(buf):
            runpy.run_path(os.path.join(SCRIPTS, 'regress_em_deficits.py'), run_name='__main__')
        text = buf.getvalue()
        blocks = text.split("=" * 100)
        def r2_of(title_part):
            blk = [b for b in blocks if title_part in b.splitlines()[1 if b.startswith("\n") else 0]][0]
            line = [l for l in blk.splitlines() if "global R2" in l][0]
            return float(line.split("global R2 =")[1].split()[0])
        self.assertGreater(r2_of("E~ from the data"), 0.97)
        self.assertGreater(r2_of("density term on"), 0.97)
        # the stress deficit is dominated by the quadratic term, so the density term of E~ changes its score only a little
        self.assertGreater(r2_of("density term off"), 0.95)

    def test_gradient_scan_summary(self):
        # two "widths" through the label mechanism the scan scripts use, then the summarizer
        root = os.path.join(self.tmp, 'gscan')
        for w, seed in (('2', 4), ('4', 5)):
            self.run_script('regress_gradient_model.py', self.write_pickle('p%s.pickle' % w, seed=seed),
                            '--block', '6', '6', '3', '--min-points', '20', '--n-null', '3', '--label', w,
                            '--outdir', os.path.join(root, 'w' + w))
        with np.load(os.path.join(root, 'w4', 'gradient_model.npz')) as _z:
            d = {k: _z[k] for k in _z.files}
        for key in ('fit_names', 'fit_R2', 'fit_null95', 'model_vB_coeff', 'joint_dens_coeff', 'pred_null95', 'best_scale'):
            self.assertIn(key, d)
        self.assertEqual(str(d['label']), '4')
        sys.argv = ['summarize_gradient_scan.py', root]
        runpy.run_path(os.path.join(SCRIPTS, 'summarize_gradient_scan.py'), run_name='__main__')


if __name__ == '__main__':
    unittest.main()
