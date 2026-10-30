"""Lorentz transformation of primitive frames (master_files/FrameBoost.py): the kinematics against analytic fields, the interpolation order, the repair of failed cells and the file layout."""
import os
import shutil
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'master_files'))
import FrameBoost as FB
from FileReaders import METHOD_HDF5
from MicroModels import IdealMHD_3D

TWO_PI = 2.0 * np.pi


def fields(t, x, y, z):
    """Smooth periodic (period 1 in x, y, z) analytic primitives of the rest frame, |v| < 0.5."""
    s = np.sin
    c = np.cos
    rho = 1.0 + 0.2 * s(TWO_PI * (x - 0.3 * t)) * c(TWO_PI * (z + 0.1 * t)) + 0.1 * c(TWO_PI * (y + z))
    p = 0.5 + 0.1 * s(TWO_PI * (x + y)) * c(TWO_PI * (z - 0.2 * t))
    v = np.stack([0.3 * s(TWO_PI * (y + 0.2 * t)) * c(TWO_PI * z), 0.25 * c(TWO_PI * (x - z + 0.1 * t)), 0.3 * s(TWO_PI * (x + y - 0.15 * t))], axis=-1)
    B = 0.3 * np.stack([s(TWO_PI * (y + z)), c(TWO_PI * (z + 0.3 * t)), s(TWO_PI * (x + 0.1 * t))], axis=-1)
    return rho, p, v, B


def write_rest_frames(directory, n, h, t_end):
    grid = np.arange(n) / n
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing='ij')
    times = np.arange(0.0, t_end + 1e-9, h)
    for k, t in enumerate(times):
        rho, p, v, B = fields(t, X, Y, Z)
        with h5py.File(os.path.join(directory, 'frame.checkpoint.%d.hdf5' % k), 'w') as f:
            d = f.create_group('Domain')
            for name, val in (('xmin', 0.0), ('ymin', 0.0), ('zmin', 0.0), ('xmax', 1.0), ('ymax', 1.0), ('zmax', 1.0), ('dx', 1.0 / n), ('dy', 1.0 / n), ('dz', 1.0 / n)):
                d.attrs[name] = np.array([val])
            for name in ('nx', 'ny', 'nz'):
                d.attrs[name] = np.array([n])
            d.attrs['endTime'] = np.array([t])
            g = f.create_group('Primitive')
            for name, a in zip(FB.NAMES, (rho, p, v[..., 0], v[..., 1], v[..., 2], B[..., 0], B[..., 1], B[..., 2])):
                g.create_dataset(name, data=a)
    return times


def exact_boosted(n, xs, ys, beta, T, nzp):
    """The state in S' on the grid of boosted_frame, from the analytic fields at the corresponding events of S."""
    g = 1.0 / np.sqrt(1.0 - beta ** 2)
    gx = (np.arange(n) / n)[xs]
    gy = (np.arange(n) / n)[ys]
    zp = (np.arange(nzp) - nzp // 2) / n
    X, Y, ZP = np.meshgrid(gx, gy, zp, indexing='ij')
    rho, p, v, B = fields(g * (T + beta * ZP), X, Y, g * (ZP + beta * T))
    rho, p, v, B = FB.boost_state(rho, p, v, B, beta)
    return np.concatenate([rho[None], p[None], np.moveaxis(v, -1, 0), np.moveaxis(B, -1, 0)], axis=0)


class TestBoostState(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.v = rng.uniform(-0.4, 0.4, (50, 3))
        self.B = rng.normal(size=(50, 3))
        self.rho = np.ones(50)
        self.p = np.ones(50)

    def test_ideal_mhd_electric_field_is_kept(self):
        rho, p, v, B = FB.boost_state(self.rho, self.p, self.v, self.B, 0.7)
        # in S' the electric field of the transformed state must again be -v' x B', i.e. E'.B' = 0 and the Lorentz invariant B^2 - E^2 is conserved
        E = -np.cross(self.v, self.B)
        g = 1 / np.sqrt(1 - 0.7 ** 2)
        u = np.array([0, 0, 0.7])
        Ep = g * (E + np.cross(u, self.B)) - g * g / (g + 1) * u * np.sum(u * E, axis=-1, keepdims=True)
        np.testing.assert_allclose(Ep, -np.cross(v, B), atol=1e-12)
        np.testing.assert_allclose(np.sum(B ** 2, -1) - np.sum(Ep ** 2, -1), np.sum(self.B ** 2, -1) - np.sum(E ** 2, -1), atol=1e-12)

    def test_round_trip(self):
        r = FB.boost_state(self.rho, self.p, self.v, self.B, 0.6)
        back = FB.boost_state(r[0], r[1], r[2], r[3], -0.6)
        np.testing.assert_allclose(back[2], self.v, atol=1e-13)
        np.testing.assert_allclose(back[3], self.B, atol=1e-13)

    def test_velocity_addition(self):
        v = np.zeros((1, 3)); v[0, 2] = 0.5
        _, _, vp, _ = FB.boost_state(np.ones(1), np.ones(1), v, np.ones((1, 3)), 0.5)
        self.assertAlmostEqual(vp[0, 2], 0.0, places=14)                    # a fluid moving with the frame is at rest in it


class TestBoostedFrame(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def error(self, n, h, beta=0.6, T=0.5):
        d = os.path.join(self.tmp, 'n%d' % n)
        os.makedirs(d)
        write_rest_frames(d, n, h, 2.5)
        xs, ys = slice(n // 4, n // 2), slice(n // 8, 3 * n // 8)
        rest = FB.RestFrames(sorted(os.path.join(d, f) for f in os.listdir(d)), xs, ys)
        nzp = n // 2
        got = FB.boosted_frame(rest, beta, T, nzp)
        want = exact_boosted(n, xs, ys, beta, T, nzp)
        return float(np.max(np.abs(got - want))), rest

    def test_matches_analytic_fields_and_converges_at_fourth_order(self):
        e1, _ = self.error(16, 0.2)
        e2, _ = self.error(32, 0.1)
        self.assertLess(e2, 2e-3)
        self.assertGreater(e1 / e2, 9.0)                                      # cubic interpolation: a factor 16 for 2x finer in time and space

    def test_no_boost_reproduces_the_frame_at_a_node(self):
        d = os.path.join(self.tmp, 'id')
        os.makedirs(d)
        write_rest_frames(d, 16, 0.25, 2.0)
        xs, ys = slice(0, 16), slice(0, 16)
        rest = FB.RestFrames(sorted(os.path.join(d, f) for f in os.listdir(d)), xs, ys)
        got = FB.boosted_frame(rest, 0.0, 1.0, 16)                              # z' = (m - 8)/16: a grid point of the rest frame (z mod 1)
        grid = np.arange(16) / 16
        X, Y, ZP = np.meshgrid(grid, grid, (np.arange(16) - 8) / 16, indexing='ij')
        rho, p, v, B = fields(1.0, X, Y, ZP)
        np.testing.assert_allclose(got[0], rho, atol=1e-12)
        np.testing.assert_allclose(got[4], v[..., 2], atol=1e-12)


class TestRepair(unittest.TestCase):
    def test_a_failed_cell_is_replaced_by_the_mean_of_its_neighbours(self):
        a = np.ones((8, 5, 5, 5))
        a[2:5] = 0.1
        a[0] = np.arange(125.0).reshape(5, 5, 5) * 0.0 + 2.0
        a[4, 2, 2, 2] = 0.99                                                   # |v| ~ 1: superluminal with the other components
        a[2, 2, 2, 2] = 0.5
        a[3, 2, 2, 2] = 0.5
        n = FB.repair_bad_cells(a)
        self.assertEqual(n, 1)
        np.testing.assert_allclose(a[:, 2, 2, 2], np.concatenate([[2.0, 1.0], [0.1] * 3, [1.0] * 3]))

    def test_nan_cell_counted(self):
        a = np.ones((8, 4, 4, 4)); a[2:5] = 0.1
        a[5, 1, 1, 1] = np.nan
        self.assertEqual(FB.repair_bad_cells(a), 1)
        self.assertTrue(np.all(np.isfinite(a)))


class TestFrameFile(unittest.TestCase):
    def test_written_frame_is_read_by_the_pipeline_reader(self):
        tmp = tempfile.mkdtemp()
        try:
            src = os.path.join(tmp, 'src')
            os.makedirs(src)
            write_rest_frames(src, 16, 0.25, 2.0)
            xs, ys = slice(2, 10), slice(3, 9)
            rest = FB.RestFrames(sorted(os.path.join(src, f) for f in os.listdir(src)), xs, ys)
            out = os.path.join(tmp, 'out')
            os.makedirs(out)
            for k, T in enumerate((0.9, 1.0, 1.1)):
                FB.write_frame(os.path.join(out, 'frame.checkpoint.%.6f.hdf5' % T), FB.boosted_frame(rest, 0.5, T, 8), T, rest.attrs, xs, ys, 8, rest.dz)
            m = IdealMHD_3D()
            METHOD_HDF5(out + os.sep).read_in_data3D(m, adiabatic_index=4.0 / 3.0)
            dv = m.domain_vars
            self.assertEqual((dv['nx'], dv['ny'], dv['nz']), (8, 6, 8))
            np.testing.assert_allclose(dv['t'], [0.9, 1.0, 1.1])
            self.assertAlmostEqual(dv['xmin'], 2.0 / 16)
            self.assertAlmostEqual(dv['ymin'], 3.0 / 16)
            self.assertAlmostEqual(dv['zmin'], -4.0 / 16)
            self.assertAlmostEqual(dv['dz'], 1.0 / 16)
        finally:
            shutil.rmtree(tmp)


class TestScript(unittest.TestCase):
    def test_cluster_of_frames(self):
        import runpy
        tmp = tempfile.mkdtemp()
        saved = sys.argv
        try:
            src = os.path.join(tmp, 'src')
            os.makedirs(src)
            write_rest_frames(src, 16, 0.1, 2.5)
            out = os.path.join(tmp, 'out')
            sys.argv = ['boost_frames.py', src, '--beta', '0.6', '--center-rest', '1.25', '--n-side', '2', '--spacing', '0.02', '--crop-x', '4', '12', '--crop-y', '2', '10',
                        '--nz', '8', '--outdir', out]
            runpy.run_path(os.path.join(ROOT, 'filter_scripts', 'boost_frames.py'), run_name='__main__')
            files = sorted(os.listdir(out))
            self.assertEqual(len(files), 5)
            with h5py.File(os.path.join(out, files[2]), 'r') as f:
                self.assertAlmostEqual(float(f['Domain'].attrs['endTime'][0]), 1.25 * 0.8, places=9)       # center-rest / gamma, gamma = 1.25
                self.assertEqual(f['Primitive/rho'].shape, (8, 8, 8))
            sys.argv = ['boost_frames.py', src, '--beta', '0.6', '--times', '0.1', '--crop-x', '4', '12', '--crop-y', '2', '10', '--nz', '8', '--outdir', out + '2']
            with self.assertRaises(SystemExit):                                                              # the rest frames do not reach back to the rest time needed
                runpy.run_path(os.path.join(ROOT, 'filter_scripts', 'boost_frames.py'), run_name='__main__')
        finally:
            sys.argv = saved
            shutil.rmtree(tmp)


if __name__ == '__main__':
    unittest.main()
