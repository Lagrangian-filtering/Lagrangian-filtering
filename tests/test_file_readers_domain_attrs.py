import os
import shutil
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'master_files'))
from FileReaders import METHOD_HDF5
from MicroModels import IdealHD_3D


class TestReadInData3DDomainAttrs(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._write_fake_file(os.path.join(self.tmpdir, 'data0.hdf5'), t=0.0)
        self._write_fake_file(os.path.join(self.tmpdir, 'data1.hdf5'), t=1.0)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_fake_file(self, path, t, shape=(2, 2, 2), ramp=False):
        with h5py.File(path, 'w') as f:
            dom = f.create_group('Domain')
            # METHOD writes attributes as shape-(1,) arrays (H5LTset_attribute_*, size 1)
            dom.attrs['nx'] = np.array([shape[0]])
            dom.attrs['ny'] = np.array([shape[1]])
            dom.attrs['nz'] = np.array([shape[2]])
            for name, val in [('xmin', 0.0), ('xmax', 1.0), ('ymin', 0.0), ('ymax', 1.0),
                               ('zmin', 0.0), ('zmax', 1.0), ('dx', 0.5), ('dy', 0.5), ('dz', 0.5)]:
                dom.attrs[name] = np.array([val])
            dom.attrs['endTime'] = np.array([t])

            prim = f.create_group('Primitive')
            aux = f.create_group('Auxiliary')
            data = np.arange(np.prod(shape), dtype=float).reshape(shape) if ramp else np.ones(shape)
            for name in ('rho', 'vx', 'vy', 'vz', 'p'):
                prim.create_dataset(name, data=data)
            for name in ('W', 'h', 'e'):
                aux.create_dataset(name, data=data)

    def test_domain_metadata_read_from_attrs(self):
        reader = METHOD_HDF5(os.path.join(self.tmpdir, ''))
        micro_model = IdealHD_3D()
        reader.read_in_data3D(micro_model)

        self.assertEqual(micro_model.domain_vars['nx'], 2)
        self.assertEqual(micro_model.domain_vars['ny'], 2)
        self.assertEqual(micro_model.domain_vars['nz'], 2)
        self.assertAlmostEqual(micro_model.domain_vars['dx'], 0.5)
        self.assertAlmostEqual(micro_model.domain_vars['xmax'], 1.0)
        np.testing.assert_allclose(micro_model.domain_vars['t'], [0.0, 1.0])

    def test_crop_and_stride(self):
        tmp = tempfile.mkdtemp()
        try:
            for n, t in enumerate((0.0, 1.0)):
                self._write_fake_file(os.path.join(tmp, 'data%d.hdf5' % n), t=t, shape=(6, 4, 2), ramp=True)
            full = np.arange(48, dtype=float).reshape(6, 4, 2)

            micro_model = IdealHD_3D()
            METHOD_HDF5(os.path.join(tmp, '')).read_in_data3D(
                micro_model, crop=((1, 5), (None, None), (None, None)), stride=2)

            expected = full[1:5:2, 0:4:2, 0:2:2]
            self.assertEqual(micro_model.prim_vars['n'].shape, (2,) + expected.shape)
            np.testing.assert_array_equal(micro_model.prim_vars['n'][0], expected)
            d = micro_model.domain_vars
            self.assertEqual((d['nx'], d['ny'], d['nz']), expected.shape)
            self.assertAlmostEqual(d['dx'], 1.0)   # 0.5 * stride 2
            self.assertAlmostEqual(d['xmin'], 0.5)  # 0 + 1 cell * 0.5
            np.testing.assert_allclose(d['x'], [0.5, 1.5])
            np.testing.assert_allclose(d['y'], [0.0, 1.0])
            self.assertAlmostEqual(d['xmax'], 0.5 + 2 * 1.0)
        finally:
            shutil.rmtree(tmp)

    def test_empty_crop_raises(self):
        reader = METHOD_HDF5(os.path.join(self.tmpdir, ''))
        with self.assertRaises(ValueError):
            reader.read_in_data3D(IdealHD_3D(), crop=((1, 1), (None, None), (None, None)))

    def _minimal_dir(self):
        """Frames as METHOD's OUTPUT_MINIMAL writes them: primitive variables only (smooth subluminal fields), no Auxiliary group."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        rng = np.random.default_rng(0)
        shape = (3, 3, 3)
        for n, t in enumerate((0.0, 1.0)):
            path = os.path.join(tmp, 'frame%d.hdf5' % n)
            self._write_fake_file(path, t=t, shape=shape)
            with h5py.File(path, 'a') as f:
                del f['Auxiliary']
                prim = f['Primitive']
                for name, val in (('rho', 1.0 + rng.random(shape)), ('p', 0.5 + rng.random(shape)), ('vx', 0.3 * rng.random(shape)),
                                  ('vy', -0.3 * rng.random(shape)), ('vz', 0.2 * rng.random(shape))):
                    del prim[name]
                    prim.create_dataset(name, data=val)
        return os.path.join(tmp, '')

    def test_missing_auxiliary_is_derived_from_primitives(self):
        d, gamma = self._minimal_dir(), 4.0 / 3.0
        m = IdealHD_3D()
        METHOD_HDF5(d).read_in_data3D(m, adiabatic_index=gamma)
        pv = m.prim_vars
        W = 1.0 / np.sqrt(1.0 - pv['vx'] ** 2 - pv['vy'] ** 2 - pv['vz'] ** 2)
        e = pv['p'] / (pv['n'] * (gamma - 1.0))
        np.testing.assert_allclose(m.aux_vars['W'], W)
        np.testing.assert_allclose(m.aux_vars['e'], e)
        np.testing.assert_allclose(m.aux_vars['h'], 1.0 + e + pv['p'] / pv['n'])
        self.assertEqual(m.aux_vars['W'].shape, pv['n'].shape)

    def test_missing_auxiliary_without_adiabatic_index_raises(self):
        with self.assertRaises(ValueError):
            METHOD_HDF5(self._minimal_dir()).read_in_data3D(IdealHD_3D())

    def test_present_auxiliary_is_not_overwritten(self):
        m = IdealHD_3D()
        METHOD_HDF5(os.path.join(self.tmpdir, '')).read_in_data3D(m, adiabatic_index=1.4)
        np.testing.assert_array_equal(m.aux_vars['W'], np.ones((2, 2, 2, 2)))


if __name__ == '__main__':
    unittest.main()
