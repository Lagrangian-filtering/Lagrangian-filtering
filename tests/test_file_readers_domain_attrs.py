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

    def _write_fake_file(self, path, t):
        shape = (2, 2, 2)
        with h5py.File(path, 'w') as f:
            dom = f.create_group('Domain')
            dom.attrs['nx'] = shape[0]
            dom.attrs['ny'] = shape[1]
            dom.attrs['nz'] = shape[2]
            for name, val in [('xmin', 0.0), ('xmax', 1.0), ('ymin', 0.0), ('ymax', 1.0),
                               ('zmin', 0.0), ('zmax', 1.0), ('dx', 0.5), ('dy', 0.5), ('dz', 0.5)]:
                dom.attrs[name] = val
            dom.attrs['endTime'] = t

            prim = f.create_group('Primitive')
            aux = f.create_group('Auxiliary')
            data = np.ones(shape)
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


if __name__ == '__main__':
    unittest.main()
