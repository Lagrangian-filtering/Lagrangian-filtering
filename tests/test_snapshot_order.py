"""METHOD checkpoint names are not zero padded: the snapshots must be used in the order of their stored times, not of their names."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'master_files'))
import SnapshotOrder as SO
from FileReaders import METHOD_HDF5

TIMES = [8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5]


def write_frame(path, t, as_dataset=False, with_time=True):
    with h5py.File(path, 'w') as f:
        d = f.create_group('Domain')
        for k, val in (('dx', 0.1), ('xmin', 0.0), ('ymin', 0.0), ('zmin', 0.0)):
            d.attrs[k] = np.array([val])
        if with_time:
            if as_dataset:
                d.create_dataset('endTime', data=np.array([t]))
            else:
                d.attrs['endTime'] = np.array([t])
        p = f.create_group('Primitive')
        p['rho'] = np.ones((4, 4, 4))
        for c in 'xyz':
            p['v' + c] = np.zeros((4, 4, 4))


class TestSnapshotOrder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def make(self, **kw):
        for t in TIMES:
            write_frame(os.path.join(self.tmp, 'frame.checkpoint.%.2f.hdf5' % t), t, **kw)

    def test_name_order_is_not_time_order(self):
        self.make()
        by_name = sorted(os.listdir(self.tmp))
        self.assertNotEqual([SO.end_time(os.path.join(self.tmp, n)) for n in by_name], TIMES)       # the trap exists

    def test_snapshot_files_in_time_order(self):
        self.make()
        got = [SO.end_time(f) for f in SO.snapshot_files(self.tmp)]
        self.assertEqual(got, TIMES)

    def test_time_stored_as_dataset(self):
        self.make(as_dataset=True)
        self.assertEqual([SO.end_time(f) for f in SO.snapshot_files(self.tmp)], TIMES)

    def test_files_without_time_keep_name_order(self):
        self.make(with_time=False)
        got = SO.snapshot_files(self.tmp)
        self.assertEqual(got, sorted(got))

    def test_reader_uses_time_order(self):
        self.make()
        reader = METHOD_HDF5(self.tmp + os.sep)
        self.assertEqual([SO.end_time(f.filename) for f in reader.hdf5_files], TIMES)
        for f in reader.hdf5_files:
            f.close()

    def test_make_mhd_config_sees_uniform_increasing_times(self):
        self.make()
        for t in (12.0, 12.5):                                    # more frames so that the config generator has room for its slices
            write_frame(os.path.join(self.tmp, 'frame.checkpoint.%.2f.hdf5' % t), t)
        res = subprocess.run([sys.executable, os.path.join(ROOT, 'filter_scripts', 'make_mhd_config.py'), self.tmp, '--name', 'x', '--crop-size', '4', '4', '4',
                              '--filter-width', '1', '--box-len', '1', '--num-t-slices', '1', '--force', '--outdir', self.tmp], capture_output=True, text=True)
        self.assertIn('spacing 0.5 (', res.stdout)
        self.assertIn('uniform: True', res.stdout)
        self.assertNotIn('spacing -', res.stdout)


if __name__ == '__main__':
    unittest.main()
