"""Tests of make_mhd_config.py (window check, margin and meso ranges) on tiny synthetic METHOD output."""
import contextlib
import io
import json
import os
import re
import runpy
import sys
import tempfile
import unittest

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'filter_scripts', 'make_mhd_config.py')
DX = 0.1


def write_run(path, n_snap=41, spacing=0.05, shape=(40, 40, 32), v=0.3):
    for i in range(n_snap):
        with h5py.File(os.path.join(path, "data_serial%03d.hdf5" % i), "w") as f:
            d = f.create_group("Domain")
            for k, val in (("endTime", 5.0 + i * spacing), ("dx", DX), ("xmin", 0.0), ("ymin", 0.0), ("zmin", 0.0)):
                d.attrs[k] = np.array([val])           # METHOD writes shape-(1,) attributes
            p = f.create_group("Primitive")
            p["rho"] = np.ones(shape)
            for c in "xyz":
                p["v" + c] = np.full(shape, v / np.sqrt(3.0))


@unittest.skipUnless(h5py, "h5py not installed")
class TestMakeMhdConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.run = os.path.join(self.tmp, "run")
        os.makedirs(self.run)
        write_run(self.run)
        self._argv = sys.argv

    def tearDown(self):
        sys.argv = self._argv

    def call(self, *args):
        sys.argv = ["make_mhd_config.py", self.run, "--name", "t", "--outdir", self.tmp] + list(args)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            runpy.run_path(SCRIPT, run_name="__main__")
        return buf.getvalue()

    def test_config_and_ranges(self):
        out = self.call("--crop-size", "16", "16", "12", "--filter-width", "2", "--box-len", "3",
                        "--num-t-slices", "3", "--coarse", "2")
        self.assertIn("window span", out)
        with open(os.path.join(self.tmp, "config_t.txt")) as f:
            text = f.read()
        opts = json.loads(re.search(r"^meso_grid_smart = (.*)$", text, re.M).group(1))
        self.assertEqual(opts["num_T_slices"], 3)
        self.assertEqual(opts["coarse_grain_factor"], 2)
        # centred crop starts at cell 12 (40-16)/2; the reach is 1.85 dx -> margin ceil(1.85)+2 = 4 cells
        for rng_, n_crop in zip((opts["x_range"], opts["y_range"]), (16, 16)):
            self.assertAlmostEqual(rng_[0], (12 + 4) * DX, places=4)
            self.assertAlmostEqual(rng_[1], (12 + n_crop - 1 - 4) * DX, places=4)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "submit_t.sh")))

    def test_window_check_refuses_too_wide_a_filter(self):
        with self.assertRaises(SystemExit) as cm:
            self.call("--crop-size", "36", "36", "30", "--filter-width", "14", "--box-len", "14")
        self.assertEqual(cm.exception.code, 1)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "config_t.txt")))

    def test_force_writes_anyway(self):
        self.call("--crop-size", "36", "36", "30", "--filter-width", "14", "--box-len", "14", "--force")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "config_t.txt")))

    def test_crop_too_small_for_the_margin(self):
        with self.assertRaises(SystemExit):
            self.call("--crop-size", "8", "8", "12", "--filter-width", "2", "--box-len", "3",
                      "--num-t-slices", "3", "--coarse", "2")


if __name__ == "__main__":
    unittest.main()
