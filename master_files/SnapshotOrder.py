"""
Order of the METHOD snapshots in a directory: by the time stored in each file, not by the file name.

METHOD names its checkpoints without zero padding (frame.checkpoint.8.97.hdf5, frame.checkpoint.10.48.hdf5), so sorting the names as text puts t = 10.48 before t = 8.97 and the meso
time slices would be built from frames in the wrong order. Files without a readable time keep the order of their names (all of them, so a directory of such files behaves as before).
"""
import glob
import os

import h5py
import numpy as np


def end_time(path):
    """The simulation time of a snapshot: Domain attribute 'endTime' (METHOD writes shape-(1,) attributes), else the dataset Domain/endTime; None if neither exists."""
    with h5py.File(path, 'r') as f:
        dom = f['Domain']
        if 'endTime' in dom.attrs:
            return float(np.ravel(dom.attrs['endTime'])[0])
        if 'endTime' in dom:
            return float(np.ravel(dom['endTime'][()])[0])
    return None


def snapshot_files(directory):
    """The *.hdf5 files of a directory in time order (ties and files without a time in the order of their names)."""
    names = sorted(glob.glob(os.path.join(directory, '') + '*.hdf5'))
    times = [end_time(n) for n in names]
    if any(t is None for t in times):
        return names
    return [n for _, n in sorted(zip(times, names), key=lambda x: x[0])]
