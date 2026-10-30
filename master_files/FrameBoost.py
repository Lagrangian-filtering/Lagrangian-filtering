"""
Lorentz transformation of METHOD primitive frames to a frame moving along z: the same flow seen in two frames.

The rest frame S holds frames of the primitives rho, p, v = (vx, vy, vz), B = (Bx, By, Bz) (lab-frame B, ideal MHD, c = 1) on a periodic grid
x_i = xmin + i dx, ... at times t_k. The frame S' moves with velocity beta along +z relative to S, with t' = gamma (t - beta z), z' = gamma (z - beta t),
x' = x, y' = y; the inverse is t = gamma (t' + beta z'), z = gamma (z' + beta t'). The state at the event (T', x, y, z') of S' is the state of S at the
event (t, x, y, z) above, transformed:

    rho, p                scalars,
    v'_z = (v_z - beta) / (1 - beta v_z),   v'_perp = v_perp / (gamma (1 - beta v_z)),
    E = -v x B,   E' = gamma (E + u x B) - gamma^2/(gamma + 1) u (u.E),   B' = gamma (B - u x E) - gamma^2/(gamma + 1) u (u.B),   u = beta e_z.

The state at a non-grid event is interpolated from the frames of S: cubic Lagrange in time between the four nearest frames and cubic Lagrange along z
(periodic); x and y are grid points (x' = x). The error is O(dt^4 + dz^4) of the fourth derivatives of the fields. Cells with a failed primitive recovery in a stored frame
(non-finite, |v| >= 1, rho or p not positive) are replaced by the mean of their six neighbours before the interpolation and counted.
"""
import os

import h5py
import numpy as np

NAMES = ('rho', 'p', 'vx', 'vy', 'vz', 'Bx', 'By', 'Bz')


def boost_state(rho, p, v, B, beta):
    """State in the frame S' moving with velocity beta e_z relative to S. v, B: (..., 3). Returns rho, p, v', B' (E' = -v' x B' holds)."""
    g = 1.0 / np.sqrt(1.0 - beta ** 2)
    u = np.array([0.0, 0.0, beta])
    E = -np.cross(v, B)
    vz = v[..., 2]
    den = 1.0 - beta * vz
    vp = np.empty_like(v)
    vp[..., 0] = v[..., 0] / (g * den)
    vp[..., 1] = v[..., 1] / (g * den)
    vp[..., 2] = (vz - beta) / den
    k = g * g / (g + 1.0)
    Bp = g * (B - np.cross(u, E)) - k * u * np.sum(u * B, axis=-1, keepdims=True)
    return rho, p, vp, Bp


def lagrange_weights(nodes, t):
    """Weights of the cubic Lagrange interpolation at t on four nodes (nodes: (4,) increasing)."""
    w = np.ones(4)
    for i in range(4):
        for j in range(4):
            if j != i:
                w[i] *= (t - nodes[j]) / (nodes[i] - nodes[j])
    return w


def repair_bad_cells(a):
    """a: (8, nx, ny, nz) in the order NAMES. Replaces cells that are non-finite, superluminal or with rho or p not positive by the mean of their six neighbours
    (periodic roll; a repeated pass if neighbours are bad too). Returns the number of cells repaired."""
    v2 = a[2] ** 2 + a[3] ** 2 + a[4] ** 2
    bad = ~np.all(np.isfinite(a), axis=0) | ~(v2 < 1.0 - 1e-12) | ~(a[0] > 0.0) | ~(a[1] > 0.0)
    n = int(bad.sum())
    for _ in range(5):
        if not bad.any():
            break
        good = ~bad
        acc = np.zeros(a.shape)
        cnt = np.zeros(a.shape[1:])
        for ax in range(3):
            for s in (1, -1):
                g = np.roll(good, s, axis=ax)
                acc += np.where(g, np.roll(a, s, axis=ax + 1), 0.0)
                cnt += g
        fill = bad & (cnt > 0)
        a[:, fill] = (acc[:, fill] / cnt[fill])
        bad = bad & ~fill
    return n


class RestFrames(object):
    """The frames of S in a directory (sorted by their stored endTime), read on demand for the cells ix, iy (slices) and all z, with a cache."""

    def __init__(self, paths, xs, ys, repair=True):
        self.paths = list(paths)
        self.xs, self.ys = xs, ys
        self.repair = repair
        self.cache = {}
        self.repaired = {}
        self.times = []
        for p in self.paths:
            with h5py.File(p, 'r') as f:
                self.times.append(float(np.ravel(f['Domain'].attrs['endTime'])[0]))
        order = np.argsort(self.times)
        self.paths = [self.paths[i] for i in order]
        self.times = np.array(self.times)[order]
        with h5py.File(self.paths[0], 'r') as f:
            self.shape = f['Primitive/rho'].shape
            self.attrs = {k: np.array(v) for k, v in f['Domain'].attrs.items()}
        self.dz = float(np.ravel(self.attrs['dz'])[0])
        self.zmin = float(np.ravel(self.attrs['zmin'])[0])
        self.nz = self.shape[2]

    def get(self, k):
        if k not in self.cache:
            with h5py.File(self.paths[k], 'r') as f:
                a = np.stack([np.asarray(f['Primitive/' + n][self.xs, self.ys, :], dtype=float) for n in NAMES])
            self.repaired[k] = repair_bad_cells(a) if self.repair else 0
            self.cache[k] = a
        return self.cache[k]

    def drop_except(self, keep):
        for k in list(self.cache):
            if k not in keep:
                del self.cache[k]

    def sample_column(self, t, z):
        """State (8, nx, ny) at the event (t, z) for every cell (x, y) of the crop."""
        k = int(np.searchsorted(self.times, t, side='right')) - 1
        k = min(max(k, 1), len(self.times) - 3)
        if not (self.times[k] - 1e-9 <= t <= self.times[k + 1] + 1e-9):
            raise ValueError('the time %.6g is outside the frames (%.6g .. %.6g)' % (t, self.times[1], self.times[-2]))
        nodes = self.times[k - 1:k + 3]
        wt = lagrange_weights(nodes, t)
        s = (z - self.zmin) / self.dz
        i0 = int(np.floor(s))
        wz = lagrange_weights(np.array([-1.0, 0.0, 1.0, 2.0]), s - i0)
        iz = [(i0 + d) % self.nz for d in (-1, 0, 1, 2)]
        out = 0.0
        for a, wa in zip(range(k - 1, k + 3), wt):
            arr = self.get(a)
            out = out + wa * sum(wb * arr[..., j] for wb, j in zip(wz, iz))
        return out


def boosted_frame(rest, beta, T, nzp, dz=None):
    """The state in S' at the time T' = T on the grid (crop x, crop y, z'_m = (m - nzp//2) dz), m = 0..nzp-1, as an array (8, nx, ny, nzp) in the order NAMES."""
    g = 1.0 / np.sqrt(1.0 - beta ** 2)
    dz = rest.dz if dz is None else dz
    cols = []
    for m in range(nzp):
        zp = (m - nzp // 2) * dz
        t, z = g * (T + beta * zp), g * (zp + beta * T)
        cols.append(rest.sample_column(t, z))
    s = np.stack(cols, axis=-1)                                    # (8, nx, ny, nzp)
    rho, p = s[0], s[1]
    v = np.moveaxis(s[2:5], 0, -1)
    B = np.moveaxis(s[5:8], 0, -1)
    rho, p, v, B = boost_state(rho, p, v, B, beta)
    return np.concatenate([rho[None], p[None], np.moveaxis(v, -1, 0), np.moveaxis(B, -1, 0)], axis=0)


def write_frame(path, state, T, attrs, xs, ys, nzp, dz):
    """Writes a frame of METHOD layout: Domain attributes copied from a source frame and rewritten for the crop (xs, ys: slices of the source grid; z' cells -(nzp//2) dz ..)."""
    nx, ny = state.shape[1], state.shape[2]
    dx, dy = float(np.ravel(attrs['dx'])[0]), float(np.ravel(attrs['dy'])[0])
    xmin = float(np.ravel(attrs['xmin'])[0]) + xs.start * dx
    ymin = float(np.ravel(attrs['ymin'])[0]) + ys.start * dy
    zmin = -(nzp // 2) * dz
    new = {'nx': nx, 'ny': ny, 'nz': nzp, 'xmin': xmin, 'ymin': ymin, 'zmin': zmin, 'dz': dz,
           'xmax': xmin + nx * dx, 'ymax': ymin + ny * dy, 'zmax': zmin + nzp * dz, 'endTime': T}
    with h5py.File(path, 'w') as f:
        d = f.create_group('Domain')
        for k, v in attrs.items():
            d.attrs[k] = v
        for k, val in new.items():
            d.attrs[k] = np.array([val], dtype=int if k in ('nx', 'ny', 'nz') else float)
        g = f.create_group('Primitive')
        for n, a in zip(NAMES, state):
            g.create_dataset(n, data=a)
