#!/usr/bin/env python
"""
Fit of the covariant second-order model of E_tilde on one or several meso pickles, to compare runs in different frames (the Lorentz-boost test of the fits).

The window of the pipeline's filter is a spatial ball in the rest space of the observer u~, so the covariance of two fields is sigma^2 h^{mu nu} grad_mu A grad_nu B with
h = g + u~ u~ (OhmsRegression.py, header of the covariance section). The model of the fluid-frame sub-filter electric field (OhmsRegression.covariant_emf)

    E~^a = s eps^{abcd} u~_b sigma^2 h^{mu nu} [ d_mu u_c d_nu b_d - zeta (d_mu n / n~) d_nu u_c b~_d ]

uses the 4-velocity u~ and the comoving field b~ (from the filtered Faraday tensor), 4-gradients with the time derivative from the three meso slices, and no assumption on the speed of the
flow, so its coefficients should be the same in every frame. Fitted: the target E~ = F~^{ab} u~_b on the operators 'vB' (the zeta = 0 part, expected +1) and 'dens' (the zeta term, expected +1).
For comparison the lab-frame 3-vector model of regress_gradient_model.py ('lab3': v = u~^i/u~^0, B in the analysis frame; its expected coefficients are -1 and +1) is fitted on the same
points; it equals the covariant one to O(v^2) and departs from it at relativistic speeds. The spatial-only variants (d_t = 0) show what the time slices contribute.

    python regress_covariant_emf.py P1.pickle [P2.pickle ...] [--labels beta0 beta0.5 ...] [--block 12 12 6] [--sigma2-factor 1.12] [--common-region]

Uses the central time slice (three slices are needed for the time derivative; with fewer the time derivative is zero and the variant 'cov' is the spatial-only one) and drops one cell at the
spatial edges. With --common-region every pickle is fitted on the same physical box (the intersection of the meso boxes, after the edge drop; the cells must coincide
between the pickles, i.e. the same spacing and offsets that are whole cells), so that the coefficients are not affected by the region. Prints one block per pickle and a summary table of the joint coefficients; writes covariant_emf.npz. Does not modify the pickles.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'master_files'))
import OhmsRegression as OR
from MesoModels import *            # noqa: F401  (needed so the pickled classes can be loaded)
from system.BaseFunctionality import Base


def operators(m, sigma2):
    """The model operators on the whole meso grid and the target, for one pickle: dict variant -> {name: array}, target, info."""
    v, st, dv = m.meso_vars, m.meso_structures, m.domain_vars
    g = np.asarray(m.metric, dtype=float)
    eps = OR.levi_civita_up(g)
    s = OR.faraday_ideal_sign(g)
    u = np.asarray(v['u_tilde'], dtype=float)
    F = np.asarray(st['FaradayTensor'], dtype=float)
    n = np.asarray(v['n_tilde'], dtype=float)
    Nt = u.shape[0]
    spacings = (float(dv['Dx']), float(dv['Dy']), float(dv['Dz']))
    dt = float(dv['Dt']) if Nt >= 3 else None

    ul = np.einsum('ab,...b->...a', g, u)
    Fl = np.einsum('ac,bd,...cd->...ab', g, g, F)
    target = np.einsum('...ab,...b->...a', F, ul)                                                    # E~^a = F~^{ab} u~_b
    bt = -np.einsum('...ab,...b->...a', 0.5 * np.einsum('abcd,...cd->...ab', eps, Fl), ul)          # b~^a = -u~_b *F~^{ab}

    variants = {}
    for label, step in (('cov', dt), ('cov_spatial', None)):
        if label == 'cov_spatial' and dt is None:
            continue
        o_vb = OR.covariant_emf(u, bt, n, spacings, sigma2, g, dt=step, zeta=0.0, s=s)
        o_full = OR.covariant_emf(u, bt, n, spacings, sigma2, g, dt=step, zeta=1.0, s=s)
        variants[label] = {'vB': o_vb, 'dens': o_full - o_vb}

    _, B_lab = Base.observer_frame_fields(st['FaradayTensor'], np.array([1.0, 0.0, 0.0, 0.0]), m.metric)
    vel = u[..., 1:] / u[..., 0:1]
    D = st['BC'][..., 0]
    lab = OR.gradient_model_operators(vel, B_lab[..., 1:], D, u, spacings, sigma2, dt)
    variants['lab3'] = {'vB': lab['model_vB'], 'dens': lab['model_dens']}

    gamma = u[..., 0]
    info = {'n_slices': Nt, 'spacing': spacings[0], 'dt': dt if dt else 0.0, 'gamma_max': float(gamma.max()), 'gamma_median': float(np.median(gamma)),
            'E_pickled_mismatch': float(np.sqrt(np.sum((target - np.asarray(v['E_tilde'], dtype=float)) ** 2) / max(np.sum(target ** 2), 1e-300)))}
    return variants, target, info


def central(shape):
    Nt = shape[0]
    t_sl = slice(Nt // 2, Nt // 2 + 1) if Nt >= 3 else slice(None)
    return (t_sl,) + tuple(slice(1, -1) if k >= 3 else slice(None) for k in shape[1:4])


def trimmed_axes(m):
    """The meso coordinates X, Y, Z of the cells kept by central() (one cell dropped at each end)."""
    return [np.asarray(m.domain_vars[k], dtype=float)[1:-1] for k in ('X', 'Y', 'Z')]


def read_axes(path):
    with open(path, 'rb') as fh:
        return trimmed_axes(pickle.load(fh))


def common_box(all_axes):
    """The physical box [lo, hi] per direction common to the meso boxes of all the pickles."""
    lo = [max(ax[d][0] for ax in all_axes) for d in range(3)]
    hi = [min(ax[d][-1] for ax in all_axes) for d in range(3)]
    return lo, hi


def box_slices(axes, box):
    """Index ranges (into the trimmed axes) of the cells inside the box, per direction."""
    lo, hi = box
    out = []
    for d, ax in enumerate(axes):
        tol = 1e-3 * abs(ax[1] - ax[0])
        idx = np.nonzero((ax >= lo[d] - tol) & (ax <= hi[d] + tol))[0]
        if idx.size == 0:
            raise SystemExit("the meso boxes of the pickles do not overlap in direction %d" % d)
        out.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return out


def analyse(path, a, box=None):
    with open(path, 'rb') as fh:
        m = pickle.load(fh)
    sigma2 = a.sigma2_factor * float(m.filter.filter_width) ** 2 / 12.0
    variants, target, info = operators(m, sigma2)
    sl = central(target.shape[:4])
    if box is not None:
        axes = trimmed_axes(m)
        sl = (sl[0],) + tuple(slice(1 + r.start, 1 + r.stop) for r in box_slices(axes, box))
    E = target[sl]
    out = {'info': info, 'sigma2': sigma2, 'variants': {}}
    names = ('vB', 'dens')
    for label, ops in variants.items():
        ops = {k: x[sl] for k, x in ops.items()}
        fit = OR.fit_region(E, ops, names)
        try:
            null = OR.shift_null(E, ops, names, n_perm=a.n_null, min_shift=3)
        except ValueError as err:                                  # a meso grid too small to shift by 3 cells: no null
            print("  (%s: no shift null: %s)" % (label, err))
            null = np.array([np.nan])
        tiles = OR.tile_fits(E, ops, names, tuple(a.block), min_points=a.min_points)
        single = {n: OR.fit_region(E, ops, (n,)) for n in names}
        size = {n: float(np.sqrt(OR.mink_dot(ops[n], ops[n]).sum() / OR.mink_dot(E, E).sum())) for n in names}
        out['variants'][label] = {
            'R2': fit['R2'], 'null95': float(np.nanpercentile(null, 95)) if np.isfinite(null).any() else float('nan'), 'coef': {n: fit[n] for n in names},
            'R2_single': {n: single[n]['R2'] for n in names}, 'coef_single': {n: single[n][n] for n in names}, 'size': size,
            'tiles': {n: OR.summarize([t[n] for t in tiles]) for n in names} if tiles else None, 'n_tiles': len(tiles)}
    out['n_points'] = int(np.prod(E.shape[:4]))
    out['shape'] = tuple(E.shape[1:4])
    return out


def report(label, res):
    i = res['info']
    print("=" * 100)
    print("%s: %d meso slices, spacing %.4g, dt %.4g, gamma max %.3f (median %.3f), sigma^2 %.3g, %d points (%s cells)" % (
        label, i['n_slices'], i['spacing'], i['dt'], i['gamma_max'], i['gamma_median'], res['sigma2'], res['n_points'], 'x'.join(str(k) for k in res['shape'])))
    print("  pickled E_tilde against F~ u~: relative mismatch %.2e (should be ~0)" % i['E_pickled_mismatch'])
    print("  %-12s %7s %9s | %-26s | %-30s | %-30s" % ("variant", "R2", "(null95)", "joint coefficients vB, dens", "vB alone: R2, coefficient", "tiles: vB median [p16, p84]"))
    for name, r in res['variants'].items():
        t = r['tiles']['vB'] if r['tiles'] else None
        print("  %-12s %7.3f (%7.3f) | %12.3f %13.3f | %9.3f %19.3f | %s" % (
            name, r['R2'], r['null95'], r['coef']['vB'], r['coef']['dens'], r['R2_single']['vB'], r['coef_single']['vB'],
            "%8.3f [%7.3f, %7.3f] n=%d" % (t['median'], t['p16'], t['p84'], r['n_tiles']) if t else "no tiles"))
    print("  expected: cov*: +1, +1;  lab3: -1, +1")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pickle_files", nargs='+')
    p.add_argument("--labels", nargs='*', default=None, help="one label per pickle (default: the file names)")
    p.add_argument("--block", type=int, nargs=3, default=[12, 12, 6])
    p.add_argument("--min-points", type=int, default=100)
    p.add_argument("--n-null", type=int, default=30)
    p.add_argument("--sigma2-factor", type=float, default=1.0, help="multiply the nominal sigma^2 = W^2/12 (see regress_gradient_model.py)")
    p.add_argument("--common-region", action="store_true", help="fit every pickle on the physical box common to all of them (their meso boxes differ with the run)")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()
    labels = a.labels or [os.path.basename(f) for f in a.pickle_files]
    if len(labels) != len(a.pickle_files):
        p.error("give one label per pickle")

    box = None
    if a.common_region:
        all_axes = [read_axes(f) for f in a.pickle_files]
        box = common_box(all_axes)
        picks = [[ax[d][r] for d, r in enumerate(box_slices(ax, box))] for ax in all_axes]
        for k, pick in enumerate(picks[1:], 1):
            for d in range(3):
                if len(pick[d]) != len(picks[0][d]) or not np.allclose(pick[d], picks[0][d], rtol=0.0, atol=1e-3 * abs(picks[0][d][1] - picks[0][d][0])):
                    p.error("the meso cells of %s do not coincide with those of %s in direction %d (different spacing or a non-integer offset)" % (labels[k], labels[0], d))
        print("common region: x [%.6g, %.6g]  y [%.6g, %.6g]  z [%.6g, %.6g]  (%s cells)" % (
            box[0][0], box[1][0], box[0][1], box[1][1], box[0][2], box[1][2], 'x'.join(str(len(q)) for q in picks[0])))

    results = {}
    for lab, f in zip(labels, a.pickle_files):
        results[lab] = analyse(f, a, box)
        report(lab, results[lab])

    print("\n" + "=" * 100 + "\nSUMMARY: joint coefficient of the vB operator (and R2); covariant model: expected +1; lab-frame 3-vector model: expected -1")
    print("%-28s %8s | %-24s | %-24s | %-24s" % ("run", "gamma", "cov", "cov_spatial", "lab3"))
    for lab, r in results.items():
        cells = []
        for v in ('cov', 'cov_spatial', 'lab3'):
            x = r['variants'].get(v)
            cells.append("%8.3f (R2 %.3f)" % (x['coef']['vB'], x['R2']) if x else "-")
        print("%-28s %8.3f | %-24s | %-24s | %-24s" % (lab[:28], r['info']['gamma_median'], *cells))

    os.makedirs(a.outdir, exist_ok=True)
    keys = [(v, k) for v in ('cov', 'cov_spatial', 'lab3') for k in ('R2', 'null95')]
    np.savez(os.path.join(a.outdir, "covariant_emf.npz"), labels=np.array(list(results)), n_points=np.array([r['n_points'] for r in results.values()]), gamma_median=np.array([r['info']['gamma_median'] for r in results.values()]),
             **{"%s_%s" % (v, k): np.array([r['variants'][v][k] if v in r['variants'] else np.nan for r in results.values()]) for v, k in keys},
             **{"%s_coef_%s" % (v, n): np.array([r['variants'][v]['coef'][n] if v in r['variants'] else np.nan for r in results.values()])
                for v in ('cov', 'cov_spatial', 'lab3') for n in ('vB', 'dens')})
    print("\nWrote covariant_emf.npz to %s" % a.outdir)


if __name__ == '__main__':
    main()
