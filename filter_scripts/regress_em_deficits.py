#!/usr/bin/env python
"""
Real-data check of the 4-covariant electromagnetic stress-energy deficit of the closure (OhmsRegression, "4-covariant form") on a pickled
resMHD3d meso model that contains the filtered electromagnetic stress-energy tensor SET_EM (made after 2026-10-21).

With F_r^{ab} = eps^{abcd} u~_c b~_d the resolved ideal Faraday tensor (b~ from <F> in the observer's frame), the deficit
    Delta^{ab} = <T_EM^{ab}> - T_EM^{ab}(F_r) = 2 Theta^{ab}(F_r, u~ ^ E~)  +  sigma^2 h^{mu nu} Theta^{ab}(grad_mu F_r, grad_nu F_r)
is the part of the stress-energy of the filtered electromagnetic field that a coarse run must add (its source is -d_j Delta^{jb}). This
script computes it from the data (<T_EM> - T_EM(F_r), all 16 lab components) and from the model, with no fitted constant:
  * the model with E~ taken from the data (isolates the two terms, expected coefficients +1, +1),
  * the model with E~ from the covariant EMF formula (the closure as a whole), with and without the Favre density term,
and scores each: R2 = 1 - |data - model|^2 / |data|^2 over all components (Euclidean norm on the lab components), best amplitude scale,
shift-null 95th percentile, tile spread of the coefficients, and the same by component group (stress ij, Poynting 0j, energy 00).
It also scores the sources -d_j Delta^{jb} (momentum, energy) that the closure adds. Then the same with the time derivatives dropped
(d_t = 0, what the lab-frame code uses). The central time slice is scored, edge cells dropped.

    python regress_em_deficits.py /path/to/resMHD3d_khi576_w4_3sl_c2_setem.pickle --sigma2-factor 1.12

Needs 3 time slices for the time derivatives (otherwise only the spatial model is available). Does not modify the pickle.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'master_files'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import OhmsRegression as OR
from MesoModels import *            # noqa: F401  (needed so the pickled classes can be loaded)
from model_report import central_slice, report

GROUPS = {'stress ij': [(i, j) for i in (1, 2, 3) for j in (1, 2, 3)],
          'Poynting 0j': [(0, j) for j in (1, 2, 3)] + [(j, 0) for j in (1, 2, 3)],
          'energy 00': [(0, 0)]}


def flat(x):
    return x.reshape(x.shape[:-2] + (16,))


def group_r2(target, pred):
    out = {}
    for name, comps in GROUPS.items():
        idx = [a * 4 + b for a, b in comps]
        out[name] = OR.prediction_r2(target[..., idx], pred[..., idx], np.ones(len(idx)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pickle_file")
    p.add_argument("--block", type=int, nargs=3, default=[12, 12, 6])
    p.add_argument("--min-points", type=int, default=100)
    p.add_argument("--n-null", type=int, default=50)
    p.add_argument("--sigma2-factor", type=float, default=1.0)
    p.add_argument("--label", default="")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()

    with open(a.pickle_file, 'rb') as fh:
        m = pickle.load(fh)
    st, dv, mv = m.meso_structures, m.domain_vars, m.meso_vars
    if 'SET_EM' not in st or not np.any(np.asarray(st['SET_EM']) != 0):
        sys.exit("this pickle has no filtered SET_EM: make it again with the current MesoModels (see the planning notes)")
    g = np.asarray(m.metric, dtype=float)
    ginv = np.linalg.inv(g)
    eps = OR.levi_civita_up(g)
    s = OR.faraday_ideal_sign(g)
    u = np.asarray(mv['u_tilde'], dtype=float)
    F = np.asarray(st['FaradayTensor'], dtype=float)
    Tfilt = np.asarray(st['SET_EM'], dtype=float)
    n = np.asarray(mv['n_tilde'], dtype=float)
    Nt = u.shape[0]
    spacings = (float(dv['Dx']), float(dv['Dy']), float(dv['Dz']))
    dt = float(dv['Dt']) if Nt >= 3 else None
    sigma2 = a.sigma2_factor * float(m.filter.filter_width) ** 2 / 12.0
    print("meso grid %s, filter width %.4g (sigma^2 = %.3g, factor %.3g), spacing %.4g%s" % (
        u.shape[:4], m.filter.filter_width, sigma2, a.sigma2_factor, spacings[0], "" if dt else "; ONE time slice: no time derivatives"))

    ul = np.einsum('ab,...b->...a', g, u)
    Fl = np.einsum('ac,bd,...cd->...ab', g, g, F)
    E_data = np.einsum('...ab,...b->...a', F, ul)                                    # E~^a = F~^{ab} u~_b
    bt = -np.einsum('...ab,...b->...a', 0.5 * np.einsum('abcd,...cd->...ab', eps, Fl), ul)   # b~^a = -u~_b *F~^{ab}
    Fr = s * np.einsum('abcd,...c,...d->...ab', eps, ul, np.einsum('ab,...b->...a', g, bt))
    Fr_l = np.einsum('...ab,bc->...ac', Fr, g)
    Tr = np.einsum('...ac,...bc->...ab', Fr, Fr_l) - 0.25 * ginv * np.einsum('...ab,...cd,ac,bd->...', Fr, Fr, g, g)[..., None, None]
    target = Tfilt - Tr                                                              # Delta^{ab} from the data

    variants = {}
    for label, step in (("full (with d_t)", dt), ("spatial only (d_t = 0)", None)):
        if label.startswith("full") and dt is None:
            continue
        for zeta in (1.0, 0.0):
            E_model = OR.covariant_emf(u, bt, n, spacings, sigma2, g, dt=step, zeta=zeta, s=s)
            lin_m, quad = OR.covariant_em_deficit_parts(Fr, u, E_model, spacings, sigma2, g, step)
            variants[(label, zeta)] = {'E_model': E_model, 'lin_model': lin_m, 'quad': quad}
        lin_d, quad = OR.covariant_em_deficit_parts(Fr, u, E_data, spacings, sigma2, g, step)
        variants[(label, 'dataE')] = {'lin_data': lin_d, 'quad': quad}

    sl = central_slice(u.shape[:4])
    tg = flat(target[sl])
    w = np.ones(16)
    size = np.sqrt(np.sum(tg ** 2, axis=-1))
    print("points used: %d (central time slice, edge cells dropped)" % int(np.prod(tg.shape[:4])))
    print("sizes: median |Delta| / |T_EM(F_r)| = %.3g; median |E~| / |B~| = %.3g\n" % (
        np.median(size / np.maximum(np.sqrt(np.sum(flat(Tr[sl]) ** 2, axis=-1)), 1e-300)),
        np.median(np.sqrt(np.sum(E_data[sl] ** 2, -1)) / np.maximum(np.sqrt(np.sum(bt[sl] ** 2, -1)), 1e-300))))

    rows = {}
    for key, v in variants.items():
        label, zeta = key
        if zeta == 'dataE':
            ops = {'lin': flat(v['lin_data'][sl]), 'quad': flat(v['quad'][sl])}
            name = "Delta, E~ from the data, %s" % label
            expected = "+1, +1"
        else:
            ops = {'lin': flat(v['lin_model'][sl]), 'quad': flat(v['quad'][sl])}
            name = "Delta, closure E~ (density term %s), %s" % ("on" if zeta == 1.0 else "off", label)
            expected = "+1, +1"
        print("=" * 100 + "\n" + name)
        pred = ops['lin'] + ops['quad']
        print("  by component group (R2 of the parameter-free prediction): " + "  ".join(
            "%s %.3f" % (k, r) for k, r in group_r2(tg, pred).items()))
        report(name, tg, ops, w, expected, tuple(a.block), a.min_points, a.n_null, rows)

    # the sources the closure adds: -d_j Delta^{jb}, from the data and from the model
    def source(D):
        return -sum(np.gradient(D[..., l + 1, :], spacings[l], axis=l + 1) for l in range(3))

    spatial_sl = (slice(None),) + tuple(slice(2, -2) if k >= 5 else slice(None) for k in u.shape[1:4])   # derivatives of derivatives: trim 2
    cs = central_slice(u.shape[:4])[0]
    src_t = source(target)[cs][spatial_sl]
    print("=" * 100 + "\nSOURCES -d_j Delta^{jb} (energy b = 0, momentum b = 1-3), R2 of the parameter-free model")
    out_src = {}
    for key, v in variants.items():
        label, zeta = key
        if zeta == 'dataE':
            D_model = v['lin_data'] + v['quad']
        else:
            D_model = v['lin_model'] + v['quad']
        src_m = source(D_model)[cs][spatial_sl]
        r_en = OR.prediction_r2(src_t[..., 0:1], src_m[..., 0:1], np.ones(1))
        r_mo = OR.prediction_r2(src_t[..., 1:], src_m[..., 1:], np.ones(3))
        out_src[key] = (r_en, r_mo)
        print("  %-34s %-22s energy R2 %.3f   momentum R2 %.3f" % (label, "closure E~ (zeta=%g)" % zeta if zeta != 'dataE' else "E~ from the data", r_en, r_mo))

    os.makedirs(a.outdir, exist_ok=True)
    out = dict(label=a.label, sigma2=sigma2, filter_width=float(m.filter.filter_width), n_slices=Nt)
    for i, (k, v) in enumerate(rows.items()):
        out["row%d_name" % i] = k
        out.update({"row%d_%s" % (i, kk): vv for kk, vv in v.items()})
    out['sources'] = np.array([[r[0], r[1]] for r in out_src.values()])
    out['source_keys'] = np.array([str(k) for k in out_src])
    np.savez(os.path.join(a.outdir, "em_deficits.npz"), **out)
    print("\nWrote em_deficits.npz to %s" % a.outdir)


if __name__ == "__main__":
    main()
