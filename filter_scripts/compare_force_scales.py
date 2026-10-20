#!/usr/bin/env python
"""
How large is the sub-filter Lorentz force F_closure compared with the other forces on the filtered fluid?

The filtered momentum equation is d_a <T^{ab}> = <f^b>. On a pickled resMHD3d meso model (central time slice, edge cells
dropped) this prints, for the spatial components b = 1..3, the distribution of
  |F_closure|, |<f>| (the filtered Lorentz force), |grad p~| and |d_a <T^{ai}>| (the total force on the filtered fluid,
  with a = t, x, y, z by centred differences on the meso grid),
their ratios to F_closure, and a consistency check (d_a<T^{ai}> against <f^i>, which should agree to the accuracy of the
meso-grid differences). The sub-filter force is the one the gradient model predicts; this shows whether it matters dynamically.

    python compare_force_scales.py /path/to/resMHD3d_khi576_w4_3sl_c2.pickle

Norms are Euclidean over the three spatial components. Needs 3 time slices for the time derivative of <T^{0i}> (otherwise it
is dropped and the divergence check is only approximate). Does not modify the pickle.
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


def pct(x):
    p = np.percentile(x, [16, 50, 84])
    return "median %.3g [16%%: %.3g, 84%%: %.3g]" % (p[1], p[0], p[2])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pickle_file")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()
    with open(a.pickle_file, 'rb') as fh:
        m = pickle.load(fh)
    st, dv, mv = m.meso_structures, m.domain_vars, m.meso_vars
    Nt = mv['u_tilde'].shape[0]
    spacings = (float(dv['Dx']), float(dv['Dy']), float(dv['Dz']))
    dt = float(dv['Dt']) if Nt >= 3 else None
    T, p_t = np.asarray(st['SET'], dtype=float), np.asarray(mv['p_tilde'], dtype=float)

    # d_a <T^{a i}> for i = 1..3: a = 0 (time), 1..3 (space)
    div = np.zeros(T.shape[:4] + (3,))
    if dt:
        div += np.gradient(T[..., 0, 1:], dt, axis=0)
    for l in range(3):
        div += np.gradient(T[..., l + 1, 1:], spacings[l], axis=l + 1)
    gradp = np.stack([np.gradient(p_t, spacings[l], axis=l + 1) for l in range(3)], axis=-1)

    t_sl = slice(Nt // 2, Nt // 2 + 1) if Nt >= 3 else slice(None)
    s_sl = tuple(slice(1, -1) if n >= 3 else slice(None) for n in T.shape[1:4])
    sl = (t_sl,) + s_sl
    nrm = lambda x: np.sqrt(np.sum(x[sl] ** 2, axis=-1)).ravel()
    Fc, f_lor = np.asarray(mv['F_closure'], dtype=float)[..., 1:], np.asarray(st['LorentzForceDensity'], dtype=float)[..., 1:]
    n_F, n_f, n_p, n_d = nrm(Fc), nrm(f_lor), nrm(gradp), nrm(div)
    n_chk = np.sqrt(np.sum((div - f_lor)[sl] ** 2, axis=-1)).ravel()
    ok = n_F > 0
    print("points used: %d%s" % (n_F.size, "" if dt else "  (single time slice: d_t<T^{0i}> omitted)"))
    print("\nmagnitudes (spatial components):")
    for name, x in (("|F_closure|", n_F), ("|<f>| (filtered Lorentz force)", n_f), ("|grad p~|", n_p), ("|d_a <T^{ai}>| (total force)", n_d)):
        print("  %-32s %s" % (name, pct(x)))
    print("\nratios to the sub-filter force:")
    for name, x in (("|F_closure| / |<f>|", n_F / np.maximum(n_f, 1e-300)), ("|F_closure| / |grad p~|", n_F / np.maximum(n_p, 1e-300)),
                    ("|F_closure| / |d_a<T^{ai}>|", n_F / np.maximum(n_d, 1e-300))):
        print("  %-32s %s" % (name, pct(x[ok])))
    print("\nconsistency of the meso differences: |d_a<T^{ai}> - <f^i>| / |<f>| = %s" % pct(n_chk / np.maximum(n_f, 1e-300)))
    print("fraction of points where |F_closure| exceeds 10%% of |d_a<T^{ai}>|: %.1f%%" % (100.0 * np.mean(n_F > 0.1 * n_d)))
    os.makedirs(a.outdir, exist_ok=True)
    np.savez(os.path.join(a.outdir, "force_scales.npz"), F_closure=n_F, f_lorentz=n_f, grad_p=n_p, div_T=n_d, check=n_chk)
    print("\nWrote force_scales.npz to %s" % a.outdir)


if __name__ == "__main__":
    main()
