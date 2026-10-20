"""Shared labels, colours and helpers for the Ohm's-law winner plots (plot_ohms_winner.py, mosaic_ohms_winner.py)."""
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

LABELS = ['none', 'R (resistive)', 'alpha (dynamo)', 'gamma (Hall)']
COLOURS = ['#d9d9d9', '#d62728', '#2ca02c', '#1f77b4']


def winner_cmap_norm():
    """Colour map and norm for winner codes -2 (outside, white), -1 (none), 0 (R), 1 (alpha), 2 (gamma)."""
    cmap = ListedColormap(['white'] + COLOURS)
    return cmap, BoundaryNorm(np.arange(-2.5, 3.5, 1.0), cmap.N)


def winner_legend(ax):
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(COLOURS, LABELS)], loc='upper right', fontsize=8)


def sym_limit(f):
    """98th percentile of |f| over its finite values (inf as well as NaN must not reach colour limits); 1 if empty."""
    v = np.abs(np.asarray(f, dtype=float))
    v = v[np.isfinite(v)]
    return (np.percentile(v, 98) if v.size else 0.0) or 1.0
