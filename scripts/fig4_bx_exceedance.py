"""Fig. 4 -- exceedance distributions of |b_x|.

P(|b_x| > x) for observations (a) within the mixed layer, (b) below the mixed
layer and (c) over the full water column, for the thick ice-front period,
the thin-ice period and the full trajectory.

Output: ``Figures/ct176-04-23_bx_exceedance_MLD_below_all.png``
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from sealfronts import config, io
from sealfronts.plotting import PERIOD_COLORS_TAB, panel_label, savefig, use_font_size

MIN_ABS_BX = 1e-12      # s-2, values below are treated as zeros and dropped
LABELS = {"thick": "Ice front", "thin": "Thin ice", "full": "Full trajectory"}
LAYERS = [("ml", "Within MLD"), ("below", "Below MLD"), ("all", "All depths")]


def abs_bx_samples(ds, layer):
    """Valid |b_x| values within / below the mixed layer or at all depths."""
    bx = io.masked(ds, "bx")
    mld = io.mld(ds)
    depth = bx["depth"].broadcast_like(bx)
    if layer == "ml":
        bx = bx.where(depth <= mld)
    elif layer == "below":
        bx = bx.where(depth > mld)
    else:  # MLD must be defined, as for the two other layers
        bx = bx.where(np.isfinite(mld))
    vals = np.abs(bx.values.ravel())
    vals = vals[np.isfinite(vals)]
    return vals[vals > MIN_ABS_BX]


def exceedance(vals):
    """Sorted values and empirical exceedance probability P(X > x)."""
    v = np.sort(vals)
    return v, 1.0 - np.arange(v.size) / v.size


def main():
    use_font_size(10)
    ds = io.load_omega()
    subsets = {r["key"]: io.subset_time(ds, r["start"], r["end"]) for r in io.regimes()}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, lab, (layer, title) in zip(axes, "abc", LAYERS):
        n_total = 0  # sum of the three sample sizes, as printed in the paper
        for key, sub in subsets.items():
            vals = abs_bx_samples(sub, layer)
            n_total += vals.size
            x, p = exceedance(vals)
            ax.plot(x, p, lw=2.5, color=PERIOD_COLORS_TAB[key],
                    label=f"{LABELS[key]} (N={vals.size:,})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-8, None)
        ax.set_ylim(1e-5, None)
        ax.grid(True, which="both", lw=0.5, alpha=0.5)
        ax.text(0.03, 0.90, f"{title}\nN={n_total:,}", transform=ax.transAxes,
                ha="left", va="top", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2))
        ax.set_xlabel(r"$|b_x|$ [s$^{-2}$]")
        panel_label(ax, lab, fontsize=13, y=1.04)
        ax.legend(fontsize=8, frameon=True)
    axes[0].set_ylabel(r"$P(|b_x| > x)$")
    plt.tight_layout()
    savefig(fig, f"{config.SEAL_ID}_bx_exceedance_MLD_below_all.png")


if __name__ == "__main__":
    main()
