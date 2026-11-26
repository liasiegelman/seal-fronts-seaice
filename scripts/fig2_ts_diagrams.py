"""Fig. 2 -- TS diagrams coloured by depth.

(a) full trajectory with labelled water masses, (b) thick ice-front period,
(c) thin-ice period. Points in the upper 10 m are shown as white dots; the
black line is the surface freezing line.

Output: ``Figures/TS_3panels.png``
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gsw
import matplotlib.pyplot as plt
import numpy as np

from sealfronts import config, io
from sealfronts.plotting import panel_label, savefig, use_font_size

DEP_MAX = 600       # m, deepest level shown
SURF_MAX = 10       # m, "surface" points drawn as white dots
SA_LIM = (31.8, 35.0)
CT_LIM = (-2.0, 2.0)
WATER_MASS_LABELS = [  # (SA, CT, text, fontsize) on panel a
    (32.4, -1.3, "Surface meltwater", 11),
    (33.6, -1.5, "Winter Water", 11),
    (34.45, 0.4, "mCDW", 12),
    (34.65, 1.4, "CDW", 11),
]


def ts_points(ds):
    """Flattened (SA, CT, depth) of all valid points in the upper DEP_MAX m."""
    sub = ds.sel(depth=slice(0, DEP_MAX))
    sa = io.masked(sub, "salinity")
    ct = io.masked(sub, "temperature")
    z = sub.depth.broadcast_like(ct).where(sub.maskprof > 0)
    sa, ct, z = (np.asarray(a.values).ravel() for a in (sa, ct, z))
    good = np.isfinite(sa) & np.isfinite(ct) & np.isfinite(z)
    return sa[good], ct[good], z[good]


def ts_panel(ax, ds, title):
    """Draw one TS diagram on ``ax``; returns the depth-coloured scatter."""
    sa, ct, z = ts_points(ds)
    surf = z <= SURF_MAX

    sc = ax.scatter(sa[~surf], ct[~surf], c=z[~surf], s=6, alpha=0.28,
                    vmin=0, vmax=DEP_MAX, cmap="plasma", rasterized=True, zorder=1)
    ax.scatter(sa[surf], ct[surf], c="white", edgecolors="grey", linewidths=0.45,
               s=8, alpha=0.4, label=f"0–{SURF_MAX} m", zorder=4)

    # sigma_0 isopycnals and surface freezing line
    sa_grid = np.linspace(31.8, 35.5, 200)
    ct_grid = np.linspace(-2.2, 2.2, 200)
    sa2, ct2 = np.meshgrid(sa_grid, ct_grid)
    sigma0 = gsw.sigma0(sa2, ct2)
    ax.contour(sa2, ct2, sigma0, levels=np.arange(25, 29, 0.1), colors="0.82",
               linewidths=0.45, zorder=0)
    cs = ax.contour(sa2, ct2, sigma0, levels=np.arange(25, 29, 0.5), colors="0.55",
                    linewidths=0.9, zorder=0)
    ax.clabel(cs, fmt="%.1f", fontsize=8, inline=True, inline_spacing=5)
    ct_freeze = gsw.CT_freezing(sa_grid, np.zeros_like(sa_grid), saturation_fraction=0)
    ax.plot(sa_grid, ct_freeze, color="black", lw=2.4, label="Freezing line", zorder=3)

    ax.set_xlim(*SA_LIM)
    ax.set_ylim(*CT_LIM)
    ax.set_xlabel(r"Absolute Salinity [g.kg$^{-1}$]", fontsize=14)
    ax.set_ylabel("Conservative Temperature [°C]", fontsize=14)
    ax.set_title(title, fontsize=14, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.12, linewidth=0.6)
    ax.legend(fontsize=12, loc="upper left", framealpha=0.7, frameon=True,
              facecolor="white", edgecolor="none")
    return sc


def main():
    use_font_size(10)
    ds = io.load_omega()
    panels = [
        ("a", "Full trajectory", ds),
        ("b", "Thick ice front", io.subset_time(ds, *config.THICK_ICE_PERIOD)),
        ("c", "Thin ice", io.subset_time(ds, *config.THIN_ICE_PERIOD)),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7, 16), sharex=True)
    for ax, (lab, title, sub) in zip(axes, panels):
        sc = ts_panel(ax, sub, title)
        panel_label(ax, lab, fontsize=14, y=1.01)
    bbox = dict(facecolor="white", alpha=0.7, edgecolor="none")
    for sa, ct, text, fs in WATER_MASS_LABELS:
        axes[0].text(sa, ct, text, fontsize=fs, bbox=bbox, zorder=10)
    for ax in axes[:-1]:
        ax.set_xlabel("")

    cax = fig.add_axes([0.99, 0.3, 0.025, 0.5])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Depth [m]", fontsize=14)
    cbar.ax.invert_yaxis()
    cbar.solids.set_alpha(1)
    plt.tight_layout()
    savefig(fig, "TS_3panels.png")


if __name__ == "__main__":
    main()
