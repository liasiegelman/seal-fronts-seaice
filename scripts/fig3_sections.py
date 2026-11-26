"""Fig. 3 -- along-track ocean-ice-atmosphere sections for the two regimes.

Left column: thick ice-front regime; right column: thin ice near a coastal
polynya. Rows: (a,g) SIC and ERA5 2 m air temperature; (b,h) temperature;
(c,i) salinity; (d,j) b_x; (e,k) w; (f,l) VHT. Thick line: MLD; contours:
sigma_0 every 0.1 kg m-3.

Output: ``Figures/two_sections.png``
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cmocean
import gsw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sealfronts import config, io
from sealfronts.plotting import (add_date_top_axis, add_isopycnals, below_freezing,
                                 finish_section_axis, panel_label, savefig,
                                 scatter_below_freezing, section, section_legend,
                                 sci_colorbar, set_distance_axis, use_font_size)

# Colour limits shared by both columns
LIMITS = dict(t2m=(-12, 2), temp=(-2, 1.7), salt=(33.6, 34.7), bx=4e-7,
              w=(-20, 20), vht=(-500, 500))
REGIMES = [
    dict(period=config.THICK_ICE_PERIOD, dep_max=500, title="Thick ice front edge",
         labels="abcdef"),
    dict(period=config.THIN_ICE_PERIOD, dep_max=300,
         title="Thin ice area near coastal polynya", labels="ghijkl"),
]


def prepare(ds, era5, period, dep_max):
    """Subset the section and collect everything a column needs."""
    sub = io.subset_time(ds, *period).sel(depth=slice(0, dep_max))
    sub = sub.load()
    times = pd.to_datetime(sub.time.values)
    temp = io.masked(sub, "temperature").values
    salt = io.masked(sub, "salinity").values
    return dict(
        depth=sub.depth.values.astype(float),
        times=times,
        x=np.arange(times.size),
        dist_km=sub.distance.values / 1000.0,
        temp=temp,
        salt=salt,
        bx=io.masked(sub, "bx").values,
        w=io.masked(sub, "w").values * 86400.0,      # m day-1
        vht=io.masked(sub, "VHF").values,
        sigma0=gsw.sigma0(salt, temp),
        freezing=below_freezing(temp, salt, sub.depth.values, sub.lat.values),
        sic=io.surface_sic(sub),
        mld=io.mld(sub).values,
        t2m=io.interp_to_times(era5["t2m"], times),   # hourly ERA5, unsmoothed
    )


def plot_column(fig, axes, caxes, d, title, labels):
    """Draw the six panels of one regime."""
    x, depth = d["x"], d["depth"]
    xmin, xmax = x[0], x[-1]
    axes[0].set_title(title, fontsize=20, pad=35)
    for ax, lab in zip(axes, labels):
        panel_label(ax, lab, fontsize=18)

    # SIC + t2m
    ax = axes[0]
    ax.fill_between(x, 0, d["sic"], color="0.65", alpha=0.45, linewidth=0,
                    label="SIC", zorder=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("SIC")
    ax.grid(True, lw=0.5)
    ax.set_xlim(xmin, xmax)
    ax_t = ax.twinx()
    ax_t.plot(x, d["t2m"], color="tab:red", lw=2.5, label="ERA5 t2m")
    ax_t.axhline(0, color="tab:red", lw=1, ls="--", alpha=0.6)
    ax_t.set_ylabel(r"ERA5 t2m [$^\circ$C]", color="tab:red")
    ax_t.tick_params(axis="y", colors="tab:red")
    ax_t.set_xlim(xmin, xmax)
    ax_t.set_ylim(LIMITS["t2m"])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_t.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", frameon=True,
              facecolor="white", framealpha=0.9)
    caxes[0].axis("off")
    add_date_top_axis(ax, x, d["times"], xmin, xmax)

    # temperature
    ax = axes[1]
    im = section(ax, d["temp"], depth, xmin, xmax, cmocean.cm.thermal, *LIMITS["temp"])
    add_isopycnals(ax, d["sigma0"], x, depth, color="white")
    ax.plot(x, d["mld"], color="white", lw=2.5, label="MLD", zorder=8)
    scatter_below_freezing(ax, d["freezing"], x, depth, zorder=9)
    section_legend(ax)
    finish_section_axis(ax, xmin, xmax)
    fig.colorbar(im, cax=caxes[1], label="Temperature [°C]")

    # salinity
    ax = axes[2]
    im = section(ax, d["salt"], depth, xmin, xmax, cmocean.cm.haline, *LIMITS["salt"])
    add_isopycnals(ax, d["sigma0"], x, depth, color="k")
    ax.plot(x, d["mld"], color="white", lw=2.5, zorder=8)
    finish_section_axis(ax, xmin, xmax)
    fig.colorbar(im, cax=caxes[2], label="Salinity [PSU]")

    # b_x, w, VHT (diverging)
    diverging = [
        (d["bx"], (-LIMITS["bx"], LIMITS["bx"]), r"$b_x$ [s$^{-2}$]", True),
        (d["w"], LIMITS["w"], r"w [m day$^{-1}$]", False),
        (d["vht"], LIMITS["vht"], r"VHT [W m$^{-2}$]", False),
    ]
    for ax, cax, (field, lims, label, sci) in zip(axes[3:], caxes[3:], diverging):
        im = section(ax, field, depth, xmin, xmax, "RdBu_r", *lims)
        add_isopycnals(ax, d["sigma0"], x, depth, color="k")
        ax.plot(x, d["mld"], color="k", lw=2.5, zorder=8)
        finish_section_axis(ax, xmin, xmax)
        if sci:
            sci_colorbar(fig, im, cax, label)
        else:
            fig.colorbar(im, cax=cax, label=label)

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    set_distance_axis(axes[-1], x, d["dist_km"], xmin, xmax)


def main():
    use_font_size(18)
    ds = io.load_omega()
    era5 = io.load_era5()

    fig = plt.figure(figsize=(50, 34))
    gs = fig.add_gridspec(6, 5, width_ratios=[55, 0.5, 6.0, 55, 0.5],
                          height_ratios=[3, 4.5, 4.5, 4.5, 4.5, 4.5],
                          hspace=0.18, wspace=0.05)
    for col, reg in enumerate(REGIMES):
        axes = [fig.add_subplot(gs[i, 3 * col]) for i in range(6)]
        caxes = [fig.add_subplot(gs[i, 3 * col + 1]) for i in range(6)]
        for ax in axes[1:]:
            ax.sharex(axes[0])
        d = prepare(ds, era5, reg["period"], reg["dep_max"])
        plot_column(fig, axes, caxes, d, reg["title"], reg["labels"])

    savefig(fig, "two_sections.png")


if __name__ == "__main__":
    main()
