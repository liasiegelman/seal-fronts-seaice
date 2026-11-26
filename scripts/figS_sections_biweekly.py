"""ED Figs. 4-8 -- full along-track sections in five ~two-week windows.

(a) SIC, ERA5 t2m and surface fluxes (Qnet, Rnet, THF; 24 h rolling means);
(b) temperature (cyan: below freezing); (c) salinity; (d) buoyancy; (e) b_x;
(f) N^2; (g) w; (h) VHT. MLD overlaid on panels b-h. Upper 600 m.

Output: ``Figures_SI/ct176-04-23_sic_t2m_fluxes_temp_salt_buoy_bx_N2_w_VHT_0_600_<t0>_<t1>.png``
where t0/t1 are the dates of the first and last profile in the window.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sealfronts import config, io
from sealfronts.plotting import (THF_COLOR, add_date_top_axis, below_freezing,
                                 finish_section_axis, panel_label, savefig,
                                 scatter_below_freezing, section, section_legend,
                                 sci_colorbar, set_distance_axis, use_font_size)

DEP_MAX = 600
FLUX_HOURS = 24
BX_LIM = 4e-7               # s-2
N2_LIMS = (0, 3e-4)         # s-2


def symmetric(values, pct):
    """(-v, v) with v the ``pct`` percentile of |values|."""
    v = np.nanpercentile(np.abs(values), pct)
    return -v, v


def plot_window(ds, era5_smooth, start, end):
    """Build the eight-panel section figure for one time window."""
    sub = io.subset_time(ds, start, end).sel(depth=slice(0, DEP_MAX)).load()
    depth = sub.depth.values.astype(float)
    times = pd.to_datetime(sub.time.values)
    x = np.arange(times.size)
    xmin, xmax = x[0], x[-1]

    temp = io.masked(sub, "temperature").values
    salt = io.masked(sub, "salinity").values
    buoy = io.masked(sub, "buoyancy").values
    bx = io.masked(sub, "bx").values
    n2 = io.masked(sub, "N2_smooth").values
    w = io.masked(sub, "w").values * 86400.0
    vht = io.masked(sub, "VHF").values
    mld = io.mld(sub).values
    sic = io.surface_sic(sub, clip=False)
    era = {k: io.interp_to_times(era5_smooth[k], times)
           for k in ("t2m", "qnet", "rnet", "thf")}

    fig = plt.figure(figsize=(28, 44))
    gs = fig.add_gridspec(8, 2, width_ratios=[40, 0.4],
                          height_ratios=[3, 4, 4, 4, 4, 4, 4, 4], hspace=0.18, wspace=0.04)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(8)]
    caxes = [fig.add_subplot(gs[i, 1]) for i in range(8)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    caxes[0].axis("off")
    for ax, lab in zip(axes, "abcdefgh"):
        panel_label(ax, lab, fontsize=18)

    # (a) SIC, t2m and fluxes on three y axes
    ax = axes[0]
    ax.fill_between(x, 0, sic, color="0.65", alpha=0.55, label="SIC")
    ax.set_ylim(0, 1)
    ax.set_ylabel("SIC")
    ax.grid(True, lw=0.5)
    ax.set_xlim(xmin, xmax)
    ax_t = ax.twinx()
    ax_t.plot(x, era["t2m"], color="tab:red", lw=2.5, label="t2m")
    ax_t.axhline(0, color="tab:red", lw=1, ls="--", alpha=0.6)
    ax_t.set_ylabel("ERA5 t2m [°C]", color="tab:red")
    ax_t.tick_params(axis="y", colors="tab:red")
    ax_t.set_xlim(xmin, xmax)
    ax_f = ax.twinx()
    ax_f.spines["right"].set_position(("axes", 1.07))
    ax_f.plot(x, era["qnet"], lw=3.0, color="k", label="Qnet")
    ax_f.plot(x, era["rnet"], lw=2.0, color="0.45", label="Rnet")
    ax_f.plot(x, era["thf"], lw=2.0, color=THF_COLOR, label="THF")
    ax_f.axhline(0, color="0.3", lw=1, alpha=0.6)
    ax_f.set_ylabel(r"ERA5 Flux [W m$^{-2}$]", color="k")
    ax_f.set_xlim(xmin, xmax)
    handles, labels = [], []
    for a in (ax, ax_t, ax_f):
        h, l = a.get_legend_handles_labels()
        handles += h
        labels += l
    leg = ax.legend(handles, labels, loc="best", frameon=True, facecolor="white",
                    edgecolor="none", framealpha=0.7)
    ax.set_zorder(3)
    ax.patch.set_alpha(0)
    leg.set_zorder(100)

    # (b) temperature
    ax = axes[1]
    im = section(ax, temp, depth, xmin, xmax, cmocean.cm.thermal,
                 np.nanmin(temp), np.nanmax(temp))
    ax.plot(x, mld, color="white", lw=2.5, label="MLD", zorder=6)
    scatter_below_freezing(ax, below_freezing(temp, salt, depth, sub.lat.values),
                           x, depth, zorder=5)
    finish_section_axis(ax, xmin, xmax)
    section_legend(ax)
    fig.colorbar(im, cax=caxes[1], label="Temperature [°C]")

    # (c) salinity
    ax = axes[2]
    im = section(ax, salt, depth, xmin, xmax, cmocean.cm.haline,
                 *np.nanpercentile(salt, [2, 98]))
    ax.plot(x, mld, color="white", lw=2.5, zorder=6)
    finish_section_axis(ax, xmin, xmax)
    fig.colorbar(im, cax=caxes[2], label="Salinity [PSU]")

    # (d-h) buoyancy, b_x, N2, w, VHT with black MLD
    panels = [
        (buoy, cmocean.cm.dense_r, np.nanpercentile(buoy, [2, 98]), "Buoyancy [m s$^{-2}$]", True),
        (bx, "RdBu_r", (-BX_LIM, BX_LIM), r"$b_x$ [s$^{-2}$]", True),
        (n2, cmocean.cm.matter, N2_LIMS, r"N$^2$ [s$^{-2}$]", True),
        (w, "RdBu_r", symmetric(w, 98), r"w [m day$^{-1}$]", True),
        (vht, "RdBu_r", symmetric(vht, 95), r"VHT [W m$^{-2}$]", False),
    ]
    for ax, cax, (field, cmap, lims, label, sci) in zip(axes[3:], caxes[3:], panels):
        im = section(ax, field, depth, xmin, xmax, cmap, *lims)
        ax.plot(x, mld, color="k", lw=2.5)
        finish_section_axis(ax, xmin, xmax)
        if sci:
            sci_colorbar(fig, im, cax, label)
        else:
            fig.colorbar(im, cax=cax, label=label)

    set_distance_axis(axes[-1], x, sub.distance.values / 1000.0, xmin, xmax)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    add_date_top_axis(axes[0], x, times, xmin, xmax)

    t0, t1 = times[0].strftime("%Y-%m-%d"), times[-1].strftime("%Y-%m-%d")
    savefig(fig, f"{config.SEAL_ID}_sic_t2m_fluxes_temp_salt_buoy_bx_N2_w_VHT_0_{DEP_MAX}_{t0}_{t1}.png")
    plt.close(fig)


def main():
    use_font_size(18)
    ds = io.load_omega()
    era5_smooth = io.rolling_mean(io.load_era5(), FLUX_HOURS)
    bounds = config.BIWEEKLY_BOUNDS
    for start, end in zip(bounds[:-1], bounds[1:]):
        plot_window(ds, era5_smooth, start, end)


if __name__ == "__main__":
    main()
