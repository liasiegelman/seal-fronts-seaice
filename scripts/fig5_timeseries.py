"""Fig. 5 -- surface forcing, upper-ocean heat transport and atmospheric
conditions along the full seal trajectory.

(a) ERA5 Qnet, Rnet, THF and the positive / negative VHT averaged over
0-200 m, with SIC on the right axis; (b) ERA5 t2m and MLD; (c) 10 m wind
vectors; (d) mean VHT profiles (+-1 standard error) for the thick ice-front
period, the thin-ice period and the full trajectory. Atmospheric variables
and MLD: 24 h rolling means; VHT: 12 h rolling mean.

Output: ``Figures/ts.png``
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sealfronts import config, io
from sealfronts.plotting import PERIOD_COLORS_CB, THF_COLOR, panel_label, savefig, use_font_size

VHT_DEPTH = (0, 200)        # m, depth range of <VHT+-> in panel a
FLUX_HOURS = 24             # h, rolling mean of ERA5 fields and MLD
VHT_HOURS = 12              # h, rolling mean of <VHT+->
WIND_ARROW_HOURS = 10       # h, one wind arrow per bin
WIND_REF_SPEED = 7          # m s-1, quiver key
ARROW_SCALE = 0.18 * (10 / 7)
PROFILE_DEP_MAX = 500       # m, depth axis of panel d
PROFILE_LABELS = {"thick": "Thick ice front", "thin": "Thin ice", "full": "Full trajectory"}


def vht_pos_neg(ds, dep_min, dep_max):
    """Depth averages of the positive and negative parts of VHT over a layer.

    Each part is summed over the layer and divided by the number of levels
    (missing levels count as zero), then smoothed with a VHT_HOURS rolling mean.
    """
    vht = ds["VHF"].sel(depth=slice(dep_min, dep_max))
    n = vht.sizes["depth"]
    df = pd.DataFrame(
        {
            "pos": (vht.where(vht > 0).sum("depth") / n).values,
            "neg": (vht.where(vht < 0).sum("depth") / n).values,
        },
        index=pd.to_datetime(ds["time"].values),
    ).sort_index()
    return io.rolling_mean(df, VHT_HOURS)


def mean_profile(ds, start, end, dep_max):
    """Mean VHT profile and its standard error over a period."""
    sub = io.subset_time(ds, start, end).sel(depth=slice(0, dep_max))
    vht = io.masked(sub, "VHF")
    mean = vht.mean("distance")
    sem = vht.std("distance") / np.sqrt(vht.count("distance"))
    return sub.depth.values, mean.values, sem.values


def wind_arrows(era5):
    """Wind arrows, one per WIND_ARROW_HOURS bin.

    Direction: bin mean of the FLUX_HOURS-smoothed (u10, v10); length: bin
    mean of the smoothed wind speed.
    """
    wind = io.rolling_mean(era5[["u10", "v10"]], FLUX_HOURS)
    wind["wspd"] = np.hypot(wind["u10"], wind["v10"])
    wind = wind.resample(f"{WIND_ARROW_HOURS}h").mean().dropna()
    norm = np.hypot(wind["u10"], wind["v10"])
    wind, norm = wind[norm > 0], norm[norm > 0]
    return (wind.index, (wind["u10"] / norm * wind["wspd"]).values,
            (wind["v10"] / norm * wind["wspd"]).values)


def main():
    use_font_size(10)
    ds = io.load_omega()
    era5 = io.load_era5()

    vht = vht_pos_neg(ds, *VHT_DEPTH)
    flux = io.rolling_mean(era5, FLUX_HOURS)
    mld = io.rolling_mean(
        pd.Series(io.mld(ds).values, index=pd.to_datetime(ds.time.values)).sort_index()
        .to_frame("mld"), FLUX_HOURS)["mld"]
    sic_series = pd.Series(io.surface_sic(ds, fill_gaps=False, clip=False),
                           index=pd.to_datetime(ds.time.values)).dropna()
    sic = np.clip(io.interp_to_times(sic_series, flux.index), 0, 1)

    fig = plt.figure(figsize=(21, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[4.2, 1.15], height_ratios=[1.5, 0.8, 0.8],
                          hspace=0.30, wspace=0.15)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax_prof = fig.add_subplot(gs[0:2, 1])
    for ax, lab in zip([ax1, ax2, ax3, ax_prof], "abcd"):
        panel_label(ax, lab, fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
    for ax in (ax1, ax2, ax3):
        ax.axvline(pd.Timestamp(config.ICE_GROWTH_ONSET), color="0.4", ls="--",
                   lw=1.5, zorder=50)

    # (a) fluxes, VHT and SIC
    sub = rf"{VHT_DEPTH[0]}-{VHT_DEPTH[1]}\,\mathrm{{m}}"
    ax1.plot(vht.index, vht["pos"], lw=2.4, c="#d73027",
             label=rf"$<\mathrm{{VHT}}^+>_{{{sub}}}$", zorder=21)
    ax1.plot(vht.index, vht["neg"], lw=2.4, c="#4575b4",
             label=rf"$<\mathrm{{VHT}}^->_{{{sub}}}$", zorder=22)
    ax1.plot(flux.index, flux["qnet"], lw=2.0, c="k", label="Qnet", zorder=20)
    ax1.plot(flux.index, flux["rnet"], lw=1.5, c="0.45", label="Rnet")
    ax1.plot(flux.index, flux["thf"], lw=1.5, c=THF_COLOR, label="THF")
    ax1.axhline(0, color="0.2", lw=1.2)
    ax1.grid(True)
    ax1.set_ylabel(r"Flux [W m$^{-2}$]", fontsize=11)
    ax1.set_ylim(-280, 320)
    ax1b = ax1.twinx()
    ax1b.fill_between(flux.index, 0, sic, color="0.7", alpha=0.4, label="SIC", zorder=0)
    ax1b.set_ylim(0, 1)
    ax1b.set_ylabel("SIC", fontsize=11)
    ax1b.tick_params(axis="y", labelsize=11)
    leg = ax1.legend(title=f"Flux {FLUX_HOURS} h; VHT {VHT_HOURS} h", frameon=True,
                     facecolor="white", edgecolor="None", framealpha=0.9, fontsize=9,
                     title_fontsize=9, loc="upper right")
    leg.set_zorder(100)
    ax1.add_artist(leg)
    leg2 = ax1.legend(*ax1b.get_legend_handles_labels(), frameon=True, facecolor="white",
                      edgecolor="None", framealpha=0.9, fontsize=9, loc="lower left")
    leg2.set_zorder(100)

    # (b) t2m and MLD
    ax2.plot(flux.index, flux["t2m"], lw=2.0, label="t2m", color="k")
    ax2.grid(True)
    ax2.set_ylim(-10, 1)
    ax2.set_ylabel(r"t2m [$^\circ$C]", fontsize=11)
    ax2b = ax2.twinx()
    ax2b.plot(mld.index, mld.values, lw=2.0, color="tab:orange", label="MLD")
    ax2b.invert_yaxis()
    ax2b.set_ylabel("MLD [m]", fontsize=11, color="tab:orange")
    ax2b.tick_params(axis="y", colors="tab:orange", labelsize=11)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, title=f"{FLUX_HOURS} h mean", fontsize=9,
               title_fontsize=9, loc="best")
    ax2.axhline(0, color="0.5", lw=1.2)

    # (c) wind vectors
    t, u, v = wind_arrows(era5)
    q = ax3.quiver(t, np.zeros(len(t)), u * ARROW_SCALE, v * ARROW_SCALE,
                   angles="uv", scale_units="xy", scale=1.0, width=0.001,
                   headwidth=2.5, headlength=2.2, headaxislength=2.0, minlength=0.05,
                   color="k", pivot="middle", zorder=5)
    ax3.axhline(0, color="0.5", lw=0.8)
    ax3.set_ylim(-2.5, 2.5)
    ax3.set_yticks([])
    ax3.set_ylabel("Wind", fontsize=11)
    ax3.grid(True, axis="x")
    ax3.quiverkey(q, X=0.05, Y=0.85, U=WIND_REF_SPEED * ARROW_SCALE,
                  label=rf"{WIND_REF_SPEED} m s$^{{-1}}$", labelpos="E",
                  coordinates="axes", fontproperties={"size": 9})
    ax3.legend([q], ["Wind vectors"], title=f"{FLUX_HOURS} h mean", fontsize=9,
               title_fontsize=9, loc="lower left")

    # (d) mean VHT profiles
    for reg in io.regimes():
        z, mean, sem = mean_profile(ds, reg["start"], reg["end"], reg["dep_max"])
        color = PERIOD_COLORS_CB[reg["key"]]
        ax_prof.plot(mean, z, lw=3, color=color, label=PROFILE_LABELS[reg["key"]])
        ax_prof.fill_betweenx(z, mean - sem, mean + sem, color=color, alpha=0.20)
    ax_prof.axvline(0, color="k", lw=1.2)
    ax_prof.grid(True, lw=0.5)
    ax_prof.set_ylim(PROFILE_DEP_MAX, 0)
    ax_prof.set_xlabel(r"$\langle$VHT$\rangle$ [W m$^{-2}$]", fontsize=11)
    ax_prof.set_ylabel("Depth [m]", fontsize=11)
    ax_prof.legend(title=r"$\langle$VHT$\rangle$ by period", frameon=True,
                   facecolor="white", framealpha=0.9, fontsize=9, title_fontsize=9,
                   loc="best")

    # time axis and highlighted regimes
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(pd.Timestamp(config.FULL_TRACK_START), flux.index.max())
    spans = [(config.THICK_ICE_PERIOD, "tab:purple", "Thick ice front"),
             (config.THIN_ICE_PERIOD, "tab:orange", "Thin ice")]
    for (start, end), color, text in spans:
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        for ax in (ax1, ax2, ax3):
            ax.axvspan(t0, t1, color=color, alpha=0.10)
        ax1.text(t0 + (t1 - t0) / 2, ax1.get_ylim()[1] * 1.08, text, ha="center",
                 va="top", fontsize=10, color=color,
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    ax1.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="x", labelbottom=False)

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.96)
    savefig(fig, "ts.png")


if __name__ == "__main__":
    main()
