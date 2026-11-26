"""ED Fig. 3 (SI Figure S3): Monte Carlo uncertainty of strain, w and VHT.

(a) Along-track strain realizations: ensemble median, 5-95 % range and the
    reference strain, with the thick- and thin-ice periods shaded.
(b) w_rms, (c) VHT_rms and (d) mean VHT profiles for the thick-ice, thin-ice
    and full-trajectory periods: ensemble median (solid), 5-95 % range
    (shading) and reference solution (dashed). Statistics use sampled points
    only (maskprof > 0); the ensemble includes the reference realization.

Inputs : config.OMEGA_NC, config.OMEGA_MC_NC (scripts/06_omega_vht.py)
Output : FIG_DIR/<SEAL_ID>_MC_summary.png

Usage: python scripts/figS_mc_summary.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from sealfronts import config, omega  # noqa: E402

DEP_MIN, DEP_MAX = 0, 500             # depth range of panels b-d [m]
THIN_ICE_DEP_MAX = 300                # thin-ice profiles limited by sampling [m]
COLORS = {
    "Thick ice": "#0072B2",
    "Thin ice": "#D55E00",
    "Full trajectory": "#009E73",
}
SECONDS_PER_DAY = 86400


def load_ensemble():
    """MC ensemble with time coordinate, sampling mask and reference fields."""
    ref = xr.open_dataset(config.OMEGA_NC)
    ds = xr.open_dataset(config.OMEGA_MC_NC).assign_coords(time=ref["time"])
    ds["maskprof"] = ref["maskprof"]
    # reference (dashed) solution = control member of the ensemble
    ds["w"] = ds["w_realization"].sel(realization=omega.CONTROL_STRAIN, drop=True)
    ds["VHF"] = ds["VHF_realization"].sel(realization=omega.CONTROL_STRAIN, drop=True)
    return ds, ref["ess_smooth"]


def plot_strain_panel(ax, ds, reference_strain):
    """Panel a: strain ensemble envelope and the two regimes."""
    strain_mc = ds["strain_rate_realization"].drop_sel(
        realization=omega.CONTROL_STRAIN)
    x_km = strain_mc["distance"].values / 1000.0

    ax.fill_between(x_km, strain_mc.quantile(0.05, "realization").values,
                    strain_mc.quantile(0.95, "realization").values,
                    color="0.75", alpha=0.50, label="MC 5–95%")
    ax.plot(x_km, strain_mc.median("realization").values,
            color="k", lw=1.5, label="MC median")
    ax.plot(reference_strain["distance"].values / 1000.0,
            reference_strain.values, color="tab:blue", lw=0.7, alpha=0.9,
            label="Original strain")
    ax.axhline(0, color="0.4", lw=0.8)

    y0, y1 = ax.get_ylim()
    for label in ("Thick ice", "Thin ice"):
        period = omega.select_period(ds, omega.PERIODS[label])
        x_start = float(period.distance.isel(time=0)) / 1000.0
        x_end = float(period.distance.isel(time=-1)) / 1000.0
        ax.axvspan(x_start, x_end, color=COLORS[label], alpha=0.12, lw=0)
        ax.text(0.5 * (x_start + x_end), y1 + 0.02 * (y1 - y0), label,
                color=COLORS[label], ha="center", va="bottom", fontsize=10,
                clip_on=False)

    ax.set_xlabel("Along-track distance [km]")
    ax.set_xlim(0, 3880)
    ax.set_ylabel(r"Strain [s$^{-1}$]")
    ax.grid(True)
    ax.legend(loc="best")


def plot_profile(ax, depth, ens, ref, color, scale=1.0, label=None):
    """Ensemble median, 5-95 % range and reference profile."""
    ax.fill_betweenx(depth, ens.quantile(0.05, "realization") * scale,
                     ens.quantile(0.95, "realization") * scale,
                     color=color, alpha=0.20)
    ax.plot(ens.median("realization") * scale, depth, color=color, lw=2.5,
            label=label)
    ax.plot(ref * scale, depth, color=color, lw=1.0, ls="--", alpha=0.9)


def main():
    ds_mc, reference_strain = load_ensemble()

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.6], hspace=0.30,
                          wspace=0.25)
    axa = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[1, 1], sharey=axb)
    axd = fig.add_subplot(gs[1, 2], sharey=axb)

    plot_strain_panel(axa, ds_mc, reference_strain)

    for label, time_slice in omega.PERIODS.items():
        dep_max = THIN_ICE_DEP_MAX if label == "Thin ice" else DEP_MAX
        ds = omega.select_period(ds_mc.sel(depth=slice(DEP_MIN, dep_max)),
                                 time_slice)
        mask = ds["maskprof"] > 0
        w_mc, vht_mc = ds["w_realization"].where(mask), ds["VHF_realization"].where(mask)
        w_ref, vht_ref = ds["w"].where(mask), ds["VHF"].where(mask)
        along = "distance" if "distance" in w_mc.dims else "time"

        def rms(da):
            return np.sqrt((da ** 2).mean(dim=along, skipna=True))

        color = COLORS[label]
        plot_profile(axb, ds.depth, rms(w_mc), rms(w_ref), color,
                     scale=SECONDS_PER_DAY, label=label)
        plot_profile(axc, ds.depth, rms(vht_mc), rms(vht_ref), color)
        plot_profile(axd, ds.depth, vht_mc.mean(dim=along, skipna=True),
                     vht_ref.mean(dim=along, skipna=True), color)

    axb.set_title(r"$w_{\rm rms}$")
    axc.set_title(r"VHT$_{\rm rms}$")
    axd.set_title(r"$\langle$VHT$\rangle$")
    axb.set_xlabel(r"[m day$^{-1}$]")
    axc.set_xlabel(r"[W m$^{-2}$]")
    axd.set_xlabel(r"[W m$^{-2}$]")
    axb.set_ylabel("Depth [m]")
    for ax in (axb, axc, axd):
        ax.grid(True)
        ax.set_ylim(DEP_MAX, DEP_MIN)
    axb.legend()

    for ax, letter in [(axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")]:
        ax.text(0.0, 1.08, letter, transform=ax.transAxes, fontweight="bold",
                fontsize=14, va="bottom", ha="left", clip_on=False)

    plt.tight_layout()
    outname = config.FIG_DIR / f"{config.SEAL_ID}_MC_summary.png"
    fig.savefig(outname, dpi=300, bbox_inches="tight")
    print(f"saved {outname}")


if __name__ == "__main__":
    main()
