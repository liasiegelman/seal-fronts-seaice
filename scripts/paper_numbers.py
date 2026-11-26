"""Recompute the quantitative statements of the paper derived from the seal
section and omega-equation output, and print them next to the value quoted
in the manuscript (main text Results, Methods "Seal data", figure captions).

Numbers from the Monte Carlo analysis, the SIC/SAR map (Fig. 1) and the raw
seal track before gridding (dive spacing) are produced by other scripts.

Usage: python scripts/paper_numbers.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from sealfronts import config, io

BX_STRONG = 1e-6           # s-2, "strong" buoyancy gradient (Fig. 4)
LATE_SUMMER = ("2024-01-20", "2024-02-11 23:59")   # Results, Fig. 5 paragraph
PRE_ONSET_DAYS = 10        # days before ICE_GROWTH_ONSET used for "before" values

rows = []


def report(section, quantity, text, value):
    """Store one line of the comparison table."""
    rows.append((section, quantity, text, value))


def fmt(x, nd=2):
    """Fixed-point formatting with ``nd`` decimals."""
    return f"{x:.{nd}f}"


def masked_values(ds, var, dep_max=None):
    """Finite values of a masked field (optionally above ``dep_max``), flattened."""
    da = io.masked(ds, var)
    if dep_max is not None:
        da = da.sel(depth=slice(0, dep_max))
    v = da.values.ravel()
    return v[np.isfinite(v)]


def layer_abs_bx(ds, layer):
    """|b_x| within (``"ml"``) or below (``"below"``) the mixed layer."""
    bx = io.masked(ds, "bx")
    mld = io.mld(ds)
    depth = bx["depth"].broadcast_like(bx)
    bx = bx.where(depth <= mld) if layer == "ml" else bx.where(depth > mld)
    v = np.abs(bx.values.ravel())
    return v[np.isfinite(v) & (v > 1e-12)]


def period_stats(era5, key, sub):
    """Statistics quoted for one regime (Fig. 3 and associated text)."""
    name = {"thick": "Thick ice front", "thin": "Thin ice", "full": "Full trajectory"}[key]
    times = pd.to_datetime(sub.time.values)
    t2m = io.interp_to_times(era5["t2m"], times)
    sic = io.surface_sic(sub)
    mld = io.mld(sub).values

    w = masked_values(sub, "w") * 86400.0
    vht = masked_values(sub, "VHF")
    k_w = kurtosis(w, fisher=False, bias=False)
    k_vht = kurtosis(vht, fisher=False, bias=False)
    quoted_k = {"thick": ("4.37", "10.11"), "thin": ("5.16", "23.17"), "full": ("-", "-")}[key]
    report(name, "kurtosis of w (Pearson, all depths)", quoted_k[0], fmt(k_w))
    report(name, "kurtosis of VHT (Pearson, all depths)", quoted_k[1], fmt(k_vht))
    if key == "full":
        return

    if key == "thick":
        report(name, "SIC range (min-max)", "~0.3-0.85", f"{sic.min():.2f}-{sic.max():.2f}")
        report(name, "ERA5 t2m range [degC] (hourly)", "+1 to -3", f"{t2m.max():+.1f} to {t2m.min():+.1f}")
        report(name, "mean MLD [m]", "~20", fmt(np.nanmean(mld), 1))
        report(name, "|w| max [m/day]", "locally approach 50", fmt(np.nanmax(np.abs(w)), 1))
        report(name, "fraction of |w| > 20 m/day", "'often exceed +-20'",
               f"{100 * np.mean(np.abs(w) > 20):.1f}%")
        report(name, "VHT max / min [W m-2]", "exceeding +-1000",
               f"{vht.max():.0f} / {vht.min():.0f}")
        report(name, "fraction of |VHT| > 1000 W m-2", "-", f"{100 * np.mean(np.abs(vht) > 1000):.2f}%")
        bx = np.abs(masked_values(sub, "bx", 500))
        report(name, "|b_x| 90th / 99th percentile [s-2] (0-500 m)", "typical ~4e-7",
               f"{np.percentile(bx, 90):.1e} / {np.percentile(bx, 99):.1e}")
    else:
        report(name, "SIC range (min-max)", "0.2-0.6", f"{sic.min():.2f}-{sic.max():.2f}")
        report(name, "ERA5 t2m range [degC] (hourly)", "-3 to below -10",
               f"{t2m.max():+.1f} to {t2m.min():+.1f}")
        report(name, "mean MLD [m]", "~50", fmt(np.nanmean(mld), 1))
        report(name, "|w| max [m/day]", "at most 15", fmt(np.nanmax(np.abs(w)), 1))
        report(name, "|w| 99.9th percentile [m/day]", "-", fmt(np.percentile(np.abs(w), 99.9), 1))
        bx = np.abs(masked_values(sub, "bx", 300))
        report(name, "|b_x| 90th / 99th percentile [s-2] (0-300 m)", "up to +-4e-7",
               f"{np.percentile(bx, 90):.1e} / {np.percentile(bx, 99):.1e}")
        vht_ml = io.masked(sub, "VHF").where(sub.depth <= io.mld(sub)).values
        report(name, "max |VHT| within the mixed layer [W m-2]", "'does not penetrate the ML'",
               fmt(np.nanmax(np.abs(vht_ml)), 1))

    report(name, "time window (UTC)", "", f"{times[0]:%Y-%m-%d %H:%M} - {times[-1]:%Y-%m-%d %H:%M}")


def exceedance_numbers(sub_thick, sub_thin):
    """P(|b_x| > 1e-6) within / below the mixed layer (Fig. 4)."""
    p = {}
    for key, sub in (("thick", sub_thick), ("thin", sub_thin)):
        for layer in ("ml", "below"):
            v = layer_abs_bx(sub, layer)
            p[key, layer] = np.mean(v > BX_STRONG)
    report("Fig. 4", "P(|bx|>1e-6) within ML, thick ice", "~10%", f"{100 * p['thick', 'ml']:.2f}%")
    report("Fig. 4", "P(|bx|>1e-6) within ML, thin ice", "<1%", f"{100 * p['thin', 'ml']:.3f}%")
    report("Fig. 4", "P(|bx|>1e-6) below ML, thick / thin", "",
           f"{100 * p['thick', 'below']:.3f}% / {100 * p['thin', 'below']:.3f}%")
    report("Fig. 4", "ratio thick/thin of P(|bx|>1e-6) below ML", "~4",
           fmt(p["thick", "below"] / p["thin", "below"], 1))
    for key, sub in (("thick", sub_thick), ("thin", sub_thin)):
        v = layer_abs_bx(sub, "ml")
        report("Fig. 4", f"|bx| exceeded by 10% of ML obs., {key}", "(1e-6 for thick)",
               f"{np.percentile(v, 90):.1e}")


def mean_profile_numbers(ds):
    """Extrema of the mean VHT profiles (Fig. 5d)."""
    for reg in io.regimes():
        sub = io.subset_time(ds, reg["start"], reg["end"]).sel(depth=slice(0, reg["dep_max"]))
        prof = io.masked(sub, "VHF").mean("distance")
        z = sub.depth.values
        imax, imin = int(prof.argmax()), int(prof.argmin())
        if reg["key"] == "thick":
            report("Fig. 5d", "thick: max of <VHT> [W m-2] @ depth [m]", "~50 @ ~200 m",
                   f"{prof[imax].item():.1f} @ {z[imax]} m")
            report("Fig. 5d", "thick: min of <VHT> [W m-2] @ depth [m]", "~-50 @ ~300 m",
                   f"{prof[imin].item():.1f} @ {z[imin]} m")
        elif reg["key"] == "thin":
            report("Fig. 5d", "thin: range of <VHT> 0-300 m [W m-2]", "within +-10",
                   f"{prof.min().item():.1f} to {prof.max().item():.1f}")
        else:
            report("Fig. 5d", "full: range of <VHT> 0-500 m [W m-2]", "between end members",
                   f"{prof.min().item():.1f} to {prof.max().item():.1f}")


def time_series_numbers(ds, era5):
    """Statements about the full-trajectory time series (Fig. 5a-c)."""
    flux = io.rolling_mean(era5, 24)
    mld = io.rolling_mean(pd.Series(io.mld(ds).values, index=pd.to_datetime(ds.time.values))
                          .sort_index().to_frame("mld"), 24)["mld"]
    onset = pd.Timestamp(config.ICE_GROWTH_ONSET)

    ls = flux.loc[LATE_SUMMER[0]:LATE_SUMMER[1]]
    ls_mld = mld.loc[LATE_SUMMER[0]:LATE_SUMMER[1]]
    report("Fig. 5", "20 Jan-11 Feb: fraction of time Qnet(24h) > 0", "positive",
           f"{100 * np.mean(ls['qnet'] > 0):.0f}% (mean {ls['qnet'].mean():.0f} W m-2)")
    report("Fig. 5", "20 Jan-11 Feb: t2m(24h) mean [min, max] [degC]", "around 0",
           f"{ls['t2m'].mean():.1f} [{ls['t2m'].min():.1f}, {ls['t2m'].max():.1f}]")
    report("Fig. 5", "20 Jan-11 Feb: MLD(24h) median; % time < 20 m", "typically < 20 m",
           f"{ls_mld.median():.1f} m; {100 * np.mean(ls_mld < 20):.0f}%")

    before = flux.loc[onset - pd.Timedelta(days=PRE_ONSET_DAYS):onset]
    after = flux.loc[onset:]
    after_mld = mld.loc[onset:]
    report("Fig. 5", f"t2m(24h) {PRE_ONSET_DAYS} d before onset [min, max]", "~0-1",
           f"[{before['t2m'].min():.1f}, {before['t2m'].max():.1f}]")
    report("Fig. 5", "t2m(24h) minimum after onset", "below -8", fmt(after["t2m"].min(), 1))
    report("Fig. 5", "fraction of time Qnet(24h) < 0 after onset", "predominantly negative",
           f"{100 * np.mean(after['qnet'] < 0):.0f}%")
    report("Fig. 5", "mean THF(24h) before / after onset [W m-2]", "turbulent loss increases",
           f"{flux.loc[:onset, 'thf'].mean():.0f} / {after['thf'].mean():.0f}")
    report("Fig. 5", "MLD(24h) after onset, 5th-95th pct / max [m]", "40-80",
           f"{after_mld.quantile(0.05):.0f}-{after_mld.quantile(0.95):.0f} / "
           f"{after_mld.max():.0f}")

    # 3 February event: <VHT+>_0-200m (12 h mean) vs Qnet (24 h mean)
    vht = ds["VHF"].sel(depth=slice(0, 200))
    pos = pd.Series((vht.where(vht > 0).sum("depth") / vht.sizes["depth"]).values,
                    index=pd.to_datetime(ds.time.values)).sort_index()
    pos = pos.rolling("12h", center=True, min_periods=1).mean()
    day = slice("2024-02-03", "2024-02-03 23:59")
    report("Fig. 5", "3 Feb: max <VHT+>_0-200m (12h) vs max Qnet (24h) [W m-2]",
           "comparable", f"{pos.loc[day].max():.0f} vs {flux.loc[day, 'qnet'].max():.0f}")


def methods_numbers(ds):
    """Methods 'Seal data' and ED Fig. 2."""
    has_data = ds["maskprof"] > 0
    depth = ds.depth.broadcast_like(ds["maskprof"]).where(has_data)
    deepest = depth.max("depth").values
    times = pd.to_datetime(ds.time.values)
    report("Methods", "number of 1-km profiles", "3880", f"{ds.sizes['distance']}")
    report("Methods", "total along-track distance [km]", "3879 (cumulative, raw track)",
           fmt(float(ds.distance[-1]) / 1000, 0))
    report("Methods", "maximal depth [m]", "807", fmt(np.nanmax(deepest), 0))
    report("Methods", "median of profile max depth [m] (gridded)", "322 (raw dives)",
           fmt(np.nanmedian(deepest), 0))
    report("Methods", "first / last profile (UTC)",
           "01/12/2024 04:09:21 - 03/22/2024 05:19:52",
           f"{times[0]:%m/%d/%Y %H:%M:%S} - {times[-1]:%m/%d/%Y %H:%M:%S}")
    report("Methods", "duration [weeks]", ">10", fmt((times[-1] - times[0]).days / 7, 1))
    n = has_data.sum("distance")
    report("ED Fig. 2", "profiles reaching 300 / 500 / 600 m", "drops below ~500 m",
           f"{int(n.sel(depth=300))} / {int(n.sel(depth=500))} / {int(n.sel(depth=600))}")


def caption_numbers():
    report("Fig. 3 caption", "thick ice-front window", "1-5 February",
           f"{config.THICK_ICE_PERIOD[0]} - {config.THICK_ICE_PERIOD[1]}")
    report("Fig. 3 caption", "thin-ice window", "14-22 March",
           f"{config.THIN_ICE_PERIOD[0]} - {config.THIN_ICE_PERIOD[1]}")


def main():
    ds = io.load_omega().load()
    era5 = io.load_era5()
    subs = {r["key"]: io.subset_time(ds, r["start"], r["end"]) for r in io.regimes()}

    for key, sub in subs.items():
        period_stats(era5, key, sub)
    exceedance_numbers(subs["thick"], subs["thin"])
    mean_profile_numbers(ds)
    time_series_numbers(ds, era5)
    methods_numbers(ds)
    caption_numbers()

    table = pd.DataFrame(rows, columns=["where", "quantity", "paper", "computed"])
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 70,
                           "display.width", 250):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
