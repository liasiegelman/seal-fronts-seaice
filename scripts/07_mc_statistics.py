"""Kurtosis of w and VHT and Monte Carlo profile statistics quoted in the paper.

Prints
  1. the kurtosis (Pearson, Gaussian = 3) of w and VHT of the reference
     solution over sampled points, for the thick-ice, thin-ice and
     full-trajectory periods (main text, "Results");
  2. the median and 5-95 % range of these kurtoses across the Monte Carlo
     ensemble;
  3. extrema of the ensemble-median w_rms, VHT_rms and mean-VHT profiles
     (0-500 m; thin ice 0-300 m) discussed in the SI text on ED Fig. 3.

Inputs: config.OMEGA_NC, config.OMEGA_MC_NC
Usage : python scripts/07_mc_statistics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from sealfronts import config, omega  # noqa: E402

DEP_MAX = {"Thick ice": 500, "Thin ice": 300, "Full trajectory": 500}  # [m]


def profile_extrema(ds_mc):
    """Depth and value of extrema of the ensemble-median profiles."""
    rows = []
    for label, tsl in omega.PERIODS.items():
        ds = omega.select_period(ds_mc.sel(depth=slice(0, DEP_MAX[label])), tsl)
        mask = ds["maskprof"] > 0
        w = ds["w_realization"].where(mask)
        vht = ds["VHF_realization"].where(mask)
        along = "distance" if "distance" in w.dims else "time"
        profs = {
            "w_rms [m/day]": np.sqrt((w ** 2).mean(along)) * 86400,
            "VHT_rms [W/m2]": np.sqrt((vht ** 2).mean(along)),
            "<VHT> [W/m2]": vht.mean(along),
        }
        for name, p in profs.items():
            med = p.median("realization")
            p05 = p.quantile(0.05, "realization")
            p95 = p.quantile(0.95, "realization")
            imax, imin = int(med.argmax()), int(med.argmin())
            rows.append({
                "period": label, "profile": name,
                "max_median": float(med[imax]), "z_max [m]": int(med.depth[imax]),
                "p95_at_max": float(p95[imax]),
                "min_median": float(med[imin]), "z_min [m]": int(med.depth[imin]),
                "p05_at_min": float(p05[imin]),
            })
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 200)
    ref = xr.open_dataset(config.OMEGA_NC)
    print("Kurtosis of the reference solution (sampled points only)")
    print(omega.kurtosis_periods(ref).round(2).to_string(index=False))

    ds_mc = xr.open_dataset(config.OMEGA_MC_NC).assign_coords(time=ref["time"])
    ds_mc["maskprof"] = ref["maskprof"]
    print(f"\nKurtosis across the {ds_mc.sizes['realization']} ensemble members")
    print(omega.mc_kurtosis_periods(ds_mc).round(2)
          .to_string(index=False))

    print("\nExtrema of ensemble-median profiles (5-95 % bounds at the extremum)")
    print(profile_extrema(ds_mc).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
