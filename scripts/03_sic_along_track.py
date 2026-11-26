"""Sea-ice concentration along the seal track.

Input
    config.INTERP_NPZ                    seal track (lon, lat, day)
    config.SIC_DIR / sic_pss25_*_v06r00.nc
                                         NSIDC G02202 v6 daily SIC (25 km, south)
Output
    config.SIC_NPZ                       ``sic``: SIC (fraction 0-1) at each profile

For every profile, the SIC map of the nearest day is bilinearly interpolated
at the seal position (EPSG:3412 x/y). NaN where the grid is masked
(land/coast or no data).

Usage
    python scripts/03_sic_along_track.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr

from sealfronts import config
from sealfronts.along_track import load_track, sample_sic_along_track, to_datetime64


def main():
    files = sorted(config.SIC_DIR.glob(config.SIC_FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No SIC files {config.SIC_FILE_GLOB} in {config.SIC_DIR}")
    ds = xr.open_mfdataset(files, combine="by_coords", parallel=True, chunks={"time": 50})
    sic = ds["cdr_seaice_conc"]

    track = load_track(config.INTERP_NPZ)
    time = to_datetime64(track["day"])
    t0, t1 = sic["time"].values[[0, -1]]
    if time.min() < t0 - np.timedelta64(1, "D") or time.max() > t1 + np.timedelta64(1, "D"):
        print(f"warning: track ({time.min()} to {time.max()}) extends beyond SIC files ({t0} to {t1})")

    ses_sic = sample_sic_along_track(sic, track["lon"], track["lat"], time)
    np.savez(config.SIC_NPZ, sic=ses_sic)
    print(f"saved {config.SIC_NPZ}")


if __name__ == "__main__":
    main()
