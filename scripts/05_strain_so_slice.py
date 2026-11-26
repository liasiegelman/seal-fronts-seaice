"""SO-SLICE SLA-derived geostrophic strain along the seal track.

Input
    config.INTERP_NPZ        seal track (lon, lat, day)
    config.SO_SLICE_FILE     SO-SLICE daily SLA on a 0.25 deg grid ("hr" product,
                             50 km / 50 km / 10 d mapping scales)
Output
    config.STRAIN_NPZ
        sn, ss, strain       normal, shear strain and strain magnitude (s-1),
                             nearest grid cell and day to each profile
        theta, theta_smooth  local track heading (rad from east), raw and
                             smoothed (Gaussian, config.STRAIN_THETA_SMOOTH_SIGMA samples)
        ess, ess_smooth      along-track strain 0.5 sn cos(2 theta) + 0.5 ss sin(2 theta)
        day                  profile times

Processing
    1. keep SLA days spanning the track (+/- config.STRAIN_TIME_PAD_DAYS) inside
       the box config.STRAIN_CROP;
    2. fill SLA gaps (under sea ice) day by day with local optimal
       interpolation (config.STRAIN_OI);
    3. sn = -2 (g/f) d2SLA/dxdy, ss = (g/f)(d2SLA/dx2 - d2SLA/dy2);
    4. sample at the profiles (nearest neighbour; profiles before the first SLA
       day take the first day).

Usage
    python scripts/05_strain_so_slice.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr

from sealfronts import config
from sealfronts.along_track import (
    add_sla_geostrophic_strain,
    along_track_strain,
    load_track,
    normalize_longitude,
    oi_fill_sla,
    sample_nearest_sel,
    to_datetime64,
    track_angle,
)


def main():
    track = load_track(config.INTERP_NPZ)
    lon, lat, day = track["lon"], track["lat"], track["day"]
    time = to_datetime64(day)

    pad = np.timedelta64(int(config.STRAIN_TIME_PAD_DAYS), "D")
    t_min = time[~np.isnat(time)].min() - pad
    t_max = time[~np.isnat(time)].max() + pad

    ds = normalize_longitude(xr.open_mfdataset([config.SO_SLICE_FILE], combine="by_coords"))
    box = config.STRAIN_CROP
    ds = ds.sel(time=slice(t_min, t_max)).where(
        (ds["latitude"] < box["lat_max"])
        & (ds["longitude"] > box["lon_min"])
        & (ds["longitude"] < box["lon_max"]),
        drop=True,
    )
    if ds.sizes.get("time", 0) == 0:
        raise ValueError("No SLA times overlap the track time range.")
    print(f"SLA subset: {dict(ds.sizes)}")

    n_out = int(np.count_nonzero((time < ds["time"].values.min()) | (time > ds["time"].values.max())))
    if n_out:
        print(f"warning: {n_out} profiles outside the SLA time range; nearest end day used")

    print("OI gap filling of SLA")
    ds = ds.assign(sla=oi_fill_sla(ds["sla"], **config.STRAIN_OI))
    print("computing geostrophic strain")
    ds = add_sla_geostrophic_strain(ds)

    sampled = sample_nearest_sel(ds, ("sn", "ss", "strain"), lon, lat, time)
    sn, ss = sampled["sn"], sampled["ss"]

    theta = track_angle(lon, lat)
    theta_smooth = track_angle(lon, lat, smooth_sigma=config.STRAIN_THETA_SMOOTH_SIGMA)

    np.savez(
        config.STRAIN_NPZ,
        ss=ss,
        sn=sn,
        strain=sampled["strain"],
        ess=along_track_strain(sn, ss, theta),
        ess_smooth=along_track_strain(sn, ss, theta_smooth),
        theta=theta,
        theta_smooth=theta_smooth,
        day=day,
    )
    print(f"saved {config.STRAIN_NPZ}")


if __name__ == "__main__":
    main()
