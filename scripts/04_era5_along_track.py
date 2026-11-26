"""ERA5 surface atmosphere and air-sea fluxes along the seal track.

Input
    config.INTERP_NPZ          seal track (lon, lat, day)
    config.ERA5_UNZIPPED_DIR   ERA5 hourly single-level netCDF files
                               (data_stream-oper_stepType-{avg,instant}.nc);
                               if absent, they are extracted from config.ERA5_ZIP
                               into WORK_DIR (see scripts/00_download_era5.py)
Output
    config.ERA5_NPZ            hourly ERA5 values at each profile from
                               config.ERA5_TRACK_START onwards (nearest
                               neighbour in time, latitude and longitude):

    u10, v10, wspd (m s-1), t2m (K)
    msshf, mslhf          sensible / latent heat flux, W m-2, ERA5 sign (positive downward)
    msdwlwrf, msdwswrf    downward long / shortwave radiation, W m-2
    msnlwrf, msnswrf      net long / shortwave radiation, W m-2 (positive downward)
    thf_down = msshf + mslhf
    rad_net  = msnlwrf + msnswrf
    qnet_down = rad_net + thf_down
    shf_up = -msshf, lhf_up = -mslhf, thf_up = shf_up + lhf_up  (positive = ocean heat loss)
    qnet_ocean_gain = rad_net - thf_up
    lon, lat, day         track positions/times of the retained profiles

Usage
    python scripts/04_era5_along_track.py
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr

from sealfronts import config
from sealfronts.along_track import load_track, sample_nearest_along_track

# ERA5 (new CDS) variable names -> names used in the output
RENAME = {
    "valid_time": "time",
    "avg_slhtf": "mslhf",
    "avg_ishf": "msshf",
    "avg_sdlwrf": "msdwlwrf",
    "avg_sdswrf": "msdwswrf",
    "avg_snlwrf": "msnlwrf",
    "avg_snswrf": "msnswrf",
}
VARIABLES = ["u10", "v10", "t2m", "msshf", "mslhf", "msdwlwrf", "msdwswrf", "msnlwrf", "msnswrf"]


def era5_files():
    """ERA5 netCDF files, extracting the CDS zip archive into WORK_DIR if needed."""
    files = sorted(config.ERA5_UNZIPPED_DIR.glob("*.nc"))
    if files:
        return files
    if not config.ERA5_ZIP.exists():
        raise FileNotFoundError(f"Neither {config.ERA5_UNZIPPED_DIR} nor {config.ERA5_ZIP} found")
    if not zipfile.is_zipfile(config.ERA5_ZIP):  # single plain netCDF file
        return [config.ERA5_ZIP]
    out_dir = config.WORK_DIR / config.ERA5_UNZIPPED_DIR.name
    with zipfile.ZipFile(config.ERA5_ZIP) as z:
        z.extractall(out_dir)
    return sorted(out_dir.glob("*.nc"))


def load_era5():
    """Open ERA5, rename variables, latitude increasing, longitude in [-180, 180)."""
    ds = xr.open_mfdataset(era5_files(), combine="by_coords")
    ds = ds.rename({k: v for k, v in RENAME.items() if k in ds.variables})[VARIABLES]
    if np.any(np.diff(ds.latitude.values) < 0):
        ds = ds.sortby("latitude")
    if ds.longitude.max() > 180:
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")
    return ds


def main():
    ds = load_era5()

    track = load_track(config.INTERP_NPZ)
    lon = track["lon"].astype(float)
    lat = track["lat"].astype(float)
    time = track["day"].astype("datetime64[ns]")

    keep = time >= np.datetime64(config.ERA5_TRACK_START)
    lon, lat, time = lon[keep], lat[keep], time[keep]

    buf = config.ERA5_CROP_BUFFER_DEG
    ds = ds.sel(
        longitude=ds.longitude[(ds.longitude >= lon.min() - buf) & (ds.longitude <= lon.max() + buf)],
        latitude=ds.latitude[(ds.latitude >= lat.min() - buf) & (ds.latitude <= lat.max() + buf)],
    )

    out = sample_nearest_along_track(ds, VARIABLES, lon, lat, time)
    out["wspd"] = np.sqrt(out["u10"] ** 2 + out["v10"] ** 2)

    # ERA5 convention: positive downward (into the ocean)
    out["thf_down"] = out["msshf"] + out["mslhf"]
    out["rad_net"] = out["msnlwrf"] + out["msnswrf"]
    out["qnet_down"] = out["rad_net"] + out["thf_down"]

    # Oceanographic convention: turbulent heat loss positive upward
    out["shf_up"] = -out["msshf"]
    out["lhf_up"] = -out["mslhf"]
    out["thf_up"] = out["shf_up"] + out["lhf_up"]
    out["qnet_ocean_gain"] = out["rad_net"] - out["thf_up"]

    order = ["u10", "v10", "wspd", "t2m", "msshf", "mslhf", "msdwlwrf", "msdwswrf",
             "msnlwrf", "msnswrf", "thf_down", "rad_net", "qnet_down",
             "shf_up", "lhf_up", "thf_up", "qnet_ocean_gain"]
    np.savez(config.ERA5_NPZ, **{k: out[k] for k in order}, lon=lon, lat=lat, day=time)
    print(f"saved {config.ERA5_NPZ}")


if __name__ == "__main__":
    main()
