"""Loading of the omega / VHT output and of the along-track ERA5 series.

The figure and statistics scripts only need two inputs:

* ``config.OMEGA_NC``: the along-track dataset written by the omega-equation
  stage (dims ``depth`` x ``distance``; 1 km x 1 m grid). Besides the
  hydrography (``temperature``, ``salinity``, ``buoyancy``, ``N2_smooth``,
  ``bx``, ``maskprof``) it carries the mixed-layer depth (``MLD``), the
  along-track sea-ice concentration (``sic``), the vertical velocity ``w``
  [m s-1] and the vertical heat transport ``VHF`` [W m-2] (called VHT in the
  paper).
* ``config.ERA5_NPZ``: ERA5 surface fields sampled along the seal track.
"""
import numpy as np
import pandas as pd
import xarray as xr

from sealfronts import config


def load_omega(path=None):
    """Open the omega / VHT dataset (lazily) with ``distance`` as along-track dim."""
    return xr.open_dataset(path or config.OMEGA_NC)


def subset_time(ds, start=None, end=None):
    """Return the profiles with ``start <= time <= end`` (inclusive).

    ``None`` bounds leave that side open. Works on the ``distance`` dimension
    (``time`` is a non-index coordinate along it).
    """
    t = pd.to_datetime(ds["time"].values)
    keep = np.ones(t.size, dtype=bool)
    if start is not None:
        keep &= t >= pd.Timestamp(start)
    if end is not None:
        keep &= t <= pd.Timestamp(end)
    return ds.isel({ds["time"].dims[0]: np.where(keep)[0]})


def masked(ds, var):
    """Field ``var`` with the points outside the seal profiles set to NaN."""
    return ds[var].where(ds["maskprof"] > 0)


def mld(ds):
    """Mixed-layer depth [m] as a 1-D series along track.

    MLD (0.03 kg m-3 density increase from 15 m, see Methods) is stored
    broadcast over depth in the omega file, masked where the profile has no
    data; it is read at 100 m, the level with the fewest gaps.
    """
    out = ds["MLD"]
    if "depth" in out.dims:
        out = out.sel(depth=100, method="nearest", drop=True)
    return out


def surface_sic(ds, fill_gaps=True, clip=True):
    """Along-track sea-ice concentration (1-D numpy array).

    Gaps are linearly interpolated over the profile index and values clipped
    to [0, 1], as in the published figures.
    """
    sic = ds["sic"]
    if "depth" in sic.dims:
        sic = sic.isel(depth=0)
    vals = np.asarray(sic.values, dtype=float)
    good = np.isfinite(vals)
    if fill_gaps and good.sum() >= 2:
        vals = np.interp(np.arange(vals.size), np.where(good)[0], vals[good])
    if clip:
        vals = np.clip(vals, 0, 1)
    return vals


def load_era5(path=None):
    """ERA5 along-track series as a time-indexed DataFrame.

    Columns: ``shf``, ``lhf`` (mean surface sensible / latent heat flux),
    ``lw_net``, ``sw_net`` (net long / short wave), ``t2m`` [deg C], ``u10``,
    ``v10`` [m s-1], and the derived ``thf = shf + lhf`` (turbulent heat
    flux), ``rnet = lw_net + sw_net`` and ``qnet = thf + rnet``. All fluxes are
    ERA5 mean rates [W m-2], positive downward (ocean/ice heat gain).
    """
    era = np.load(path or config.ERA5_NPZ, allow_pickle=True)
    df = pd.DataFrame(
        {
            "shf": np.asarray(era["msshf"], dtype=float),
            "lhf": np.asarray(era["mslhf"], dtype=float),
            "lw_net": np.asarray(era["msnlwrf"], dtype=float),
            "sw_net": np.asarray(era["msnswrf"], dtype=float),
            "t2m": np.asarray(era["t2m"], dtype=float),
            "u10": np.asarray(era["u10"], dtype=float),
            "v10": np.asarray(era["v10"], dtype=float),
        },
        index=pd.DatetimeIndex(pd.to_datetime(era["day"])),
    ).sort_index()
    if np.nanmean(df["t2m"]) > 100:  # Kelvin -> Celsius
        df["t2m"] = df["t2m"] - 273.15
    df["thf"] = df["shf"] + df["lhf"]
    df["rnet"] = df["lw_net"] + df["sw_net"]
    df["qnet"] = df["thf"] + df["rnet"]
    return df


def rolling_mean(df, hours):
    """Centred time-based rolling mean over ``hours`` (``None``: unchanged)."""
    if hours is None:
        return df
    return df.rolling(f"{hours}h", center=True, min_periods=1).mean()


def interp_to_times(series, times):
    """Linearly interpolate a time-indexed Series onto ``times``."""
    return np.interp(
        pd.DatetimeIndex(times).asi8,
        series.index.asi8,
        series.to_numpy(dtype=float),
    )


def regimes():
    """The three periods compared throughout the paper.

    Returns a list of dicts with ``key``, ``start``, ``end`` (``None`` = open)
    and the depth range over which mean profiles are shown (the thin-ice
    profiles rarely exceed 300 m, see ED Fig. 2).
    """
    return [
        dict(key="thick", start=config.THICK_ICE_PERIOD[0],
             end=config.THICK_ICE_PERIOD[1], dep_max=500),
        dict(key="thin", start=config.THIN_ICE_PERIOD[0],
             end=config.THIN_ICE_PERIOD[1], dep_max=300),
        dict(key="full", start=None, end=None, dep_max=500),
    ]
