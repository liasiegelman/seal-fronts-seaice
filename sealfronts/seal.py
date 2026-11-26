"""Preprocessing of the elephant-seal CTD profiles (MEOP high-resolution files).

Functions used by ``scripts/01_interpolate_seal.py`` (regular along-track grid,
buoyancy gradients, Ertel PV, Ri^-1, MLD) and ``scripts/02_clean_seal.py``
(salinity spike QC, re-computed and smoothed N2).

Conventions
-----------
* The vertical axis is the pressure-bin index of the MEOP files (1 dbar bins,
  0..999), called ``depth`` [m] in the outputs.
* The along-track grid is regular in cumulative distance: first a 100 m grid
  (10 points per km), then sub-sampled every 1 km after a 1-km running mean.
* 2-D arrays are (depth, profile) unless stated otherwise.
"""
from itertools import takewhile

import gsw
import numpy as np
import pandas as pd
import xarray as xr
from astropy.time import Time
from geopy.distance import geodesic
from netCDF4 import Dataset
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage import gaussian_filter

from sealfronts import config

JULD_REF = 2433282.5   # Julian day of the MEOP/Argo time origin (1950-01-01)
NPTS_PER_KM = 10       # points per km of the fine along-track grid (100 m)


# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------
def raw_nc_path(tag):
    """Raw MEOP profile file of a tag (``config.SEAL_NC`` for the paper seal)."""
    return config.SEAL_DIR / f"{tag}_fr1_prof.nc"


def interp_npz_path(tag):
    """Interpolated-section file of a tag (``config.INTERP_NPZ`` for the paper seal)."""
    return config.WORK_DIR / f"interpolated_data_{tag}_fr1_prof.npz"


def sic_npz_path(tag):
    """Along-track sea-ice concentration file of a tag (``config.SIC_NPZ``)."""
    return config.WORK_DIR / f"sic_{tag}.npz"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def gaussian_filter_nan(a, sigma, mode="nearest", truncate=4.0):
    """NaN-safe Gaussian filter (normalised convolution).

    NaNs are given zero weight; points whose kernel contains no valid data
    are returned as NaN.

    Parameters
    ----------
    a : array_like
        Input array, may contain NaNs.
    sigma : float or sequence
        Gaussian standard deviation(s) in grid points.
    mode, truncate :
        Passed to :func:`scipy.ndimage.gaussian_filter`.
    """
    a = np.asarray(a, dtype=float)
    valid = np.isfinite(a).astype(float)
    a_filled = np.where(np.isfinite(a), a, 0.0)

    num = gaussian_filter(a_filled, sigma=sigma, mode=mode, truncate=truncate)
    den = gaussian_filter(valid, sigma=sigma, mode=mode, truncate=truncate)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def running_mean_edge_padded(x, n):
    """Centred n-point running mean of a 1-D series, padded with its end values.

    The series is padded with ``n`` copies of its first/last value and the
    full convolution is cropped back to ``len(x)`` (same indexing as the
    original analysis code).
    """
    xx = np.concatenate((np.repeat(x[0], n), x, np.repeat(x[-1], n)))
    start = int(np.floor((n - 1) + n / 2 - 1 / 2))
    stop = int(np.floor(-n - 0.5 - n / 2))
    return np.convolve(xx, np.ones((n,)) / n)[start:stop]


def juld_to_datetime(juld):
    """Convert MEOP JULD (days since 1950-01-01) to an array of datetimes."""
    t = Time(juld + JULD_REF, format="jd", scale="utc")
    t.format = "iso"
    return t.to_datetime()


# ---------------------------------------------------------------------------
# Raw profiles
# ---------------------------------------------------------------------------
def load_profiles(path, adjusted=True):
    """Read a MEOP ``*_prof.nc`` file.

    Profiles with the 99999 missing-location flag are removed.

    Parameters
    ----------
    path : path-like
        MEOP netCDF file.
    adjusted : bool
        Use the delayed-mode ``*_ADJUSTED`` variables (True, used for all the
        science) or the raw ones (False, used only to rebuild the depth mask).

    Returns
    -------
    dict with masked arrays ``lat``, ``lon``, ``juld`` (n_prof,) and
    ``pres``, ``temp``, ``psal`` (n_prof, n_levels).
    """
    sfx = "_ADJUSTED" if adjusted else ""
    with Dataset(path) as nc:
        lat = nc.variables["LATITUDE"][:]
        lon = nc.variables["LONGITUDE"][:]
        juld = nc.variables["JULD_LOCATION"][:]
        pres = nc.variables["PRES" + sfx][:]
        temp = nc.variables["TEMP" + sfx][:]
        psal = nc.variables["PSAL" + sfx][:]

    good = np.argwhere(lat != 99999).squeeze()
    return dict(lat=lat[good], lon=lon[good], juld=juld[good],
                pres=pres[good, :], temp=temp[good, :], psal=psal[good, :])


def along_track_distance(lat, lon):
    """Geodesic distance between consecutive profiles and its cumulative sum [km].

    Steps involving a NaN position are set to 0.
    """
    dist = np.zeros(len(lat))
    for k in range(len(lat) - 1):
        if (~np.isnan(lat[k]) and ~np.isnan(lon[k])
                and ~np.isnan(lat[k + 1]) and ~np.isnan(lon[k + 1])):
            dist[k + 1] = geodesic((lat[k], lon[k]), (lat[k + 1], lon[k + 1])).km
    return dist, dist.cumsum(axis=0)


def profile_depth_mask(prof_raw, distcum, pmax=1001):
    """Depth mask on the 100 m along-track grid (1 above the deepest valid sample, NaN below).

    Rebuilds the profile depth mask of the original analysis, which uses the
    *non-adjusted* PRES/TEMP/PSAL. For each grid point, the nearest profile
    gives the maximum depth (last level where buoyancy is finite).

    Parameters
    ----------
    prof_raw : dict
        Output of :func:`load_profiles` with ``adjusted=False``.
    distcum : ndarray
        Cumulative along-track distance of the profiles [km].
    """
    xmin = 0
    xmax = int(np.ceil(np.max(distcum)))
    sa = prof_raw["psal"][:, :pmax]
    ct = prof_raw["temp"][:, :pmax]
    grav = gsw.grav(prof_raw["lat"], prof_raw["pres"].T)[:pmax, :]
    rho_ano = (gsw.sigma0(sa, ct) + 1000) / 1025
    b = grav * (1 - rho_ano.T)

    # deepest valid level of each profile
    c = pd.DataFrame(b).apply(pd.Series.last_valid_index)
    c = c.replace(to_replace=[None], value=np.nan)

    # NB: linspace, so the spacing is only approximately 100 m (kept as is,
    # the interpolation step indexes this grid as if it were exactly 100 m).
    dist1 = np.linspace(xmin, xmax, (xmax - xmin + 1) * NPTS_PER_KM)
    m_prof = np.full((rho_ano.shape[1], dist1.shape[0]), np.nan)
    for k in range(len(dist1)):
        prof_max = c[np.abs(dist1[k] - distcum).argmin()]
        if np.isfinite(prof_max):
            m_prof[0:int(prof_max), k] = 1
    return m_prof


# ---------------------------------------------------------------------------
# Interpolation onto the regular along-track grid
# ---------------------------------------------------------------------------
def _regular_segment(distcum_valid, xmin):
    """Integer-km-bounded 100 m grid spanning the valid profiles, and its column slice."""
    dist = np.arange(int(np.ceil(distcum_valid[0])),
                     int(np.floor(distcum_valid[-1])) + 0.1, 0.1)
    i0 = int(NPTS_PER_KM * (dist[0] - xmin))
    return dist, slice(i0, i0 + len(dist))


def interp_to_regular(distcum, values, valid, xmin, ngrid):
    """Linear interpolation of each depth level onto the 100 m along-track grid.

    For each level, only profiles where ``valid`` is True are used, and the
    output is filled between the first and last of them (rounded inwards to
    whole km); elsewhere it is NaN. Levels with fewer than 3 valid profiles
    are left NaN.

    Parameters
    ----------
    distcum : (n_prof,) cumulative distance [km]
    values : (n_prof,) or (n_depth, n_prof) field to interpolate
        A 1-D field (e.g. Coriolis f) is interpolated with each level's mask.
    valid : (n_depth, n_prof) bool
    xmin : int, start of the grid [km]
    ngrid : int, number of grid points

    Returns
    -------
    out : (n_depth, ngrid) interpolated field
    grid : (n_depth, ngrid) grid distance where ``out`` is defined, NaN elsewhere
    """
    nz = valid.shape[0]
    out = np.full((nz, ngrid), np.nan)
    grid = np.full((nz, ngrid), np.nan)
    for j in range(nz):
        i = np.argwhere(np.isfinite(distcum) & valid[j, :]).squeeze()
        if i.size > 2:
            dist, sl = _regular_segment(distcum[i], xmin)
            v = values[i] if values.ndim == 1 else values[j, i]
            out[j, sl] = InterpolatedUnivariateSpline(distcum[i], v, k=1)(dist)
            grid[j, sl] = dist
    return out, grid


def running_mean_rows(a, n, valid, min_valid=3):
    """Apply :func:`running_mean_edge_padded` to the valid points of each row.

    Rows with fewer than ``min_valid`` valid points are left NaN.
    """
    out = np.full(a.shape, np.nan)
    for j in range(len(a)):
        i = np.argwhere(valid[j, :]).squeeze()
        if i.size >= min_valid:
            out[j, i] = running_mean_edge_padded(a[j, i].squeeze(), n)
    return out


def mixed_layer_depth(bK, grav_surf, m_profK, thres=0.03, ref_level=7):
    """Mixed-layer depth from a 0.03 kg m^-3 density threshold.

    The reference is the buoyancy at ``ref_level`` (7 m); the MLD is the
    first level below it where b drops by more than g*thres/1025. Profiles
    whose MLD reaches the bottom of the profile (within 1 m) are set to NaN.
    """
    def n_above(col, b_ref):
        return len([x for x in takewhile(lambda x: x[1] > b_ref, enumerate(col))])

    b_surf = bK[ref_level, :]
    delta = -grav_surf * thres / 1025
    mld = np.full(b_surf.shape[0], np.nan)
    for j in range(b_surf.shape[0]):
        s = n_above(bK[:, j], b_surf[j] + delta[j])
        if s != bK.shape[0] and np.isfinite(bK[s, j]):
            mld[j] = int(s + ref_level)

    c = pd.DataFrame(m_profK).apply(pd.Series.last_valid_index)  # profile max depth
    for j in range(len(mld)):
        if np.isfinite(mld[j]) and mld[j] >= c[j] - 1:
            mld[j] = np.nan
    return mld


def smooth_n2_section(n2K, sigma=4):
    """2-D NaN-safe Gaussian smoothing of an N2 section (1 km x 1 m grid).

    Each row is first extended by sigma**2 points of edge padding (written at
    the row's valid indices plus 2*sigma**2 points beyond the last one, i.e.
    shifted by sigma**2 columns) and the padding is cropped afterwards, as in
    the original analysis.
    """
    K = sigma ** 2
    n2_new = np.full((n2K.shape[0], n2K.shape[1] + 2 * K), np.nan)
    for j in range(len(n2K)):
        i = np.argwhere(np.isfinite(n2K[j, :]))
        if len(i) > 1:
            bb = np.concatenate((np.repeat(n2K[j, i[0]], K), n2K[j, i].squeeze(),
                                 np.repeat(n2K[j, i[-1]], K))).squeeze()
            n2_new[j, np.hstack((i.squeeze(), np.arange(1, 2 * K + 1) + i[-1]))] = bb
    return gaussian_filter_nan(n2_new, sigma=sigma)[:, K:-K]


def interpolate_section(prof, m_prof, pmin=0, pmax=1000, n_ma=10, m_ma=3):
    """Interpolate a seal record onto a regular 1 km x 1 m section and derive dynamics.

    Steps: buoyancy b = g (1 - rho/1025); linear interpolation of b, CT, SA,
    N2, f, g onto a 100 m along-track grid; 1-km running mean of b and
    sub-sampling every km; MLD; smoothed N2; lateral buoyancy gradient b_x;
    thermal-wind velocity v (integrated upward from the bottom, assuming zero
    velocity at the deepest level), its running mean and dv/dx; Ertel PV
    q = (f + v_x) N2 - b_x^2 / f; and smoothed inverse Richardson number
    Ri^-1 = M4 / (f^2 N2).

    Parameters
    ----------
    prof : dict
        Output of :func:`load_profiles` (adjusted variables).
    m_prof : (n_levels, n_grid) depth mask from :func:`profile_depth_mask`.
    pmin, pmax : vertical index range kept.
    n_ma : running-mean window for b, in 100 m points (10 = 1 km); also the
        sub-sampling step.
    m_ma : running-mean window for v, in 1 km points.

    Returns
    -------
    dict of arrays saved to the interpolated npz file.
    """
    lat, lon, days = prof["lat"], prof["lon"], prof["juld"]
    pres, ct, sa = prof["pres"], prof["temp"], prof["psal"]

    _, distcum = along_track_distance(lat, lon)

    # Whole record: from 0 km to the last profile, rounded up to a whole km.
    if np.any(np.diff(days) <= 0):
        raise ValueError("profile times are not strictly increasing")
    xmin = int(np.floor(distcum[0]))
    xmax = int(np.ceil(distcum[-1]))
    ind_min = np.abs(distcum - xmin).argmin()
    ind_max = np.abs(distcum - xmax).argmin()
    sel = slice(ind_min, ind_max + 1)
    ngrid = NPTS_PER_KM * (xmax - xmin + 1)

    pres, ct, sa = pres[sel, pmin:pmax], ct[sel, pmin:pmax], sa[sel, pmin:pmax]
    distcum, lat, lon, days = distcum[sel], lat[sel], lon[sel], days[sel]
    m_prof = m_prof[pmin:pmax, NPTS_PER_KM * xmin:NPTS_PER_KM * xmin + ngrid]

    f = gsw.f(lat)
    grav = gsw.grav(lat, pres.T)
    N2, _ = gsw.Nsquared(sa.T, ct.T, pres.T, lat)
    # N2 is defined at mid-levels (n-1 values); the original code completes it
    # to n levels by prepending its *last* (deepest) row on top. Kept for
    # reproducibility; it only affects the top 1 m before smoothing.
    N2 = np.vstack((N2[-1, :], N2))

    # NB: the analysis uses PSAL/TEMP directly as SA/CT in gsw.
    rho_ano = (gsw.sigma0(sa.T, ct.T) + 1000) / 1025
    b = grav * (1 - rho_ano)

    # --- interpolation onto the 100 m grid ---------------------------------
    valid_rho = np.isfinite(rho_ano)
    f_reg, distreg = interp_to_regular(distcum, f, valid_rho, xmin, ngrid)
    grav_reg, _ = interp_to_regular(distcum, grav, valid_rho, xmin, ngrid)

    valid_b = np.isfinite(b)
    b_r, _ = interp_to_regular(distcum, b, valid_b, xmin, ngrid)
    ct_reg, _ = interp_to_regular(distcum, ct.T, valid_b, xmin, ngrid)
    sa_reg, _ = interp_to_regular(distcum, sa.T, valid_b, xmin, ngrid)

    n2_reg, _ = interp_to_regular(distcum, N2, np.isfinite(N2), xmin, ngrid)

    # --- 1 km running mean of b, then keep one point per km -----------------
    b_ma = running_mean_rows(b_r, n_ma, np.isfinite(distreg) & np.isfinite(b_r))

    bK = b_ma[:, ::n_ma]
    fK = f_reg[:, ::n_ma]
    n2K = n2_reg[:, ::n_ma]
    ctK = ct_reg[:, ::n_ma]
    saK = sa_reg[:, ::n_ma]
    m_profK = m_prof[:, ::n_ma]

    # positions/times on the 1 km grid
    dkm = np.arange(0, bK.shape[1])
    lon_reg = InterpolatedUnivariateSpline(distcum, lon, k=1)(dkm)
    lat_reg = InterpolatedUnivariateSpline(distcum, lat, k=1)(dkm)
    days_reg = InterpolatedUnivariateSpline(distcum, days, k=1)(dkm)
    day_reg = juld_to_datetime(days_reg)

    mld = mixed_layer_depth(bK, grav_reg[0, ::n_ma], m_profK)

    # --- smoothed stratification (negative N2 set to 1e-8) ------------------
    N2_smooth = smooth_n2_section(n2K, sigma=4)
    N2_smooth[N2_smooth < 0] = 1e-8

    # --- gradients (grid step 1 km = 1000 m) --------------------------------
    dx = n_ma * 100  # [m]
    dbdx = np.gradient(bK, n_ma / 10, axis=1) / dx

    # thermal wind, integrated from the bottom (level of no motion)
    v = np.flipud(np.nancumsum(np.flipud(dbdx / fK), axis=0))
    v_ma = running_mean_rows(v, m_ma, np.isfinite(v), min_valid=2)
    dvdx = np.gradient(v_ma, n_ma / 10, axis=1) / dx

    M4 = abs(dbdx) ** 2
    qertel = (fK + dvdx) * N2_smooth - M4 / fK

    # inverse Richardson number, smoothed with a 2-D Gaussian (NaN -> 0)
    inv_ri = (fK ** 2 * N2_smooth / M4) ** (-1)
    inv_ri[~np.isfinite(inv_ri)] = 0
    inv_ri_smooth = gaussian_filter(inv_ri, 0.7)

    return dict(lon=lon_reg, lat=lat_reg, day=day_reg, sa=saK, ct=ctK,
                mask_prof=m_profK, b=bK, f=fK, N2=N2_smooth, mld=mld,
                bx=dbdx, M4=M4, qertel=qertel, inv_ri=inv_ri_smooth)


# ---------------------------------------------------------------------------
# Dataset assembly and QC (02_clean_seal.py)
# ---------------------------------------------------------------------------
# npz key -> dataset variable name, (depth, time) fields
_SECTION_VARS = {
    "sa": "salinity", "mask_prof": "maskprof", "b": "buoyancy", "f": "f",
    "N2": "N2", "bx": "bx", "M4": "M4", "qertel": "qErtel", "inv_ri": "inv_Ri",
}


def build_tag_dataset(tag, interp_npz, sic_npz):
    """Assemble the interpolated section and along-track SIC of one tag into an xarray Dataset."""
    d = np.load(interp_npz, allow_pickle=True)
    sic = np.load(sic_npz, allow_pickle=True)["sic"]
    mld = d["mld"]

    ds = xr.Dataset({"temperature": (("depth", "time"), d["ct"])})
    ds.coords["time"] = d["day"]
    ds.coords["depth"] = np.arange(0, d["ct"].shape[0])
    ds = ds.assign_coords(lat=("time", d["lat"]), lon=("time", d["lon"]),
                          name=("time", [tag] * len(mld)))

    parts = [ds]
    for key, var in _SECTION_VARS.items():
        parts.append(xr.Dataset({var: (("depth", "time"), d[key])}))
        if var == "N2":
            parts.append(xr.Dataset({"MLD": (("time",), mld)}))
    parts.append(xr.Dataset({"sic": (("time",), sic)}))
    return xr.merge(parts, combine_attrs="drop_conflicts")


def select_under_ice(ds, sic_min=0.02, lat_max=-64.6):
    """Keep the profiles in sea ice (SIC >= sic_min) or south of lat_max."""
    return ds.where((ds["sic"] >= sic_min) | (ds["lat"] <= lat_max), drop=True)


def qc_salinity_spikes(ds, sal_var="salinity", time_window=5, depth_window=5,
                       thresh=0.4, shallow_limit=7, fresh_limit=30,
                       expand_time=3, expand_down=2):
    """Remove salinity spikes based on local consistency in time and depth.

    A point is flagged when it departs by more than ``thresh`` from the
    running median along time or along depth. QC is only applied in the
    upper ``shallow_limit`` m or where salinity < ``fresh_limit``. The flag
    is then expanded by +-``expand_time`` profiles and, iteratively, by
    ``expand_down`` levels up and down (net reach +-3 levels for 2). Flagged
    points are set to NaN in *all* variables (1-D variables such as MLD and
    sic therefore become (time, depth)).

    Defaults are the values used for the paper.

    Returns
    -------
    ds_clean, bad, bad_raw
    """
    S = ds[sal_var]

    S_med_t = S.rolling(time=time_window, center=True,
                        min_periods=max(3, time_window // 2)).median()
    S_med_z = S.rolling(depth=depth_window, center=True,
                        min_periods=max(3, depth_window // 2)).median()

    bad_raw = (np.abs(S - S_med_t) > thresh) | (np.abs(S - S_med_z) > thresh)
    apply_qc = (ds["depth"] <= shallow_limit).broadcast_like(S) | (S < fresh_limit)
    bad = bad_raw & apply_qc

    if expand_time > 0:
        win = 2 * expand_time + 1
        bad = (bad.astype(int).rolling(time=win, center=True, min_periods=1)
               .max().astype(bool))

    for i in range(1, expand_down + 1):
        bad = bad | bad.shift(depth=i, fill_value=False)   # downward
        bad = bad | bad.shift(depth=-i, fill_value=False)  # upward

    return ds.where(~bad), bad, bad_raw


def add_smoothed_n2(ds, sigma=3.0):
    """Recompute N2 from the QC'd T/S and add a Gaussian-smoothed version as ``N2_smooth``.

    Pressure is obtained from the depth index with ``gsw.p_from_z``. N2
    (time, depth_mid) is smoothed with a NaN-safe 2-D Gaussian of ``sigma``
    grid points (profiles x 1 m) and completed to the full depth axis by
    repeating its deepest level.

    Note: the smoothing runs along the time axis of ``ds`` as a whole; if ``ds``
    holds several tags concatenated in time, the first/last ~4*sigma profiles
    of each tag are blended with the neighbouring tag (as in the original
    multi-tag ``ds_clean.nc``).
    """
    p = gsw.p_from_z(-ds["depth"], ds["lat"])
    N2, _ = xr.apply_ufunc(
        gsw.Nsquared, ds["salinity"], ds["temperature"], p, ds["lat"],
        input_core_dims=[["depth"], ["depth"], ["depth"], []],
        output_core_dims=[["depth_mid"], ["depth_mid"]],
        vectorize=True,
    )
    N2_smooth = gaussian_filter_nan(N2, sigma=sigma)
    N2_smooth = np.hstack((N2_smooth, N2_smooth[:, -1][:, None]))

    ds["N2_smooth"] = xr.DataArray(
        N2_smooth.T, dims=("depth", "time"),
        coords={"depth": ds["depth"], "time": ds["time"]},
    )
    return ds
