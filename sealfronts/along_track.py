"""Helpers to sample gridded auxiliary datasets along the seal track.

Used by
    scripts/03_sic_along_track.py     sea-ice concentration (NSIDC G02202 v6)
    scripts/04_era5_along_track.py    ERA5 surface winds / air temperature / fluxes
    scripts/05_strain_so_slice.py     SO-SLICE SLA-derived geostrophic strain
"""
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

EARTH_RADIUS_M = 6_371_000.0
EARTH_ROTATION_RATE = 7.2921159e-5  # s-1
G = 9.81  # m s-2


# ---------------------------------------------------------------------------
# Seal track
# ---------------------------------------------------------------------------
def load_track(npz_path):
    """Load the seal track from the interpolated profile file.

    Returns a dict with ``lon`` and ``lat`` (float arrays) and ``day`` (the
    profile times exactly as stored, i.e. an object array of datetimes).
    """
    data = np.load(npz_path, allow_pickle=True)
    return {"lon": data["lon"], "lat": data["lat"], "day": data["day"]}


def to_datetime64(day):
    """Convert an array of datetimes to ``datetime64[ns]``."""
    return np.asarray(day).astype("datetime64[ns]")


# ---------------------------------------------------------------------------
# Sea-ice concentration
# ---------------------------------------------------------------------------
def sample_sic_along_track(sic, lon, lat, time):
    """Sample a polar-stereographic SIC field along the track.

    The SIC map of the nearest day is linearly interpolated in (x, y) at each
    seal position, after projecting lon/lat to NSIDC Sea Ice Polar
    Stereographic South (EPSG:3412).

    Parameters
    ----------
    sic : xarray.DataArray (time, y, x)
    lon, lat : array-like, degrees
    time : array of datetime64[ns]
    """
    from pyproj import CRS, Transformer

    tfm = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3412), always_xy=True)
    x_pts, y_pts = tfm.transform(lon, lat)
    pts = xr.Dataset(
        {
            "time": ("points", time),
            "x": ("points", x_pts),
            "y": ("points", y_pts),
        }
    )
    sic_along = (
        sic.sel(time=pts["time"], method="nearest")
        .interp(x=pts["x"], y=pts["y"], method="linear")
    )
    return sic_along.compute().to_numpy()


# ---------------------------------------------------------------------------
# Nearest-neighbour sampling on a regular lon/lat/time grid (ERA5)
# ---------------------------------------------------------------------------
def nearest_index(grid, values):
    """Index of the nearest ``grid`` element for each element of ``values``.

    Ties resolve to the first grid element (same as ``np.argmin``).
    """
    grid = np.asarray(grid)
    values = np.asarray(values)
    return np.abs(values[:, None] - grid[None, :]).argmin(axis=1)


def sample_nearest_along_track(ds, var_names, lon, lat, time):
    """Nearest-neighbour (time, latitude, longitude) samples of ``var_names``.

    ``ds`` must have 1-D ``time``, ``latitude`` and ``longitude`` coordinates
    and variables with dims ``(time, latitude, longitude)``.
    """
    it = nearest_index(ds["time"].values, time)
    iy = nearest_index(ds["latitude"].values, lat)
    ix = nearest_index(ds["longitude"].values, lon)
    return {name: ds[name].values[it, iy, ix] for name in var_names}


# ---------------------------------------------------------------------------
# SLA -> geostrophic strain
# ---------------------------------------------------------------------------
def wrap_longitude(lon):
    """Wrap longitudes to [-180, 180)."""
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def normalize_longitude(ds):
    """Wrap the ``longitude`` coordinate to [-180, 180) and sort it."""
    return ds.assign_coords(longitude=wrap_longitude(ds["longitude"])).sortby("longitude")


def coriolis_parameter(latitude):
    """Coriolis parameter f (s-1)."""
    return 2.0 * EARTH_ROTATION_RATE * np.sin(np.deg2rad(latitude))


def meters_per_degree_longitude(latitude):
    """Length (m) of one degree of longitude at ``latitude``."""
    return EARTH_RADIUS_M * np.cos(np.deg2rad(latitude)) * np.deg2rad(1.0)


def add_sla_geostrophic_strain(ds, sla_name="sla"):
    """Add SLA-derived geostrophic strain components to ``ds``.

    Same convention as the NeurOST product metadata::

        sn     = -2 (g/f) d2SLA/dxdy                (normal strain)
        ss     =    (g/f) (d2SLA/dx2 - d2SLA/dy2)   (shear strain)
        strain = sqrt(sn^2 + ss^2)

    Derivatives are second-order centred differences on the lon/lat grid
    (``xarray.DataArray.differentiate`` with ``edge_order=2``).
    """
    lat = ds["latitude"]
    f = xr.DataArray(coriolis_parameter(lat), coords={"latitude": lat}, dims=("latitude",))
    f = f.where(np.abs(f) > 1e-10)
    dx_per_deg = xr.DataArray(
        meters_per_degree_longitude(lat), coords={"latitude": lat}, dims=("latitude",)
    )
    dy_per_deg = EARTH_RADIUS_M * np.deg2rad(1.0)

    sla = ds[sla_name]
    deta_dx = sla.differentiate("longitude", edge_order=2) / dx_per_deg
    deta_dy = sla.differentiate("latitude", edge_order=2) / dy_per_deg
    d2eta_dx2 = deta_dx.differentiate("longitude", edge_order=2) / dx_per_deg
    d2eta_dy2 = deta_dy.differentiate("latitude", edge_order=2) / dy_per_deg
    d2eta_dxdy = deta_dx.differentiate("latitude", edge_order=2) / dy_per_deg

    scale = G / f
    sn = (-2.0 * d2eta_dxdy * scale).transpose(*sla.dims)
    ss = ((d2eta_dx2 - d2eta_dy2) * scale).transpose(*sla.dims)
    strain = np.hypot(sn, ss).transpose(*sla.dims)

    ds = ds.assign(
        sn=sn.astype("float32"),
        ss=ss.astype("float32"),
        strain=strain.astype("float32"),
    )
    ds["sn"].attrs.update(long_name="normal strain from SLA", description="-2*(g/f)*d2SLA/dxdy", units="s-1")
    ds["ss"].attrs.update(long_name="shear strain from SLA", description="(g/f)*(d2SLA/dx2-d2SLA/dy2)", units="s-1")
    ds["strain"].attrs.update(long_name="strain magnitude from SLA", description="sqrt(sn^2+ss^2)", units="s-1")
    return ds


def oi_fill_sla_one_day(da2d, lx, ly, noise, n_neighbors):
    """Fill NaNs of one daily SLA map by local optimal interpolation (OI).

    Gaussian covariance with decorrelation scales ``lx``/``ly`` (degrees) and
    normalised noise variance ``noise``; each gap cell uses its
    ``n_neighbors`` nearest valid cells (in scaled lon/lat space). Only NaNs
    north of the southernmost valid latitude of that day are filled; valid
    values are left unchanged.
    """
    da2d = da2d.transpose("latitude", "longitude")
    lat = da2d["latitude"].values
    lon = da2d["longitude"].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    values = np.asarray(da2d.values, dtype=float).copy()
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 3:
        return da2d.astype("float32")

    min_valid_lat = float(np.nanmin(lat2d[finite]))
    fill_needed = (~finite) & (lat2d >= min_valid_lat)
    if not np.any(fill_needed):
        return da2d.astype("float32")

    obs_xy = np.column_stack((lon2d[finite] / lx, lat2d[finite] / ly))
    obs_values = values[finite]
    target_xy = np.column_stack((lon2d[fill_needed] / lx, lat2d[fill_needed] / ly))

    k = min(int(n_neighbors), obs_values.size)
    _, neighbor_indices = cKDTree(obs_xy).query(target_xy, k=k)
    if k == 1:
        neighbor_indices = neighbor_indices[:, None]

    filled_values = np.full(target_xy.shape[0], np.nan, dtype=float)
    for i, idx in enumerate(neighbor_indices):
        nxy = obs_xy[idx]
        obs_dist = np.linalg.norm(nxy[:, None, :] - nxy[None, :, :], axis=2)
        obs_cov = np.exp(-0.5 * obs_dist**2) + noise * np.eye(idx.size)
        target_cov = np.exp(-0.5 * np.linalg.norm(nxy - target_xy[i], axis=1) ** 2)
        try:
            weights = np.linalg.solve(obs_cov, target_cov)
            filled_values[i] = np.sum(weights * obs_values[idx])
        except np.linalg.LinAlgError:
            filled_values[i] = np.nanmean(obs_values[idx])

    values[fill_needed] = filled_values
    return xr.DataArray(
        values.astype("float32"), coords=da2d.coords, dims=da2d.dims,
        attrs=da2d.attrs, name=da2d.name,
    )


def oi_fill_sla(da, lx, ly, noise, n_neighbors):
    """Apply :func:`oi_fill_sla_one_day` to every day of a (time, lat, lon) SLA."""
    original_dims = da.dims
    da = da.transpose("time", "latitude", "longitude")
    filled = xr.concat(
        [oi_fill_sla_one_day(da.sel(time=t), lx, ly, noise, n_neighbors) for t in da["time"].values],
        dim=da["time"],
    )
    filled.name = da.name
    filled.attrs.update(da.attrs)
    filled.attrs["gap_fill"] = (
        "NaNs filled by local optimal interpolation before SLA derivative calculation; "
        f"lx={lx} deg, ly={ly} deg, noise={noise}, n_neighbors={n_neighbors}."
    )
    return filled.transpose(*original_dims)


def sample_nearest_sel(ds, var_names, lon, lat, time):
    """Nearest-neighbour samples of ``var_names`` at (time, lat, lon) track points
    using xarray vectorised ``sel(method="nearest")``."""
    dim = "track"
    sel = {
        "longitude": xr.DataArray(wrap_longitude(lon), dims=dim),
        "latitude": xr.DataArray(np.asarray(lat, dtype=float), dims=dim),
        "time": xr.DataArray(np.asarray(time, dtype="datetime64[ns]"), dims=dim),
    }
    sampled = ds[list(var_names)].sel(sel, method="nearest")
    return {name: sampled[name].values for name in var_names}


# ---------------------------------------------------------------------------
# Track geometry and along-track strain
# ---------------------------------------------------------------------------
def track_angle(lon, lat, smooth_sigma=None):
    """Local track heading ``theta`` (rad, counter-clockwise from east).

    Positions are projected on a local tangent plane centred on the mean
    track position; the unit tangent is from ``np.gradient`` of x, y. If
    ``smooth_sigma`` > 0, the unit tangent components are Gaussian-smoothed
    (``sigma`` in samples, ``mode="nearest"``) and renormalised.
    """
    lon_rad = np.unwrap(np.deg2rad(np.asarray(lon, dtype=float)))
    lat_rad = np.deg2rad(np.asarray(lat, dtype=float))
    lon0, lat0 = np.mean(lon_rad), np.mean(lat_rad)

    x = EARTH_RADIUS_M * np.cos(lat0) * (lon_rad - lon0)
    y = EARTH_RADIUS_M * (lat_rad - lat0)
    dx, dy = np.gradient(x), np.gradient(y)
    ds = np.hypot(dx, dy)
    ds = np.where(ds == 0.0, np.nan, ds)
    sx, sy = dx / ds, dy / ds

    if smooth_sigma is not None and smooth_sigma > 0:
        sx = gaussian_filter1d(sx, sigma=smooth_sigma, mode="nearest")
        sy = gaussian_filter1d(sy, sigma=smooth_sigma, mode="nearest")
        norm = np.hypot(sx, sy)
        norm = np.where(norm == 0.0, np.nan, norm)
        sx, sy = sx / norm, sy / norm

    return np.arctan2(sy, sx)


def along_track_strain(sn, ss, theta):
    """Along-track strain rate Ess = 0.5 sn cos(2 theta) + 0.5 ss sin(2 theta)."""
    return 0.5 * sn * np.cos(2.0 * theta) + 0.5 * ss * np.sin(2.0 * theta)

