"""Vertical velocity and vertical heat transport from the 2D QG omega equation.

Vertical velocity ``w`` is diagnosed along the seal track from the
two-dimensional (x, z) quasi-geostrophic omega equation (Methods, "Vertical
velocities"; Siegelman et al. 2020)

    N^2 w_xx + f^2 w_zz = -2 (u_x b_x)_x ,

where x is the along-track direction, ``u_x`` the along-track strain rate
estimated from SO-SLICE altimetry (assumed constant with depth), ``b_x`` the
along-track buoyancy gradient and ``N^2`` the stratification measured by the
seal. The elliptic problem is solved by successive over-relaxation with the
``xinvert`` package (Xu et al. 2023). The vertical heat transport is

    VHT = rho0 * cp * w * CT' ,

with CT' the Conservative Temperature anomaly with respect to the along-track
mean profile (Methods, "Vertical heat transport").

The module also implements the Monte Carlo propagation of strain and seal
orientation uncertainty (Methods, "Monte Carlo uncertainty propagation").
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis
from xinvert import FiniteDiff
from xinvert.apps import __template as _xinvert_template
from xinvert.apps import __mask_FS as _xinvert_mask_FS
from xinvert.apps import inv_standard2D

from sealfronts import config

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
STRAIN_SOURCE = "so_slice_oi"        # SO-SLICE (50-km OI mapping) strain
CONTROL_STRAIN = "ess_smooth"        # along-track strain forcing the reference run
OMEGA_STRAIN_SCALE = 1.0             # multiplicative factor on the strain forcing

RHO0 = 1025.0                        # reference density [kg m-3]
CP = 3985.0                          # specific heat capacity [J kg-1 K-1]
DX = 1000.0                          # along-track spacing of ds_clean profiles [m]

# The section (3880 km) is inverted in independent along-track chunks of
# 550 km; a trailing chunk shorter than this is merged with the previous one.
CHUNK_SIZE = 550                     # [grid points] = [km]

# Profile indices used to build the background bx and N2 profiles that fill
# the water column below the deepest valid seal measurement.
BACKGROUND_SLICE = slice(1600, 1900)
N2_MIN = 1e-8                        # floor on N2 [s-2]

# Two patches with spuriously strong bx at the edge of the sampled domain are
# replaced by the background bx profile: (distance range [m], depth range [m]).
BX_EDGE_PATCHES = [
    ((3840e3, 3850e3), (150, 410)),
    ((3566e3, 3572e3), (150, 325)),
]

# Boundary conditions: reflective (w_z = 0) in depth, fixed (w = 0) at the
# along-track ends of each chunk.
ITER_PARAMS = {
    "BCs": ["reflect", "fixed"],
    "undef": np.nan,
    "mxLoop": 2000,
    "tolerance": 1e-10,
    "printInfo": False,
}

# Monte Carlo ensemble (Methods, "Monte Carlo uncertainty propagation")
MC_PARAMS = dict(
    n_mc=50,                 # number of realizations
    theta_std_deg=10.0,      # std of trajectory-angle perturbations [deg]
    amp_range=(0.5, 2.0),    # A ~ U(0.5, 2)
    smooth_sigma=5,          # Gaussian smoothing of perturbations [grid points = km]
    clip_abs=5e-5,           # clipping of the strain realizations [s-1]
    seed=42,                 # numpy default_rng seed
)

# Dynamical regimes discussed in the paper (UTC)
PERIODS = {
    "Thick ice": slice(*config.THICK_ICE_PERIOD),
    "Thin ice": slice(*config.THIN_ICE_PERIOD),
    "Full trajectory": None,
}

_UNDEF = -9.99e8  # internal undefined value used by xinvert


# ---------------------------------------------------------------------------
# 2D omega-equation solver on top of xinvert
# ---------------------------------------------------------------------------
def invert_omega_2D(F, dims, mParams, iParams, coords="cartesian", icbc=None):
    """Invert the 2D (z, x) QG omega equation  f^2 w_zz + N^2 w_xx = F.

    Parameters
    ----------
    F : xarray.DataArray
        Forcing -2 (u_x b_x)_x [s-3] on dims ``dims`` (depth first).
    dims : list of str
        ``[vertical_dim, along_track_dim]``.
    mParams : dict
        ``{"f": f [s-1], "N2": N2 [s-2]}`` (scalars or DataArrays like F).
    iParams : dict
        xinvert iteration parameters (boundary conditions, tolerance...).
    coords : str
        Only ``"cartesian"`` is supported.

    Returns
    -------
    xarray.DataArray
        Vertical velocity w [m s-1].
    """
    return _xinvert_template(_coeffs_omega_2D, inv_standard2D, 2, F, dims,
                             coords, icbc, ["f", "N2"], mParams, iParams)


def _coeffs_omega_2D(force, dims, coords, mParams, iParams, icbc):
    """Coefficients A w_zz + B w_zx + C w_xx = F of the 2D omega equation."""
    if coords.lower() != "cartesian":
        raise ValueError(f"unsupported coords {coords!r}; only 'cartesian'")

    maskF, initS, zero = _xinvert_mask_FS(force, dims, iParams, icbc)
    A = zero + mParams["f"] ** 2
    B = zero
    C = zero + mParams["N2"]
    F = maskF.where(maskF != _UNDEF, _UNDEF)
    return F, initS, (A, B, C)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_seal_section(ds_clean_path, strain_npz_path, seal_id):
    """Load the cleaned seal section and attach the along-track strain.

    Parameters
    ----------
    ds_clean_path : path
        ``ds_clean.nc`` (all tags, dims depth x time).
    strain_npz_path : path
        SO-SLICE strain interpolated on the seal track (``sn``, ``ss``,
        ``ess``, ``ess_smooth``, ``theta_smooth``, ``day``).
    seal_id : str
        MEOP tag to analyse.

    Returns
    -------
    ds : xarray.Dataset
        Section of ``seal_id`` (dims depth x time) with strain variables
        ``sn``, ``ss`` [s-1], ``ess``, ``ess_smooth`` (along-track strain
        [s-1]) and a ``distance`` coordinate [m] on the time dimension.
    theta_smooth : xarray.DataArray
        Smoothed trajectory angle [rad] on the same time axis (used only by
        the Monte Carlo ensemble).
    """
    ds = xr.open_dataset(ds_clean_path).drop_vars("depth_mid", errors="ignore")
    if seal_id not in np.unique(ds["name"]):
        raise ValueError(f"tag {seal_id!r} not found in {ds_clean_path}")
    ds = ds.where(ds.coords["name"] == seal_id, drop=True)

    npz = np.load(strain_npz_path, allow_pickle=True)
    strain_vars = ["sn", "ss", "ess", "ess_smooth", "theta_smooth"]
    strain = xr.Dataset(
        {k: ("time", npz[k]) for k in strain_vars},
        coords={"time": npz["day"]},
    ).sel(time=ds.time, method="nearest", tolerance=pd.Timedelta("1h"))

    for k in strain_vars:
        ds[k] = strain[k]
    ds.attrs["strain_source"] = STRAIN_SOURCE

    distance = np.arange(ds.sizes["time"]) * DX
    ds = ds.assign_coords(distance=("time", distance))
    ds["distance"].attrs["units"] = "m"
    theta_smooth = ds["theta_smooth"]
    return ds.drop_vars("theta_smooth"), theta_smooth


def _fill_below_deepest(field, maskprof):
    """Fill a (depth, time) field below the deepest valid profile sample.

    The masked field is linearly interpolated (and extrapolated) in depth,
    and values below the deepest valid sample of each profile are replaced
    by the median profile over ``BACKGROUND_SLICE``.

    Returns the filled field and the background profile.
    """
    f0 = (field * maskprof).interpolate_na(
        dim="depth", method="linear", fill_value="extrapolate")
    background = f0.isel(time=BACKGROUND_SLICE).median("time")

    deepest_valid = maskprof.depth.where(~maskprof.isnull()).max("depth")
    depth2d, _ = xr.broadcast(field.depth, field.time)
    _, deepest2d = xr.broadcast(field.depth, deepest_valid)
    bg2d, _ = xr.broadcast(background, field.time)
    return xr.where(depth2d > deepest2d, bg2d, f0), background


def prepare_omega_inputs(ds):
    """Build the fields entering the omega equation (dims depth x distance).

    Parameters
    ----------
    ds : xarray.Dataset
        Output of :func:`load_seal_section`.

    Returns
    -------
    dict
        ``bx`` [s-2], ``N2`` [s-2], ``f`` [s-1] (surface value), ``ds_dist``
        (``ds`` indexed by distance) and the gap-filled along-track strain
        options ``ess`` and ``ess_smooth`` [s-1].
    """
    bx_fill, bx_bg = _fill_below_deepest(ds["bx"], ds["maskprof"])
    N2_fill, _ = _fill_below_deepest(ds["N2_smooth"], ds["maskprof"])
    N2_fill = xr.where(N2_fill < N2_MIN, N2_MIN, N2_fill)

    bx = bx_fill.swap_dims({"time": "distance"})
    for (x0, x1), (z0, z1) in BX_EDGE_PATCHES:
        target = bx.sel(distance=slice(x0, x1), depth=slice(z0, z1))
        bx.loc[dict(distance=target.distance, depth=target.depth)] = (
            bx_bg.sel(depth=target.depth)
            .expand_dims(distance=target.distance)
            .transpose(*target.dims)
        )

    def _gapfill(da):
        return (da.interpolate_na(dim="time").ffill("time").bfill("time")
                .swap_dims({"time": "distance"}))

    return {
        "bx": bx,
        "N2": N2_fill.swap_dims({"time": "distance"}),
        "f": ds["f"].sel(depth=0).interpolate_na(dim="time")
                    .swap_dims({"time": "distance"}),
        "ess": _gapfill(ds["ess"]),
        "ess_smooth": _gapfill(ds["ess_smooth"]),
        "ds_dist": ds.swap_dims({"time": "distance"}),
    }


def _chunk_slices(n, step=CHUNK_SIZE):
    """Along-track index slices for the chunked inversion."""
    starts = list(range(0, n, step))
    if len(starts) > 1 and (n - starts[-1]) < step:
        starts = starts[:-1]
    return [slice(i, n) if i == starts[-1] else slice(i, i + step)
            for i in starts]


# ---------------------------------------------------------------------------
# w and VHT
# ---------------------------------------------------------------------------
def compute_w_vht(inputs, strain, label="", verbose=True):
    """Solve the omega equation for a given along-track strain.

    Parameters
    ----------
    inputs : dict
        Output of :func:`prepare_omega_inputs`.
    strain : xarray.DataArray
        Along-track strain u_x [s-1] on the ``distance`` dimension.
    label : str
        Label used in progress messages.

    Returns
    -------
    W, VHF, WB, WS : xarray.DataArray
        Vertical velocity w [m s-1], vertical heat transport
        rho0 cp w CT' [W m-2], buoyancy flux w b' [m2 s-3] and salt flux
        w S' [g kg-1 m s-1] (dims depth x distance). Anomalies are taken
        with respect to the along-track mean profile of each inversion chunk.
    """
    fd = FiniteDiff({"Z": "depth", "X": "distance"},
                    BCs={"Z": "reflect", "X": "fixed"}, coords="cartesian")

    Q = inputs["bx"] * strain * OMEGA_STRAIN_SCALE         # u_x b_x [s-3]
    divQ = -2 * fd.divg(Q, dims=["X"])                     # -2 (u_x b_x)_x
    forcing = xr.where(np.isfinite(divQ), divQ, np.nan).load()

    ds_dist = inputs["ds_dist"]
    out = {k: [] for k in ("W", "VHF", "WB", "WS")}
    for sl in _chunk_slices(ds_dist.sizes["distance"]):
        W = invert_omega_2D(
            forcing.isel(distance=sl),
            dims=["depth", "distance"],
            coords="cartesian",
            iParams=ITER_PARAMS,
            mParams={"N2": inputs["N2"].isel(distance=sl),
                     "f": inputs["f"].isel(distance=sl)},
        )
        if verbose:
            print(f"{label}: max |w| = "
                  f"{np.round(np.nanmax(np.abs(W.values * 86400)), 2)} m/day")

        anom = {}
        for v in ("temperature", "buoyancy", "salinity"):
            sub = ds_dist[v].isel(distance=sl)
            anom[v] = sub - sub.mean(dim="distance")

        out["W"].append(W)
        out["VHF"].append(W * anom["temperature"] * CP * RHO0)
        out["WB"].append(W * anom["buoyancy"])
        out["WS"].append(W * anom["salinity"])

    return tuple(xr.concat(out[k], dim="distance")
                 for k in ("W", "VHF", "WB", "WS"))


def build_omega_dataset(ds, inputs, verbose=True):
    """Reference solution forced by the smoothed SO-SLICE along-track strain.

    Returns ``ds`` (indexed by distance) with ``w``, ``VHF``, ``WB``, ``WS``
    and the forcing ``strain_rate`` added; this is the content of
    ``config.OMEGA_NC``.
    """
    strain = inputs[CONTROL_STRAIN]
    W, VHF, WB, WS = compute_w_vht(inputs, strain, CONTROL_STRAIN, verbose)

    ds_out = ds.swap_dims({"time": "distance"}).copy()
    ds_out["w"] = W
    ds_out["VHF"] = VHF
    ds_out["WB"] = WB
    ds_out["WS"] = WS
    ds_out["strain_rate"] = strain
    ds_out.attrs["strain_source"] = STRAIN_SOURCE
    ds_out.attrs["omega_strain_input_name"] = CONTROL_STRAIN
    ds_out.attrs["omega_strain_scale"] = OMEGA_STRAIN_SCALE
    return ds_out


# ---------------------------------------------------------------------------
# Monte Carlo strain realizations
# ---------------------------------------------------------------------------
def make_projected_strain_mc(sn, ss, theta, sigma_eps, n_mc=50,
                             theta_std_deg=10.0, amp_range=(0.5, 2.0),
                             smooth_sigma=5, clip_abs=5e-5, seed=42):
    """Monte Carlo realizations of the along-track strain.

    For each realization i (Methods, "Monte Carlo uncertainty propagation"):

    * theta_i = theta + G_sigma[N(0, theta_std)]  (perturbed track angle)
    * eps_i   = 0.5 (sn cos 2 theta_i + ss sin 2 theta_i)
    * strain_i = A_i eps_i + G_sigma[N(0, sigma_eps^2)],  A_i ~ U(amp_range)
    * clipped to +/- clip_abs

    where G_sigma is a 1D Gaussian filter of ``smooth_sigma`` grid points
    (1 point = 1 km). Random numbers come from
    ``numpy.random.default_rng(seed)`` drawn in the order dtheta, A, noise
    for each realization.

    Parameters
    ----------
    sn, ss : xarray.DataArray
        Normal and shear strain [s-1] along the track (1D).
    theta : xarray.DataArray
        Smoothed trajectory angle [rad] relative to east.
    sigma_eps : float
        Standard deviation of the noise [s-1] (std of the reference strain).

    Returns
    -------
    xarray.DataArray
        ``strain_parallel_mc`` with dims (realization, <dim of sn>) [s-1];
        NaN where sn, ss or theta are missing.
    """
    rng = np.random.default_rng(seed)
    sn_v, ss_v, th_v = sn.values, ss.values, theta.values
    valid = np.isfinite(sn_v) & np.isfinite(ss_v) & np.isfinite(th_v)

    out = []
    for _ in range(n_mc):
        dtheta = gaussian_filter1d(
            rng.normal(0.0, np.deg2rad(theta_std_deg), size=th_v.shape),
            sigma=smooth_sigma, mode="nearest")
        theta_mc = th_v + dtheta
        eps = 0.5 * (sn_v * np.cos(2 * theta_mc) + ss_v * np.sin(2 * theta_mc))

        amp = rng.uniform(amp_range[0], amp_range[1])
        noise = gaussian_filter1d(
            rng.normal(0.0, sigma_eps, size=eps.shape),
            sigma=smooth_sigma, mode="nearest")

        strain_k = np.where(valid, amp * eps + noise, np.nan)
        out.append(np.clip(strain_k, -clip_abs, clip_abs))

    dim = sn.dims[0]
    mc = xr.DataArray(np.stack(out), dims=("realization", dim),
                      coords={"realization": np.arange(n_mc), dim: sn[dim]},
                      name="strain_parallel_mc")
    if "distance" in sn.coords and dim != "distance":
        mc = mc.assign_coords(distance=(dim, sn["distance"].values))
    return mc


def omega_strain_realizations(control_strain, strain_mc):
    """Stack the reference strain and the gap-filled MC realizations.

    Returns a DataArray ``strain_rate`` (realization, distance) [s-1] whose
    first realization is the reference (labelled ``CONTROL_STRAIN``) followed
    by ``mc_000`` ... ``mc_NNN``. MC realizations are linearly interpolated
    across gaps and extended at the ends, as the reference forcing.
    """
    control = (control_strain.rename("strain_rate").transpose("distance")
               .expand_dims(realization=[CONTROL_STRAIN]))
    mc = (strain_mc.swap_dims({"time": "distance"})
          .transpose("realization", "distance")
          .interpolate_na(dim="distance").ffill("distance").bfill("distance")
          .rename("strain_rate"))
    mc = mc.assign_coords(
        realization=[f"mc_{int(r):03d}" for r in mc["realization"].values])
    return xr.concat([control, mc], dim="realization")


_WORKER_INPUTS = None


def _init_worker(inputs):
    """Store the omega inputs once per worker process."""
    global _WORKER_INPUTS
    _WORKER_INPUTS = inputs


def _solve_realization(args):
    """Solve one (label, strain) realization; returns float32 w and VHT."""
    label, strain = args
    W, VHF, _, _ = compute_w_vht(_WORKER_INPUTS, strain, label)
    return W.astype("float32"), VHF.astype("float32")


def compute_w_vht_realizations(inputs, strain_realizations, workers=1):
    """Solve the omega equation for every strain realization.

    Realizations are independent; with ``workers > 1`` they are solved in
    parallel processes (results are identical to a serial run).

    Returns a Dataset with ``w_realization`` [m s-1], ``VHF_realization``
    [W m-2] and ``strain_rate_realization`` [s-1], all float32 (as in the
    analysis used for the paper).
    """
    real = strain_realizations["realization"]
    tasks = [(str(r), strain_realizations.sel(realization=r))
             for r in real.values]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers, initializer=_init_worker,
                                 initargs=(inputs,)) as ex:
            results = list(ex.map(_solve_realization, tasks))
    else:
        _init_worker(inputs)
        results = [_solve_realization(t) for t in tasks]

    return xr.Dataset({
        "w_realization": xr.concat([r[0] for r in results], dim=real),
        "VHF_realization": xr.concat([r[1] for r in results], dim=real),
        "strain_rate_realization": strain_realizations.astype("float32"),
    })


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def select_period(ds, time_slice):
    """Select a time window of a distance-indexed dataset (None = all)."""
    if time_slice is None:
        return ds
    return ds.swap_dims({"distance": "time"}).sel(time=time_slice)


def _pearson_kurtosis(da):
    """Pearson (non-excess), bias-corrected kurtosis of finite values."""
    v = da.values.ravel()
    return kurtosis(v[np.isfinite(v)], fisher=False, bias=False)


def kurtosis_periods(ds, periods=PERIODS):
    """Kurtosis of w and VHT over sampled points (maskprof > 0) per period.

    Returns a DataFrame with columns period, N_w, N_VHT, kurtosis_w,
    kurtosis_VHT (Pearson definition: 3 for a Gaussian).
    """
    rows = []
    for label, tsl in periods.items():
        sub = select_period(ds, tsl)
        mask = sub["maskprof"] > 0
        w, vht = sub["w"].where(mask), sub["VHF"].where(mask)
        rows.append({
            "period": label,
            "N_w": int(np.isfinite(w.values).sum()),
            "N_VHT": int(np.isfinite(vht.values).sum()),
            "kurtosis_w": _pearson_kurtosis(w),
            "kurtosis_VHT": _pearson_kurtosis(vht),
        })
    return pd.DataFrame(rows)


def mc_kurtosis_periods(ds_mc, periods=PERIODS):
    """Median and 5-95 % range across realizations of the w/VHT kurtosis.

    ``ds_mc`` holds ``w_realization``, ``VHF_realization`` and the sampling
    mask ``maskprof``, with a ``time`` coordinate on distance.
    """
    rows = []
    for label, tsl in periods.items():
        sub = select_period(ds_mc, tsl)
        mask = sub["maskprof"] > 0
        w = sub["w_realization"].where(mask)
        vht = sub["VHF_realization"].where(mask)
        kw = np.array([_pearson_kurtosis(w.sel(realization=r))
                       for r in sub.realization.values])
        kv = np.array([_pearson_kurtosis(vht.sel(realization=r))
                       for r in sub.realization.values])
        rows.append({
            "period": label,
            "w_kurt_median": np.nanmedian(kw),
            "w_kurt_p05": np.nanpercentile(kw, 5),
            "w_kurt_p95": np.nanpercentile(kw, 95),
            "VHT_kurt_median": np.nanmedian(kv),
            "VHT_kurt_p05": np.nanpercentile(kv, 5),
            "VHT_kurt_p95": np.nanpercentile(kv, 95),
        })
    return pd.DataFrame(rows)
