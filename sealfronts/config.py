"""Paths and global settings shared by all scripts.

Set the environment variables below (or edit the defaults) to point to your
local copies of the input data and to where outputs should be written.

    SEALFRONTS_DATA   root folder holding the raw input data (see README)
    SEALFRONTS_WORK   folder for intermediate products (created if missing)
    SEALFRONTS_FIGS   folder for figures (created if missing)
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("SEALFRONTS_DATA", REPO_ROOT / "data"))
WORK_DIR = Path(os.environ.get("SEALFRONTS_WORK", REPO_ROOT / "work"))
FIG_DIR = Path(os.environ.get("SEALFRONTS_FIGS", REPO_ROOT / "figures"))

for _d in (WORK_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Seal deployment analysed in the paper (MEOP tag id)
SEAL_ID = "ct176-04-23"

# ---------------------------------------------------------------------------
# Raw inputs (relative to DATA_DIR)
# ---------------------------------------------------------------------------
SEAL_DIR = DATA_DIR / "seal"                                   # MEOP high-resolution profiles
SEAL_NC = SEAL_DIR / f"{SEAL_ID}_fr1_prof.nc"
SEAL_MASK = SEAL_DIR / f"mask_prof_{SEAL_ID}_fr1_prof.npy"     # optional: rebuilt by scripts/01 by default
SIC_DIR = DATA_DIR / "sic"                                     # NSIDC G02202 v6 daily SIC
ERA5_DIR = DATA_DIR / "era5"                                   # ERA5 single levels
SO_SLICE_DIR = DATA_DIR / "so_slice"                           # SO-SLICE sea level anomaly
RTOPO_FILE = DATA_DIR / "rtopo2" / "RTopo-2.0.1_30sec_bedrock_topography.nc"

# ---------------------------------------------------------------------------
# Intermediate products (relative to WORK_DIR)
# ---------------------------------------------------------------------------
INTERP_NPZ = WORK_DIR / f"interpolated_data_{SEAL_ID}_fr1_prof.npz"
DS_CLEAN = WORK_DIR / "ds_clean.nc"
SIC_NPZ = WORK_DIR / f"sic_{SEAL_ID}.npz"
ERA5_NPZ = WORK_DIR / f"interpolated_era5_{SEAL_ID}.npz"
STRAIN_NPZ = WORK_DIR / f"interpolated_so_slice_hr_ts_sn_ss_{SEAL_ID}.npz"
OMEGA_NC = WORK_DIR / f"omega_vht_so_slice_oi_{SEAL_ID}_ess_smooth.nc"

# ---------------------------------------------------------------------------
# Map figure (Fig. 1): Sentinel-1 SAR inputs
# ---------------------------------------------------------------------------
# Sentinel-1 EW HH "SAR roughness" 3-day composites exported (.npz) from the
# OceanDataLab Syntool portal (https://aviso-calval.oceandatalab.com/).
SAR_DIR = DATA_DIR / "sar"                                     # Fig. 1b

# ---------------------------------------------------------------------------
# Auxiliary data along the seal track (scripts 00, 03, 04, 05)
# ---------------------------------------------------------------------------
# NSIDC G02202 v6 daily SIC, southern hemisphere 25 km polar stereographic
SIC_FILE_GLOB = "sic_pss25_*_v06r00.nc"

# ERA5 hourly single levels, Jan-Mar 2024, area [N, W, S, E] = [-59, 30.11, -69.95, 65.26].
# The CDS returns a zip archive holding one file per stream (stepType-avg,
# stepType-instant); scripts/04 unzips it if needed.
ERA5_ZIP = ERA5_DIR / "era5_single_levels_2024JanMar.zip"
ERA5_UNZIPPED_DIR = ERA5_DIR / "era5_single_levels_2024JanMar"
ERA5_TRACK_START = "2024-01-05"   # ERA5 series only from this date onwards
ERA5_CROP_BUFFER_DEG = 1.0        # lon/lat buffer around the track before sampling

# SO-SLICE SLA on a 0.25 deg grid (50 km / 50 km / 10 d mapping scales) -- the
# "hr" product used in the paper.
SO_SLICE_FILE = SO_SLICE_DIR / "so_slice_sla_025deg_daily.nc"
STRAIN_TIME_PAD_DAYS = 1                       # SLA days kept around the track
STRAIN_CROP = dict(lat_max=-40.0, lon_min=-5.0, lon_max=115.0)  # computation box (exclusive)
STRAIN_OI = dict(lx=2.0, ly=0.5, noise=0.02, n_neighbors=30)    # SLA gap filling (deg)
STRAIN_THETA_SMOOTH_SIGMA = 3                  # samples, track-heading smoothing

# ---------------------------------------------------------------------------
# Regimes and time windows used by the figures and paper numbers
# (Figs. 2-5, ED Figs. 2 and 4-8, scripts/paper_numbers.py)
# ---------------------------------------------------------------------------
# (start, end) of the two contrasted regimes, inclusive, UTC
THICK_ICE_PERIOD = ("2024-01-30 20:30", "2024-02-05 03:00")   # thick ice front (Lutzow-Holm Bay)
THIN_ICE_PERIOD = ("2024-03-13 21:30", "2024-03-22 05:38")    # thin ice near coastal polynya
FULL_TRACK_START = "2024-01-12"                               # start of the analysed track
ICE_GROWTH_ONSET = "2024-02-22"                               # dashed line in Fig. 5
# Bounds of the five ~two-week along-track sections (ED Figs. 4-8)
BIWEEKLY_BOUNDS = ["2024-01-12", "2024-01-26", "2024-02-09",
                   "2024-02-23", "2024-03-08", "2024-03-22 05:38"]

# ---------------------------------------------------------------------------
# Omega equation / Monte Carlo outputs (scripts/06_omega_vht.py)
# ---------------------------------------------------------------------------
# Monte Carlo ensemble: w, VHT and strain forcing for the reference strain
# plus 50 perturbed strain realizations (ED Fig. 3)
OMEGA_MC_NC = WORK_DIR / f"omega_vht_so_slice_oi_{SEAL_ID}_ess_smooth_mc.nc"
