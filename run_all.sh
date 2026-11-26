#!/usr/bin/env bash
# Run the full pipeline: raw data -> intermediate products -> figures -> paper numbers.
# Set SEALFRONTS_DATA / SEALFRONTS_WORK / SEALFRONTS_FIGS first (see README).
# Runtime: ~45 min on a 14-core laptop (05 ~25 min, 06 ~18 min with 10 workers).
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-4}"

# 1. Preprocessing
python scripts/01_interpolate_seal.py      # raw MEOP profiles -> regular 1 km x 1 m grid
python scripts/03_sic_along_track.py       # NSIDC SIC along the track (needed by 02)
python scripts/02_clean_seal.py            # under-ice selection, salinity QC, smoothed N2
python scripts/04_era5_along_track.py      # ERA5 surface fluxes along the track
python scripts/05_strain_so_slice.py       # SO-SLICE geostrophic strain along the track

# 2. Omega equation, vertical heat transport, Monte Carlo
python scripts/06_omega_vht.py --workers "$WORKERS"
python scripts/07_mc_statistics.py

# 3. Figures
python scripts/fig1_map_sic_sar.py
python scripts/fig2_ts_diagrams.py
python scripts/fig3_sections.py
python scripts/fig4_bx_exceedance.py
python scripts/fig5_timeseries.py
python scripts/figS_n_profiles.py
python scripts/figS_mc_summary.py
python scripts/figS_sections_biweekly.py

# 4. Numbers quoted in the text
python scripts/paper_numbers.py
