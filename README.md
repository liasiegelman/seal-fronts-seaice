# Deep submesoscale ocean fronts beneath Antarctic sea ice: seal-borne observations

Code to reproduce the analysis and figures of

> Siegelman, L., et al. *Deep submesoscale ocean fronts connect warm Circumpolar Deep Water to Antarctic sea ice during the melt season.* Submitted to *Science Advances*.

The analysis uses sub-kilometer CTD profiles recorded by an instrumented southern elephant seal (MEOP tag `ct176-04-23`, January–March 2024) in the East Antarctic marginal ice zone. It diagnoses vertical velocity and vertical heat transport (VHT) at submesoscale fronts with a 2D quasi-geostrophic omega equation.

## Repository layout

```
sealfronts/            shared Python package
  config.py            all paths, the seal ID, regime windows and parameters
  seal.py              gridding, gradients, mixed-layer depth and salinity QC of the seal profiles
  along_track.py       sampling of SIC, ERA5 and SO-SLICE strain along the track
  omega.py             2D omega-equation solver (xinvert), w/VHT, Monte Carlo strain ensemble
  io.py, plotting.py   loading helpers and shared plotting utilities
  maps.py              map helpers for Fig. 1
scripts/               numbered processing steps, then one script per figure
run_all.sh             runs the full pipeline
environment.yml        conda environment
```

| Script | Produces |
|---|---|
| `00_download_era5.py` | (optional) ERA5 download from the Copernicus CDS |
| `01_interpolate_seal.py` | seal profiles gridded at 1 km × 1 m, plus b, N², b_x, MLD, … |
| `03_sic_along_track.py` | NSIDC sea-ice concentration along the track |
| `02_clean_seal.py` | under-ice selection, salinity QC, smoothed N² → `ds_clean.nc` (needs step 03) |
| `04_era5_along_track.py` | ERA5 surface fluxes, 2 m air temperature and 10 m winds along the track |
| `05_strain_so_slice.py` | SO-SLICE geostrophic strain along the track |
| `06_omega_vht.py` | w and VHT from the omega equation, plus the 50-member Monte Carlo ensemble |
| `07_mc_statistics.py` | kurtosis and Monte Carlo statistics quoted in the paper |
| `fig1_map_sic_sar.py` | Fig. 1 |
| `fig2_ts_diagrams.py` | Fig. 2 |
| `fig3_sections.py` | Fig. 3 |
| `fig4_bx_exceedance.py` | Fig. 4 |
| `fig5_timeseries.py` | Fig. 5 |
| `figS_n_profiles.py` | Extended Data Fig. 2 |
| `figS_mc_summary.py` | Extended Data Fig. 3 |
| `figS_sections_biweekly.py` | Extended Data Figs. 4–8 |
| `paper_numbers.py` | the numbers quoted in the Results, recomputed and printed next to the text values |

## Installation

```bash
conda env create -f environment.yml
conda activate sealfronts
```

## Input data

Download the data below and arrange it under one folder, `DATA_DIR`:

```
DATA_DIR/
  seal/ct176-04-23_fr1_prof.nc                     seal CTD profiles (MEOP)
  sic/sic_pss25_YYYYMMDD_F17_v06r00.nc             NSIDC daily SIC files
  era5/era5_single_levels_2024JanMar.zip           CDS download (or its unzipped folder
                                                   era5/era5_single_levels_2024JanMar/)
  so_slice/so_slice_sla_025deg_daily.nc            SO-SLICE daily SLA
  rtopo2/RTopo-2.0.1_30sec_bedrock_topography.nc
  sar/sar_composite_20240203-05.npz                Fig. 1b SAR composites
  sar/sar_composite_20240317-19.npz
```

| Dataset | Source | Used for |
|---|---|---|
| Seal CTD profiles, tag `ct176-04-23` (high-resolution, recovered tag) | MEOP, https://meop.net/database/ | everything |
| Sea-ice concentration, NSIDC G02202 version 6, daily 25 km, Southern Hemisphere, 24 Dec 2023 – 23 Mar 2024 | https://nsidc.org/data/g02202/versions/6 | along-track SIC, Fig. 1a |
| ERA5 hourly single levels, 1 Jan – 31 Mar 2024, 59–69.95°S, 30.11–65.26°E: 10 m wind, 2 m temperature, mean surface latent and sensible heat flux, mean surface downward and net long- and shortwave radiation | Copernicus CDS (`scripts/00_download_era5.py`) | surface fluxes, Figs. 3 and 5 |
| SO-SLICE sea level anomaly, 0.25° daily grid, 50 km / 50 km / 10 d mapping scales (Mosneron Dupin et al., 2026) | https://doi.org/10.5281/zenodo.17467407 | geostrophic strain for the omega equation |
| RTopo-2.0.1 bedrock topography, 30 arc-second | https://doi.org/10.1594/PANGAEA.856844 | 1000 m isobath, Fig. 1 |
| Sentinel-1 EW HH "SAR roughness" 3-day composites (3–5 Feb 2024, 34.02–47.98°E, 69.97–66.60°S; 17–19 Mar 2024, 27.99–35.98°E, 69.99–67.40°S), exported as `.npz` from OceanDataLab Syntool | https://aviso-calval.oceandatalab.com/ | Fig. 1b |

The profile mask that limits gridding to depths actually sampled by each dive is rebuilt from the raw profiles in step 01. A precomputed mask can be passed with `--mask-file` instead.

## Running

```bash
export SEALFRONTS_DATA=/path/to/DATA_DIR
export SEALFRONTS_WORK=/path/to/work      # intermediate products (default: ./work)
export SEALFRONTS_FIGS=/path/to/figures   # figures (default: ./figures)
WORKERS=8 ./run_all.sh
```

The full run takes about 45 minutes on a 14-core laptop. Step 05 (gap-filling the SLA by optimal interpolation) takes about 25 minutes. Step 06 takes about 18 minutes with 10 workers. Each figure script takes seconds once steps 01–06 have run.

## Notes on reproducibility

- **Single seal by default.** `02_clean_seal.py` processes only the paper seal. In the original processing the N² smoothing was applied to five seal records concatenated in time. This changes `N2_smooth` in the first and last 12 profiles of the track (about 0.6% of the record). To reproduce the original exactly, run
  `python scripts/02_clean_seal.py --tags ct158-198-19 ct158-199-19 ct176-04-23 ct176-06-23 ct176-09-23`,
  with the other tags' gridded files in `WORK_DIR`.
- **Fixed Monte Carlo seed.** The ensemble uses `numpy.random.default_rng(42)`, so results are deterministic.
- **xinvert version.** The omega solver relies on internal helpers of `xinvert` 0.1.7. Use that exact version.

## Citation

If you use this code, please cite the paper above. <!-- TODO: add DOI and Zenodo archive DOI on acceptance -->

## License

MIT (see `LICENSE`).
