"""Optional: download the ERA5 data used by scripts/04_era5_along_track.py.

Requires a Copernicus Climate Data Store account and the ``cdsapi`` package,
with your personal API key in ``~/.cdsapirc`` (see
https://cds.climate.copernicus.eu/how-to-api). No credentials are stored here.

Downloads hourly ERA5 single-level fields for Jan-Mar 2024 over the seal
region to config.ERA5_ZIP (the CDS delivers a zip archive with one netCDF per
stream) and extracts it to config.ERA5_UNZIPPED_DIR.

Usage
    python scripts/00_download_era5.py
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sealfronts import config

REQUEST = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_surface_latent_heat_flux",
        "mean_surface_sensible_heat_flux",
        "2m_temperature",
        "mean_surface_downward_long_wave_radiation_flux",
        "mean_surface_downward_short_wave_radiation_flux",
        "mean_surface_net_long_wave_radiation_flux",
        "mean_surface_net_short_wave_radiation_flux",
    ],
    "year": ["2024"],
    "month": ["01", "02", "03"],
    "day": [f"{d:02d}" for d in range(1, 32)],
    "time": [f"{h:02d}:00" for h in range(24)],
    "area": [-59.00, 30.11, -69.95, 65.26],  # N, W, S, E
    "data_format": "netcdf",
    "download_format": "unarchived",
}


def main():
    import cdsapi

    config.ERA5_DIR.mkdir(parents=True, exist_ok=True)
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", REQUEST, str(config.ERA5_ZIP))
    if zipfile.is_zipfile(config.ERA5_ZIP):
        with zipfile.ZipFile(config.ERA5_ZIP) as z:
            z.extractall(config.ERA5_UNZIPPED_DIR)
    print(f"ERA5 data in {config.ERA5_UNZIPPED_DIR}")


if __name__ == "__main__":
    main()
