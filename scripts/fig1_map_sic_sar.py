"""Fig. 1 -- regional map: sea-ice concentration and SAR composites.

(a) Mean NSIDC v6 sea-ice concentration (SIC) during the seal's voyage,
    averaged in time over pixels within 100 km of the seal position of the
    day, with the seal track coloured by week, the 1000 m RTopo-2 isobath,
    the two SAR boxes of panel (b) and an Antarctic inset.
(b) Two 3-day Sentinel-1 SAR composites (OceanDataLab Syntool extractions)
    mosaicked side by side, with the seal track over the matching periods
    coloured by day.

Inputs : config.DS_CLEAN, config.SIC_DIR, config.RTOPO_FILE,
         config.SAR_DIR (the two files listed in SAR_COMPOSITES)
Output : <FIG_DIR>/combined_SIC_SAR_map.png

Run from the repository root:  python scripts/fig1_map_sic_sar.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cmocean
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from pyproj import CRS, Transformer
from scipy.ndimage import gaussian_filter

from sealfronts import config
from sealfronts import maps

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# 3-day SAR composites (Sentinel-1 EW HH, "SAR roughness" layer) exported from
# https://aviso-calval.oceandatalab.com/ (Syntool). Order matters: the second
# composite is drawn on top of the first where both have data.
SAR_COMPOSITES = [
    # 17-18-19 March 2024 ("Thin ice")
    dict(file="sar_composite_20240317-19.npz",
         bbox=dict(lat_min=-69.9868, lat_max=-67.3990, lon_min=27.9932, lon_max=35.9802)),
    # 3-4-5 February 2024 ("Thick ice front")
    dict(file="sar_composite_20240203-05.npz",
         bbox=dict(lat_min=-69.9680, lat_max=-66.6007, lon_min=34.0247, lon_max=47.9773)),
]

TRACK_START = pd.Timestamp("2024-01-12")  # first day of the weekly bins
WEEK = 7                                   # bin length (days)
SIC_BUFFER_KM = 100                        # radius around the seal for SIC mean

TOP_XLIM = (28.9, 60.1)
TOP_YLIM = (-70.5, -64.5)
BOT_XMAX = 43.05                           # panel (b) starts at the SAR western edge
BOT_YLIM = (-69.6, -67.2)
BOUNDARY_LON = 34.0                        # limit between the two composites

# Dashed boxes in (a) outlining the SAR areas of (b): (lon0, lat0, width, height)
SAR_BOXES = [
    (29.5, -69.6, 34 - 29.5, -67.4 - (-69.6)),
    (34.0, -69.6, 43.05 - 34.0, -67.2 - (-69.6)),
]

# Brightening of the darker February composite (34-40E) for visibility
ENHANCE_BOX = dict(lon=(34.0, 40.0), lat=(-69.6, -67.4))
ENHANCE_GAIN, ENHANCE_OFFSET, ENHANCE_SMOOTH = 1.45, 0.05, 4

# Seal-track periods shown in (b) and positions of their colorbars
DAILY_TRACKS = [
    dict(start="2024-03-13 21:30", end="2024-03-22 05:38",
         cbar_box=[0.12, 0.485, 0.18, 0.012], zorder=38),
    dict(start="2024-01-30 20:30", end="2024-02-05 03:00",
         cbar_box=[0.665, 0.32, 0.22, 0.014], zorder=39),
]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def mean_sic_near_track(track, sic_dir, buffer_km):
    """Time-mean SIC over pixels within ``buffer_km`` of the seal.

    For each daily SIC map between the first and last seal positions, only
    pixels within ``buffer_km`` of the seal position closest in time are
    kept; the masked maps are then averaged in time.
    """
    t0, t1 = track["time"].min(), track["time"].max()
    files = [f for f in sorted(Path(sic_dir).glob("sic_pss25_*.nc"))
             if t0.floor("D") <= pd.Timestamp(f.name.split("_")[2]) <= t1]
    ds = xr.open_mfdataset(files, combine="by_coords", data_vars="minimal",
                           coords="minimal", compat="override")
    sic = ds["cdr_seaice_conc"]
    sic = sic.where((sic >= 0) & (sic <= 1)).sel(time=slice(t0, t1)).load()

    to_xy = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3412),
                                 always_xy=True)
    seal_x, seal_y = to_xy.transform(track["lon"].values, track["lat"].values)
    x2d, y2d = np.meshgrid(sic["x"].values, sic["y"].values)
    seal_time = pd.DatetimeIndex(track["time"])

    masks = []
    for tt in pd.to_datetime(sic["time"].values):
        i = np.argmin(np.abs(seal_time - tt))
        dist2 = (x2d - seal_x[i]) ** 2 + (y2d - seal_y[i]) ** 2
        masks.append(dist2 <= (buffer_km * 1e3) ** 2)
    mask = xr.DataArray(np.stack(masks), dims=("time", "y", "x"),
                        coords={"time": sic["time"]})
    return sic.where(mask).mean(dim="time", skipna=True)


def mosaic_sar(composites):
    """Mosaic SAR extractions on a common regular lon/lat grid.

    The grid resolution is that of the first image; each image is
    nearest-neighbour interpolated and masked outside its own footprint.
    Later images take priority where images overlap.
    """
    das = [maps.load_syntool_sar(config.SAR_DIR / c["file"], c["bbox"]).astype(float)
           for c in composites]
    dlat = float(np.abs(np.diff(das[0].latitude)).mean())
    dlon = float(np.abs(np.diff(das[0].longitude)).mean())
    lat = np.arange(min(float(d.latitude.min()) for d in das),
                    max(float(d.latitude.max()) for d in das) + dlat, dlat)
    lon = np.arange(min(float(d.longitude.min()) for d in das),
                    max(float(d.longitude.max()) for d in das) + dlon, dlon)

    mosaic = None
    for da in das:
        on_grid = da.interp(latitude=lat, longitude=lon, method="nearest",
                            kwargs={"fill_value": np.nan})
        on_grid = on_grid.where(
            (on_grid.latitude >= da.latitude.min())
            & (on_grid.latitude <= da.latitude.max())
            & (on_grid.longitude >= da.longitude.min())
            & (on_grid.longitude <= da.longitude.max()))
        mosaic = on_grid if mosaic is None else on_grid.combine_first(mosaic)
    return mosaic


def enhance_region(sar):
    """Brighten SAR inside ENHANCE_BOX with smoothed edges.

    Returns the display image and the (vmin, vmax) display range taken as
    the 1st and 99.5th percentiles of the raw mosaic.
    """
    raw = sar.values
    lon2d, lat2d = np.meshgrid(sar.longitude.values, sar.latitude.values)
    box = ((lon2d >= ENHANCE_BOX["lon"][0]) & (lon2d <= ENHANCE_BOX["lon"][1])
           & (lat2d >= ENHANCE_BOX["lat"][0]) & (lat2d <= ENHANCE_BOX["lat"][1]))
    weight = np.clip(gaussian_filter(box.astype(float), sigma=ENHANCE_SMOOTH), 0, 1)
    lo, hi = np.nanpercentile(raw, 1), np.nanpercentile(raw, 99.5)
    enhanced = np.clip(raw * ENHANCE_GAIN + ENHANCE_OFFSET, lo, hi)
    return raw * (1 - weight) + enhanced * weight, (lo, hi)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def add_gridlines(ax, nx):
    """Dashed grey gridlines with lon/lat labels on the bottom and left."""
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = cticker.LongitudeLocator(nbins=nx)
    gl.ylocator = cticker.LatitudeLocator(nbins=10)
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}


def label(ax, x, y, text, **kw):
    """Text at a lon/lat position (PlateCarree)."""
    ax.text(x, y, text, transform=ccrs.PlateCarree(), **kw)


def plot_panel_a(fig, ax, track, sic_mean, bathy):
    """SIC, weekly-binned track, isobath, SAR boxes, labels and inset."""
    maps.add_panel_letter(ax, "a")

    im = ax.pcolormesh(sic_mean["x"].values, sic_mean["y"].values,
                       sic_mean.values, cmap=cmocean.cm.ice,
                       norm=PowerNorm(gamma=0.5, vmin=0, vmax=1),
                       shading="auto", transform=ccrs.epsg(3412), zorder=0,
                       alpha=0.9)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.005,
                        fraction=0.015)
    cbar.set_label("Mean sea ice concentration")

    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "land", "10m", edgecolor="black", facecolor="lightgray",
        alpha=0.5), zorder=1)
    ax.coastlines(resolution="10m", linewidth=0.6, color="black", zorder=2)

    for x0, y0, w, h in SAR_BOXES:
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none",
                               edgecolor="0.55", linewidth=1.6,
                               linestyle="--", alpha=0.85,
                               transform=ccrs.PlateCarree(), zorder=3))

    # Weekly bins starting on TRACK_START
    day = track["time"].dt.floor("D")
    week = ((day - TRACK_START).dt.days // WEEK).values
    keep = week >= 0
    n_weeks = week[keep].max() + 1
    cmap = mcolors.ListedColormap(
        plt.get_cmap("plasma")(np.linspace(0.1, 1, n_weeks)))
    bounds = np.arange(n_weeks + 1) - 0.5
    sc = ax.scatter(track["lon"].values[keep], track["lat"].values[keep],
                    c=week[keep], cmap=cmap,
                    norm=mcolors.BoundaryNorm(bounds, n_weeks), s=10,
                    linewidth=0, transform=ccrs.PlateCarree(), zorder=6)

    ax.set_xlim(*TOP_XLIM)
    ax.set_ylim(*TOP_YLIM)
    maps.set_lat_aspect(ax, *TOP_YLIM)
    maps.add_isobath_line(ax, bathy, level=-1000, color="white", lw=1.2,
                          zorder=4, min_length=250)
    add_gridlines(ax, nx=20)

    cax = fig.add_axes([0.49, 0.627, 0.35, 0.014])
    cbar_track = fig.colorbar(sc, cax=cax, orientation="horizontal",
                              boundaries=bounds, ticks=np.arange(n_weeks),
                              spacing="uniform")
    ticks = []
    for k in range(n_weeks):
        d0 = TRACK_START + pd.Timedelta(days=WEEK * k)
        d1 = min(d0 + pd.Timedelta(days=WEEK - 1), day.max())
        ticks.append(f"{d0:%m/%d}-{d1:%m/%d}")
    cbar_track.ax.set_xticklabels(ticks, rotation=45, ha="right")
    cbar_track.ax.tick_params(labelsize=8)
    cbar_track.set_label("Seal track, weekly bins")

    maps.add_antarctica_inset(fig, [0.14, 0.72, 0.21, 0.17], TOP_XLIM, TOP_YLIM)

    no_box = dict(facecolor="none", edgecolor="none", pad=2)
    label(ax, 37.2, -69, "Lützow-Holm Bay", fontsize=11, color="black",
          ha="center", bbox=no_box, zorder=50)
    label(ax, 38.9, -70.4, "Queen Maud Land", fontsize=11, color="black",
          ha="center", bbox=no_box, zorder=40)
    label(ax, 53.4, -67, "Enderby Land", fontsize=11, color="black",
          ha="center", bbox=no_box, zorder=40)
    label(ax, 38.5, -70, "Shirase Glacier", fontsize=10, color="black",
          ha="center", bbox=dict(facecolor="white", alpha=0.7,
                                 edgecolor="none", pad=2), zorder=50)


def plot_panel_b(fig, ax, track, sar, bathy):
    """SAR mosaic, isobath, daily-binned tracks and annotations."""
    maps.add_panel_letter(ax, "b")

    img, (vmin, vmax) = enhance_region(sar)
    ax.pcolormesh(sar.longitude.values, sar.latitude.values, img,
                  shading="auto", cmap="gray", vmin=vmin, vmax=vmax,
                  transform=ccrs.PlateCarree())
    xmin = float(sar.longitude.min())
    ax.set_xlim(xmin, BOT_XMAX)
    ax.set_ylim(*BOT_YLIM)

    # White separator (black-edged) between the two composites
    for color, lw, z in (("black", 4.5, 9), ("white", 3, 10)):
        ax.plot([BOUNDARY_LON, BOUNDARY_LON], BOT_YLIM, color=color,
                linewidth=lw, transform=ccrs.PlateCarree(), zorder=z)

    add_gridlines(ax, nx=25)
    maps.set_lat_aspect(ax, *BOT_YLIM)
    maps.add_isobath_line(ax, bathy, level=-1000, color="white", lw=1.2,
                          zorder=20, min_length=250)

    # Feature annotations
    pc = ccrs.PlateCarree()._as_mpl_transform(ax)
    dark_box = dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3)
    ax.annotate("mesoscale eddy\n(cyclone)", xy=(37.0, -68.25),
                xytext=(35.6, -67.75), xycoords=pc, textcoords=pc,
                color="white", fontsize=10, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.2,
                                shrinkA=4, shrinkB=4),
                bbox=dark_box, zorder=60)

    filament_label = (38.5, -67.6)
    label(ax, *filament_label, "submesoscale filaments", color="white",
          fontsize=10, ha="center", va="bottom", bbox=dark_box, zorder=19)
    for xy in [(37.5, -68.25), (37.6, -68.0), (39.1, -67.75), (40.3, -67.9)]:
        ax.annotate("", xy=xy, xytext=filament_label, xycoords=pc,
                    textcoords=pc, zorder=19,
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.2))

    title_box = dict(facecolor="black", alpha=0.55, edgecolor="none", pad=4)
    common = dict(color="white", fontsize=10, ha="center", va="top",
                  bbox=title_box, zorder=20)
    label(ax, 0.5 * (xmin + BOUNDARY_LON), -67.27, "Thin ice",
          fontweight="bold", **common)
    label(ax, 38.5, -67.27, "Thick ice front", fontweight="bold", **common)
    label(ax, 32.87, -69.4, "3-day SAR composite\n17–18–19 March 2024", **common)
    label(ax, 42, -69.4, "3-day SAR composite\n3–4–5 February 2024", **common)

    for kw in DAILY_TRACKS:
        maps.add_daily_seal_track(ax, fig, track, kw["start"], kw["end"],
                                  kw["cbar_box"], point_size=10,
                                  zorder=kw["zorder"])


def main():
    track = maps.load_seal_track(config.DS_CLEAN, config.SEAL_ID)
    sic_mean = mean_sic_near_track(track, config.SIC_DIR, SIC_BUFFER_KM)
    bathy = maps.load_bathymetry(config.RTOPO_FILE, (28, 61), (-71, -64))
    sar = mosaic_sar(SAR_COMPOSITES)

    fig = plt.figure(figsize=(14, 18))
    ax_top = fig.add_axes([0.07, 0.500, 0.82, 0.43], projection=ccrs.PlateCarree())
    ax_bot = fig.add_axes([0.07, 0.17, 0.82, 0.43], projection=ccrs.PlateCarree())
    plot_panel_a(fig, ax_top, track, sic_mean, bathy)
    plot_panel_b(fig, ax_bot, track, sar, bathy)

    out = config.FIG_DIR / "combined_SIC_SAR_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
