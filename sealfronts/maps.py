"""Shared helpers for the map figure (Fig. 1).

Covers loading the seal track, Syntool SAR extractions and RTopo-2
bathymetry, and a few cartopy drawing utilities.
"""

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.patches import Polygon as MplPolygon
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_seal_track(ds_clean_path, seal_id):
    """Return the positions of one seal as a DataFrame (time, lon, lat).

    Only the ``time``, ``lon``, ``lat`` and ``name`` coordinates of the
    cleaned dataset are read.
    """
    with xr.open_dataset(ds_clean_path) as ds:
        name = ds["name"].values
        keep = name == seal_id
        track = pd.DataFrame({
            "time": pd.to_datetime(ds["time"].values[keep]),
            "lon": ds["lon"].values[keep],
            "lat": ds["lat"].values[keep],
        })
    if track.empty:
        raise ValueError(f"No profiles for seal {seal_id} in {ds_clean_path}")
    return track


def load_syntool_sar(path, bbox):
    """Load a Syntool SAR extraction as a (latitude, longitude) DataArray.

    The first layer stored in the .npz (``3857_SAR_roughness``, an RGBA uint8
    image) is used, and its first (grey-level) channel is returned. The image
    rows are flipped so that latitude increases with row index. Longitude and
    latitude are taken as evenly spaced between the bounds of ``bbox``
    (dict with lat_min, lat_max, lon_min, lon_max).
    """
    with np.load(path, allow_pickle=True) as f:
        data = f[f.files[0]]
    img = data[:, :, 0] if data.ndim == 3 else data
    ny, nx = img.shape
    b = bbox
    da = xr.DataArray(
        img[::-1, :],
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.linspace(b["lat_min"], b["lat_max"], ny),
            "longitude": np.linspace(b["lon_min"], b["lon_max"], nx),
        },
        name="sar",
    )
    return da


def load_bathymetry(rtopo_file, lon_range, lat_range):
    """Read RTopo-2 bedrock topography (m) over a lon/lat box."""
    with xr.open_dataset(rtopo_file) as ds:
        lon = ds["lon"].values
        lat = ds["lat"].values
        ilon = np.where((lon >= lon_range[0]) & (lon <= lon_range[1]))[0]
        ilat = np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
        bathy = ds["bedrock_topography"].isel(
            londim=slice(ilon[0], ilon[-1] + 1),
            latdim=slice(ilat[0], ilat[-1] + 1),
        ).load()
    return bathy


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def set_lat_aspect(ax, lat_min, lat_max):
    """Set a lon/lat axis aspect ratio that is locally isotropic in km."""
    lat0 = 0.5 * (lat_min + lat_max)
    ax.set_aspect(1 / np.cos(np.deg2rad(lat0)), adjustable="box")


def add_panel_letter(ax, letter):
    """Bold panel letter just above the top-left corner of ``ax``."""
    ax.text(0.0, 1.01, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", ha="left", va="bottom", clip_on=False)


def add_isobath_line(ax, bathy, level=-1000, color="white", lw=1.2,
                     zorder=4, min_length=250):
    """Draw one bathymetric contour, dropping segments shorter than
    ``min_length`` vertices (small loops/fragments)."""
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(bathy["lon"].values, bathy["lat"].values,
                        bathy.values, levels=[level])
    for seg in cs.allsegs[0]:
        if len(seg) < min_length:
            continue
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=lw,
                transform=ccrs.PlateCarree(), zorder=zorder)
    plt.close(fig_tmp)


def add_antarctica_inset(fig, rect, lon_range, lat_range, lim=3.2e6):
    """Add a circular south-polar inset map with the study region in red.

    Natural Earth 110 m land south of 55S is drawn in polar stereographic
    coordinates (EPSG:3031).
    """
    ax = fig.add_axes(rect)
    tfm = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    shp = shpreader.natural_earth(resolution="110m", category="physical",
                                  name="land")

    edge_patches, fill_patches = [], []
    for geom in shpreader.Reader(shp).geometries():
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            lon, lat = (np.asarray(v) for v in poly.exterior.xy)
            if np.nanmax(lat) > -55:
                continue
            coords = np.column_stack(tfm.transform(lon, lat))
            if np.allclose(coords[0], coords[-1]):
                coords = coords[:-1]
            edge_patches.append(MplPolygon(coords, closed=False))
            fill_patches.append(MplPolygon(coords, closed=True))

    ax.add_collection(PatchCollection(edge_patches, facecolor="none",
                                      edgecolor="black", linewidth=0.6,
                                      zorder=2))
    ax.add_collection(PatchCollection(fill_patches, facecolor="lightgray",
                                      edgecolor="none", zorder=3))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    lon0, lon1 = lon_range
    lat0, lat1 = lat_range
    rx, ry = tfm.transform(np.array([lon0, lon1, lon1, lon0, lon0]),
                           np.array([lat0, lat0, lat1, lat1, lat0]))
    ax.plot(rx, ry, color="red", linewidth=1.5, zorder=5)

    bg = Circle((0.5, 0.5), 0.5, transform=ax.transAxes, facecolor="white",
                edgecolor="none", zorder=-10)
    ax.add_patch(bg)
    clip = Circle((0.5, 0.5), 0.5, transform=ax.transAxes, facecolor="none",
                  edgecolor="none")
    for artist in ax.get_children():
        if artist is not bg:
            artist.set_clip_path(clip)
    ax.add_patch(Circle((0.5, 0.5), 0.5, transform=ax.transAxes,
                        facecolor="none", edgecolor="black", linewidth=0.8,
                        zorder=100))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def add_daily_seal_track(ax, fig, track, start, end, cbar_box,
                         label="Seal track, daily bins\nZoomed time period",
                         point_size=10, zorder=30):
    """Scatter the seal positions between ``start`` and ``end`` coloured by
    day (plasma), with a white-styled horizontal colorbar at ``cbar_box``
    (figure coordinates)."""
    sel = track[(track["time"] >= pd.Timestamp(start))
                & (track["time"] <= pd.Timestamp(end))]
    day = sel["time"].dt.floor("D")
    unique_days = np.sort(day.unique())
    day_index = np.searchsorted(unique_days, day.values)
    n_days = len(unique_days)

    cmap = plt.get_cmap("plasma", n_days)
    norm = mcolors.BoundaryNorm(np.arange(n_days + 1) - 0.5, n_days)
    sc = ax.scatter(sel["lon"], sel["lat"], c=day_index, cmap=cmap, norm=norm,
                    s=point_size, edgecolor=None, linewidth=0.25,
                    transform=ccrs.PlateCarree(), zorder=zorder)

    cax = fig.add_axes(cbar_box)
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal",
                        boundaries=np.arange(n_days + 1) - 0.5,
                        ticks=np.arange(n_days), spacing="uniform")
    cbar.ax.set_xticklabels(
        [pd.Timestamp(d).strftime("%m/%d") for d in unique_days],
        rotation=25, ha="right")
    for tick_label in cbar.ax.get_xticklabels():
        tick_label.set_bbox(dict(facecolor="black", alpha=0.5,
                                 edgecolor="none", pad=1))
    cbar.ax.tick_params(which="both", colors="white", labelcolor="white",
                        labelsize=8)
    cbar.ax.minorticks_off()
    cbar.set_label(label, fontsize=9, color="white")
    cbar.outline.set_edgecolor("white")
    cbar.ax.set_facecolor((0, 0, 0, 0.3))
    return sc, cbar
