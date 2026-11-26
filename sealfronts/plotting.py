"""Shared plotting helpers for the along-track section and statistics figures."""
import gsw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from sealfronts import config

# Colours of the three periods
PERIOD_COLORS_TAB = {"thick": "tab:blue", "thin": "tab:orange", "full": "tab:green"}   # Fig. 4
PERIOD_COLORS_CB = {"thick": "#0072B2", "thin": "#D55E00", "full": "#009E73"}          # Fig. 5d
THF_COLOR = "#6A3D9A"


def savefig(fig, filename, dpi=300):
    """Save ``fig`` as ``config.FIG_DIR / filename`` and report the path."""
    out = config.FIG_DIR / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out}")
    return out


def panel_label(ax, label, fontsize=18, y=1.02):
    """Bold panel letter at the top-left corner, just above the axes."""
    ax.text(0.0, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", ha="left", va="bottom", clip_on=False)


def section(ax, field, depth, xmin, xmax, cmap, vmin, vmax):
    """Draw a (depth, profile) section with one pixel per profile and metre.

    The x axis is the profile index (``xmin``..``xmax``); the y axis is
    depth, increasing downward.
    """
    im = ax.imshow(np.asarray(field), origin="lower", aspect="auto",
                   extent=[xmin, xmax, depth[0], depth[-1]], cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="none")
    return im


def finish_section_axis(ax, xmin, xmax):
    """Common styling of a section axis (depth downward, grid, x limits)."""
    ax.invert_yaxis()
    ax.set_ylabel("Depth [m]")
    ax.grid(True, lw=0.5)
    ax.set_xlim(xmin, xmax)


def sci_colorbar(fig, im, cax, label):
    """Colorbar with a shared x10^n offset (for b, b_x, N^2)."""
    cbar = fig.colorbar(im, cax=cax, label=label)
    cbar.formatter = ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
    return cbar


def add_isopycnals(ax, sigma0, x, depth, color="k"):
    """Overlay labelled sigma_0 contours every 0.1 kg m-3."""
    finite = sigma0[np.isfinite(sigma0)]
    if finite.size == 0:
        return None
    levels = np.arange(np.floor(finite.min() * 10) / 10,
                       np.ceil(finite.max() * 10) / 10, 0.1)
    if levels.size == 0:
        return None
    cs = ax.contour(x, depth, sigma0, levels=levels, colors=color,
                    linewidths=0.7, alpha=0.70, zorder=7)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
    return cs


def below_freezing(temp, salt, depth, lat):
    """Boolean (depth, profile) mask of water colder than the freezing point.

    ``temp`` is Conservative Temperature and ``salt`` Absolute Salinity
    (2-D arrays); the freezing CT is computed for air-free sea water at the
    local pressure.
    """
    p = gsw.p_from_z(-np.asarray(depth)[:, None], np.asarray(lat)[None, :])
    ct_freeze = gsw.CT_freezing(salt, p, saturation_fraction=0)
    return ((temp - ct_freeze) < 0) & np.isfinite(temp)


def scatter_below_freezing(ax, mask, x, depth, zorder=9):
    """Mark below-freezing points as small cyan dots."""
    yy, xx = np.where(mask)
    if xx.size:
        ax.scatter(x[xx], np.asarray(depth)[yy], s=8, color="cyan", linewidth=0,
                   alpha=0.9, label="Below freezing", zorder=zorder)


def section_legend(ax):
    """White-on-black legend used on the temperature sections."""
    ax.legend(loc="lower right", frameon=True, facecolor="black",
              edgecolor="none", framealpha=0.7, labelcolor="white")


def add_date_top_axis(ax, x, times, xmin, xmax):
    """Twin x axis with a tick at each midnight (``mm-dd``) inside the section."""
    ax_top = ax.twiny()
    ax_top.set_xlim(xmin, xmax)
    t = pd.DatetimeIndex(times)
    first = t.min().normalize()
    if t.min() > first:
        first += pd.Timedelta(days=1)
    days = pd.date_range(first, t.max().normalize(), freq="D")
    pos = [x[np.argmin(np.abs(t - d))] for d in days]
    ax_top.set_xticks(pos)
    ax_top.set_xticklabels([d.strftime("%m-%d") for d in days], rotation=25, ha="left")
    ax_top.set_xlabel("Date")
    ax_top.set_xlim(xmin, xmax)
    return ax_top


def set_distance_axis(ax, x, dist_km, xmin, xmax):
    """Relabel the profile-index ticks of ``ax`` with along-track distance [km]."""
    pos = ax.get_xticks()
    pos = pos[(pos >= xmin) & (pos <= xmax)]
    idx = np.clip(np.round(pos).astype(int), 0, len(x) - 1)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{dist_km[i]:.0f}" for i in idx])
    ax.set_xlabel("Along-track distance [km]")


def use_font_size(size):
    """Set the default matplotlib font size for the figure being built."""
    plt.rcParams["font.size"] = size
