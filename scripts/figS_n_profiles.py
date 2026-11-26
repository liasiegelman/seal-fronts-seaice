"""ED Fig. 2 -- number of seal profiles as a function of depth.

Output: ``Figures_SI/ct176-04-23_n_profiles_vs_depth_temperature.png``
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from sealfronts import config, io
from sealfronts.plotting import savefig, use_font_size


def profile_count(ds):
    """Number of profiles with data at each depth level."""
    return (ds["maskprof"] > 0).sum("distance")


def main():
    use_font_size(18)
    ds = io.load_omega()
    n = profile_count(ds)

    fig, ax = plt.subplots(figsize=(5, 8))
    ax.plot(n.values, ds.depth.values, lw=2, c="k")
    ax.set_ylim(0, 805)
    ax.invert_yaxis()
    ax.grid(True, lw=0.5)
    ax.set_xlabel("Number of profiles")
    ax.set_ylabel("Depth [m]")
    ax.set_xscale("log")
    ax.set_xlim(1, 5000)
    plt.tight_layout()
    savefig(fig, f"{config.SEAL_ID}_n_profiles_vs_depth_temperature.png")


if __name__ == "__main__":
    main()
