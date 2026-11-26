"""Step 2 - quality-control the seal salinity and build the analysis dataset.

Inputs : <WORK_DIR>/interpolated_data_<tag>_fr1_prof.npz   (step 1, config.INTERP_NPZ)
         <WORK_DIR>/sic_<tag>.npz                          (along-track SIC, config.SIC_NPZ)
Output : <WORK_DIR>/ds_clean.nc                            (config.DS_CLEAN)

Steps
1. Assemble each tag's section and SIC into an xarray Dataset on (depth, time),
   with ``lat``, ``lon`` and ``name`` (tag id) coordinates along time.
2. Keep profiles in sea ice (SIC >= 2 %) or south of 64.6 S.
3. Remove near-surface / fresh salinity spikes (see seal.qc_salinity_spikes);
   flagged points are set to NaN in all variables.
4. Recompute N2 from the cleaned T/S and add a smoothed version, ``N2_smooth``.

By default only the paper seal (config.SEAL_ID) is processed. Passing several
tags concatenates them along time, in the given order, before step 4 (this is
how the original multi-tag ds_clean.nc was made).

Usage (from the repository root):
    python scripts/02_clean_seal.py [--tags TAG ...] [--qc-plots]
"""
import argparse
import sys
import warnings
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sealfronts import config, seal  # noqa: E402


def plot_salinity_qc(ds_raw, ds_clean, tag, chunk=550, zmax=15):
    """Save near-surface salinity sections before/after QC (one figure per chunk of profiles)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = ds_raw["salinity"].sel(depth=slice(0, zmax))
    cln = ds_clean["salinity"].sel(depth=slice(0, zmax))
    for i in range(0, ds_raw.sizes["time"], chunk):
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        for ax, da, title in zip(axes, (raw, cln), ("raw", "after QC")):
            sub = da.isel(time=slice(i, i + chunk))
            img = ax.imshow(sub.values, origin="lower", aspect="auto",
                            extent=(i, i + sub.sizes["time"], zmax + 0.5, -0.5))
            ax.set_ylabel("Depth [m]")
            ax.set_title(f"{tag} salinity, {title}")
            fig.colorbar(img, ax=ax, label="S [psu]")
        axes[-1].set_xlabel("Profile index (1 km spacing)")
        out = config.FIG_DIR / f"qc_salinity_{tag}_{i:05d}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags", nargs="+", default=[config.SEAL_ID],
                        help="MEOP tag ids to include (default: %(default)s)")
    parser.add_argument("--qc-plots", action="store_true",
                        help="save before/after salinity QC sections in FIG_DIR")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    ds = xr.concat(
        [seal.build_tag_dataset(tag, seal.interp_npz_path(tag), seal.sic_npz_path(tag))
         for tag in args.tags],
        dim="time",
    )
    ds_ice = seal.select_under_ice(ds)

    clean_tags = []
    for tag in args.tags:
        ds_tag = ds_ice.where(ds_ice["name"] == tag, drop=True)
        ds_tag_clean, bad, _ = seal.qc_salinity_spikes(ds_tag)
        print(f"[{tag}] {ds_tag.sizes['time']} profiles under ice, "
              f"{int(bad.sum())} points flagged by the salinity QC")
        if args.qc_plots:
            plot_salinity_qc(ds_tag, ds_tag_clean, tag)
        clean_tags.append(ds_tag_clean)
    ds_clean = xr.concat(clean_tags, dim="time")

    ds_clean = seal.add_smoothed_n2(ds_clean, sigma=3.0)

    ds_clean.to_netcdf(config.DS_CLEAN)
    print(f"written {config.DS_CLEAN}")


if __name__ == "__main__":
    main()
