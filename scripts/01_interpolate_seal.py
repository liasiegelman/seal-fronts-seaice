"""Step 1 - interpolate the seal CTD profiles onto a regular along-track section.

Input : raw MEOP profiles  <DATA_DIR>/seal/<tag>_fr1_prof.nc   (config.SEAL_NC)
Output: <WORK_DIR>/interpolated_data_<tag>_fr1_prof.npz          (config.INTERP_NPZ)

The npz holds, on a 1 km (along-track) x 1 m (depth, 0-999) grid:
lon, lat, day (1 km grid), ct, sa, mask_prof, b, f, N2 (smoothed),
mld, bx, M4, qertel (Ertel PV) and inv_ri (smoothed Ri^-1).

The depth mask (1 above the deepest valid sample of the nearest profile,
NaN below) is rebuilt from the raw file; it is identical to the
``seal/mask_prof_<tag>_fr1_prof.npy`` file of the original analysis
(config.SEAL_MASK), which can be used instead with ``--mask-file``.

Usage (from the repository root):
    python scripts/01_interpolate_seal.py                 # paper seal (config.SEAL_ID)
    python scripts/01_interpolate_seal.py --tags ct176-06-23 ct176-09-23
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sealfronts import config, seal  # noqa: E402


def process_tag(tag, mask_file=None):
    """Interpolate one tag and write its npz file."""
    nc_path = seal.raw_nc_path(tag)
    print(f"[{tag}] reading {nc_path}")
    prof = seal.load_profiles(nc_path, adjusted=True)

    if mask_file is not None:
        m_prof = np.load(mask_file)
    else:
        _, distcum = seal.along_track_distance(prof["lat"], prof["lon"])
        m_prof = seal.profile_depth_mask(seal.load_profiles(nc_path, adjusted=False),
                                         distcum)

    out = seal.interpolate_section(prof, m_prof)
    out_path = seal.interp_npz_path(tag)
    np.savez(out_path, **out)
    print(f"[{tag}] {out['b'].shape[1]} km section written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags", nargs="+", default=[config.SEAL_ID],
                        help="MEOP tag ids to process (default: %(default)s)")
    parser.add_argument("--mask-file", action="store_true",
                        help="use the precomputed seal/mask_prof_<tag>_fr1_prof.npy "
                             "instead of rebuilding the depth mask")
    args = parser.parse_args()

    # gsw / NaN-slice warnings on the many empty levels are expected
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for tag in args.tags:
        mask_file = (config.SEAL_DIR / f"mask_prof_{tag}_fr1_prof.npy"
                     if args.mask_file else None)
        process_tag(tag, mask_file)


if __name__ == "__main__":
    main()
