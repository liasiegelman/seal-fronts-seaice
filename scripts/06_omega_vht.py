"""Vertical velocity and vertical heat transport from the 2D QG omega equation.

Inputs : config.DS_CLEAN, config.STRAIN_NPZ
Outputs: config.OMEGA_NC     reference solution (w, VHF, WB, WS) forced by the
                             smoothed SO-SLICE along-track strain (Figs. 2-5)
         config.OMEGA_MC_NC  Monte Carlo ensemble (reference + 50 perturbed
                             strain realizations) of w and VHT (ED Fig. 3)

Usage (from the repository root):
    python scripts/06_omega_vht.py [--workers N] [--skip-mc]

Each realization takes ~1.5 min on one core (7 chunks x 2000 SOR
iterations); the 51 inversions of the ensemble are independent and can be
spread over ``--workers`` processes.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sealfronts import config, omega  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel processes for the Monte Carlo ensemble")
    parser.add_argument("--skip-mc", action="store_true",
                        help="only compute the reference solution")
    args = parser.parse_args()

    ds, theta_smooth = omega.load_seal_section(
        config.DS_CLEAN, config.STRAIN_NPZ, config.SEAL_ID)
    inputs = omega.prepare_omega_inputs(ds)

    # --- reference solution -------------------------------------------------
    ds_out = omega.build_omega_dataset(ds, inputs)
    ds_out.to_netcdf(config.OMEGA_NC)
    print(f"saved {config.OMEGA_NC}")
    if args.skip_mc:
        return

    # --- Monte Carlo ensemble ----------------------------------------------
    # noise std = std of the reference along-track strain [s-1]
    sigma_eps = np.nanstd(ds["ess_smooth"].values)
    strain_mc = omega.make_projected_strain_mc(
        ds["sn"], ds["ss"], theta_smooth, sigma_eps, **omega.MC_PARAMS)
    strain_real = omega.omega_strain_realizations(
        inputs[omega.CONTROL_STRAIN], strain_mc)
    print(f"solving the omega equation for "
          f"{strain_real.sizes['realization']} strain realizations")

    ds_mc = omega.compute_w_vht_realizations(
        inputs, strain_real, workers=args.workers)
    ds_mc.attrs.update(
        strain_source=omega.STRAIN_SOURCE,
        omega_strain_input_name=omega.CONTROL_STRAIN,
        omega_n_realizations=int(strain_real.sizes["realization"]),
        omega_control_realization=omega.CONTROL_STRAIN,
        mc_sigma_eps=float(sigma_eps),
        **{f"mc_{k}": str(v) for k, v in omega.MC_PARAMS.items()},
    )
    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for v in ds_mc.data_vars}
    ds_mc.to_netcdf(config.OMEGA_MC_NC, encoding=encoding)
    print(f"saved {config.OMEGA_MC_NC}")


if __name__ == "__main__":
    main()
