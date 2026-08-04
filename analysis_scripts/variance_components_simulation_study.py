"""
Simulation study for the multi-mirror variance-components likelihood.

Section 8.3 of the soiling model notes argues that mirrors measured together at one site
do not receive independent deposition noise, and that treating them as independent
understates the uncertainty in the fitted mean parameter. This script checks that claim,
and quantifies how much data is needed before the common/mirror-specific split is worth
reporting.

Data are simulated from the generative process (``simulate_reflectance_data``), not from
the covariance the likelihood assembles, so agreement between the two is evidence about
the implementation rather than a tautology.

Three questions:

1. Consistency. Does the bias in the fitted parameters shrink as experiments are added?
2. Coverage. Do the 95% Wald intervals from the transformed-space Hessian actually cover
   at 95%, for a range of kappa? And does the scalar model under-cover the mean parameter
   on the same data, which is the concrete claim section 8.3 makes?
3. Design. How many mirrors are needed before kappa is usefully identified?

The model fitted is ConstantMeanDeposition: it needs only tilts, a dust concentration and
a reference density, so the inputs are fully synthetic. The claim under test concerns the
covariance structure rather than how the mean is computed, so it carries over to hrz0.

Runtime is of order 15 minutes at the default settings. Results are written to
``results/variance_components_study/``.

Usage:
    python analysis_scripts/variance_components_simulation_study.py
    python analysis_scripts/variance_components_simulation_study.py --replicates 50
"""

import argparse
import time
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heliosoil.base_models import ConstantMeanBase
from heliosoil.fitting import ConstantMeanDeposition


# Ground truth. mu_tilde is small enough that the accumulated loss over an experiment
# leaves the mirrors reflective; see _mean_parameter_bounds for the admissible range.
MU_TILDE = 3.0e-4
SIGMA_DEP = 6.0e-4
NOMINAL_REFLECTANCE = 0.95
INCIDENCE_ANGLE = 15.0
DUST_DENSITY = 40.0
MEASUREMENT_SIGMA = 1.2e-3
NUMBER_OF_MEASUREMENTS = 4.0

# One experiment is a week of hourly data with daily measurements.
TIMESTEPS = 169
N_MEASUREMENTS = 8

CONFIDENCE = 1.96  # two-sided 95%


def build_experiment(n_experiments, n_mirrors, rng):
    """A synthetic model and simulation inputs with ``n_experiments`` experiments."""
    model = ConstantMeanDeposition.__new__(ConstantMeanDeposition)
    ConstantMeanBase.__init__(model)
    model.set_variance_model("components")
    model.mu_tilde, model.sigma_dep = MU_TILDE, SIGMA_DEP
    model.helios.nominal_reflectance = NOMINAL_REFLECTANCE

    keys = range(n_experiments)
    model.helios.tilt = {f: rng.uniform(0.0, 60.0, size=(n_mirrors, TIMESTEPS)) for f in keys}
    model.helios.incidence_angle = {f: INCIDENCE_ANGLE for f in keys}
    model.helios.inc_ref_factor = {
        f: np.array(2.0 / np.cos(np.radians(INCIDENCE_ANGLE))) for f in keys
    }
    for name in ("delta_soiled_area", "delta_soiled_area_variance",
                 "soiling_factor", "soiling_factor_prediction_variance"):
        setattr(model.helios, name, {})

    simulation_inputs = types.SimpleNamespace(
        files=[f"synthetic_{f}" for f in keys],
        time={f: np.arange(TIMESTEPS, dtype=float) for f in keys},
        dust=types.SimpleNamespace(PM10={f: DUST_DENSITY for f in keys}),
        # A diurnal-looking dust signal, so the loading varies within an experiment.
        dust_concentration={
            f: 40.0 + 25.0 * np.sin(np.arange(TIMESTEPS) / 6.0) + rng.uniform(0, 10, TIMESTEPS)
            for f in keys
        },
        dust_type={f: "PM10" for f in keys},
    )
    indices = np.linspace(0, TIMESTEPS - 1, N_MEASUREMENTS + 1).astype(int).tolist()
    return model, simulation_inputs, indices


def fit_once(n_experiments, n_mirrors, kappa, rng):
    """Simulate one dataset and fit it under both variance models.

    Returns:
        dict: Estimates and interval half-widths on the fitting scale, or None if either
        fit failed. Failures are counted rather than retried; a study that silently drops
        hard cases would overstate coverage.
    """
    model, simulation_inputs, indices = build_experiment(n_experiments, n_mirrors, rng)
    truth = [MU_TILDE, SIGMA_DEP, kappa]

    data = model.simulate_reflectance_data(
        simulation_inputs,
        truth,
        indices,
        MEASUREMENT_SIGMA,
        number_of_measurements=NUMBER_OF_MEASUREMENTS,
        rng=rng,
    )

    result = {}
    try:
        model.set_variance_model("components")
        estimate, covariance = model.fit_mle(simulation_inputs, data, verbose=False)
        errors = np.sqrt(np.diag(covariance))
        result["components"] = (estimate, errors)

        model.set_variance_model("scalar")
        scalar_estimate, scalar_covariance = model.fit_mle(
            simulation_inputs, data, verbose=False
        )
        result["scalar"] = (scalar_estimate, np.sqrt(np.diag(scalar_covariance)))
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        return None

    finite = all(np.all(np.isfinite(np.concatenate(v))) for v in result.values())
    return result if finite else None


def covers(estimate, error, true_value):
    """Does the Wald interval on the fitting scale contain the truth?"""
    return abs(estimate - true_value) <= CONFIDENCE * error


def run_configuration(n_experiments, n_mirrors, kappa, replicates, seed):
    """Fit ``replicates`` synthetic datasets and summarise estimates and coverage."""
    rng = np.random.default_rng(seed)
    true_transformed = np.array(
        [np.log(MU_TILDE), np.log(SIGMA_DEP), np.log(kappa) - np.log1p(-kappa)]
    )

    rows, failures = [], 0
    for _ in range(replicates):
        outcome = fit_once(n_experiments, n_mirrors, kappa, rng)
        if outcome is None:
            failures += 1
            continue

        estimate, error = outcome["components"]
        scalar_estimate, scalar_error = outcome["scalar"]
        rows.append(
            {
                "log_mu_tilde": estimate[0],
                "log_sigma_dep": estimate[1],
                "logit_kappa": estimate[2],
                "covers_mu_tilde": covers(estimate[0], error[0], true_transformed[0]),
                "covers_sigma_dep": covers(estimate[1], error[1], true_transformed[1]),
                "covers_kappa": covers(estimate[2], error[2], true_transformed[2]),
                "half_width_mu_tilde": CONFIDENCE * error[0],
                "scalar_log_mu_tilde": scalar_estimate[0],
                "scalar_covers_mu_tilde": covers(
                    scalar_estimate[0], scalar_error[0], true_transformed[0]
                ),
                "scalar_half_width_mu_tilde": CONFIDENCE * scalar_error[0],
            }
        )

    frame = pd.DataFrame(rows)
    n = len(frame)
    if n == 0:
        return None

    def coverage(column):
        rate = frame[column].mean()
        return rate, np.sqrt(rate * (1 - rate) / n)  # binomial standard error

    mu_rate, mu_se = coverage("covers_mu_tilde")
    scalar_rate, scalar_se = coverage("scalar_covers_mu_tilde")
    return {
        "n_experiments": n_experiments,
        "n_mirrors": n_mirrors,
        "kappa": kappa,
        "replicates": n,
        "failures": failures,
        "bias_log_mu_tilde": frame["log_mu_tilde"].mean() - true_transformed[0],
        "bias_log_sigma_dep": frame["log_sigma_dep"].mean() - true_transformed[1],
        "bias_logit_kappa": frame["logit_kappa"].mean() - true_transformed[2],
        "sd_log_mu_tilde": frame["log_mu_tilde"].std(ddof=1),
        "coverage_mu_tilde": mu_rate,
        "coverage_mu_tilde_se": mu_se,
        "coverage_sigma_dep": coverage("covers_sigma_dep")[0],
        "coverage_kappa": coverage("covers_kappa")[0],
        "median_half_width_mu_tilde": frame["half_width_mu_tilde"].median(),
        "scalar_coverage_mu_tilde": scalar_rate,
        "scalar_coverage_mu_tilde_se": scalar_se,
        "scalar_median_half_width_mu_tilde": frame["scalar_half_width_mu_tilde"].median(),
    }


def plot_results(consistency, coverage, mirrors, output_dir):
    """Three figures: consistency, coverage against kappa, and identification against P."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].axhline(0.0, color="0.6", lw=0.8)
    for name, label in (
        ("bias_log_mu_tilde", r"$\log\tilde\mu$"),
        ("bias_log_sigma_dep", r"$\log\sigma_{dep}$"),
        ("bias_logit_kappa", r"logit $\kappa$"),
    ):
        axes[0].plot(consistency["n_experiments"], consistency[name], "o-", label=label)
    axes[0].set_xlabel("experiments $L$")
    axes[0].set_ylabel("bias on the fitting scale")
    axes[0].set_title("Consistency")
    axes[0].legend()

    axes[1].axhline(0.95, color="0.6", lw=0.8, label="nominal")
    axes[1].errorbar(
        coverage["kappa"], coverage["coverage_mu_tilde"],
        yerr=CONFIDENCE * coverage["coverage_mu_tilde_se"], fmt="o-", label="components",
    )
    axes[1].errorbar(
        coverage["kappa"], coverage["scalar_coverage_mu_tilde"],
        yerr=CONFIDENCE * coverage["scalar_coverage_mu_tilde_se"], fmt="s--", label="scalar",
    )
    axes[1].set_xlabel(r"true $\kappa$")
    axes[1].set_ylabel(r"coverage of the 95% interval for $\log\tilde\mu$")
    axes[1].set_title("Coverage of the mean parameter")
    axes[1].legend()

    axes[2].axhline(0.95, color="0.6", lw=0.8)
    axes[2].plot(mirrors["n_mirrors"], mirrors["coverage_kappa"], "o-", label=r"$\kappa$")
    axes[2].plot(mirrors["n_mirrors"], mirrors["coverage_mu_tilde"], "s--", label=r"$\tilde\mu$")
    axes[2].set_xlabel("mirrors per experiment $P$")
    axes[2].set_ylabel("coverage")
    axes[2].set_title("Identification against mirror count")
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "variance_components_study.png", dpi=150)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--output", type=Path, default=Path("results/variance_components_study")
    )
    arguments = parser.parse_args()

    arguments.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # 1. Consistency: bias against the number of experiments, at a moderate kappa.
    print("Consistency sweep over L ...")
    consistency = pd.DataFrame(
        [
            run_configuration(L, 6, 0.45, arguments.replicates, arguments.seed + L)
            for L in (1, 2, 4, 8)
        ]
    )

    # 2. Coverage across the range of kappa, including near the boundaries where the
    #    split is hardest to identify.
    print("Coverage sweep over kappa ...")
    coverage = pd.DataFrame(
        [
            run_configuration(4, 6, k, arguments.replicates, arguments.seed + int(1000 * k))
            for k in (0.1, 0.5, 0.9)
        ]
    )

    # 3. How many mirrors are needed: kappa is identified only by across-mirror contrasts.
    print("Mirror-count sweep over P ...")
    mirrors = pd.DataFrame(
        [
            run_configuration(4, P, 0.45, arguments.replicates, arguments.seed + 10 * P)
            for P in (2, 4, 8)
        ]
    )

    for name, frame in (
        ("consistency", consistency),
        ("coverage", coverage),
        ("mirrors", mirrors),
    ):
        frame.to_csv(arguments.output / f"{name}.csv", index=False)

    plot_results(consistency, coverage, mirrors, arguments.output)

    print(f"\nTruth: mu_tilde={MU_TILDE:.2e}, sigma_dep={SIGMA_DEP:.2e}")
    print(f"Elapsed {time.time() - started:.0f}s. Written to {arguments.output}\n")

    print("Consistency (kappa = 0.45, P = 6):")
    print(consistency[["n_experiments", "bias_log_mu_tilde", "sd_log_mu_tilde",
                       "coverage_mu_tilde", "failures"]].to_string(index=False))

    print("\nCoverage of log(mu_tilde), components vs scalar (L = 4, P = 6):")
    print(coverage[["kappa", "coverage_mu_tilde", "scalar_coverage_mu_tilde",
                    "median_half_width_mu_tilde", "scalar_median_half_width_mu_tilde"]]
          .to_string(index=False))

    print("\nIdentification against mirror count (L = 4, kappa = 0.45):")
    print(mirrors[["n_mirrors", "bias_logit_kappa", "coverage_kappa",
                   "coverage_mu_tilde"]].to_string(index=False))


if __name__ == "__main__":
    main()
