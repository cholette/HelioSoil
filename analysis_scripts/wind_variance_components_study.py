"""
Simulation study for the multi-mirror variance-components likelihood on the wind model.

Section 8.3 of the soiling model notes argues that mirrors measured together at one site do
not receive independent deposition noise, and that treating them as independent understates
the uncertainty in the fitted mean parameters. On the Yadnarie campaigns the common fraction
fits at kappa-hat = 0.84, which if taken at face value means the ten mirrors carry about the
information of 1.2 independent ones. This study asks whether that number can be trusted and
what may be reported on the strength of it.

Four questions, in the order they matter:

1. COVERAGE NEAR THE MEASURED KAPPA. Do the 95% Wald intervals cover at 95% when kappa is
   around 0.85? This is the regime where the deposition covariance is closest to singular --
   at kappa = 1 it loses rank in every direction where two mirrors are indistinguishable --
   and it is the one question no amount of design analysis can answer.

2. THE DESIGN EFFECT, AS A PREDICTION. With P mirrors sharing a fraction kappa,
   D = 1 + (P-1) kappa and the independent model's interval on a mean parameter should be
   too narrow by a factor sqrt(D). At the Yadnarie values that is about 2.5x. The study
   measures the ratio of interval widths and compares it against sqrt(D), which is a sharper
   test than asking merely whether coverage falls below 95%.

3. WHAT THE GEOMETRY ALLOWS. kappa is identified only by contrasts BETWEEN mirrors, so the
   design variable is the number of mirrors a mechanism can actually tell apart, not the
   number present. Two mirrors at the same tilt have identical loadings under every
   mechanism; two at the same tilt with azimuths 180 degrees apart are identical to the
   normal-wind noise as well, because that loading sums the windward and leeward bases and
   so depends on the wind angle only through |cos|. The sweep therefore contrasts distinct
   tilts, duplicated tilts, and Yadnarie's real layout -- which has ten mirrors and six
   distinguishable noise profiles.

4. BOUNDARY BEHAVIOUR. How often does kappa-hat land on 0 or 1, and what happens to the
   reported interval when it does? A Wald interval is not valid at a bound (plan D9), and a
   study that quietly drops those replicates would overstate how well the method works.

Data are simulated from the generative process (`random_delta_soiled_area` via
`simulate_reflectance_data`), not from the covariance the likelihood assembles. That the two
agree is already established as a unit test -- `tests/test_simulated_deposition.py` compares
20000 simulated datasets against `_experiment_covariance` -- so this study does not need to
re-establish it and can spend its replicates on the questions above.

Usage:
    python analysis_scripts/wind_variance_components_study.py
    python analysis_scripts/wind_variance_components_study.py --replicates 50
    python analysis_scripts/wind_variance_components_study.py --quick        # a smoke run
"""

import argparse
import time
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from heliosoil.base_models import ConstantMeanBase
from heliosoil.horizontal_impaction import ConstantMeanWindBase, ConstantMeanWindDeposition

# ----------------------------------------------------------------------------------
# Ground truth, set to the values fitted on Yadnarie so the study answers questions
# about the campaign that motivated it rather than about a generic rig.
# ----------------------------------------------------------------------------------

COMPONENTS = ["gravitational", "normal_wind"]
TRUE_MEANS = {"mu_tilde": 1.70e-05, "omega_windward": 6.42e-07, "omega_leeward": 8.80e-07}
TRUE_SIGMAS = {"sigma_dep": 1.85e-04, "sigma_dep_gamma": 4.60e-06}
MEASURED_KAPPA = 0.85  # fitted 0.842 on both campaigns, all ten mirrors

NOMINAL_REFLECTANCE = 0.95
INCIDENCE_ANGLE = 15.0
DUST_DENSITY = 40.0
MEASUREMENT_SIGMA = 1.5e-3
NUMBER_OF_MEASUREMENTS = 9.0

# One experiment is a week at five-minute resolution with daily measurements, which is the
# shape of a Yadnarie campaign.
TIMESTEPS = 289
N_MEASUREMENTS = 10

CONFIDENCE = 1.96
# kappa-hat within this of a bound counts as pinned. The logit keeps it strictly interior,
# so exact equality never happens and a tolerance is the only workable test.
BOUNDARY_TOLERANCE = 1e-4


# ----------------------------------------------------------------------------------
# Geometries. The point of the sweep: mirrors are only worth counting if some mechanism
# can distinguish them.
# ----------------------------------------------------------------------------------

GEOMETRIES = {
    # Eight mirrors, eight distinct tilts, azimuths spread. The friendly case.
    "distinct-8": {
        "tilts": [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 110.0],
        "azimuths": [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
    },
    # Eight mirrors, four distinct tilts, each duplicated with azimuths 180 degrees apart --
    # so the pairs are identical to gravitational settling AND to the normal-wind noise.
    "duplicated-8": {
        "tilts": [30.0, 30.0, 60.0, 60.0, 90.0, 90.0, 0.0, 0.0],
        "azimuths": [135.0, 315.0, 135.0, 315.0, 135.0, 315.0, 45.0, 225.0],
    },
    # Yadnarie as measured: ten mirrors, six distinguishable noise profiles.
    "yadnarie": {
        "tilts": [0.0, 180.0, 90.0, 60.0, 30.0, 0.0, 90.0, 60.0, 30.0, 0.0],
        "azimuths": [45.0, 225.0, 315.0, 315.0, 315.0, 315.0, 135.0, 135.0, 135.0, 135.0],
    },
}


def design_effect(n_loaded, kappa):
    """D = 1 + (P - 1) kappa, the factor by which the common component inflates the variance
    of a mean estimated from P equally loaded mirrors (notes eq. for the design effect).

    Exact only for equal loadings; here it is the prediction the study tests, not an input.
    """
    return 1.0 + (n_loaded - 1) * kappa


def build_experiment(geometry, n_experiments, rng):
    """A synthetic wind model, its simulation inputs, and the measurement indices."""
    tilts = np.asarray(GEOMETRIES[geometry]["tilts"], dtype=float)
    azimuths = np.asarray(GEOMETRIES[geometry]["azimuths"], dtype=float)
    n_mirrors = len(tilts)

    model = ConstantMeanWindDeposition.__new__(ConstantMeanWindDeposition)
    ConstantMeanBase.__init__(model)
    ConstantMeanWindBase.__init__(model, components=list(COMPONENTS))
    model.verbose = False
    for name, value in {**TRUE_MEANS, **TRUE_SIGMAS}.items():
        setattr(model, name, value)

    keys = range(n_experiments)
    # Tilts are STATIC. Re-drawing them each timestep would make every mirror
    # distinguishable by construction and flatter the identifiability of any real rig.
    model.helios.nominal_reflectance = NOMINAL_REFLECTANCE
    model.helios.tilt = {f: np.repeat(tilts[:, None], TIMESTEPS, axis=1) for f in keys}
    model.helios.azimuth = {f: np.repeat(azimuths[:, None], TIMESTEPS, axis=1) for f in keys}
    model.helios.incidence_angle = {f: INCIDENCE_ANGLE for f in keys}
    model.helios.inc_ref_factor = {f: np.array(2.0 / np.cos(np.radians(INCIDENCE_ANGLE))) for f in keys}
    for name in ("delta_soiled_area", "delta_soiled_area_variance", "soiling_factor", "soiling_factor_prediction_variance"):
        setattr(model.helios, name, {})

    steps = np.arange(TIMESTEPS)
    simulation_inputs = types.SimpleNamespace(
        files=[f"synthetic_{f}" for f in keys],
        time={f: steps.astype(float) for f in keys},
        dust=types.SimpleNamespace(PM10={f: DUST_DENSITY for f in keys}),
        # Diurnal dust and a wind series that both rises and swings, so the mechanisms'
        # loadings vary within an experiment rather than being proportional to each other.
        dust_concentration={f: 40.0 + 25.0 * np.sin(steps / 24.0) + rng.uniform(0, 10, TIMESTEPS) for f in keys},
        dust_type={f: "PM10" for f in keys},
        wind_speed={f: 4.0 + 3.0 * np.sin(steps / 17.0) + rng.uniform(0, 2, TIMESTEPS) for f in keys},
        wind_direction={f: (180.0 + 140.0 * np.sin(steps / 31.0)) % 360.0 for f in keys},
    )
    indices = np.linspace(0, TIMESTEPS - 1, N_MEASUREMENTS + 1).astype(int).tolist()
    return model, simulation_inputs, indices, n_mirrors


def truth_vector(model, kappa):
    """Ground truth in ``parameter_names`` order for the model's current variance model."""
    values = [TRUE_MEANS[name] for name in model._mean_param_names]
    values += [TRUE_SIGMAS[name] for name in model._sigma_param_names]
    values += [kappa] * len(model.fitted_kappa_names)
    return values


def fit_once(geometry, n_experiments, kappa, rng, variance_models):
    """Simulate one dataset and fit it under each variance model.

    Returns a dict of ``{variance_model: (fitted estimate, fitted standard errors)}``, or
    None if any fit failed. Failures are counted rather than retried: a study that silently
    drops the hard cases would overstate how well the method works.
    """
    model, simulation_inputs, indices, _ = build_experiment(geometry, n_experiments, rng)

    model.set_variance_model("shared_kappa", endpoint_correction=True)
    data = model.simulate_reflectance_data(
        simulation_inputs,
        truth_vector(model, kappa),
        indices,
        MEASUREMENT_SIGMA,
        number_of_measurements=NUMBER_OF_MEASUREMENTS,
        rng=rng,
    )

    outcome = {}
    for variance_model in variance_models:
        # "independent" keeps the endpoint correction on so that it differs from the others
        # in the common fraction ALONE, which is what makes the comparison a test of kappa.
        model.set_variance_model(variance_model, endpoint_correction=True)
        try:
            estimate, covariance = model.fit_mle(simulation_inputs, data, verbose=False)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            return None
        diagonal = np.diag(np.asarray(covariance, dtype=float))
        errors = np.sqrt(np.where(diagonal > 0.0, diagonal, np.nan))
        if not np.all(np.isfinite(estimate)):
            return None
        outcome[variance_model] = (np.asarray(estimate, dtype=float), errors, list(model.parameter_names))
    return outcome


def covers(estimate, error, true_value):
    """Does the Wald interval on the fitting scale contain the truth?

    A non-finite standard error -- what a parameter on its bound produces -- is NOT counted
    as covering. Treating it as a miss is the conservative reading and keeps those replicates
    visible in the coverage rate instead of quietly removing them.
    """
    if not np.isfinite(error):
        return False
    return abs(estimate - true_value) <= CONFIDENCE * error


def run_configuration(geometry, n_experiments, kappa, replicates, seed, variance_models, progress):
    """Fit `replicates` synthetic datasets and summarise estimates, coverage and boundaries."""
    rng = np.random.default_rng(seed)
    reference, simulation_inputs, indices, n_mirrors = build_experiment(geometry, n_experiments, rng)
    reference.set_variance_model("shared_kappa", endpoint_correction=True)

    # Ground truth on each variance model's own fitting scale. Computed once, per model,
    # because the vectors differ in length: per_mechanism carries one kappa per mechanism.
    fitted_truth = {}
    for variance_model in variance_models:
        reference.set_variance_model(variance_model, endpoint_correction=True)
        fitted_truth[variance_model] = np.asarray(
            reference.transform_scale(truth_vector(reference, kappa), direction="forward"), dtype=float
        )
    reference.set_variance_model("shared_kappa", endpoint_correction=True)

    def reference_for(variance_model):
        """`reference`, switched to `variance_model` so transform_scale accepts that
        model's vector length. Kept as a closure rather than a fresh model per call:
        building one costs a full geometry setup."""
        reference.set_variance_model(variance_model, endpoint_correction=True)
        return reference

    rows, failures = [], 0
    for _ in range(replicates):
        outcome = fit_once(geometry, n_experiments, kappa, rng, variance_models)
        progress.update(1)
        if outcome is None:
            failures += 1
            continue

        row = {}
        for variance_model, (estimate, errors, names) in outcome.items():
            truth = fitted_truth[variance_model]
            for index, name in enumerate(names):
                row[f"{variance_model}:{name}"] = estimate[index]
                row[f"{variance_model}:covers_{name}"] = covers(estimate[index], errors[index], truth[index])
                row[f"{variance_model}:half_width_{name}"] = CONFIDENCE * errors[index]
            # Every common fraction this model fits, by its own name -- not just the pooled
            # one. Hardcoding "common_variance_fraction" here left the per-mechanism kappas
            # unrecorded, which is precisely what the geometry sweep exists to measure.
            natural = np.asarray(reference_for(variance_model).transform_scale(list(estimate)), dtype=float)
            for index, name in enumerate(names):
                if not name.startswith(("kappa", "common_variance_fraction")):
                    continue
                kappa_hat = float(natural[index])
                row[f"{variance_model}:hat_{name}"] = kappa_hat
                row[f"{variance_model}:pinned_{name}"] = kappa_hat < BOUNDARY_TOLERANCE or kappa_hat > 1.0 - BOUNDARY_TOLERANCE
                row[f"{variance_model}:unavailable_{name}"] = not np.isfinite(errors[index])
        rows.append(row)

    if not rows:
        return None

    frame = pd.DataFrame(rows)
    n = len(frame)
    mechanism_mirrors = reference.loaded_mirror_counts(simulation_inputs, _stub_reflectance(n_experiments))
    predicted = design_effect(max(mechanism_mirrors.values()), kappa)

    summary = {
        "geometry": geometry,
        "n_mirrors": n_mirrors,
        "n_experiments": n_experiments,
        "kappa": kappa,
        "replicates": n,
        "failures": failures,
        "loaded_mirrors_max": max(mechanism_mirrors.values()),
        "design_effect_predicted": predicted,
        "width_ratio_predicted": np.sqrt(predicted),
    }
    # Report coverage for every parameter each model actually fitted, discovered from the
    # frame rather than listed here, so a model with a longer vector cannot go unmeasured.
    for variance_model in variance_models:
        for column in [c for c in frame.columns if c.startswith(f"{variance_model}:covers_")]:
            name = column.split(":covers_", 1)[1]
            rate = frame[column].mean()
            summary[f"{variance_model}_coverage_{name}"] = rate
            # Prefixed, not suffixed: "..._coverage_mu_tilde_se" sitting next to
            # "..._coverage_mu_tilde" in a 40-column header is trivially misread as the
            # coverage itself, and at n = 10 a binomial standard error is about 0.1
            # whatever the coverage is, so the misreading looks alarming and is not.
            summary[f"se_of_{variance_model}_coverage_{name}"] = np.sqrt(rate * (1 - rate) / n)
        for column in [c for c in frame.columns if c.startswith(f"{variance_model}:hat_")]:
            name = column.split(":hat_", 1)[1]
            summary[f"{variance_model}_{name}_mean"] = frame[column].mean()
            summary[f"{variance_model}_{name}_sd"] = frame[column].std(ddof=1)
            summary[f"{variance_model}_{name}_pinned_rate"] = frame[f"{variance_model}:pinned_{name}"].mean()
            summary[f"{variance_model}_{name}_unavailable_rate"] = frame[f"{variance_model}:unavailable_{name}"].mean()
        column = f"{variance_model}:half_width_mu_tilde"
        if column in frame:
            summary[f"{variance_model}_half_width_mu_tilde"] = frame[column].median()

    # Back-compatible aliases for the plot and for reading at a glance.
    if "shared_kappa_common_variance_fraction_mean" in summary:
        summary["kappa_hat_mean"] = summary["shared_kappa_common_variance_fraction_mean"]
        summary["kappa_hat_sd"] = summary["shared_kappa_common_variance_fraction_sd"]
        summary["kappa_pinned_rate"] = summary["shared_kappa_common_variance_fraction_pinned_rate"]

    # The prediction under test: independent intervals narrower by sqrt(D).
    wide, narrow = "shared_kappa_half_width_mu_tilde", "independent_half_width_mu_tilde"
    if wide in summary and narrow in summary and summary[narrow] > 0:
        summary["width_ratio_observed"] = summary[wide] / summary[narrow]
    return summary


def _stub_reflectance(n_experiments):
    """The minimum `loaded_mirror_counts` needs: something with a `times` mapping."""
    return types.SimpleNamespace(times={f: None for f in range(n_experiments)})


def plot_results(coverage, geometry_frame, output_dir):
    """Three figures: coverage against kappa, the design-effect prediction, and geometry."""
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    axes[0].axhline(0.95, color="0.4", lw=0.9, ls="--", label="nominal 95%")
    for variance_model, marker in (("shared_kappa", "o"), ("independent", "s")):
        column = f"{variance_model}_coverage_mu_tilde"
        if column in coverage:
            axes[0].errorbar(
                coverage["kappa"], coverage[column], yerr=CONFIDENCE * coverage[f"se_of_{column}"], marker=marker, capsize=3, label=variance_model
            )
    axes[0].set_xlabel(r"true $\kappa$")
    axes[0].set_ylabel(r"coverage of $\tilde\mu$")
    axes[0].set_title("1-2. Coverage, and whether independent under-covers")
    axes[0].legend(fontsize=8)

    if "width_ratio_observed" in coverage:
        axes[1].plot(coverage["kappa"], coverage["width_ratio_predicted"], "k--", label=r"predicted $\sqrt{D}$")
        axes[1].plot(coverage["kappa"], coverage["width_ratio_observed"], "o-", label="observed")
        axes[1].set_xlabel(r"true $\kappa$")
        axes[1].set_ylabel("interval width, shared / independent")
        axes[1].set_title("2. The design effect as a prediction")
        axes[1].legend(fontsize=8)

    # Every common fraction any model fitted, so a mechanism whose kappa is unrecoverable
    # is visible rather than absent. A spread approaching 0.29 is that of a uniform draw on
    # [0, 1] -- i.e. the data said nothing.
    kappa_columns = [c for c in geometry_frame.columns if c.endswith("_sd") and "kappa" in c or c.endswith("common_variance_fraction_sd")]
    positions = np.arange(len(geometry_frame))
    width = 0.8 / max(len(kappa_columns), 1)
    for offset, column in enumerate(kappa_columns):
        label = column.removesuffix("_sd").replace("per_mechanism_", "per-mech ").replace("shared_kappa_common_variance_fraction", "shared")
        axes[2].bar(positions + offset * width, geometry_frame[column], width=width, label=label)
    axes[2].axhline(1.0 / np.sqrt(12.0), color="0.3", ls="--", lw=0.9, label="uniform on [0,1]")
    axes[2].set_xticks(positions + 0.4 - width / 2)
    axes[2].set_xticklabels(geometry_frame["geometry"], rotation=20, ha="right")
    axes[2].set_ylabel(r"sd of $\hat\kappa$ across replicates")
    axes[2].set_title("3. What the geometry allows")
    axes[2].legend(fontsize=7)

    figure.tight_layout()
    figure.savefig(output_dir / "wind_variance_components_study.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path, default=Path("results/wind_variance_components_study"))
    parser.add_argument("--quick", action="store_true", help="A few replicates and one geometry, to check the script runs.")
    arguments = parser.parse_args()

    replicates = 3 if arguments.quick else arguments.replicates
    kappa_grid = (0.5, MEASURED_KAPPA) if arguments.quick else (0.1, 0.5, 0.85, 0.95)
    geometry_grid = ("yadnarie",) if arguments.quick else tuple(GEOMETRIES)
    both = ("independent", "shared_kappa")

    arguments.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # Every fit that will be run, so one progress bar can carry a meaningful estimate of
    # the time remaining rather than restarting per sweep.
    # Questions 1, 2 and 4 sweep kappa on Yadnarie's geometry; question 3 sweeps geometry at
    # the measured kappa and adds per_mechanism, since whether the split is recoverable is a
    # question about the geometry. Running all three variance models in the geometry sweep
    # rather than as a fourth sweep saves a third of the wall clock.
    all_three = ("independent", "shared_kappa", "per_mechanism")
    coverage_jobs = [("yadnarie", 2, k, both, "coverage") for k in kappa_grid]
    geometry_jobs = [(g, 2, MEASURED_KAPPA, all_three, "geometry") for g in geometry_grid]
    all_jobs = coverage_jobs + geometry_jobs
    total = replicates * len(all_jobs)

    coverage_rows, geometry_rows = [], []
    with tqdm(total=total, unit="rep", desc="starting", dynamic_ncols=True, smoothing=0.05) as progress:
        for job_index, (geometry, n_experiments, kappa, variance_models, label) in enumerate(all_jobs):
            progress.set_description(f"{label}: {geometry}, kappa={kappa}")
            # A fixed offset per job, NOT hash(): Python randomises string hashing per
            # process unless PYTHONHASHSEED is set, which would make --seed a lie.
            summary = run_configuration(
                geometry, n_experiments, kappa, replicates, arguments.seed + 1000 * job_index, variance_models, progress
            )
            if summary is None:
                continue
            (coverage_rows if label == "coverage" else geometry_rows).append(summary)
            if "kappa_hat_mean" in summary:
                progress.set_postfix(kappa_hat=f"{summary['kappa_hat_mean']:.3f}", fails=summary["failures"])

    frames = {}
    for name, rows in (("coverage", coverage_rows), ("geometry", geometry_rows)):
        frame = pd.DataFrame(rows)
        frame.to_csv(arguments.output / f"{name}.csv", index=False)
        frames[name] = frame

    # A tidy long-format view alongside the wide ones. Forty-odd columns of
    # "<model>_coverage_<parameter>" is not a readable table; one row per
    # (configuration, variance model, parameter) with the coverage and its uncertainty
    # side by side is.
    tidy = []
    for sweep, frame in frames.items():
        for _, row in frame.iterrows():
            for column in [c for c in frame.columns if "_coverage_" in c and not c.startswith("se_of_")]:
                variance_model, parameter = column.split("_coverage_", 1)
                rate = row[column]
                error = row.get(f"se_of_{column}", np.nan)
                tidy.append(
                    {
                        "sweep": sweep,
                        "geometry": row["geometry"],
                        "kappa": row["kappa"],
                        "variance_model": variance_model,
                        "parameter": parameter,
                        "replicates": row["replicates"],
                        "coverage": rate,
                        "coverage_se": error,
                        "coverage_reported": f"{rate:.2f} +/- {error:.2f}",
                    }
                )
    tidy_frame = pd.DataFrame(tidy)
    tidy_frame.to_csv(arguments.output / "coverage_long.csv", index=False)
    frames["coverage_long"] = tidy_frame

    if not frames["coverage"].empty and not frames["geometry"].empty:
        plot_results(frames["coverage"], frames["geometry"], arguments.output)

    # At n replicates a coverage estimate carries a binomial standard error of about
    # sqrt(0.95 * 0.05 / n): 0.07 at n = 10, 0.015 at n = 200. Distinguishing a true 0.85
    # from the nominal 0.95 therefore needs a couple of hundred replicates; below that the
    # coverage columns are for spotting a collapse, not for measuring a shortfall.
    nominal_se = np.sqrt(0.95 * 0.05 / max(replicates, 1))
    print(f"\nCoverage is estimated from {replicates} replicates, so its own standard error is about {nominal_se:.3f}.")
    print("Columns named se_of_<...> are that uncertainty, NOT the coverage. See coverage_long.csv for the readable view.")
    print(f"\nTruth: {TRUE_MEANS}, {TRUE_SIGMAS}, kappa={MEASURED_KAPPA}")
    for name, frame in frames.items():
        if frame.empty:
            continue
        print(f"\n--- {name} ---")
        columns = [c for c in frame.columns if not c.endswith("_se")]
        print(frame[columns].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nElapsed {time.time() - started:.0f}s. Written to {arguments.output}\n")


if __name__ == "__main__":
    main()
