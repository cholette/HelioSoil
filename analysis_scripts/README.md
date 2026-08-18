# `analysis_scripts.cli` — quick reference (Yadnarie workflows)

One entry point for the three analysis workflows. Examples use new standardised naming conventions, ensure input files for soiling_* and parameters* datafiles are setup and named correctly contact Cody for more information.

Run from the repository root:

```bash
python -m analysis_scripts.cli simulate   --help   # fit one model + report it
python -m analysis_scripts.cli select     --help   # compare models by cross-validation
python -m analysis_scripts.cli experiment --help   # weather / measured-soiling summary (no model)
```

`simulate` and `select` share every site, physics and run-control option and differ only in
what they sweep. Defaults come from `model_pipeline.PipelineConfig`, so that dataclass is the
single source of truth; `--help` always prints the live values.

Typical loop: **`select`** ranks models → copy the winning `model_expression` → **`simulate`**
re-fits that one model and produces the report figures.

---

## Yadnarie at a glance

| | |
|---|---|
| Campaigns (`C`) | **2** — `campaign_1` = 2024-11-11→16, `campaign_2` = 2025-02-06→12 |
| Common mirrors | 10: `ONE_M2_T00`, `OSW_M1_T180`, `ONW_M{1..4}_T{90,60,30,00}`, `OSE_M{1..4}_T{90,60,30,00}` |
| Tilts present | 0, 30, 60, 90, 180° |
| Orientations | NE, NW, SE, SW (the letters after the leading `O` in the mirror name) |
| Dust measures | `PM2.5`, `PM10`, `PM17` (the `PMT` column normalises to `PM17`) |
| Dust *specs* (with differences) | the three above plus `PM10-PM2.5`, `PM17-PM2.5`, `PM17-PM10` |
| Wind data | `WD` (meteorological, "from") + `WindSpeed` — this is what the wind mechanisms use |

With `C = 2`, both cross-validation schedules collapse to the same 2 folds
(`n_train = 1`), so `select`'s pooled row and `simulate`'s `cv_pooled` row are the same
statistic.

---

## The four options that actually change the answer

### `--daily-average`

Daily-averages the reflectance targets before fitting. **Off by default; every Yadnarie run
to date has used it** — pass it, and pass it consistently, or numbers are not comparable
across runs.

### `--fit-method {mle,ls}`

| | `mle` (default) | `ls` |
|---|---|---|
| Estimates | all mean **and** noise parameters | mean parameters only, no `sigma_*` |
| Training mirrors | model-aware (see below) | **all 10 common mirrors** (no shared-noise double-counting to avoid) |
| Prediction interval on figures | yes | no |
| `loss_vs_tilt` / `loss_distributions` | yes | **skipped** (no noise process to sample) |
| `fit_quality_test_mirrors.pdf` | when mirrors were held out | never (nothing held out) |

Both are supported by all three model types (`constant_mean`/`constant_mean_wind` are affine
in their means → one bounded linear solve; `semi_physical` uses a bounded scalar search over
`hrz0`). If a model type ever lacked `fit_ls`, it silently falls back to MLE — and the
`simulate` header and `select`'s `fit_method` column report what was *actually* run.

### `--variance-model {independent,shared_kappa,per_mechanism}`

How the deposition noise is correlated across mirrors. Mirrors at one site are measured together and soil under the same weather, so their noise is not independent.

- `independent` (default) — the historical model. Every mirror's deposition noise is its own, and the reflectance differences are treated as independent. Unchanged numbers for unchanged commands.
- `shared_kappa` — one common fraction across all mechanisms, one extra parameter. 
- `per_mechanism` — one fraction per mechanism. Might have identifiability issues with limited mirrors / campaigns. 

MLE only — least squares fits no noise parameters, so the flag has no effect on `--fit-method
ls` or on `select`. Recorded in `run_config.json`; any fitted κ appears as its own row in
`fitted_parameters.csv`.

Training-mirror default under `mle`: with `shared_kappa` or `per_mechanism` **all mirrors
are used**, because the likelihood models the between-mirror correlation directly and the
shared part is no longer counted once per mirror. With `independent` the older rule stands: a
gravitational-only / constant-mean model trains on the single lowest-tilt representative
(`ONE_M2_T00`), and a model with an active `normal_wind`, `tangential_wind` or
`impaction_retention` term uses all mirrors. Prefer `shared_kappa` for new work. 
`--train-mirrors NAME [NAME ...]` overrides all of this.

### `--model-type {constant_mean,constant_mean_wind,semi_physical,all}`

Default `all`. Only `constant_mean_wind` has wind components; `--model` is ignored by the
other two (they still run under `all`).

### `--model EXPRESSION`

The wind model, written the way the workflows report it. Mechanisms are summed; a `PM*`
prefix drives that mechanism with its own dust channel, a bare name uses the run's
`--dust-type`:

```
gravitational + normal_wind
PM10*turbulent_wind + PM10*normal_wind + PM10*tangential_wind
PM17*gravitational + (PM10-PM2.5)*tangential_wind
```

Vocabulary (5 mechanisms → 31 non-empty subsets):
`gravitational`, `turbulent_wind`, `normal_wind`, `tangential_wind`, `impaction_retention`.
See [horizontal_impaction.py](../src/heliosoil/horizontal_impaction.py) for the formulae —
notably `impaction_retention`'s sin·cos weighting, whose windward/leeward contrast peaks near
45° tilt (`normal_wind`'s grows monotonically to 90°).

---

## `simulate` — fit one model and report it

Leave-one-campaign-out by default: fold *k* trains on every campaign except *k* and tests on
*k*, so every campaign gets a genuine out-of-sample prediction. Naming `--train-experiments`
switches to a single named split instead.

```bash
# The two report runs on record
python -m analysis_scripts.cli simulate --location yadnarie --daily-average \
    --model-type constant_mean_wind \
    --model "PM10*turbulent_wind + PM10*normal_wind + PM10*tangential_wind" \
    --fit-method ls --run-name yadnarie-report-ls --force

python -m analysis_scripts.cli simulate --location yadnarie --daily-average \
    --model-type semi_physical --run-name yadnarie-report-mle --force

# All three model types, library-default wind components, MLE
python -m analysis_scripts.cli simulate --location yadnarie --daily-average --run-name baseline

# Single split instead of CV: train on campaign 0, test on campaign 1
python -m analysis_scripts.cli simulate --location yadnarie --daily-average \
    --train-experiments 0 --run-name single-split
```

`simulate` extras: `--dust-type PM10|PM2.5|PM17` (one channel for the whole run, default
`PM10`), `--train-experiments INDEX [INDEX ...]` (0-based campaign indices).

### Output — `results/simulate/yadnarie/{run_name}/{label}/`

`label` is the model's own name, e.g. `constant-mean_PM10xturbulent-wind_PM10xnormal-wind`,
so per-component channels are already in the path.

| File | What it is |
|---|---|
| `fitted_parameters.csv` | one row per (fit, parameter) with 95% CI. `fit` is `fold_1`/`fold_2`, plus `cv_mean`/`cv_std` (across-fold spread — a parameter whose spread swamps its own CI is not identified by this data) and `all_campaigns` when the extra all-data fit ran |
| `performance_stats.csv` | per-fold in/out-of-sample `N, MBE, MAE, RMSE, R2`, plus `cv_mean`/`cv_std` and **`cv_pooled`** — the single statistic over every held-out prediction at once. **`cv_pooled` is the one to quote.** MBE/MAE/RMSE score the *daily soiling rate*; R² is a reflectance goodness-of-fit |
| `all_campaigns_test.pdf` | every campaign drawn from the fold that held it out (nothing highlighted — nothing was trained on) |
| `fold_k/all_campaigns.pdf` | that fold's model over all campaigns, its training campaigns highlighted |
| `fold_k/fit_quality_{train_mirrors,test_mirrors,test_experiments}.pdf` | predicted-vs-measured daily loss, shared axis limits. `test_mirrors` is absent when the fit trained on every mirror (any `ls` run, any wind-noise model, any run with `--variance-model` other than `independent`) |
| `campaign_N/reflectance_tilt_TT_{train,test}-fK.pdf` | measured vs predicted reflectance for one tilt, annotated with that tilt's MAE/RMSE. The suffix names the fold and the campaign's role in it — `campaign_1/..._test-f1.pdf` is campaign 1 held out by fold 1 |
| `campaign_N/performance_stats.csv` | those per-tilt stats plus an `all` row, pooled over every fold that touched the campaign |
| `rate_error_by_tilt.pdf/.csv` | boxplot of \|daily soiling-rate error\| per tilt, in-sample vs out-of-sample, in p.p./day. Shows the *spread* behind each MAE. Note the in-sample side pools (C−1) folds' training campaigns, hence the per-box `n` row |
| `loss_vs_tilt.pdf/.csv` | design curve: predicted daily loss vs fixed tilt, one curve per orientation, with a parameter-uncertainty band. **Skipped for `ls` and for `semi_physical`** |
| `loss_distributions.pdf/.csv` | full distribution of a horizontal mirror's daily loss on the low/median/high/worst dust day. Same skip rule |
| `run_config.json`, `run.log` | full invocation + versions; suppressed warnings. Written at `{run_name}/`, not inside each model's `{label}/` |

The two `loss_*` outputs are design results, not CV results, so under cross-validation one
extra fit on **all** campaigns is run to back them (saved as `fitting_results`, reported as
the `all_campaigns` rows). That extra fit only happens when the figures are producible, so
`ls` and `semi_physical` runs pay nothing for it.

---

## `select` — cross-validated model comparison

Sweeps two axes — the soiling model and the dust channel — over a fixed leave-*n*-campaigns-out
schedule (every training size `n_train = 1..C-1` and every split of each; for Yadnarie that is
2 folds).

```bash
# Default sweep: (constant_mean + 31 wind combos + semi_physical) x 3 dust types = 99 runs
python -m analysis_scripts.cli select --location yadnarie --daily-average \
    --run-name yadnarie_report

# Same sweep, one dust channel, least squares -- much faster, the usual screening run
python -m analysis_scripts.cli select --location yadnarie --daily-average \
    --fit-method ls --dust PM10 --run-name yadnarie-ls --force

# Fix the training mirrors instead of the model-aware default
python -m analysis_scripts.cli select --location yadnarie --daily-average \
    --train-mirrors ONW_M3_T30 OSE_M2_T60 --run-name yadnarie-train-tilt

# Per-component dust sweep: every assignment of the 6 dust specs to these 2 mechanisms
# (6^2 = 36 runs). Needs --model, and --model must NOT already name channels.
python -m analysis_scripts.cli select --location yadnarie --daily-average \
    --model "gravitational + tangential_wind" --dust sweep --run-name dustsweep --force
```

`select` extras: `--dust {TYPE,all,sweep}` (default `all`), `-j/--jobs N` worker threads for
parallel fold fits (default `min(8, cpu_count)`; `-j 1` disables parallelism). `--model` is
optional here — omitting it compares all 31 component combinations. `--dust sweep` prints its
fit count and asks for confirmation first (`--force` runs it unattended).

### Output — `results/model_select/yadnarie/{run_name}/{dust_type}/{label}/`

| File | What it is |
|---|---|
| `cross_validation_folds.csv` | per-fold out-of-sample stats, in-sample R², and fitted parameters (NaN row for a failed fold) |
| `cross_validation_summary.csv` | per-`n_train` aggregates + `n_failed`, tagged with `model_expression` and `fit_method` |
| `cross_validation_summary.pdf` | out-of-sample RMSE and R² vs number of training campaigns |
| **`{run_name}/all_models_kfold_summary.csv`** | every (dust_type, model) summary stacked and **sorted globally by pooled out-of-sample MAE**, failed-fold runs pushed to the bottom. This is the ranking table |

Runs whose every mechanism names its own channel are dust-type independent, so they are fitted
once under `{run_name}/per-component-dust/` instead of once per dust type.

**Reading the summary — two estimators, not interchangeable:**

- `*_mean` / `*_std` reduce the per-fold statistics, weighting each **fold** equally.
- `*_pooled` scores every held-out prediction in one pass, weighting each **observation**
  equally. Defined only on the `n_train = C-1` row (the one schedule where each campaign is
  held out exactly once) and identical to `simulate`'s `cv_pooled`. **This is what the ranking
  sorts on and what you should quote.**
- **R² is reported pooled only.** Each fold's R² is normalised by its own held-out campaign's
  reflectance variance, so a mean of fold R²s divides by a different denominator every time
  and estimates nothing. The per-fold values are still plotted as a scatter and kept as
  `R2_fold_std`.

Everything is scored on the same evaluation set (all 10 common mirrors, identical reflectance
targets across dust types), so rows are comparable across **both** axes.

---

## Gotchas

- **`--daily-average` is off by default.** Yadnarie work uses it; a run without it is not
  comparable to one with it.
- **Overwriting a run folder** prompts, and aborts with no TTY. Use `--force` (or a fresh
  `--run-name`) for scripted runs. `--run-name` defaults to a `run-yy-mm-dd_hh-mm` timestamp.
- **`ls` runs produce no prediction intervals and no `loss_*` figures.** If you want the design
  curves, use `mle`.
- **`--dust sweep` is exponential** in the number of mechanisms (`len(dust_specs) ** n`): 6
  specs × 2 mechanisms = 36 runs, × 4 mechanisms = 1296. It prints the count and asks first.
- **Console vs log.** Per-fold detail and suppressed warnings go to `{run_dir}/run.log`, not
  the terminal (they would corrupt `select`'s progress bar). `--verbose` raises HelioSoil's
  console logging to INFO.
- **Wind-direction caveats for Yadnarie**: `WD` uses the meteorological "from" convention, and
  `WD = 0` in still air is a logger sentinel rather than true north (~12.6% of campaign 1) —
  it feeds every WD-driven mechanism. Worth keeping in mind when a wind model wins or loses by
  a small margin.

## See also

- [cli.py](cli.py) — the full option surface
- [simulate.py](simulate.py) / [model_selection.py](model_selection.py) — module docstrings go
  deeper than this file
- [model_pipeline.py](model_pipeline.py) — shared config, data loading, fit/evaluate kernel
- [horizontal_impaction.py](../src/heliosoil/horizontal_impaction.py) — the wind mechanisms
