# Static Analysis for Flaky Test Prioritization

An undergraduate capstone comparing rule-based, machine-learning, and adaptive
static analysis of Java test-method source. The analyzers assign risk scores
for prioritization; they do not execute the Java tests or prove flakiness.

## Dataset and setup

The benchmark uses the unmodified original FlakeBench dataset and its four
project-separated train/test pairs in `src/data/`: 8,574 tests, including 280
flaky and 8,294 non-flaky tests. Categories 0–4 are flaky; category 5 is non-flaky.
Each model trains only on the training file paired with its held-out test file.

Data and generated results are ignored by Git and must be supplied locally on
another checkout. The deleted `DELETED_ROWS.csv` audit log is not required.

Use Python 3.11 or later. Run commands from the repository root:

```bash
python3 -m pip install pandas numpy scikit-learn threadpoolctl flake8
python3 -B -m flake8 src
```

Existing repository tests additionally require `pytest`.

## Implemented analyzers

| Analyzer | Behavior |
| --- | --- |
| Rule-based | Adds heuristic weights for selected pattern presence; no training. |
| Logistic regression | Pattern counts, clipping at 5 by default, `log1p`, standardization, and balanced logistic regression. |
| Random Forest | Raw pattern counts; 100 trees, maximum depth 5, balanced class weights, seed 42. |
| Adaptive | Weighted pattern-presence screening on every test, followed by logistic-regression scoring on selected tests. |

Shared regex features cover concurrency, asynchronous waiting, time, randomness,
collections, file/network APIs, assertions, loops, and branches. Each analyzer
uses its implemented subset. Standalone rules and the adaptive screen have
different indicators and weights.

Standalone rules support `source_mode="raw"` and `source_mode="masked"`.
Masked mode excludes comments and literal contents from matching while retaining
the original source in predictions. Raw mode remains the constructor default.
ML feature extraction and adaptive screening continue to use raw source.

Adaptive routing supports these switches:

- Original defaults: rule scores at or below `0.0` receive `0.0`, scores at or
  above `0.5` receive `0.95`, and the remaining tests reach ML.
- `bypass_low=False`: send low-score tests to ML instead of assigning zero.
- `bypass_high=False`: let ML reconsider high-score tests.
- Both switches default to `True` for baseline reproduction.
- `AdaptiveRuleScreen` uses the adaptive indicators and weights without ML or
  fixed-score routing, as an optional matched baseline.

The high-score reconsideration variant keeps `bypass_low=True` and sets
`bypass_high=False`. It is the default variant in the repeated timing command.
The ordinary benchmark retains the original constructor defaults.

## Regular four-fold benchmark

```bash
python3 -B -m src.main
```

This trains and analyzes all four methods on the supplied fold pairs, without
inner validation or setting selection. It writes:

- Per-method, per-fold prediction CSVs under `src/results/`.
- `src/results/benchmark_summary.csv`: mean fold metrics, routing, and timings.
- `src/results/benchmark_summary_timings.csv`: individual fold timings.

This command overwrites its generated predictions and summaries. An optional
positional summary path changes the summary destination, but predictions still
go to `src/results/`. Use the repeated command below to isolate a new run.

The shared runner times each actual call without analyzing the data again just
to measure its duration. The older standalone rule `time_passed()` helper is
retained but is not used by the benchmark.

## Repeated timing benchmark

```bash
python3 -B -m src.timing \
  --output-dir src/results/timing/my_run \
  --repeats 5 --warmups 1 --threads 1 \
  --adaptive-policy reconsider-high
```

Choose a new output directory; existing directories are refused. All methods
run sequentially. Measured analyzer order rotates by fold and repetition, and
each run creates a fresh model. Each method receives one excluded warm-up per
fold followed by five measured repetitions by default. `--threads` consistently
limits native numerical-library thread pools.

Use `--adaptive-policy current` for the original routing policy. Optional
`--data-dir` and `--folds` choose the input directory and fold numbers. Other
settings retain their defaults, including raw standalone rules.

Measurements use wall-clock seconds from `time.perf_counter()`:

| Measurement | Boundary |
| --- | --- |
| Data loading | Read required train/test CSVs; outside training and analysis. |
| Training | Entire `train()` call, including feature extraction, preprocessing, and fitting. N/A for rules. |
| Analysis | Entire `analyze()` call, including feature extraction, adaptive screening/routing, prediction, and output-frame construction. Returns unsorted scores. |
| Training + analysis | Sum of those two durations within each repetition. |
| Evaluation | Ranking metrics, including sorting for inspection cutoffs. |
| Prediction export | Saving the prediction CSV. |

Imports, analyzer construction, final summary generation, validation, and Java
test execution are outside these timings. Adaptive still trains on the full
training fold; routing applies during analysis.

| Generated file | Contents |
| --- | --- |
| `timing_runs.csv` | Every measured repetition, order, timing, metric, and available routing diagnostic. |
| `timing_by_fold.csv` | Median, minimum, and maximum repeat durations per method/fold. |
| `quality_speed_summary.csv` | Quality metrics and unweighted means of per-fold median durations. |
| `metrics_by_fold.csv` | Quality metrics from the first measured repetition per method/fold. |
| `benchmark_summary.csv` | Formatted quality, routing, and timing summary. |
| `benchmark_summary_timings.csv` | Per-fold medians used by the formatted summary. |
| `run.json` | Status, policy, repetition settings, timer boundaries, software/hardware details, native thread pools, and source/input checksums. |
| `test_set_*_results.csv` | Predictions, overwritten within this new directory between repetitions. |

Each stage is aggregated independently. The median combined duration can differ
from the sum of separate training and analysis medians. Repetitions describe
runtime variation on the same machine, not additional independent datasets.
Use the per-fold variation when interpreting comparisons. Measurements remain
in generated reports; performance findings belong in the research report.

## Validation and setting selection

Validation is separate from the regular and timing benchmarks. It creates three
project-grouped inner splits inside each outer training file, with seed 42.
Settings are selected by mean validation Average Precision; exact ties choose
the first declared candidate. Validation does not read outer test files.

```bash
python3 -B -m src.validation validate \
  --config configs/validation_defaults.json \
  --output-dir src/results/validation/my_run

python3 -B -m src.validation evaluate \
  --run-dir src/results/validation/my_run
```

The first command saves assignments, candidate metrics, selected settings, and
source/runtime/input checksums in a new directory. The second refits selected
settings on the full outer training folds and evaluates the paired test folds.
Evaluation refuses changed source, runtime, or inputs: create a fresh validation
run after source changes. Earlier runs remain historical records.
Validation/search time is not included in analysis timings.

Available configurations:

- `configs/validation_defaults.json`: current model defaults.
- `configs/rules_source_mode.json`: raw versus masked standalone rules.
- `configs/adaptive_routing.json`: original, reconsider-low, reconsider-high,
  reconsider-both, and the matched rule-only screen.

The routing experiment keeps indicators, weights, caps, preprocessing, and the
ML backend fixed. Choosing a policy per fold is a different experiment from
applying one fixed policy across all four folds.

## Metrics

- **Average Precision (AP):** sklearn `average_precision_score`. The legacy CSV
  key `pr_auc` and formatted label `PR-AUC` refer to AP, not trapezoidal PR area.
- **ROC-AUC:** ordering of flaky versus non-flaky tests across score thresholds.
- **Recall@K%:** fraction of known flaky tests in the top K% of the ranking.
- **Precision@K%:** fraction of the selected group labeled flaky.

Quality summaries use unweighted means of the held-out fold metrics. Scores
are used for ranking; calibrated probabilities have not been established.
Routing percentages describe routing rather than measured runtime reductions.

## Source layout

```text
src/
  BaseAnalyzer.py          Shared analyzer interface
  RuleBasedAnalyzer.py    Standalone weighted rules
  java_source.py          Comment and literal masking
  features.py             Shared regex features
  MLAnalyzer.py           Logistic regression and preprocessing
  RandomForestAnalyzer.py Random Forest baseline
  AdaptiveAnalyzer.py     Configurable cascade and matched rule screen
  benchmark.py            Fold lifecycle, metrics, timings, and persistence
  main.py                 Regular benchmark CLI and compatibility helpers
  timing.py               Repeated timing CLI and run metadata
  validation.py           Inner validation and selected-setting evaluation
  metrics.py              Ranking metrics
  utility.py              Data loading and CSV export
  dataset_audit.py        Optional read-only dataset inspection
configs/                  Validation candidates
tests/                    Existing automated checks
```

Optional dataset inspection is available through `python3 -B -m src.dataset_audit`.
It writes audit reports without deleting or relabeling input records. An audit
run is not required by the benchmark.
