"""Repeat the fixed-fold benchmark with training and analysis timed separately.

No validation or parameter selection is performed. Run sequentially on the same
machine; repetitions describe runtime noise, not independent datasets.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

from .AdaptiveAnalyzer import AdaptiveAnalyzer
from .benchmark import (
    AnalyzerSpec, BenchmarkRunner, DEFAULT_ANALYZERS, DEFAULT_DATA_DIR,
)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(output_dir, manifest):
    (output_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


def run_timing(
    output_dir, data_dir=DEFAULT_DATA_DIR, folds=(1, 2, 3, 4),
    repeats=5, warmups=1, threads=1, adaptive_policy="reconsider-high",
):
    """Save raw repetitions, per-fold variation, quality and timing summaries.

    Every repetition creates and trains a fresh analyzer. Warm-ups are excluded
    from all reports. Only generated prediction files in this new directory are
    overwritten between repetitions. Timings exclude imports and construction.
    """
    if repeats < 1 or warmups < 0 or threads < 1:
        raise ValueError("Use repeats >= 1, warmups >= 0, and threads >= 1.")
    folds = tuple(folds)
    if not folds or len(set(folds)) != len(folds):
        raise ValueError("Provide at least one fold, without duplicates.")
    if adaptive_policy not in ("current", "reconsider-high"):
        raise ValueError("Unknown adaptive routing policy.")
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    inputs = [
        data_dir / f"{split}_set_{fold}.csv"
        for fold in folds for split in ("train", "test")
    ]
    input_hashes = {str(path.resolve()): checksum(path) for path in inputs}
    output_dir.mkdir(parents=True, exist_ok=False)
    analyzers = list(DEFAULT_ANALYZERS)
    analyzers[-1] = AnalyzerSpec(
        "Adaptive Hybrid",
        lambda: AdaptiveAnalyzer(
            bypass_low=True, bypass_high=(adaptive_policy == "current"),
        ),
        "_adaptive",
    )
    manifest = {
        "status": "running",
        "procedure": "Fixed outer folds; no validation or setting selection",
        "folds": folds, "repeats": repeats, "warmups_per_fold": warmups,
        "adaptive_policy": adaptive_policy,
        "other_settings": "Unchanged defaults, including raw standalone rules",
        "timer": "time.perf_counter; wall-clock seconds",
        "analysis_boundary": "Entire analyze call, returning unsorted scores",
        "training_boundary": "Entire train call, including feature extraction",
        "evaluation_boundary": "Ranking metrics, including cutoff sorting",
        "reporting_boundary": "Prediction CSV export only",
        "excluded": "Imports, analyzer construction, final summary export",
        "aggregation": "Unweighted mean of per-fold repetition medians",
        "execution": "Sequential; analyzer order rotated by fold and repeat",
        "requested_native_threads": threads,
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
        },
        "source_sha256": {
            path.name: checksum(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "input_sha256": input_hashes,
    }
    write_manifest(output_dir, manifest)
    runner = BenchmarkRunner(data_dir=data_dir, results_dir=output_dir)
    records, results = [], {}
    try:
        with threadpool_limits(limits=threads):
            manifest["native_threadpools"] = threadpool_info()
            for fold_index, fold in enumerate(folds):
                for _ in range(warmups):
                    for spec in analyzers:
                        result = runner.run_fold(
                            spec, fold, save_predictions=False,
                        )
                        if not result.metrics:
                            raise ValueError(
                                "Warm-up did not produce metrics."
                            )
                for repeat in range(repeats):
                    offset = (fold_index + repeat) % len(analyzers)
                    order = analyzers[offset:] + analyzers[:offset]
                    for position, spec in enumerate(order, start=1):
                        result = runner.run_fold(spec, fold)
                        if not result.metrics or not result.timings:
                            raise ValueError("Fold did not produce results.")
                        results.setdefault((fold, spec.name), result)
                        records.append({
                            "analyzer": spec.name, "fold": fold,
                            "repeat": repeat + 1, "order": position,
                            **result.timings, **result.metrics,
                            **{
                                f"routing_{key}": value
                                for key, value in (
                                    result.routing or {}
                                ).items()
                            },
                        })
                    print(
                        f"Fold {fold}: repetition {repeat + 1}/{repeats}",
                        flush=True,
                    )

        raw = pd.DataFrame(records)
        raw.to_csv(output_dir / "timing_runs.csv", index=False)
        timing_keys = list(next(iter(results.values())).timings)
        grouped = raw.groupby(["analyzer", "fold"], sort=False)[timing_keys]
        by_fold = grouped.agg(["median", "min", "max"])
        by_fold.columns = [
            f"{key}_{stat}" for key, stat in by_fold.columns
        ]
        by_fold.reset_index().to_csv(
            output_dir / "timing_by_fold.csv", index=False,
        )
        medians = grouped.median()
        speed = medians.groupby("analyzer", sort=False).mean()
        quality = pd.DataFrame([
            {"analyzer": name, "fold": fold, **result.metrics}
            for (fold, name), result in results.items()
        ])
        quality.to_csv(output_dir / "metrics_by_fold.csv", index=False)
        quality_mean = quality.drop(columns="fold").groupby("analyzer").mean()
        speed.join(quality_mean).reset_index().to_csv(
            output_dir / "quality_speed_summary.csv", index=False,
        )

        def median_fold_result(spec, fold):
            values = medians.loc[(spec.name, fold)]
            timings = {
                key: None if pd.isna(value) else float(value)
                for key, value in values.items()
            }
            return replace(results[(fold, spec.name)], timings=timings)

        runner.analyzers = tuple(analyzers)
        summary = runner.compare_folds(
            folds, output_dir / "benchmark_summary.csv",
            fold_runner=median_fold_result,
        )
        manifest["status"] = "complete"
        write_manifest(output_dir, manifest)
        return summary
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        write_manifest(output_dir, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4],
                        choices=[1, 2, 3, 4])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--adaptive-policy", default="reconsider-high",
                        choices=["current", "reconsider-high"])
    args = parser.parse_args()
    run_timing(**vars(args))


if __name__ == "__main__":
    main()
