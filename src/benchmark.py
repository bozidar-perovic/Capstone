"""Shared fold execution and reporting, independent of analyzer internals."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Optional, Sequence, Union

import pandas as pd

from .AdaptiveAnalyzer import AdaptiveAnalyzer
from .BaseAnalyzer import BaseAnalyzer
from .metrics import compute_ranking_metrics
from .MLAnalyzer import MLAnalyzer
from .RandomForestAnalyzer import RandomForestAnalyzer
from .RuleBasedAnalyzer import RuleBasedAnalyzer
from .utility import load_data, save_results_to_csv


# Canonical original dataset and its benchmark outputs.
DEFAULT_DATA_DIR = Path("src/data")
DEFAULT_RESULTS_DIR = Path("src/results")
DEFAULT_SUMMARY_FILE = DEFAULT_RESULTS_DIR / "benchmark_summary.csv"


@dataclass(frozen=True)
class AnalyzerSpec:
    """Benchmark metadata and factory (a fresh model for each fold)."""

    name: str
    factory: Callable[[], BaseAnalyzer]
    result_suffix: str


DEFAULT_ANALYZERS = (
    AnalyzerSpec("Rule-Based", RuleBasedAnalyzer, "_rule_based"),
    AnalyzerSpec("Logistic Reg", MLAnalyzer, ""),
    AnalyzerSpec("Random Forest", RandomForestAnalyzer, "_rf"),
    AnalyzerSpec("Adaptive Hybrid", AdaptiveAnalyzer, "_adaptive"),
)


@dataclass
class FoldResult:
    """Quality, routing, and elapsed seconds for one fold.

    Analysis includes the complete analyze call. CSV loading, model creation,
    metric calculation (including sorting), and CSV export are outside it.
    Training includes its feature extraction and fitted preprocessing.
    """

    metrics: Optional[Dict[str, float]] = None
    routing: Optional[Dict[str, Any]] = None
    timings: Optional[Dict[str, Optional[float]]] = None


class BenchmarkRunner:
    """Own dataset paths, fold lifecycle, persistence, and report formatting.

    Analyzer factories prevent fitted state leaking between folds. Only the
    training split is passed to train; held-out records are passed to analyze.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        results_dir: Union[str, Path] = DEFAULT_RESULTS_DIR,
        analyzers: Sequence[AnalyzerSpec] = DEFAULT_ANALYZERS,
        loader: Callable = load_data,
        saver: Callable = save_results_to_csv,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.analyzers = tuple(analyzers)
        self.loader = loader
        self.saver = saver

    def run_fold(
        self, spec: AnalyzerSpec, fold_number: int,
        save_predictions: bool = True,
    ) -> FoldResult:
        """Load, train if needed, score, save, and evaluate one analyzer."""
        analyzer = spec.factory()
        train_data = None
        started = perf_counter()
        if analyzer.requires_training:
            train_data = self.loader(
                self.data_dir / f"train_set_{fold_number}.csv"
            )
        test_data = self.loader(
            self.data_dir / f"test_set_{fold_number}.csv"
        )
        loading_seconds = perf_counter() - started
        if test_data is None:
            return FoldResult()
        training_seconds = None
        if analyzer.requires_training:
            if train_data is None:
                return FoldResult()
            started = perf_counter()
            analyzer.train(train_data)
            training_seconds = perf_counter() - started
        started = perf_counter()
        results = analyzer.analyze(test_data)
        analysis_seconds = perf_counter() - started
        if results is None:
            return FoldResult()
        reporting_seconds = None
        if save_predictions:
            started = perf_counter()
            self.saver(
                results,
                self.results_dir
                / f"test_set_{fold_number}{spec.result_suffix}_results.csv",
            )
            reporting_seconds = perf_counter() - started
        started = perf_counter()
        metrics = None
        if "category" in results.columns:
            y_true = (results["category"] != 5).astype(int)
            metrics = compute_ranking_metrics(y_true, results["flaky_score"])
        evaluation_seconds = perf_counter() - started
        timings = {
            "data_loading_seconds": loading_seconds,
            "training_seconds": training_seconds,
            "analysis_seconds": analysis_seconds,
            "evaluation_seconds": evaluation_seconds,
            "reporting_seconds": reporting_seconds,
            "training_and_analysis_seconds": (
                (training_seconds or 0.0) + analysis_seconds
            ),
        }
        return FoldResult(
            metrics, analyzer.get_routing_summary() or None, timings,
        )

    def compare_folds(
        self,
        folds: Sequence[int] = (1, 2, 3, 4),
        output_file: Union[str, Path] = DEFAULT_SUMMARY_FILE,
        fold_runner: Optional[
            Callable[[AnalyzerSpec, int], FoldResult]
        ] = None,
    ) -> Optional[pd.DataFrame]:
        """Aggregate fold metrics, preserving the existing benchmark format.

        An optional execution adapter supports the legacy main.py functions.
        """
        run = fold_runner or self.run_fold
        metrics = {spec.name: [] for spec in self.analyzers}
        routings = []
        timing_records = []
        for fold in folds:
            for spec in self.analyzers:
                result = run(spec, fold)
                if result.metrics:
                    metrics[spec.name].append(result.metrics)
                if result.routing and spec.name == "Adaptive Hybrid":
                    routings.append(result.routing)
                if result.timings:
                    timing_records.append({
                        "analyzer": spec.name, "fold": fold,
                        **result.timings,
                    })

        if not metrics or not all(metrics.values()):
            return None
        frames = {
            name: pd.DataFrame(values) for name, values in metrics.items()
        }
        rows = []
        for key, label in [
            ("roc_auc", "ROC-AUC"),
            ("pr_auc", "PR-AUC"),
            ("recall@5%", "Recall@5%"),
            ("recall@10%", "Recall@10%"),
            ("recall@20%", "Recall@20%"),
            ("precision@5%", "Precision@5%"),
            ("precision@10%", "Precision@10%"),
        ]:
            fmt = ".1%" if "@" in key else ".4f"
            row = {"Metric": label}
            row.update({
                name: format(frame[key].mean(), fmt)
                for name, frame in frames.items()
            })
            rows.append(row)

        if routings:
            avg = pd.DataFrame(routings).mean()
            for label, value, defaults in [
                ("Fast Clean (Bypassed ML)", avg["fast_clean_pct"],
                 ("N/A", "N/A", "N/A")),
                ("ML Routed (Evaluated)", avg["ml_deep_pct"],
                 ("0.0%", "100.0%", "100.0%")),
                ("Fast Flaky (Bypassed ML)", avg["fast_flaky_pct"],
                 ("N/A", "N/A", "N/A")),
                ("Total ML Workload Saved", 1 - avg["ml_deep_pct"],
                 ("100.0%", "0.0%", "0.0%")),
            ]:
                row = {"Metric": label}
                row.update(dict(zip(
                    ("Rule-Based", "Logistic Reg", "Random Forest"), defaults
                )))
                row["Adaptive Hybrid"] = f"{value:.1%}"
                rows.append(row)

        if timing_records:
            timing_frame = pd.DataFrame(timing_records)
            averages = timing_frame.groupby("analyzer").mean(numeric_only=True)
            for key, label in [
                ("training_seconds", "Training Time (s)"),
                ("analysis_seconds", "Analysis Time (s)"),
                ("training_and_analysis_seconds", "Training + Analysis (s)"),
                ("evaluation_seconds", "Evaluation Time (s)"),
                ("reporting_seconds", "Prediction Export Time (s)"),
                ("data_loading_seconds", "Data Loading Time (s)"),
            ]:
                row = {"Metric": label}
                for spec in self.analyzers:
                    value = (
                        averages.loc[spec.name, key]
                        if spec.name in averages.index else float("nan")
                    )
                    row[spec.name] = (
                        "N/A" if pd.isna(value) else f"{value:.6f}"
                    )
                rows.append(row)

        summary = pd.DataFrame(rows)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_file, index=False)
        if timing_records:
            timing_path = Path(output_file).with_name(
                f"{Path(output_file).stem}_timings.csv"
            )
            pd.DataFrame(timing_records).to_csv(timing_path, index=False)
        print(f'Process done, saved in the "{output_file}"')
        return summary
