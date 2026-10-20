"""Shared fold execution and reporting, independent of analyzer internals."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import pandas as pd

from .AdaptiveAnalyzer import AdaptiveAnalyzer
from .BaseAnalyzer import BaseAnalyzer
from .metrics import compute_ranking_metrics
from .MLAnalyzer import MLAnalyzer
from .RandomForestAnalyzer import RandomForestAnalyzer
from .RuleBasedAnalyzer import RuleBasedAnalyzer
from .utility import load_data, save_results_to_csv


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
    """Ranking metrics and optional analyzer diagnostics for one fold."""

    metrics: Optional[Dict[str, float]] = None
    routing: Optional[Dict[str, Any]] = None


class BenchmarkRunner:
    """Own dataset paths, fold lifecycle, persistence, and report formatting.

    Analyzer factories prevent fitted state leaking between folds. Only the
    training split is passed to train; held-out records are passed to analyze.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "src/data",
        results_dir: Union[str, Path] = "src/results",
        analyzers: Sequence[AnalyzerSpec] = DEFAULT_ANALYZERS,
        loader: Callable = load_data,
        saver: Callable = save_results_to_csv,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.analyzers = tuple(analyzers)
        self.loader = loader
        self.saver = saver

    def run_fold(self, spec: AnalyzerSpec, fold_number: int) -> FoldResult:
        """Load, train if needed, score, save, and evaluate one analyzer."""
        analyzer = spec.factory()
        train_data = None
        if analyzer.requires_training:
            train_data = self.loader(
                self.data_dir / f"train_set_{fold_number}.csv"
            )
        test_data = self.loader(
            self.data_dir / f"test_set_{fold_number}.csv"
        )
        if test_data is None:
            return FoldResult()
        if analyzer.requires_training:
            if train_data is None:
                return FoldResult()
            analyzer.train(train_data)
        results = analyzer.analyze(test_data)
        if results is None:
            return FoldResult()
        self.saver(
            results,
            self.results_dir
            / f"test_set_{fold_number}{spec.result_suffix}_results.csv",
        )
        metrics = None
        if "category" in results.columns:
            y_true = (results["category"] != 5).astype(int)
            metrics = compute_ranking_metrics(y_true, results["flaky_score"])
        return FoldResult(metrics, analyzer.get_routing_summary() or None)

    def compare_folds(
        self,
        folds: Sequence[int] = (1, 2, 3, 4),
        output_file: Union[str, Path] = "src/benchmark_summary.csv",
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
        for fold in folds:
            for spec in self.analyzers:
                result = run(spec, fold)
                if result.metrics:
                    metrics[spec.name].append(result.metrics)
                if result.routing and spec.name == "Adaptive Hybrid":
                    routings.append(result.routing)

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

        summary = pd.DataFrame(rows)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_file, index=False)
        print(f'Process done, saved in the "{output_file}"')
        return summary
