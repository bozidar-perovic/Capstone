"""CLI and backward-compatible entry points for the analyzer benchmark.

Model behavior lives in BaseAnalyzer subclasses; fold execution and summary
reporting live in BenchmarkRunner.
"""

from pathlib import Path
import sys
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import pandas as pd

from .benchmark import (
    AnalyzerSpec,
    BenchmarkRunner,
    DEFAULT_ANALYZERS,
    DEFAULT_SUMMARY_FILE,
    FoldResult,
)
from .utility import load_data, save_results_to_csv


def _run_fold(index: int, fold_number: int) -> FoldResult:
    runner = BenchmarkRunner(loader=load_data, saver=save_results_to_csv)
    return runner.run_fold(DEFAULT_ANALYZERS[index], fold_number)


def run_rule_based_fold(
    fold_number: int, *, include_timings: bool = False,
) -> Union[Optional[Dict[str, float]], FoldResult]:
    """Score and evaluate a rule-based fold without training data."""
    result = _run_fold(0, fold_number)
    return result if include_timings else result.metrics


def run_ml_fold(
    fold_number: int, *, include_timings: bool = False,
) -> Union[Optional[Dict[str, float]], FoldResult]:
    """Train and evaluate logistic regression on one fold."""
    result = _run_fold(1, fold_number)
    return result if include_timings else result.metrics


def run_rf_fold(
    fold_number: int, *, include_timings: bool = False,
) -> Union[Optional[Dict[str, float]], FoldResult]:
    """Train and evaluate Random Forest on one fold."""
    result = _run_fold(2, fold_number)
    return result if include_timings else result.metrics


def run_adaptive_fold(
    fold_number: int, *, include_timings: bool = False,
) -> Union[
    Tuple[Optional[Dict[str, float]], Optional[Dict[str, Any]]], FoldResult,
]:
    """Train and evaluate the hybrid, returning routing diagnostics too."""
    result = _run_fold(3, fold_number)
    if include_timings:
        return result
    return result.metrics, result.routing


def compare_folds(
    folds: Sequence[int] = (1, 2, 3, 4),
    output_file: Union[str, Path] = DEFAULT_SUMMARY_FILE,
) -> Optional[pd.DataFrame]:
    """Benchmark the original dataset folds and save their results."""
    runners: Dict[str, Callable[..., Any]] = {
        "Rule-Based": run_rule_based_fold,
        "Logistic Reg": run_ml_fold,
        "Random Forest": run_rf_fold,
    }

    def run(spec: AnalyzerSpec, fold: int) -> FoldResult:
        if spec.name == "Adaptive Hybrid":
            result = run_adaptive_fold(fold, include_timings=True)
            if isinstance(result, FoldResult):
                return result
            metrics, routing = result
            return FoldResult(metrics=metrics, routing=routing)
        result = runners[spec.name](fold, include_timings=True)
        return result if isinstance(result, FoldResult) else FoldResult(
            metrics=result,
        )

    return BenchmarkRunner().compare_folds(folds, output_file, fold_runner=run)


if __name__ == "__main__":
    out_path = (
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SUMMARY_FILE
    )
    compare_folds(output_file=out_path)
