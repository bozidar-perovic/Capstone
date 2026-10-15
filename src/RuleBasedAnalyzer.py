"""Rule-based static analyzer for flaky test prioritization.

Assigns transparent, heuristic risk scores to Java test methods based on the
presence of known flakiness-inducing code constructs:
- Explicit thread sleeps (`Thread.sleep`)
- Thread creation and execution (`new Thread`, `.start()`)
- Asynchronous waiting constructs (`await`)
- Concurrency managers and executors (`ExecutorService`, `Executors`)
- System and wall-clock time APIs (`System.currentTimeMillis`, `nanoTime`)
- Nondeterministic and random APIs (`Random`, `Math.random`)

Unlike count-based models, this analyzer scores feature *presence* (binary
signal per test method) to avoid skew from repeated calls within loops or
boilerplate. Thread creation and thread starting share a single weight to
prevent double-counting compound statements like `new Thread(task).start()`.
"""

from pathlib import Path
import time
from typing import List, Optional, Union

import pandas as pd

from .BaseAnalyzer import BaseAnalyzer
from .features import FEATURE_PATTERNS
from .utility import create_unique_id


class RuleBasedAnalyzer(BaseAnalyzer):
    """Keyword heuristic analyzer evaluating presence of flakiness indicators.

    Uses precompiled regex patterns from features.py. Signals contribute once
    per test method to avoid overweighting repetitive method bodies.

    Attributes:
        total_score: Running total score for the currently inspected method.
        scores: List of continuous flakiness risk scores assigned to the
            most recently analyzed batch of tests.
    """

    def __init__(self) -> None:
        """Initialize the rule-based analyzer with empty score accumulators."""
        super().__init__()
        self.total_score: float = 0.0
        self.scores: List[float] = []

    def analyze(
        self, data: Union[str, Path, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Prioritize tests by computing heuristic flakiness scores.

        Iterates through test methods, inspecting the source code for static
        indicators of concurrency, asynchronous waiting, time sensitivity, and
        nondeterminism.

        Args:
            data: Input test suite, either as a DataFrame with a 'full_code'
                column, or a path (str/Path) to a CSV dataset file.

        Returns:
            A pandas DataFrame copy containing all original columns plus a
            new 'flaky_score' column with continuous heuristic risk scores,
            or None if the dataset could not be loaded.
        """
        # Support file path strings, Path objects, and preloaded DataFrames
        df: Optional[pd.DataFrame]
        if isinstance(data, (str, Path)):
            df = create_unique_id(data)
            if df is None:
                print("Failed to load data.")
                return None
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            print("Failed to load data.")
            return None

        # Reset per-run score cache so repeated calls on different frames
        # remain strictly independent and correctly sized.
        self.scores = []
        self.total_score = 0.0

        if df.empty:
            df["flaky_score"] = []
            return df

        # Iterate through test rows and evaluate static heuristic indicators
        for row in df.itertuples(index=False):
            test_method = getattr(row, "full_code", None)
            self.total_score = 0.0

            if isinstance(test_method, str) and test_method:
                # Thread.sleep is one of the strongest empirical indicators
                # of timing flakiness and race condition masking.
                if FEATURE_PATTERNS["sleep_count"].search(test_method):
                    self.total_score += 0.35

                # Thread creation and thread start share a single weight to
                # avoid double-counting "new Thread(task).start()".
                has_thread = bool(
                    FEATURE_PATTERNS["thread_creation_count"].search(
                        test_method
                    )
                    or FEATURE_PATTERNS["thread_start_count"].search(
                        test_method
                    )
                )
                if has_thread:
                    self.total_score += 0.25

                # Asynchronous waiting constructs (e.g., Awaitility, await)
                if FEATURE_PATTERNS["await_count"].search(test_method):
                    self.total_score += 0.15

                # Managed concurrency and thread pool usage
                if FEATURE_PATTERNS["executor_count"].search(test_method):
                    self.total_score += 0.25

                # System clock and wall-clock timing sensitivity
                if FEATURE_PATTERNS["time_api_count"].search(test_method):
                    self.total_score += 0.10

                # Random number generation and shuffle operations
                if FEATURE_PATTERNS["random_api_count"].search(test_method):
                    self.total_score += 0.30

            self.scores.append(self.total_score)

        df["flaky_score"] = self.scores
        return df

    def time_passed(self, data: Union[str, Path, pd.DataFrame]) -> float:
        """Measure the execution time required to analyze test data.

        Useful for benchmarking static analysis latency against runtime
        execution costs.

        Args:
            data: Input test data to analyze.

        Returns:
            Elapsed execution duration in seconds.
        """
        start = time.perf_counter()
        self.analyze(data)
        end = time.perf_counter()
        return end - start
