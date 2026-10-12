from pathlib import Path
import time
import pandas as pd

from .BaseAnalyzer import BaseAnalyzer
from .features import FEATURE_PATTERNS
from .utility import create_unique_id


class RuleBasedAnalyzer(BaseAnalyzer):
    """Keyword analyzer evaluating presence of flakiness indicators.

    Uses precompiled regex patterns from features.py. Signals contribute
    once per method (presence, not counts) and thread creation and start
    share a single weight to avoid double-counting.
    """

    def __init__(self):
        super().__init__()
        self.total_score = 0
        self.scores = []

    def analyze(self, data):
        # Support file path strings, Path objects, and preloaded DataFrames
        if isinstance(data, (str, Path)):
            data = create_unique_id(data)
            if data is None:
                print("Failed to load data.")
                return None
        elif isinstance(data, pd.DataFrame):
            data = data.copy()
        else:
            print("Failed to load data.")
            return None

        # Reset per-run scores so repeated analyze() calls stay aligned
        self.scores = []
        self.total_score = 0

        if data.empty:
            data["flaky_score"] = []
            return data

        # Iterate through each row in the DataFrame
        for row in data.itertuples(index=False):
            test_method = getattr(row, "full_code", None)
            self.total_score = 0

            if isinstance(test_method, str) and test_method:
                if FEATURE_PATTERNS["sleep_count"].search(test_method):
                    self.total_score += 0.35

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

                if FEATURE_PATTERNS["await_count"].search(test_method):
                    self.total_score += 0.15

                if FEATURE_PATTERNS["executor_count"].search(test_method):
                    self.total_score += 0.25

                if FEATURE_PATTERNS["time_api_count"].search(test_method):
                    self.total_score += 0.10

                if FEATURE_PATTERNS["random_api_count"].search(test_method):
                    self.total_score += 0.30

            self.scores.append(self.total_score)

        data["flaky_score"] = self.scores
        return data

    def time_passed(self, data):
        start = time.perf_counter()
        self.analyze(data)
        end = time.perf_counter()
        return end - start
