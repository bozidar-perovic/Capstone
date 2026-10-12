"""Adaptive hybrid analyzer combining fast regex filtering with ML inference.

Implements the multi-stage architecture defined in the capstone guide:
1. Stage 1: Fast regex-based rule scan on all test methods.
2. Confidently clean cases (rule_score <= low_threshold) receive 0.0 risk,
   skipping ML feature extraction and inference entirely (~80% of tests).
3. Confidently high-risk cases (rule_score >= high_threshold) receive 0.95,
   flagged immediately by high-signal concurrency/timing constructs.
4. Ambiguous cases (low_threshold < rule_score < high_threshold) are routed
   to the ML analyzer for deeper statistical prioritization.
"""

from typing import Any, Dict, Optional
import numpy as np

from .BaseAnalyzer import BaseAnalyzer
from .features import FEATURE_PATTERNS
from .MLAnalyzer import MLAnalyzer

DEFAULT_RULE_WEIGHTS: Dict[str, float] = {
    "sleep_count": 0.35,
    "await_count": 0.15,
    "executor_count": 0.25,
    "future_count": 0.15,
    "latch_barrier_count": 0.20,
    "time_api_count": 0.15,
    "random_api_count": 0.30,
    "unordered_collection_count": 0.10,
    "file_network_count": 0.15,
}


class AdaptiveAnalyzer(BaseAnalyzer):
    """Tiered analyzer routing tests to fast rules or machine learning."""

    def __init__(
        self,
        low_threshold: float = 0.0,
        high_threshold: float = 0.5,
        high_score: float = 0.95,
        ml_analyzer: Optional[Any] = None,
        rule_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.high_score = high_score
        self.ml_analyzer = ml_analyzer or MLAnalyzer()
        self.rule_weights = rule_weights or DEFAULT_RULE_WEIGHTS
        self.FEATURE_PATTERNS = FEATURE_PATTERNS
        self._is_trained = False
        self.last_routing_summary: Dict[str, Any] = {}

    def compute_rule_score(self, code: Any) -> float:
        """Compute Stage 1 rule presence score using compiled regex patterns.

        Thread creation and thread start share a single weight to avoid
        double-counting 'new Thread(...).start()'.
        """
        if not isinstance(code, str) or not code:
            return 0.0

        score = 0.0

        # Shared thread presence indicator
        has_thread = bool(
            self.FEATURE_PATTERNS["thread_creation_count"].search(code)
            or self.FEATURE_PATTERNS["thread_start_count"].search(code)
        )
        if has_thread:
            score += 0.25

        for pattern_name, weight in self.rule_weights.items():
            pattern = self.FEATURE_PATTERNS.get(pattern_name)
            if pattern and pattern.search(code):
                score += weight

        return score

    def train(self, train_data):
        """Train the underlying ML analyzer on training data."""
        self.ml_analyzer.train(train_data)
        self._is_trained = True
        return self

    def analyze(self, data):
        """Prioritize tests using the tiered 3-zone routing algorithm."""
        if not self._is_trained:
            raise ValueError("Train the model before analyzing test data.")

        result = data.copy()
        n = len(data)

        # Stage 1: Fast regex scan
        rule_scores = np.array([
            self.compute_rule_score(c) for c in data["full_code"]
        ])

        final_scores = np.zeros(n, dtype=float)
        routes = np.empty(n, dtype=object)

        # Zone 1: Confidently clean
        clean_mask = rule_scores <= self.low_threshold
        final_scores[clean_mask] = 0.0
        routes[clean_mask] = "fast_clean"

        # Zone 3: Confidently high-risk
        high_mask = rule_scores >= self.high_threshold
        final_scores[high_mask] = self.high_score
        routes[high_mask] = "fast_flaky"

        # Zone 2: Uncertain / ambiguous -> ML evaluation
        mid_mask = (~clean_mask) & (~high_mask)
        routes[mid_mask] = "ml_deep"

        if mid_mask.any():
            mid_data = data.loc[mid_mask]
            ml_results = self.ml_analyzer.analyze(mid_data)
            final_scores[mid_mask] = ml_results["flaky_score"].values

        result["flaky_score"] = final_scores
        result["route"] = routes
        result["rule_score"] = rule_scores

        # Record routing metrics for cost/efficiency analysis
        clean_count = int(np.sum(clean_mask))
        high_count = int(np.sum(high_mask))
        mid_count = int(np.sum(mid_mask))

        self.last_routing_summary = {
            "total_tests": n,
            "fast_clean_count": clean_count,
            "fast_clean_pct": (clean_count / n) if n > 0 else 0.0,
            "fast_flaky_count": high_count,
            "fast_flaky_pct": (high_count / n) if n > 0 else 0.0,
            "ml_deep_count": mid_count,
            "ml_deep_pct": (mid_count / n) if n > 0 else 0.0,
        }

        return result

    def get_routing_summary(self) -> Dict[str, Any]:
        """Return the routing breakdown from the most recent analyze call."""
        return self.last_routing_summary
