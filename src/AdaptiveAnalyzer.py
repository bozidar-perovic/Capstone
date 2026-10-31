"""Adaptive hybrid analyzer combining fast regex filtering with ML inference.

Implements the multi-stage triage architecture defined in the capstone guide:
1. Stage 1: Fast regex-based rule scan on all test methods.
2. Zone 1 (Confidently Clean): Tests with rule_score <= low_threshold receive
   an immediate 0.0 risk score, completely bypassing ML feature extraction
   and model inference.
3. Zone 3 (Confidently Flaky): Tests with rule_score >= high_threshold receive
   an immediate high risk score (default: 0.95), flagged by strong
   concurrency/timing signals.
4. Zone 2 (Ambiguous): Tests with low_threshold < rule_score < high_threshold
   are routed to the ML analyzer for statistical evaluation.

Either bypass can be disabled to let the same ML backend reconsider that group.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from .BaseAnalyzer import BaseAnalyzer
from .features import FEATURE_PATTERNS
from .MLAnalyzer import MLAnalyzer

# Default presence weights for Stage 1 heuristic screening
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
    """Tiered analyzer routing tests to fast rules or machine learning.

    Screens test methods and delegates selected routes to the ML backend.

    Attributes:
        low_threshold: Cutoff below which tests bypass ML as clean.
        high_threshold: Cutoff above which tests bypass ML as high-risk.
        high_score: Fixed risk score assigned to fast-flaky tests.
        ml_analyzer: Machine learning model instance used for ambiguous tests.
        rule_weights: Mapping of feature names to Stage 1 heuristic weights.
        bypass_low: Whether low rule scores skip ML and receive zero.
        bypass_high: Whether high rule scores skip ML and receive high_score.
        FEATURE_PATTERNS: Dictionary of compiled regex patterns.
        last_routing_summary: Dict recording test counts and percentages
            routed through each zone in the most recent analysis run.
    """

    requires_training = True

    def __init__(
        self,
        low_threshold: float = 0.0,
        high_threshold: float = 0.5,
        high_score: float = 0.95,
        ml_analyzer: Optional[Any] = None,
        rule_weights: Optional[Dict[str, float]] = None,
        bypass_low: bool = True,
        bypass_high: bool = True,
    ) -> None:
        """Initialize the AdaptiveAnalyzer with thresholds and ML backend.

        Args:
            low_threshold: Upper threshold for Zone 1 (fast clean). Tests with
                rule_score <= low_threshold skip ML and receive 0.0 risk score.
            high_threshold: Lower threshold for Zone 3 (fast flaky). Tests with
                rule_score >= high_threshold skip ML and receive high_score.
            high_score: Risk score assigned to Zone 3 tests (default: 0.95).
            ml_analyzer: Optional ML analyzer backend. Defaults to MLAnalyzer()
                (logistic regression), but can accept any BaseAnalyzer instance
                such as RandomForestAnalyzer.
            rule_weights: Optional dictionary of Stage 1 heuristic weights.
                Defaults to DEFAULT_RULE_WEIGHTS.
            bypass_low: Keep the low-score shortcut when True. If False,
                send those tests to the existing ML backend instead.
            bypass_high: Keep the high-score shortcut when True. If False,
                send those tests to the existing ML backend instead.
        """
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.high_score = high_score
        self.ml_analyzer = ml_analyzer or MLAnalyzer()
        self.rule_weights = rule_weights or DEFAULT_RULE_WEIGHTS
        if not all(
            isinstance(flag, bool) for flag in (bypass_low, bypass_high)
        ):
            raise ValueError("Bypass settings must be boolean values.")
        self.bypass_low = bypass_low
        self.bypass_high = bypass_high
        self.FEATURE_PATTERNS = FEATURE_PATTERNS
        self._is_trained = False
        self.last_routing_summary: Dict[str, Any] = {}

    def compute_rule_score(self, code: Any) -> float:
        """Compute Stage 1 rule presence score using compiled regex patterns.

        Thread creation and thread start share a single weight to avoid
        double-counting 'new Thread(...).start()'.

        Args:
            code: Java method source code string.

        Returns:
            Calculated heuristic presence score (>= 0.0).
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

    def train(self, train_data: pd.DataFrame) -> "AdaptiveAnalyzer":
        """Train the underlying ML analyzer on training data.

        Args:
            train_data: Training DataFrame with 'full_code' and 'category'.

        Returns:
            Self (the fitted AdaptiveAnalyzer instance).
        """
        self.ml_analyzer.train(train_data)
        self._is_trained = True
        return self

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prioritize tests using the tiered 3-zone routing algorithm.

        1. Computes Stage 1 rule presence score for each test.
        2. Assigns 0.0 to low scores if their bypass is enabled.
        3. Assigns high_score to high scores if their bypass is enabled.
        4. Evaluates every remaining test with the existing ML backend.

        Args:
            data: DataFrame containing test records to score.

        Returns:
            A copy of the input DataFrame with added columns:
            - 'flaky_score': continuous risk score [0.0, 1.0]
            - 'route': routing decision ('fast_clean', 'ml_deep', 'fast_flaky')
            - 'rule_score': Stage 1 heuristic presence score

        Raises:
            ValueError: If called before the model has been trained.
        """
        if not self._is_trained:
            raise ValueError("Train the model before analyzing test data.")

        result = data.copy()
        n = len(data)

        # Stage 1: Fast regex scan across all test methods
        rule_scores = np.array([
            self.compute_rule_score(c) for c in data["full_code"]
        ])

        final_scores = np.zeros(n, dtype=float)
        routes = np.empty(n, dtype=object)

        # Zone 1: Confidently clean -> bypass ML evaluation entirely
        low_rule_mask = rule_scores <= self.low_threshold
        clean_mask = low_rule_mask & self.bypass_low
        final_scores[clean_mask] = 0.0
        routes[clean_mask] = "fast_clean"

        # Zone 3: Confidently high-risk -> flag immediately without ML
        high_rule_mask = rule_scores >= self.high_threshold
        high_mask = high_rule_mask & self.bypass_high
        final_scores[high_mask] = self.high_score
        routes[high_mask] = "fast_flaky"

        # Zone 2: Uncertain / ambiguous -> route to ML analyzer
        mid_mask = (~clean_mask) & (~high_mask)
        routes[mid_mask] = "ml_deep"

        if mid_mask.any():
            mid_data = data.loc[mid_mask]
            ml_results = self.ml_analyzer.analyze(mid_data)
            final_scores[mid_mask] = ml_results["flaky_score"].values

        result["flaky_score"] = final_scores
        result["route"] = routes
        result["rule_score"] = rule_scores

        # Record counts and percentages for each route.
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
            "reconsidered_low_count": int(np.sum(low_rule_mask & mid_mask)),
            "reconsidered_high_count": int(np.sum(high_rule_mask & mid_mask)),
        }

        return result

    def get_routing_summary(self) -> Dict[str, Any]:
        """Return the routing breakdown from the most recent analyze call.

        Returns:
            Dictionary containing total tests, counts, and percentages
            for 'fast_clean', 'ml_deep', and 'fast_flaky' routes.
        """
        return self.last_routing_summary


class AdaptiveRuleScreen(AdaptiveAnalyzer):
    """Stateless baseline using precisely the adaptive indicators and weights.

    Returns the unchanged rule score directly, with no fixed-score routing or
    ML fitting/scoring. This isolates the screen from the cascade policy.
    """

    requires_training = False

    def __init__(self, rule_weights: Optional[Dict[str, float]] = None):
        super().__init__(rule_weights=rule_weights)

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["rule_score"] = [
            self.compute_rule_score(code) for code in data["full_code"]
        ]
        result["flaky_score"] = result["rule_score"]
        return result
