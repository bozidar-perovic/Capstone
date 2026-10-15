"""Random Forest static analyzer for non-linear flaky test prioritization.

Implements an ensemble decision-tree baseline to capture feature interactions
(e.g., combinations of thread creation, assertions, and loop nesting) that
simple linear models might miss.

Features:
- Depth-constrained trees (max_depth=5) to prevent overfitting on small
  flaky test subsets.
- Balanced class weights to accommodate heavy class imbalance (~2-3% flaky).
- Gini impurity-based feature importance querying to identify the strongest
  static flakiness predictors across repositories.
"""

from typing import Literal, Mapping, Optional, Sequence, Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .BaseAnalyzer import BaseAnalyzer
from .features import FEATURE_PATTERNS, extract_features

ClassWeight = Union[
    Literal["balanced", "balanced_subsample"],
    Mapping,
    Sequence[Mapping],
    None,
]


class RandomForestAnalyzer(BaseAnalyzer):
    """Random Forest classifier for static test flakiness risk scoring.

    Trains an ensemble of decision trees on raw pattern occurrence counts,
    providing non-linear decision boundaries and Gini feature importances.

    Attributes:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth to constrain model complexity.
        random_state: Seed for pseudo-random number generation.
        class_weight: Weighting strategy for handling class imbalance.
        model: Fitted RandomForestClassifier instance, or None before training.
        FEATURE_PATTERNS: Dictionary of compiled regex patterns.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 5,
        random_state: Optional[int] = 42,
        class_weight: ClassWeight = "balanced",
    ) -> None:
        """Initialize the Random Forest analyzer with tree hyperparameters.

        Args:
            n_estimators: The number of trees in the forest (default: 100).
            max_depth: The maximum depth of each tree (default: 5).
                Constraining depth prevents memorizing project code idioms.
            random_state: Random state seed for reproducibility (default: 42).
            class_weight: Strategy for adjusting weights inversely proportional
                to class frequencies (default: 'balanced').
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.class_weight: ClassWeight = class_weight
        self.model: Optional[RandomForestClassifier] = None
        self.FEATURE_PATTERNS = FEATURE_PATTERNS

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract raw pattern frequency counts from test source code.

        Args:
            data: Input DataFrame containing a 'full_code' column.

        Returns:
            A DataFrame of integer occurrence counts for all feature patterns.
        """
        return extract_features(data, self.FEATURE_PATTERNS)

    def train(self, train_data: pd.DataFrame) -> "RandomForestAnalyzer":
        """Train Random Forest classifier on labeled training data.

        Converts the dataset's 'category' column into a binary ground truth,
        where category 5 is non-flaky (0) and categories 0-4 are flaky (1).

        Args:
            train_data: Training DataFrame with 'full_code' and 'category'.

        Returns:
            Self (the fitted RandomForestAnalyzer instance).
        """
        X_train = self.extract_features(train_data)
        # Category 5 is non-flaky; categories 0 through 4 are flaky.
        y_train = (train_data["category"] != 5).astype(int)

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight=self.class_weight,
        )
        self.model.fit(X_train, y_train)
        return self

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict continuous flaky risk probabilities for test cases.

        Args:
            data: DataFrame containing test records to score.

        Returns:
            A copy of the input DataFrame with an appended 'flaky_score'
            column containing predicted probabilities of being flaky [0, 1].

        Raises:
            ValueError: If called before the model has been trained.
        """
        if self.model is None:
            raise ValueError("Train the model before analyzing test data.")

        X_test = self.extract_features(data)
        result = data.copy()
        # Extract class 1 (flaky) probability estimates
        result["flaky_score"] = self.model.predict_proba(X_test)[:, 1]
        return result

    def get_feature_importances(self) -> pd.Series:
        """Return Gini feature importances sorted in descending order.

        Provides model interpretability by ranking which static patterns
        contributed most to splitting decisions across the ensemble.

        Returns:
            A pandas Series mapping feature pattern names to importance
            scores summing to 1.0.

        Raises:
            ValueError: If called before the model has been trained.
        """
        if self.model is None:
            raise ValueError("Train the model before querying importances.")

        return pd.Series(
            self.model.feature_importances_,
            index=list(self.FEATURE_PATTERNS.keys()),
        ).sort_values(ascending=False)
