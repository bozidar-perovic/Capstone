from typing import Literal, Mapping, Sequence

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .BaseAnalyzer import BaseAnalyzer
from .features import FEATURE_PATTERNS, extract_features

ClassWeight = (
    Literal["balanced", "balanced_subsample"] | Mapping | Sequence[Mapping] | None
)


class RandomForestAnalyzer(BaseAnalyzer):

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 5,
        random_state: int | None = 42,
        class_weight: ClassWeight = "balanced",
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.class_weight: ClassWeight = class_weight
        self.model = None
        self.FEATURE_PATTERNS = FEATURE_PATTERNS

    def extract_features(self, data):
        """Extract raw counts for all static code patterns."""
        return extract_features(data, self.FEATURE_PATTERNS)

    def train(self, train_data):
        """Train Random Forest classifier on labeled training data."""
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

    def analyze(self, data):
        """Predict flaky probabilities for test cases."""
        if self.model is None:
            raise ValueError("Train the model before analyzing test data.")

        X_test = self.extract_features(data)
        result = data.copy()
        result["flaky_score"] = self.model.predict_proba(X_test)[:, 1]
        return result

    def get_feature_importances(self) -> pd.Series:
        """Return Gini feature importances sorted descending."""
        if self.model is None:
            raise ValueError("Train the model before querying importances.")

        return pd.Series(
            self.model.feature_importances_,
            index=list(self.FEATURE_PATTERNS.keys()),
        ).sort_values(ascending=False)
