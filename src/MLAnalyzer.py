"""Machine learning static analyzer using regularized logistic regression.

Trains a class-weighted logistic regression classifier on static source-code
pattern counts to predict flaky test risk probabilities.

The training pipeline implements three preprocessing stages designed to handle
the peculiarities of static test-code features:
1. Feature count clipping (np.clip): Caps raw counts (e.g. assertions, loops)
   at a maximum value (default: 5) to prevent sprawling unit or integration
   tests with dozens of assertions from artificially inflating flakiness risk.
2. Logarithmic transformation (np.log1p): Normalizes heavy-tailed and skewed
   code feature count distributions.
3. Feature standardization (StandardScaler): Centers and scales features to
   unit variance so regularized coefficients are treated fairly.
4. Balanced Logistic Regression: Automatically adjusts class weights inversely
   proportional to class frequencies to counteract extreme class imbalance
   (~2-3% flaky tests).
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .BaseAnalyzer import BaseAnalyzer
from .features import FEATURE_PATTERNS, extract_features


class MLAnalyzer(BaseAnalyzer):
    """Linear machine learning model for static flaky test risk prediction.

    Extracts regex-based static code counts, transforms them via a robust
    scaling pipeline, and fits a class-weighted logistic regression model.

    Attributes:
        max_count: Maximum threshold for feature count clipping, or None.
        model: Fitted scikit-learn Pipeline instance, or None before training.
        FEATURE_PATTERNS: Dictionary of compiled regular expressions used
            for feature extraction.
    """

    requires_training = True

    def __init__(self, max_count: Optional[int] = 5) -> None:
        """Initialize MLAnalyzer with clipping and pipeline settings.

        Args:
            max_count: Upper bound to clip raw feature counts at. Capping
                at 5 prevents complex tests with numerous assertions from
                generating extreme false positive scores. Pass None to disable.
        """
        super().__init__()
        self.max_count = max_count
        self.model: Optional[Pipeline] = None
        self.FEATURE_PATTERNS = FEATURE_PATTERNS

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract raw pattern frequency counts from test source code.

        Scaling and transformation steps are intentionally deferred to the
        fitted Pipeline to avoid data leakage across train and test splits.

        Args:
            data: Input DataFrame containing a 'full_code' column.

        Returns:
            A DataFrame of integer occurrence counts for all feature patterns.
        """
        return extract_features(data, self.FEATURE_PATTERNS)

    def train(self, train_data: pd.DataFrame) -> "MLAnalyzer":
        """Train the logistic regression pipeline on labeled test code.

        Converts the dataset's 'category' column into a binary ground-truth
        indicator, where category 5 is non-flaky (0) and categories 0-4 are
        flaky (1).

        Args:
            train_data: Training DataFrame with 'full_code' and 'category'
                columns.

        Returns:
            Self (the fitted MLAnalyzer instance).
        """
        X_train = self.extract_features(train_data)
        # Category 5 is non-flaky; categories 0 through 4 are flaky.
        y_train = (train_data["category"] != 5).astype(int)

        steps = []
        # Step 1: Clip extreme counts to suppress assertion inflation
        if self.max_count is not None:
            steps.append(
                (
                    "clip",
                    FunctionTransformer(
                        np.clip,
                        kw_args={"a_min": 0, "a_max": self.max_count},
                        validate=True,
                    ),
                )
            )

        # Step 2: Log transform, standardize, and fit balanced classifier
        steps.extend(
            [
                ("log", FunctionTransformer(np.log1p, validate=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(class_weight="balanced", max_iter=1000),
                ),
            ]
        )
        self.model = Pipeline(steps)
        self.model.fit(X_train, y_train)
        return self

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict continuous flaky risk probabilities for held-out tests.

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
        # predict_proba returns [P(non-flaky), P(flaky)];
        # column 1 represents continuous flaky risk
        result["flaky_score"] = self.model.predict_proba(X_test)[:, 1]
        return result
