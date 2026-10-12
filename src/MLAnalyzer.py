import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .BaseAnalyzer import BaseAnalyzer
from .features import FEATURE_PATTERNS, extract_features


class MLAnalyzer(BaseAnalyzer):
    def __init__(self, max_count=5):
        super().__init__()
        self.max_count = max_count
        self.model = None
        self.FEATURE_PATTERNS = FEATURE_PATTERNS

    def extract_features(self, data):
        """Return raw counts; scaling belongs to the fitted model pipeline."""
        return extract_features(data, self.FEATURE_PATTERNS)

    def train(self, train_data):
        X_train = self.extract_features(train_data)
        # Category 5 is non-flaky; categories 0 through 4 are flaky.
        y_train = (train_data["category"] != 5).astype(int)

        steps = []
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
        steps.extend(
            [
                ("log", FunctionTransformer(np.log1p, validate=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000
                    ),
                ),
            ]
        )
        self.model = Pipeline(steps)
        self.model.fit(X_train, y_train)
        return self

    def analyze(self, data):
        if self.model is None:
            raise ValueError("Train the model before analyzing test data.")

        X_test = self.extract_features(data)
        result = data.copy()
        result["flaky_score"] = self.model.predict_proba(X_test)[:, 1]
        return result
