import pandas as pd
from sklearn.linear_model import LogisticRegression

from .BaseAnalyzer import BaseAnalyzer


class MLAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.model = None

    def extract_features(self, data):

        feature_rows = []

        for code in data["full_code"]:
            if not isinstance(code, str):
                code = ""

            feature_rows.append(
                {
                    "has_thread_sleep": int("Thread.sleep(" in code),
                    "has_new_thread": int("new Thread(" in code),
                    "has_await": int("await" in code),
                    "has_executor": int("Executor" in code),
                    "has_atomic": int("Atomic" in code),
                    "has_date": int("Date" in code),
                    "has_time": int("Time" in code or "time" in code),
                    "has_sync": int("sync" in code),
                    "has_random": int("Random" in code),
                    "line_count": len(code.splitlines()),
                }
            )

        return pd.DataFrame(feature_rows, index=data.index)

    def train(self, train_data):

        X_train = self.extract_features(train_data)
        # Category 5 is non-flaky; categories 0 through 4 are flaky.
        y_train = (train_data["category"] != 5).astype(int)

        self.model = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.model.fit(X_train, y_train)
        return self

    def analyze(self, data):

        if self.model is None:
            raise ValueError("Train the model before analyzing test data.")

        X_test = self.extract_features(data)
        result = data.copy()
        result["flaky_score"] = self.model.predict_proba(X_test)[:, 1]
        return result
