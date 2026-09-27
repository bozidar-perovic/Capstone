import pandas as pd

from .BaseAnalyzer import BaseAnalyzer


class MLAnalyzer(BaseAnalyzer):

    def extract_features(self, data):
        if "full_code" not in data.columns:
            raise ValueError("Input data must have a full_code column.")

        feature_rows = []

        for code in data["full_code"]:
            if not isinstance(code, str):
                code = ""

            feature_rows.append(
                {
                    "has_thread_sleep": int("Thread.sleep(" in code),
                    "has_new_thread": int("new Thread(" in code),
                    "line_count": len(code.splitlines()),
                }
            )

        return pd.DataFrame(feature_rows, index=data.index)

    def analyze(self, data):
        print("hello")
