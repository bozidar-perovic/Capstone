import pandas as pd
from sklearn.linear_model import LogisticRegression
import re
from .BaseAnalyzer import BaseAnalyzer


class MLAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.model = None
        self.FEATURE_PATTERNS = {
            "sleep_count": re.compile(r"\bThread\s*\.\s*sleep\s*\("),
            "thread_creation_count": re.compile(r"\bnew\s+Thread\s*\("),
            "await_count": re.compile(r"\bawait\b"),
            "thread_start_count": re.compile(
                r"\b\w+\s*\.\s*start\s*\("
            ),
            "executor_count": re.compile(
                r"\b(?:Executors|ExecutorService|Executor)\b"
            ),
            "future_count": re.compile(r"\b(?:Future|CompletableFuture)\b"),
            "latch_barrier_count": re.compile(
                r"\b(?:CountDownLatch|CyclicBarrier|Semaphore|Phaser)\b"
            ),
            "time_api_count": re.compile(
                r"\b(?:Date|Instant|LocalDateTime|Clock)\b"
                r"|\bSystem\s*\.\s*(?:currentTimeMillis|nanoTime)\s*\("
            ),
            "random_api_count": re.compile(
                r"\b(?:Random|ThreadLocalRandom)\b"
                r"|\bMath\s*\.\s*random\s*\("
                r"|\bshuffle\s*\("
            ),
            "unordered_collection_count": re.compile(
                r"\b(?:HashMap|HashSet|Hashtable)\b"
            ),
            "file_network_count": re.compile(
                r"\b(?:File|Path|Files|Socket|URL|HttpClient|URLConnection)\b"
            ),
            "assertion_count": re.compile(
                r"\bassert(?:That|Equals|NotEquals|True|False|Null|NotNull|"
                r"Same|NotSame|Throws)\s*\("
                r"|\bAssertions\s*\.\s*assert\w*\s*\("
            ),
            "loop_count": re.compile(r"\b(?:for|while|do)\b"),
            "branch_count": re.compile(r"\b(?:if|switch)\b"),
        }

    def extract_features(self, data):

        feature_rows = []

        for code in data["full_code"]:
            if not isinstance(code, str):
                code = ""

            feature_rows.append(
                {
                    name: len(pattern.findall(code))
                    for name, pattern in self.FEATURE_PATTERNS.items()
                }
            )

        # Build a table where each row contains the extracted feature counts
        # for one input test method. The dictionary keys become feature-column
        # names, and the values are the counts found in that method's source.
        return pd.DataFrame(
            feature_rows,
            columns=list(self.FEATURE_PATTERNS),
            # Preserve the input DataFrame's row labels so these features stay
            # aligned with the corresponding tests and labels in later steps.
            index=data.index,
        )

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
