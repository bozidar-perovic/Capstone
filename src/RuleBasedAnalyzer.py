import time

from .BaseAnalyzer import BaseAnalyzer
from .utility import create_unique_id


class RuleBasedAnalyzer(BaseAnalyzer):
    """
    Represents the keyword analyzer that analyzes the code based on
    specific keywords that indicate potential flakiness.

    Attributes:
        analyze(self,data): Analyzes the code based on specific
        keywords and calculates a score indicating potential flakiness.

    Example:
        >>> analyzer = RuleBasedAnalyzer()
        >>> analyzer.analyze("path/to/code.csv")
        0.2
        0.3
        0.5
        etc.
    """

    def __init__(self):
        super().__init__()
        self.total_score = 0
        self.scores = []

    def analyze(self, data):

        #  Load the data from the CSV file
        data = create_unique_id(data)

        if data is None:
            print("Failed to load data.")
            return 0

        # Reset per-run scores so repeated
        # analyze() calls stay aligned to rows.
        self.scores = []

        # Iterate through each row in the DataFrame
        for row in data.itertuples(index=False):

            # Extract the full_code from the row
            test_method = row.full_code

            self.total_score = 0

            if isinstance(test_method, str) and "Thread.sleep(" in test_method:
                self.total_score += 0.35

            if isinstance(test_method, str) and "new Thread(" in test_method:
                self.total_score += 0.25

            if isinstance(test_method, str) and "await" in test_method:
                self.total_score += 0.15

            if isinstance(test_method, str) and "Executor" in test_method:
                self.total_score += 0.25

            if isinstance(test_method, str) and "Atomic" in test_method:
                self.total_score += 0.1

            if isinstance(test_method, str) and "Date" in test_method:
                self.total_score += 0.1

            if isinstance(test_method, str) and (
                "Time" in test_method or "time" in test_method
            ):
                self.total_score += 0.1

            if isinstance(test_method, str) and (
                "Api" in test_method or "api" in test_method or "API" in test_method
            ):
                self.total_score += 0.05

            if isinstance(test_method, str) and "Random" in test_method:
                self.total_score += 0.3

            if isinstance(test_method, str) and (
                "Shared" in test_method or "share" in test_method
            ):
                self.total_score += 0.15

            if isinstance(test_method, str) and "sync" in test_method:
                self.total_score += 0.25

            self.scores.append(self.total_score)

        data["flaky_score"] = self.scores
        return data

    def time_passed(self, data):

        start = time.perf_counter()

        self.analyze(data)

        end = time.perf_counter()

        return end - start
