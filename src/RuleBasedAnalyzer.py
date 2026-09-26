try:
    from .BaseAnalyzer import BaseAnalyzer
    from .utility import create_unique_id
except ImportError:
    from BaseAnalyzer import BaseAnalyzer
    from utility import create_unique_id


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
        self.thread_sleep_score = 0
        self.new_thread_score = 0
        self.time_score = 0
        self.await_score = 0
        self.executor_score = 0
        self.atomic_score = 0
        self.date_score = 0
        self.api_score = 0
        self.random_score = 0
        self.shared_score = 0
        self.sync_score = 0
        self.time_score = 0
        self.total_score = 0
        self.scores = []

    def analyze(self, data):

        #  Load the data from the CSV file
        data = create_unique_id(data)

        if data is None:
            print("Failed to load data.")
            return 0

        # Reset per-run scores so repeated analyze() calls stay aligned to rows.
        self.scores = []

        # Iterate through each row in the DataFrame
        for row in data.itertuples(index=False):

            # Extract the full_code from the row
            test_method = row.full_code

            # Each time we will reset the scores to 0.
            #  Before it made no sense to do with the
            # @Test at the end which led to many bugs,
            # especially the final flaky_score
            self.total_score = 0
            self.thread_sleep_score = 0
            self.new_thread_score = 0
            self.time_score = 0
            self.await_score = 0
            self.executor_score = 0
            self.atomic_score = 0
            self.date_score = 0
            self.api_score = 0
            self.random_score = 0
            self.shared_score = 0
            self.sync_score = 0

            if isinstance(test_method, str) and "Thread.sleep(" in test_method:
                self.thread_sleep_score = 0.35

            if isinstance(test_method, str) and "new Thread(" in test_method:
                self.new_thread_score = 0.25

            if isinstance(test_method, str) and "Time" in test_method:
                self.time_score = 0.10

            if isinstance(test_method, str) and "await" in test_method:
                self.await_score = 0.15

            if isinstance(test_method, str) and "Executor" in test_method:
                self.executor_score = 0.25

            if isinstance(test_method, str) and "Atomic" in test_method:
                self.atomic_score = 0.1

            if isinstance(test_method, str) and "Date" in test_method:
                self.date_score = 0.1

            if isinstance(test_method, str) and ("Time" or "time" in test_method):
                self.time_score = 0.1

            if isinstance(test_method, str) and (
                "Api" in test_method or "api" in test_method or "API" in test_method
            ):
                self.api_score = 0.05

            if isinstance(test_method, str) and "Random" in test_method:
                self.random_score = 0.3

            if isinstance(test_method, str) and ("Shared" or "share" in test_method):
                self.shared_score = 0.15

            if isinstance(test_method, str) and "sync" in test_method:
                self.sync_score = 0.25

            self.total_score += (
                self.thread_sleep_score
                + self.new_thread_score
                + self.time_score
                + self.await_score
                + self.executor_score
                + self.atomic_score
                + self.date_score
                + self.api_score
                + self.random_score
                + self.shared_score
                + self.sync_score
                + self.time_score
            )

            self.scores.append(self.total_score)

        data["flaky_score"] = self.scores
        return data
