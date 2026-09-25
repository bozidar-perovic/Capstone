from .BaseAnalyzer import BaseAnalyzer
from .utility import load_data


class RuleBasedAnalyzer(BaseAnalyzer):

    def __init__(self):
        super().__init__()

    def analyze(self, data):

        #  Load the data from the CSV file
        data = load_data(data)

        if data is None:
            print("Failed to load data.")
            return 0

        total_score = 0

        # Iterate through each row in the DataFrame
        for row in data.itertuples(index=False):

            thread_sleep_score = 0
            new_thread_score = 0
            time_score = 0
            await_score = 0
            executor_score = 0
            atomic_score = 0
            date_score = 0
            api_score = 0
            random_score = 0
            shared_score = 0
            sync_score = 0
            time_score = 0

            # Extract the full_code from the row
            test_method = row.full_code

            if isinstance(test_method, str) and "Thread.sleep(" in test_method:
                thread_sleep_score = 0.35

            if isinstance(test_method, str) and "new Thread(" in test_method:
                new_thread_score = 0.25

            if isinstance(test_method, str) and "Time" in test_method:
                time_score = 0.10

            if isinstance(test_method, str) and "await" in test_method:
                await_score = 0.15

            if isinstance(test_method, str) and "Executor" in test_method:
                executor_score = 0.25

            if isinstance(test_method, str) and "Atomic" in test_method:
                atomic_score = 0.1

            if isinstance(test_method, str) and "Date" in test_method:
                date_score = 0.1

            if isinstance(test_method, str) and ("Time" or "time" in test_method):
                time_score = 0.1

            if isinstance(test_method, str) and (
                "Api" in test_method or "api" in test_method or "API" in test_method
            ):
                api_score = 0.05

            if isinstance(test_method, str) and "Random" in test_method:
                random_score = 0.3

            if isinstance(test_method, str) and ("Shared" or "share" in test_method):
                shared_score = 0.15

            if isinstance(test_method, str) and "sync" in test_method:
                sync_score = 0.25

            total_score += (
                thread_sleep_score
                + new_thread_score
                + time_score
                + await_score
                + executor_score
                + atomic_score
                + date_score
                + api_score
                + random_score
                + shared_score
                + sync_score
                + time_score
            )
            print(total_score)

            if isinstance(test_method, str) and "@Test" in test_method:
                total_score = 0
                thread_sleep_score = 0
                new_thread_score = 0
                time_score = 0
                await_score = 0
                executor_score = 0
                atomic_score = 0
                date_score = 0
                api_score = 0
                random_score = 0
                shared_score = 0
                sync_score = 0
