import pandas as pd


# Loading the data from the CSV file
def load_data(file_path) -> pd.DataFrame | None:
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: There was a parsing error while reading the file.")
        return None


def rule_based_solution(file_path):

    # Load the data from the CSV file
    data = load_data(file_path)

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

        # Extract the full_code from the row
        test_method = row.full_code

        if isinstance(test_method, str) and "Thread.sleep(" in test_method:
            thread_sleep_score = 1

        if isinstance(test_method, str) and "new Thread(" in test_method:
            new_thread_score = 1

        if isinstance(test_method, str) and "Time" in test_method:
            time_score = 1

        if isinstance(test_method, str) and "await" in test_method:
            await_score = 1

        if isinstance(test_method, str) and "Executor" in test_method:
            executor_score = 1

        if isinstance(test_method, str) and "Atomic" in test_method:
            atomic_score = 1

        if isinstance(test_method, str) and "Date" in test_method:
            date_score = 1

        if isinstance(test_method, str) and (
            "Api" in test_method or "api" in test_method or "API" in test_method
        ):
            api_score = 1

        if isinstance(test_method, str) and "Random" in test_method:
            random_score = 1

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
        )

    return total_score


if __name__ == "__main__":
    x = rule_based_solution("src/data/train_set_1.csv")
    print(x)
