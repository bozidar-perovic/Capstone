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


def rule_based_solution(file_path) -> int:

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

        # Extract the full_code from the row
        test_method = row.full_code

        # Convert to string and handle None values
        if isinstance(test_method, str) and "Thread.sleep(" in test_method:
            thread_sleep_score = 1
            print("Found Thread.sleep in:", row.full_code)

        # Check for new thread creation
        if isinstance(test_method, str) and "new Thread(" in test_method:
            new_thread_score = 1
            print("Found new thread in:", row.full_code)

        total_score += thread_sleep_score + new_thread_score

    return total_score


if __name__ == "__main__":
    x = rule_based_solution("src/data/train_set_1.csv")
    print(x)
