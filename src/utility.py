# Loading the data from the CSV file
import pandas as pd


# Utility functions for loading and processing data
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


def create_unique_id(data) -> pd.DataFrame | None:

    data = load_data(data)

    if data is None:
        print("No data provided to create unique IDs.")
        return None

    data["unique_identifier"] = data.apply(
        lambda row: f"{row['id']}_{row['project']}_{row['test_name']}", axis=1
    )

    return data


# Function to extract specific columns from the data for later
# comparison of flakiness and categories
def get_comparison_data(data) -> pd.DataFrame | None:

    data = create_unique_id(data)

    if data is None:
        print("Failed to load data.")
        return None

    data["is_flaky"] = data.apply(
        lambda row: "yes" if row["category"] != 5 else "no", axis=1
    )
    data_information = data[["unique_identifier", "category", "label", "is_flaky"]]

    return data_information
