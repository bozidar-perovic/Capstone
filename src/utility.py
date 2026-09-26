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


# Function to extract specific columns from the data for later comparison of flakiness and categories
def get_comparison_data(data) -> pd.DataFrame | None:

    data = load_data(data)

    if data is None:
        print("Failed to load data.")
        return None

    data_information = data[["id", "project", "category"]]

    return data_information
