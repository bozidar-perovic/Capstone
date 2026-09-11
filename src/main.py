import pandas as pd

def load_data(file_path):
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

def main():
    data = load_data("src/data/FlakeBench_dataset.csv")
    if data is not None:
        print("Data loaded successfully:")
        print(data.head())
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()