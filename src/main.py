import string

import pandas as pd

# Loading the data from the CSV file
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

# Testing the data loading and iterating over the DataFrame and implementing the keyword extraction logic
def main():
    data = load_data("src/data/train_set_1.csv")
    if data is not None:
        print("Data loaded successfully:")
        print(data.head())
    else:
        print("Failed to load data.")

    df = data[["id", "full_code"]]

    # Itterates through out the entire file and checks for the keywords in the full_code column and prints out the test method if it finds any of the keywords
    for i in range(len(df)):
        test_method = df.iloc[i]["full_code"]
        keywords = [word for word in test_method.split() if word not in string.punctuation]
        for i in range(len(keywords)):
            if keywords[i] == "Thread.sleep(50);":
                print("Found a test with thread sleep of 50 milliseconds.")
                break
            if keywords[i] == "Thread.sleep(500);":
                print("Found a test with thread sleep of 500 milliseconds.")
                break
            if keywords[i] == "Thread.sleep(100);":
                print("Found a test with thread sleep of 100 milliseconds.")
                break
            if keywords[i] == "Thread.sleep(10);":
                print("Found a test with thread sleep of 10 milliseconds.")
                break



if __name__ == "__main__":
    main()