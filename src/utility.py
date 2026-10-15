"""Data loading, preprocessing, and serialization utility functions.

Provides helper routines for:
- Robust CSV dataset ingestion with exception handling.
- Generating composite unique test identifiers (`id_project_testName`).
- Preparing comparative evaluation ground truth frames.
- Persisting scored test prediction DataFrames to disk.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def load_data(
    file_path: Union[str, Path, pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """Load a dataset from a CSV file into a pandas DataFrame.

    If a preloaded DataFrame is passed, it is returned directly.

    Args:
        file_path: Filesystem path to the CSV file (str or Path), or an
            existing pandas DataFrame.

    Returns:
        A loaded pandas DataFrame, or None if the file was not found, empty,
        or malformed.
    """
    if isinstance(file_path, pd.DataFrame):
        return file_path

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


def create_unique_id(
    data: Union[str, Path, pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """Generate a globally unique identifier for every test method.

    FlakeBench reuses numeric IDs across different projects (e.g., ID 282
    exists in both Hadoop and Cassandra). This function creates a composite
    identifier formatted as '{id}_{project}_{test_name}'.

    Args:
        data: Input DataFrame or filesystem path to a CSV file.

    Returns:
        DataFrame with an added 'unique_identifier' column, or None on failure.
    """
    df = load_data(data)

    if df is None:
        print("No data provided to create unique IDs.")
        return None

    df = df.copy()
    df["unique_identifier"] = df.apply(
        lambda row: f"{row['id']}_{row['project']}_{row['test_name']}", axis=1
    )

    return df


def get_comparison_data(
    data: Union[str, Path, pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """Extract comparative ground truth columns from a dataset.

    Maps numeric category codes (0-4: flaky, 5: non-flaky) to a binary
    'is_flaky' indicator ('yes' / 'no') and extracts identification fields.

    Args:
        data: Input DataFrame or filesystem path to a CSV file.

    Returns:
        A filtered DataFrame containing 'unique_identifier', 'category',
        'label', and 'is_flaky', or None if loading fails.
    """
    df = create_unique_id(data)

    if df is None:
        print("Failed to load data.")
        return None

    df["is_flaky"] = df.apply(
        lambda row: "yes" if row["category"] != 5 else "no", axis=1
    )
    data_information = df[
        ["unique_identifier", "category", "label", "is_flaky"]
    ]

    return data_information


def save_results_to_csv(
    results: pd.DataFrame, output_file: Union[str, Path]
) -> None:
    """Save an analyzed DataFrame containing risk scores to a CSV file.

    Ensures parent directories exist before writing.

    Args:
        results: pandas DataFrame containing test records and predicted scores.
        output_file: Destination filesystem path for the output CSV.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)
