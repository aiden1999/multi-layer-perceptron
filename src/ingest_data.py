"""Ingesting and transforming raw data for use with the perceptron."""

from numpy.typing import NDArray
import polars as pl


def get_datasets() -> list[NDArray]:
    """Return transformed datasets.

    Returns:
        List of transformed datasets.
    """
    df = pl.read_csv("data/cleaned_data.csv")
    df = standardise_data(df)
    datasets = split_data(df)
    return datasets


def standardise_data(df: pl.DataFrame) -> pl.DataFrame:
    """Standardise data to be between 0 and 1.

    Args:
        df: Raw data.

    Returns:
        Standardised DataFrame.
    """
    df = df.select((pl.all() - pl.all().min()) / (pl.all().max() - pl.all().min()))
    return df


def split_data(df: pl.DataFrame) -> list[NDArray]:
    """Splits data into different datasets for training, validation, and testing.

    Args:
        df: DataFrame containing all the data.

    Returns:
        List of DataFrames, one for each purpose.
    """
    training_ratio = 0.7
    validation_ratio = 0.15
    # as implied, 15% of the data is used for testing

    df = df.sample(fraction=1.0, shuffle=True)

    num_rows = len(df)
    training_end = int(num_rows * training_ratio)
    validation_end = training_end + int(num_rows * validation_ratio)

    training_df = df[:training_end].to_numpy()
    validation_df = df[training_end:validation_end].to_numpy()
    testing_df = df[validation_end:].to_numpy()

    return [training_df, validation_df, testing_df]
