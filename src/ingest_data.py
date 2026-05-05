"""[TODO:description]

[TODO:description]
"""

import polars as pl


def ingest_data() -> list[pl.DataFrame]:
    """[TODO:description]

    Returns:
        [TODO:return]
    """
    df = pl.read_csv("data/cleaned_data.csv")
    datasets = split_data(df)
    return datasets


def split_data(df: pl.DataFrame) -> list[pl.DataFrame]:
    """[TODO:description]

    Args:
        df: [TODO:description]

    Returns:
        [TODO:return]
    """
    training_ratio = 0.7
    validation_ratio = 0.15

    df = df.sample(fraction=1.0, shuffle=True, seed=4)

    num_rows = len(df)
    training_end = int(num_rows * training_ratio)
    validation_end = training_end + int(num_rows * validation_ratio)

    training_df = df[:training_end]
    validation_df = df[training_end:validation_end]
    testing_df = df[validation_end:]

    return [training_df, validation_df, testing_df]
