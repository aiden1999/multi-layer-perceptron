"""[TODO:description]

[TODO:description]
"""

import numpy as np
import polars as pl


def get_datasets() -> list[np.ndarray]:
    """[TODO:description]

    Returns:
        [TODO:return]
    """
    df = pl.read_csv("data/cleaned_data.csv")
    datasets = split_data(df)
    return datasets


def split_data(df: pl.DataFrame) -> list[np.ndarray]:
    """[TODO:description]

    Args:
        df: [TODO:description]

    Returns:
        [TODO:return]
    """
    training_ratio = 0.7
    validation_ratio = 0.15
    # as implied, 15% of the data is used for testing

    df = df.sample(fraction=1.0, shuffle=True, seed=4)

    num_rows = len(df)
    training_end = int(num_rows * training_ratio)
    validation_end = training_end + int(num_rows * validation_ratio)

    training_df = df[:training_end].to_numpy()
    validation_df = df[training_end:validation_end].to_numpy()
    testing_df = df[validation_end:].to_numpy()

    return [training_df, validation_df, testing_df]
