import polars as pl
import polars.testing as pl_testing
import pytest

from src.ingest_data import standardise_data


def test_standardise_data_works():
    test_data = {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}
    test_df = pl.DataFrame(test_data)
    expected_data = {"a": [0.0, 0.25, 0.5, 0.75, 1.0], "b": [0.0, 0.25, 0.5, 0.75, 1.0]}
    expected_df = pl.DataFrame(expected_data)
    actual_df = standardise_data(test_df)
    pl_testing.assert_frame_equal(expected_df, actual_df)


def test_standardise_data_throws_exception():
    test_data = {"a": ["a", "b", "c"], "b": ["d", "e", "f"]}
    test_df = pl.DataFrame(test_data)
    with pytest.raises(Exception):
        standardise_data(test_df)


def test_split_data_training_df_right_length():
    pass


def test_split_data_validation_df_right_length():
    pass


def test_split_data_testing_df_right_length():
    pass
