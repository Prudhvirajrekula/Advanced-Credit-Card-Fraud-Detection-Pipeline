# Tests/test_data_loader_mocked.py

import pytest
from unittest.mock import patch, MagicMock
from Data_Loading_Caching.data_loader import DatasetLoader

@pytest.fixture
def fake_paths(tmp_path):
    # Create mock files with expected pattern
    filenames = [
        "preprocessing_creditcard_train_features_80.csv",
        "preprocessing_creditcard_train_labels_80.csv",
        "preprocessing_creditcard_validation_features_80.csv",
        "preprocessing_creditcard_validation_labels_80.csv",
        "preprocessing_creditcard_test_features_80.csv",
        "preprocessing_creditcard_test_labels_80.csv",
    ]
    for name in filenames:
        (tmp_path / name).touch()
    return tmp_path


def test_build_data_paths_parses_correct_keys(fake_paths):
    loader = DatasetLoader(str(fake_paths))
    expected_keys = {
        "train_features", "train_labels",
        "validation_features", "validation_labels",
        "test_features", "test_labels"
    }

    assert set(loader.data_paths.keys()) == expected_keys


@patch("Data_Loading_Caching.data_loader.pl.read_csv")
@patch("Data_Loading_Caching.data_loader.mpd.read_csv")
@patch("Data_Loading_Caching.data_loader.pd.read_csv")
def test_load_all_uses_correct_loader(mock_pd, mock_modin, mock_polars, fake_paths):
    mock_pd.return_value = "pd_df"
    mock_modin.return_value = "modin_df"
    mock_polars.return_value = "polars_df"

    loader = DatasetLoader(str(fake_paths))

    # pandas
    result_pd = loader.load_all("pandas")
    assert all(v == "pd_df" for v in result_pd.values())

    # polars
    result_polars = loader.load_all("polars")
    assert all(v == "polars_df" for v in result_polars.values())

    # unknown
    with pytest.raises(ValueError):
        loader.load_all("unknown_backend")


@patch("Data_Loading_Caching.data_loader.pl.read_csv")
def test_benchmark_returns_time_and_df(mock_polars, fake_paths):
    mock_polars.return_value = "fake_df"

    loader = DatasetLoader(str(fake_paths))
    loader.loaders = {"polars": mock_polars}

    result = loader.benchmark("train_features")

    assert "polars" in result
    assert result["polars"]["df"] == "fake_df"
    assert isinstance(result["polars"]["load_time_seconds"], float)

