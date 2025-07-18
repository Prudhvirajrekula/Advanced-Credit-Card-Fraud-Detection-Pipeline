# Tests/test_data_cache_mocked.py

import pytest
from unittest.mock import patch, MagicMock
from Data_Loading_Caching.data_cache import load_data_once, get_data, reload_data
from Data_Loading_Caching import data_cache

@pytest.fixture(autouse=True)
def clear_cache_before_each_test():
    # Reset cache before each test
    data_cache._DATA_CACHE = None


@patch("Data_Loading_Caching.data_cache.DatasetLoader")
def test_mocked_data_loading(loader_mock):
    fake_data = {
        "train_features": "fake_train_X",
        "train_labels": "fake_train_y",
        "validation_features": "fake_val_X",
        "validation_labels": "fake_val_y",
        "test_features": "fake_test_X",
        "test_labels": "fake_test_y",
    }

    # Setup the mock
    instance = loader_mock.return_value
    instance.benchmark.return_value = None  # no-op
    instance.load_all.return_value = fake_data

    # Call the function under test
    data_cache.load_data_once()
    data = data_cache.get_data()

    # Check expected keys
    assert data["X_train"] == "fake_train_X"
    assert data["y_val"] == "fake_val_y"
    assert data["X_test"] == "fake_test_X"

    # Verify caching works: second call doesn’t re-trigger load
    loader_mock.reset_mock()
    data_cache.load_data_once()
    loader_mock.assert_not_called()


@patch("Data_Loading_Caching.data_cache.DatasetLoader")
def test_mocked_reload_data(loader_mock):
    instance = loader_mock.return_value
    instance.load_all.return_value = {
        "train_features": 1, "train_labels": 2,
        "validation_features": 3, "validation_labels": 4,
        "test_features": 5, "test_labels": 6,
    }

    # First load
    data_cache.load_data_once()
    old_data = data_cache.get_data()

    # Change what the mock returns
    instance.load_all.return_value = {
        "train_features": 101, "train_labels": 202,
        "validation_features": 303, "validation_labels": 404,
        "test_features": 505, "test_labels": 606,
    }

    # Reload
    data_cache.reload_data()
    new_data = data_cache.get_data()

    assert old_data != new_data
    assert new_data["X_train"] == 101

