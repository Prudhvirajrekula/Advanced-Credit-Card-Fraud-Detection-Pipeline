# Data_Loading_Caching/data_cache.py

from .data_loader import DatasetLoader
import numpy as np

# Module-level dict to store cached data.
_DATA_CACHE = None

def load_data_once():
    """
    Load data only if it hasn't been loaded before.
    If already loaded, this function does nothing.
    """
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        # Already loaded, no-op
        return

    # Otherwise, load the data
    loader = DatasetLoader("Preprocessed_Data")
    #loader.benchmark("train_features")
    data = loader.load_all("polars")

    _DATA_CACHE = {
        "X_train": data["train_features"],
        "y_train": data["train_labels"],
        "X_val":   data["validation_features"],
        "y_val":   data["validation_labels"],
        "X_test":  data["test_features"],
        "y_test":  data["test_labels"]
    }


def get_data():
    """
    Returns the cached data dictionary. If not loaded, raises an error.
    """
    if _DATA_CACHE is None:
        raise RuntimeError("Data has not been loaded. Call load_data_once() first.")
    return _DATA_CACHE

def get_numpy_data(dtype="float64"):
    """
    Returns the cached data as NumPy arrays.
    dtype can be "float32" or "float64"
    """
    if _DATA_CACHE is None:
        raise RuntimeError("Data has not been loaded. Call load_data_once() first.")
    
    if dtype not in {"float32", "float64"}:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return {
        "X_train": _DATA_CACHE["X_train"].to_numpy().astype(dtype),
        "y_train": _DATA_CACHE["y_train"].to_numpy().astype(dtype),
        "X_val":   _DATA_CACHE["X_val"].to_numpy().astype(dtype),
        "y_val":   _DATA_CACHE["y_val"].to_numpy().astype(dtype),
        "X_test":  _DATA_CACHE["X_test"].to_numpy().astype(dtype),
        "y_test":  _DATA_CACHE["y_test"].to_numpy().astype(dtype),
    }

def reload_data():
    global _DATA_CACHE
    _DATA_CACHE = None
    load_data_once()

