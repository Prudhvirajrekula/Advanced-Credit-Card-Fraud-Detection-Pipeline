from timeit import default_timer as timer
from pathlib import Path
import pandas as pd
import polars as pl
import modin.pandas as mpd


class DatasetLoader:
    def __init__(self, data_dir: str, prefix="preprocessing_creditcard", suffix="80.csv"):
        self.data_dir = Path(data_dir)
        self.prefix = prefix
        self.suffix = suffix
        self.data_paths = self._build_data_paths()

        self.loaders = {
            "pandas": pd.read_csv,
            "modin": mpd.read_csv,
            "polars": pl.read_csv,
        }

    def _build_data_paths(self):
        data_paths = {}
        for file_path in self.data_dir.glob(f"{self.prefix}_*_{self.suffix}"):
            parts = file_path.stem.split("_")
            try:
                split = parts[-3]    # train/test/validation
                dtype = parts[-2]    # features/labels
                key = f"{split}_{dtype}"
                data_paths[key] = file_path
            except IndexError:
                print(f"Skipping malformed file: {file_path.name}")
        return data_paths

    def benchmark(self, key: str):
        if key not in self.data_paths:
            raise ValueError(f"Key '{key}' not found in data_paths.")
        
        results = {}
        path = self.data_paths[key]

        for name, loader_func in self.loaders.items():
            if loader_func is None:
                print(f"Skipping {name}: required package not available.")
                continue

            start = timer()
            df = loader_func(path)
            elapsed = round(timer() - start, 4)

            results[name] = {
                "df": df,
                "load_time_seconds": elapsed
            }

            print(f"{name.capitalize()} load time: {elapsed} seconds")
        
        return results

    def load_all(self, loader_name: str):
        if loader_name not in self.loaders:
            raise ValueError(f"Loader '{loader_name}' is not defined.")

        loader_func = self.loaders[loader_name]
        if loader_func is None:
            raise ImportError(f"Loader '{loader_name}' is not available.")

        loaded_data = {}
        for key, path in self.data_paths.items():
            loaded_data[key] = loader_func(path)

        return loaded_data

