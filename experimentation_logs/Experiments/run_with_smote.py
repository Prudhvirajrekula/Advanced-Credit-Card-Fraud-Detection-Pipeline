# run_with_smote.py (example)

from Data_Loading_Caching.data_cache import load_data_once, get_data, get_numpy_data
from Core.fraud_detection_mlflow_optuna_pipeline_ext import FraudDetectionPipeline

# make sure data is loaded (cached). If it's already loaded, no disk I/O is done.
load_data_once()

# retrieve in-memory data
data = get_numpy_data(dtype="float64")
X_train = data["X_train"]
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]
X_test  = data["X_test"]
y_test  = data["y_test"]

# define or import your param search space
param_space = {
    "n_estimators":  {"method": "suggest_int",   "low": 100, "high": 200, "step": 100},
    "max_depth":     {"method": "suggest_int",   "low": 5,   "high": 7, "step": 1},
    "learning_rate": {"method": "suggest_float", "low": 0.001, "high": 0.2, "log": True},
    "subsample":     {"method": "suggest_float", "low": 0.7, "high": 0.9, "step": 0.1},
    "colsample_bytree": {"method": "suggest_float", "low": 0.8, "high": 1.0, "step": 0.1},
    "scale_pos_weight": {"method": "suggest_float", "low": 0.1, "high": 0.2, "step": 0.05},
    "min_child_weight": {"method": "suggest_int", "low": 2, "high": 3},
    "reg_alpha": {"method": "suggest_int", "low": 2, "high": 5, "step": 1},
    "gamma": {"method": "suggest_int", "low": 2, "high": 4, "step": 1}
}

pipeline = FraudDetectionPipeline(
    param_search_space=param_space,
    random_state=42,
    early_stopping_rounds=10,
    n_splits=3,
    n_trials=20
)

best_params, best_score, best_model = pipeline.train_with_optuna(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
)

