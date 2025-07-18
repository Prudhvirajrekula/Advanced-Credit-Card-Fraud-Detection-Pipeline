###############################################################################################
#                                     import the relevant dependencies                        #
###############################################################################################

import mlflow
import mlflow.sklearn
import optuna

import xgboost as xgb
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

class FraudDetectionPipeline:
    def __init__(
        self,
        param_search_space,
        random_state=42,
        early_stopping_rounds=10,
        n_splits=3,
        n_trials=20
    ):
        """
        :param param_search_space: a dict or function defining the Optuna search space.
        :param random_state: Seed for reproducibility.
        :param early_stopping_rounds: XGBoost early stopping rounds.
        :param n_splits: Number of folds for Stratified K-Fold CV.
        :param n_trials: Number of Optuna trials.
        """
        self.param_search_space = param_search_space
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.n_splits = n_splits
        self.n_trials = n_trials

        self.sampling_methods = {'smote': SMOTE(random_state=self.random_state)}
        self.selected_sampling = 'smote'

        self.best_model = None
        self.best_params = None
        self.best_score = -1.0

    def _get_sampler(self):
        return self.sampling_methods[self.selected_sampling]

    def _cross_val_score(self, params, X, y):
        """
        Perform Stratified K-Fold CV with SMOTE resampling in each fold.
        Returns mean train PR-AUC and mean val PR-AUC across folds.
        """
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        train_scores, val_scores = [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # SMOTE resampling on the fold
            sampler = self._get_sampler()
            X_res, y_res = sampler.fit_resample(X_tr, y_tr)

            model = xgb.XGBClassifier(
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='aucpr',
                random_state=self.random_state,
                early_stopping_rounds=self.early_stopping_rounds,
                **params
            )

            # We do not have a separate “early stopping” set per fold,
            # but we can pass the same fold’s val set to XGBoost for early stopping.
            model.fit(
                X_res, y_res,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            y_proba_train = model.predict_proba(X_res)[:, 1]
            y_proba_val = model.predict_proba(X_val)[:, 1]

            train_scores.append(average_precision_score(y_res, y_proba_train))
            val_scores.append(average_precision_score(y_val, y_proba_val))

        return np.mean(train_scores), np.mean(val_scores)

    def _optuna_objective(self, trial, X, y):
        """
        Objective function for Optuna.
        - Use self.param_search_space to define the parameter suggestions.
        - Return the CV validation PR-AUC.
        """
        # Build a params dict from the user-defined search space
        params = {}
        for param_name, config in self.param_search_space.items():
            method = config["method"]  # e.g. "suggest_int", "suggest_float"
            # e.g. trial.suggest_int('n_estimators', 100, 1000, step=50)
            if method == "suggest_int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    config["low"],
                    config["high"],
                    config.get("step", 1)
                )
            elif method == "suggest_float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    config["low"],
                    config["high"],
                    step=config.get("step", None),
                    log=config.get("log", False)
                )
            # Add more conditionals if you want other suggestion methods

        train_score, val_score = self._cross_val_score(params, X, y)
        
        sampling_name = self.selected_sampling

        # Log for each trial
        mlflow.log_metric(f"train_pr_auc_trial_{sampling_name}_{trial.number}", train_score)
        mlflow.log_metric(f"val_pr_auc_trial_{sampling_name}_{trial.number}", val_score)

        return val_score  # because direction='maximize' in the study

    def train_with_optuna(self, X_train, y_train, X_val, y_val, X_test, y_test):
        with mlflow.start_run(run_name="fraud_detection_optuna_smote"):

            # Log the sampling method
            mlflow.log_param("sampling_method", self.selected_sampling)
            mlflow.log_param("n_splits", self.n_splits)
            mlflow.log_param("n_trials", self.n_trials)
            mlflow.log_param("early_stopping_rounds", self.early_stopping_rounds)

            # 1) Create an Optuna study to maximize val_score
            study = optuna.create_study(direction='maximize')
            func = lambda trial: self._optuna_objective(trial, X_train, y_train)
            study.optimize(func, n_trials=self.n_trials, show_progress_bar=False)

            self.best_params = study.best_params
            self.best_score = study.best_value  # best val CV PR-AUC
            mlflow.log_params(self.best_params)
            mlflow.log_metric("best_val_pr_auc_cv", self.best_score)

            # 2) Retrain a final model on X_train (and optionally X_val if desired)
            #    using the best hyperparams, with early stopping on the official validation set.
            self.best_model = xgb.XGBClassifier(
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='aucpr',
                random_state=self.random_state,
                early_stopping_rounds=self.early_stopping_rounds,
                **self.best_params
            )
            
            # If you want SMOTE on the entire train set:
            sampler = self._get_sampler()
            X_res, y_res = sampler.fit_resample(X_train, y_train)

            self.best_model.fit(
                X_res, y_res,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Evaluate on train, validation, and test
            train_pred_proba = self.best_model.predict_proba(X_train)[:, 1]
            val_pred_proba = self.best_model.predict_proba(X_val)[:, 1]
            test_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

            train_pr_auc = average_precision_score(y_train, train_pred_proba)
            val_pr_auc = average_precision_score(y_val, val_pred_proba)
            test_pr_auc = average_precision_score(y_test, test_pred_proba)

            mlflow.log_metric("final_train_pr_auc", train_pr_auc)
            mlflow.log_metric("final_val_pr_auc", val_pr_auc)
            mlflow.log_metric("final_test_pr_auc", test_pr_auc)

            # Log other metrics as needed
            mlflow.log_metric("final_train_f1", f1_score(y_train, train_pred_proba >= 0.5))
            mlflow.log_metric("final_val_f1", f1_score(y_val, val_pred_proba >= 0.5))
            mlflow.log_metric("final_test_f1", f1_score(y_test, test_pred_proba >= 0.5))

            # Log model artifact
            mlflow.sklearn.log_model(self.best_model, artifact_path="model")

        return self.best_params, self.best_score, self.best_model

