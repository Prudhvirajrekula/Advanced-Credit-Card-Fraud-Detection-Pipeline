# %% [markdown]
# # Fraud Detection Model Training and Deployment
#
# This notebook:
# 1. Downloads preprocessed training, validation, and test data from S3.
# 2. Trains an XGBoost model using cross‑validation and Random Oversampling.
# 3. Evaluates the model on the test set.
# 4. Saves predictions and the trained model artifact to S3.
# 5. Deploys the model to a SageMaker endpoint using a separate inference script.

# %% [code]
import os
import boto3
import sagemaker
import numpy as np
import pandas as pd
import joblib
import json
import time

from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, classification_report, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb

# Import sampler for random oversampling
from imblearn.over_sampling import RandomOverSampler

# %% [code]
# SageMaker session and role configuration
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# S3 configuration (should match the values used in the bash script)
AWS_REGION = "ap-southeast-2"
BUCKET_NAME = "fraud-detection-012725"
s3_prefix = "fraud-detection"  # Base folder for the project

# Define S3 folders for data and outputs
TRAIN_FOLDER = "preprocessed/train"
VALIDATION_FOLDER = "preprocessed/validation"
TEST_FOLDER = "preprocessed/test"
PREDICTIONS_FOLDER = "predictions"
MODEL_ARTIFACTS_FOLDER = "model_artifacts"

s3 = boto3.client("s3", region_name=AWS_REGION)

# %% [code]
def download_csv_from_s3(bucket, key, local_path):
    s3.download_file(bucket, key, local_path)
    return pd.read_csv(local_path)

print("Downloading training data...")
train_key = f"{s3_prefix}/{TRAIN_FOLDER}/train.csv"
df_train = download_csv_from_s3(BUCKET_NAME, train_key, "train.csv")

print("Downloading validation data...")
val_key = f"{s3_prefix}/{VALIDATION_FOLDER}/validation.csv"
df_val = download_csv_from_s3(BUCKET_NAME, val_key, "validation.csv")

print("Downloading test data...")
test_key = f"{s3_prefix}/{TEST_FOLDER}/test.csv"
df_test = download_csv_from_s3(BUCKET_NAME, test_key, "test.csv")

# Assume the CSV files have features and a target column named 'target'
features = [col for col in df_train.columns if col != 'target']

X_train_full = df_train[features].values
y_train_full = df_train['target'].values

X_val = df_val[features].values
y_val = df_val['target'].values

X_test = df_test[features].values
y_test = df_test['target'].values

# %% [code]
# Define the FraudDetectionPipeline class for training and evaluation
class FraudDetectionPipeline:
    def __init__(self, sampling_method, random_state=42, early_stopping_rounds=10, n_splits=5):
        self.sampling_method = sampling_method
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.n_splits = n_splits

        self.best_model = None
        self.best_score = -1
        self.best_params = None

    def _get_sampler(self):
        """Return the sampler based on the sampling_method."""
        samplers = {
            'random_oversample': RandomOverSampler(random_state=self.random_state)
        }
        if self.sampling_method not in samplers:
            raise ValueError(f"Unsupported sampling method: {self.sampling_method}")
        return samplers[self.sampling_method]

    def train_with_cv(self, X, y, param_grid):
        """
        Performs Stratified K-Fold cross-validation over the given param_grid,
        selects the best hyperparameters, and retrains a final model.
        """
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        sampler = self._get_sampler()

        for params in param_grid:
            print(f"\nTesting parameters: {params}")
            pr_auc_scores = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val_fold, y_val_fold = X[val_idx], y[val_idx]

                # Resample using the sampler
                X_res, y_res = sampler.fit_resample(X_train, y_train)

                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric="aucpr",
                    early_stopping_rounds=self.early_stopping_rounds,
                    random_state=self.random_state,
                    **params
                )

                # Train with early stopping on the validation fold
                model.fit(
                    X_res, y_res,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False
                )

                print(f"Fold {fold} - Best iteration: {model.best_iteration}, Best score: {model.best_score}")
                y_proba = model.predict_proba(X_val_fold)[:, 1]
                fold_score = average_precision_score(y_val_fold, y_proba)
                pr_auc_scores.append(fold_score)
                print(f"Fold {fold} PR-AUC: {fold_score:.4f}")

            mean_score = np.mean(pr_auc_scores)
            print(f"Mean PR-AUC for parameters {params}: {mean_score:.4f}")

            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = params
                self._train_final_model(X, y)

        print(f"\nBest parameters: {self.best_params}\nBest mean PR-AUC: {self.best_score:.4f}")
        return self.best_params, self.best_score

    def _train_final_model(self, X, y):
        """
        Splits the data into a final hold-out set and retrains the model using the best parameters.
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        X_res, y_res = self._get_sampler().fit_resample(X_train, y_train)

        self.best_model = xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric="aucpr",
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            **self.best_params
        )
        self.best_model.fit(
            X_res, y_res,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    def evaluate(self, X_test, y_test):
        """Evaluates the model on test data and returns various metrics."""
        if self.best_model is None:
            raise ValueError("Train the model first using train_with_cv().")
        y_pred = self.best_model.predict(X_test)
        y_proba = self.best_model.predict_proba(X_test)[:, 1]

        metrics = {
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'pr_auc': average_precision_score(y_test, y_proba),
            'g_mean': self._g_mean(y_test, y_pred)
        }

        print(f"\nEvaluation Results using sampling method '{self.sampling_method}':")
        print(classification_report(y_test, y_pred))
        return metrics

    @staticmethod
    def _g_mean(y_true, y_pred):
        """Computes the geometric mean of sensitivity and specificity."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return np.sqrt(recall * specificity)

# %% [code]
# Set up hyperparameter grid
param_grid = [
    {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100},
    {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 150},
    # Add additional parameter combinations as needed.
]

# Combine training and validation data for cross-validation
X_full = np.concatenate([X_train_full, X_val])
y_full = np.concatenate([y_train_full, y_val])

# Initialize and train the pipeline
pipeline = FraudDetectionPipeline(sampling_method='random_oversample', random_state=42)
print("Starting cross-validation training...")
best_params, best_score = pipeline.train_with_cv(X_full, y_full, param_grid)

print("\nEvaluating the final model on test data...")
metrics = pipeline.evaluate(X_test, y_test)
print("Test Metrics:", metrics)

# Generate predictions on the test set
y_test_pred = pipeline.best_model.predict(X_test)
test_results = pd.DataFrame({
    'actual': y_test,
    'predicted': y_test_pred,
    'proba': pipeline.best_model.predict_proba(X_test)[:, 1]
})

# %% [code]
# Save predictions locally and upload to S3
predictions_file = "test_predictions.csv"
test_results.to_csv(predictions_file, index=False)
print(f"Predictions saved locally to {predictions_file}")

predictions_s3_key = f"{s3_prefix}/{PREDICTIONS_FOLDER}/{predictions_file}"
s3.upload_file(predictions_file, BUCKET_NAME, predictions_s3_key)
print(f"Predictions uploaded to s3://{BUCKET_NAME}/{predictions_s3_key}")

# %% [code]
# Save the model artifact locally and upload to S3
model_file = "model.joblib"
joblib.dump(pipeline.best_model, model_file)
print(f"Model artifact saved locally to {model_file}")

model_s3_key = f"{s3_prefix}/{MODEL_ARTIFACTS_FOLDER}/{model_file}"
s3.upload_file(model_file, BUCKET_NAME, model_s3_key)
print(f"Model artifact uploaded to s3://{BUCKET_NAME}/{model_s3_key}")

# %% [code]
# Deploy the model to a SageMaker endpoint using the separate inference script.
# The inference script (inference.py) should be in S3 under the scripts folder.

from sagemaker.sklearn.model import SKLearnModel

# S3 URI for the model artifact and the inference script
model_data = f"s3://{BUCKET_NAME}/{model_s3_key}"
inference_script_s3 = f"s3://{BUCKET_NAME}/{s3_prefix}/scripts/inference.py"

# Download the inference script locally so that the SKLearnModel container can use it.
inference_script_local = "separated_inference.py"
s3.download_file(BUCKET_NAME, f"{s3_prefix}/scripts/separated_inference.py", inference_script_local)

sklearn_model = SKLearnModel(
    model_data=model_data,
    role=role,
    entry_point=inference_script_local,
    framework_version="0.23-1",  # Adjust as needed
    py_version="py3",
    sagemaker_session=sagemaker_session
)

endpoint_name = "fraud-detection-endpoint"
predictor = sklearn_model.deploy(initial_instance_count=1, instance_type="ml.m5.large", endpoint_name=endpoint_name)
print(f"Endpoint '{endpoint_name}' is being created. It may take a few minutes...")

# %% [code]
# Test the deployed endpoint
import json

# Create sample input: take first 5 records from X_test
sample_input = X_test[:5]
sample_records = [dict(zip(features, row)) for row in sample_input]
payload = {"instances": sample_records}

runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(payload)
)
result = json.loads(response['Body'].read().decode())
print("Endpoint response:")
print(json.dumps(result, indent=2))

