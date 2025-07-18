# 🧠 Advanced Credit Card Fraud Detection System (XGBoost + AWS Deployment)

## 🔍 Overview
This is a full-scale, modular machine learning system for detecting credit card fraud. It uses PySpark preprocessing, advanced sampling strategies to combat class imbalance, XGBoost for classification, MLflow with Optuna for model tracking and tuning, and a complete AWS deployment pipeline using SageMaker, S3, and Glue.

---

## 📁 Directory Structure

```
.
├── aws_pipeline/                   # Deployment scripts and AWS infrastructure
│   ├── automated_deployment_script.sh       # Main automation script to deploy S3, Glue, SageMaker
│   ├── glue-trust-policy.json               # IAM trust policy for Glue
│   ├── optimized_fraud_detection_glue_job.py # Glue job for ETL tasks
│   ├── sagemaker-trust-policy.json          # IAM trust policy for SageMaker
│   ├── separated_inference.py               # Code for SageMaker inference endpoint
│   └── separated_sagemaker_notebook.py      # Launches SageMaker Notebook with preloaded environment

├── inference_module/              # Scripts to simulate model prediction performance
│   ├── main.py                                # Entry point for simulation and benchmarking
│   ├── optimized_simulate_WAN_test_inference.py # Optimized test inference under WAN conditions
│   └── simulate_WAN_test_inference.py         # Basic WAN test script

├── model_training/                # Model development code
│   └── PySpark_Preprocessing.py              # PySpark script for loading, transforming, and saving data

├── experimentation_logs/          # Optuna tuning + MLflow experiment tracking
│   ├── Core/
│   │   └── fraud_detection_mlflow_optuna_pipeline_ext.py   # Main MLflow + Optuna script
│   ├── Data_Loading_Caching/
│   │   ├── data_cache.py                      # Caches transformed data to speed up training
│   │   └── data_loader.py                     # Loads and splits dataset for tuning
│   ├── Evaluation/
│   │   ├── eval_classification_metrics.py     # Calculates precision, recall, G-mean, F1
│   │   └── model_report_summary.py            # Summarizes model reports
│   ├── Hyperparameter_Search/
│   │   ├── objective_func_pipeline.py         # Objective function for Optuna
│   │   └── search_config.py                   # Search space configuration
│   └── Utility/
│       ├── helper_func.py                     # Helper functions for feature engineering
│       └── settings.py                        # MLflow and pipeline settings

|
├── requirements.txt

└── README.md
```

---

## ⚙️ System Workflow

### 1. Data Preprocessing (`model_training/`)
- The script `PySpark_Preprocessing.py`:
  - Loads Kaggle/ULB fraud dataset
  - Applies log normalization on `Amount`
  - Standardizes `V1–V28` features
  - Splits data (80% train, 10% val, 10% test)
  - Outputs cleaned CSVs to be used in training

---

### 2. ML Model Training + Resampling (`experimentation_logs/`)
- `fraud_detection_mlflow_optuna_pipeline_ext.py`:
  - Triggers the full experiment pipeline
  - Logs metrics and models in MLflow
  - Uses resampling like SMOTE, ADASYN, NearMiss

- Core modules:
  - `data_loader.py`: Reads CSVs, applies caching
  - `objective_func_pipeline.py`: Defines loss metric for tuning
  - `eval_classification_metrics.py`: Evaluates recall, precision, F1, G-mean

---

### 3. Inference Simulation (`inference_module/`)
- `main.py`: Entry point for running inference benchmarks
- `simulate_WAN_test_inference.py`: Simulates inference performance in WAN settings
- `optimized_simulate_WAN_test_inference.py`: Optimized to reduce latency in cloud API calls

---

### 4. AWS Deployment (`aws_pipeline/`)
- `automated_deployment_script.sh`:
  - Provisions S3 bucket, SageMaker instance, Glue jobs
  - Uploads preprocessed data and scripts

- `optimized_fraud_detection_glue_job.py`: Glue script for production data flow

- `separated_inference.py`: Loads SageMaker model and serves live predictions

- `separated_sagemaker_notebook.py`: Boots a new SageMaker instance for retraining

---

## 🚀 Execution Steps

Follow these steps to run the complete fraud detection pipeline:

---

### 🧹 Step 1: Preprocess the Dataset
Transforms the raw Kaggle dataset using PySpark.
```bash
python model_training/PySpark_Preprocessing.py
```

---

### 🧠 Step 2: Train the Model with MLflow + Optuna
Trains an XGBoost model with automated hyperparameter tuning and experiment tracking.
```bash
python experimentation_logs/Core/fraud_detection_mlflow_optuna_pipeline_ext.py
```

---

### ☁️ Step 3: Deploy Pipeline on AWS
Uploads datasets to S3 and spins up SageMaker & Glue services.
```bash
bash aws_pipeline/automated_deployment_script.sh
```

---

### 🔍 Step 4: Test Inference under Load
Runs WAN simulation to test endpoint latency.
```bash
python inference_module/main.py
```

---

---


---

## 📦 Required Dependencies

Install all project dependencies using:
```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
pyspark==3.4.1
xgboost==1.7.6
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.25.2
matplotlib==3.7.2
seaborn==0.12.2
mlflow==2.6.0
optuna==3.2.0
boto3==1.28.2
sagemaker==2.174.0
awscli==1.27.160
joblib==1.3.2
imbalanced-learn==0.11.0
```

---
## ✅ Highlights
- 💡 Handles extreme class imbalance with 9 resampling techniques
- 🎯 Uses Optuna + MLflow for automated tuning and tracking
- 🧪 Tested for WAN inference latency using real-time endpoint
- ☁️ Fully automates deployment using AWS CLI, Glue, SageMaker
- 🧰 Modular folder structure for maintainability and clarity

---
