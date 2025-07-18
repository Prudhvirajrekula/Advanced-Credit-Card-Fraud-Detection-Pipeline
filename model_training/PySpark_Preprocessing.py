#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p
from pyspark.sql import DataFrame

# ML imports
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array

# --------------------------------------------------------------------------------
# 1. Initialize Spark
# --------------------------------------------------------------------------------

spark = SparkSession.builder \
    .appName("OptimizedStratifiedSplittingWithPreprocessing") \
    .getOrCreate()

# --------------------------------------------------------------------------------
# 2. Load and Shuffle the Dataset
# --------------------------------------------------------------------------------

DATA_PATH = "creditcard.csv"

df = spark.read.csv(DATA_PATH, header=True, inferSchema=True).sample(fraction=1.0, seed=42)

# --------------------------------------------------------------------------------
# 3. Split Data into Train, Validation, and Test
# --------------------------------------------------------------------------------

train_fraction = 0.8
val_fraction   = 0.1
test_fraction  = 0.1

splits = df.randomSplit([train_fraction, val_fraction, test_fraction], seed=42)
train_set, val_set, test_set = splits[0], splits[1], splits[2]

# --------------------------------------------------------------------------------
# 4. Utility Functions
# --------------------------------------------------------------------------------

def clean_data(df: DataFrame) -> DataFrame:
    """Drop duplicates and rows with nulls."""
    return df.dropDuplicates().dropna()

def log_normalize(df: DataFrame, column: str) -> DataFrame:
    """Apply log1p(x) to a given column (e.g., 'Amount')."""
    return df.withColumn(column, log1p(col(column)))

def build_preprocessing_pipeline(feature_cols):
    """
    Build a Spark ML Pipeline consisting of:
     1. VectorAssembler  -> to combine features into a single 'features' column
     2. StandardScaler   -> to scale/standardize the 'features'
    """
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
    pipeline = Pipeline(stages=[assembler, scaler])
    return pipeline

def explode_scaled_features(df: DataFrame, feature_cols) -> DataFrame:
    """
    Convert 'scaled_features' (Vector) into a Spark array, then
    create columns named exactly the same as feature_cols (e.g. "V1", "V2", ...).

    IMPORTANT: We first drop the old columns "V1..V28" so that we don't
    get a 'COLUMN_ALREADY_EXISTS' error when we create new columns
    with those same names.
    """
    # 1. Drop the original unscaled feature columns.
    for c in feature_cols:
        df = df.drop(c)

    # 2. Convert ML Vector to Spark Array
    df = df.withColumn("scaled_array", vector_to_array(col("scaled_features")))

    # 3. Build the list of columns to SELECT
    select_exprs = [
        col(c) for c in df.columns
        if c not in {"scaled_features", "scaled_array", "features"}
    ]
    # 4. Add each scaled feature as a new column with the old name
    select_exprs += [
        col("scaled_array")[i].alias(feature_cols[i])
        for i in range(len(feature_cols))
    ]

    # 5. Final select
    df = df.select(*select_exprs)
    return df

# --------------------------------------------------------------------------------
# 5. Main Preprocessing / Fitting Flow
# --------------------------------------------------------------------------------

def preprocess_dataset(df: DataFrame, pipeline_model, feature_cols, name: str) -> DataFrame:
    """
    Preprocess a dataset:
      - Clean (drop duplicates, na)
      - Log-normalize 'Amount'
      - Transform via fitted pipeline
      - Explode scaled features
    """
    print(f"Preprocessing {name} dataset...")
    cleaned = clean_data(df)
    normalized = log_normalize(cleaned, "Amount")
    transformed = pipeline_model.transform(normalized)
    final_df = explode_scaled_features(transformed, feature_cols)
    return final_df

def preprocess_multiple_datasets(datasets: dict, pipeline_model, feature_cols) -> dict:
    """
    Apply 'preprocess_dataset' to each dataset in the dictionary.
    """
    preprocessed_datasets = {}
    for name, df in datasets.items():
        preprocessed_datasets[name] = preprocess_dataset(df, pipeline_model, feature_cols, name)
    return preprocessed_datasets

def fit_and_transform_data(train_df: DataFrame, val_df: DataFrame, test_df: DataFrame):
    """
    1. Clean & log-normalize the training set, then fit the pipeline.
    2. Transform train, validation, and test sets.
    3. Return the preprocessed DataFrames.
    """
    pca_features = [f"V{i}" for i in range(1, 29)]

    # -- Fit the pipeline on the TRAIN set --
    train_cleaned = clean_data(train_df)
    train_cleaned = log_normalize(train_cleaned, "Amount")

    pipeline = build_preprocessing_pipeline(pca_features)
    pipeline_model = pipeline.fit(train_cleaned)

    # -- Transform all splits with the fitted model --
    datasets = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }
    preprocessed = preprocess_multiple_datasets(datasets, pipeline_model, pca_features)

    return preprocessed["train"], preprocessed["val"], preprocessed["test"]

# --------------------------------------------------------------------------------
# 6. Splitting Out Labels (Class) and Saving to CSV
# --------------------------------------------------------------------------------

def save_features_and_labels(df: DataFrame, label_col: str, features_path: str, labels_path: str):
    """
    Given a DataFrame that still contains the label column:
      1. Split into features_df (everything except label_col)
      2. Split into labels_df (only label_col)
      3. Save each to CSV
    """
    # 1) Labels only
    labels_df = df.select(label_col)
    # 2) Features only
    features_df = df.drop(label_col)

    # Save to CSV
    features_df.coalesce(1).write.csv(features_path, header=True, mode="overwrite")
    labels_df.coalesce(1).write.csv(labels_path, header=True, mode="overwrite")

def save_all_splits(
    train_df: DataFrame, val_df: DataFrame, test_df: DataFrame, label_col: str
):
    """
    Save train, val, and test splits as separate features+labels CSV files.
    """
    # You can of course rename the paths as you wish
    TRAIN_FEATURES_PATH = "preprocessed_creditcard_train_features"
    TRAIN_LABELS_PATH   = "preprocessed_creditcard_train_labels"

    VAL_FEATURES_PATH   = "preprocessed_creditcard_validation_features"
    VAL_LABELS_PATH     = "preprocessed_creditcard_validation_labels"

    TEST_FEATURES_PATH  = "preprocessed_creditcard_test_features"
    TEST_LABELS_PATH    = "preprocessed_creditcard_test_labels"

    # -- Train --
    save_features_and_labels(train_df, label_col, TRAIN_FEATURES_PATH, TRAIN_LABELS_PATH)
    # -- Validation --
    save_features_and_labels(val_df, label_col, VAL_FEATURES_PATH, VAL_LABELS_PATH)
    # -- Test --
    save_features_and_labels(test_df, label_col, TEST_FEATURES_PATH, TEST_LABELS_PATH)

    print("Preprocessed data splits (features + labels) saved to CSV:")
    print(f"  Train features: {TRAIN_FEATURES_PATH}, Train labels: {TRAIN_LABELS_PATH}")
    print(f"  Val features:   {VAL_FEATURES_PATH}, Val labels:   {VAL_LABELS_PATH}")
    print(f"  Test features:  {TEST_FEATURES_PATH}, Test labels:  {TEST_LABELS_PATH}")

# --------------------------------------------------------------------------------
# 7. Putting It All Together
# --------------------------------------------------------------------------------

start_time = time.time()

# 7.1 Fit the pipeline on the train set, transform all splits
train_preprocessed, val_preprocessed, test_preprocessed = fit_and_transform_data(
    train_set, val_set, test_set
)

# 7.2 Save Features and Labels Separately
#     (We assume the label is in a column named "Class")
save_all_splits(train_preprocessed, val_preprocessed, test_preprocessed, label_col="Class")

end_time = time.time()
print(f"Preprocessing + saving completed in {end_time - start_time:.2f} seconds")

def rename_csv_files(parent_directory):
    """
    Navigate through subdirectories in the parent directory, find the CSV file
    starting with 'part', and rename it to match the name of the subdirectory.

    Args:
        parent_directory (str): Path to the parent directory containing subdirectories.

    Returns:
        None
    """
    # Iterate through all items in the parent directory
    for subdirectory in os.listdir(parent_directory):
        # Construct full path for each subdirectory
        subdirectory_path = os.path.join(parent_directory, subdirectory)
        
        # Check if the path is a directory
        if os.path.isdir(subdirectory_path):
            # Look for files in the subdirectory
            for file_name in os.listdir(subdirectory_path):
                if file_name.startswith("part") and file_name.endswith(".csv"):
                    # Construct old and new file paths
                    old_file_path = os.path.join(subdirectory_path, file_name)
                    new_file_name = f"{subdirectory}.csv"
                    new_file_path = os.path.join(subdirectory_path, new_file_name)
                    
                    # Rename the file
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed '{old_file_path}' to '{new_file_path}'")
                    break  # Only one file should exist; no need to check further

# Below is an Example
parent_dir = "."  # Replace with your actual parent directory path
rename_csv_files(parent_dir)

