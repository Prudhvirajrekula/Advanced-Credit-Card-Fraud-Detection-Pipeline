import time
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, log1p
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame

# --------------------------------------------------------------------------------
# 1. Initialize Glue Context
# --------------------------------------------------------------------------------
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# S3 paths
BUCKET_NAME = "fraud-detection-012725"
RAW_DATA_PATH = f"s3://{BUCKET_NAME}/raw_data/"
PROCESSED_DATA_PATH = f"s3://{BUCKET_NAME}/processed_data/"

# --------------------------------------------------------------------------------
# 2. Load Dataset from the Data Catalog
# --------------------------------------------------------------------------------
# We'll assume your crawler created a table named "raw_data" in the "fraud_detection_db" database.

DATABASE_NAME = "fraud_detection_db"
TABLE_NAME = "raw_data"

raw_dynamic_frame = glueContext.create_dynamic_frame.from_catalog(
    database=DATABASE_NAME,
    table_name=TABLE_NAME,
    transformation_ctx="raw_df"
)

# (Optional) Shuffle the dataset by sampling with fraction=1.0 and a seed
df = raw_dynamic_frame.toDF().sample(fraction=1.0, seed=42)

# --------------------------------------------------------------------------------
# 3. Split Data into Train, Validation, and Test
# --------------------------------------------------------------------------------
train_fraction = 0.7
val_fraction = 0.15
test_fraction = 0.15

splits = df.randomSplit([train_fraction, val_fraction, test_fraction], seed=42)
train_set, val_set, test_set = splits[0], splits[1], splits[2]

# --------------------------------------------------------------------------------
# 4. Utility Functions
# --------------------------------------------------------------------------------
def clean_data(df: DataFrame) -> DataFrame:
    return df.dropDuplicates().dropna()

def log_normalize(df: DataFrame, column: str) -> DataFrame:
    return df.withColumn(column, log1p(col(column)))

def build_preprocessing_pipeline(feature_cols):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withMean=True,
        withStd=True
    )
    pipeline = Pipeline(stages=[assembler, scaler])
    return pipeline

def explode_scaled_features(df: DataFrame, feature_cols) -> DataFrame:
    # Drop original columns (v1..v28) before adding scaled versions
    for c in feature_cols:
        df = df.drop(c)
    df = df.withColumn("scaled_array", vector_to_array(col("scaled_features")))
    select_exprs = [
        col(c) for c in df.columns
        if c not in {"scaled_features", "scaled_array", "features"}
    ]
    select_exprs += [
        col("scaled_array")[i].alias(feature_cols[i])
        for i in range(len(feature_cols))
    ]
    return df.select(*select_exprs)

# --------------------------------------------------------------------------------
# 5. Main Preprocessing Flow
# --------------------------------------------------------------------------------
def preprocess_dataset(df: DataFrame, pipeline_model, feature_cols, name: str) -> DataFrame:
    """
    Cleans, log-normalizes 'amount', applies the pipeline, and explodes scaled features.
    """
    cleaned = clean_data(df)
    normalized = log_normalize(cleaned, "amount")
    transformed = pipeline_model.transform(normalized)
    final_df = explode_scaled_features(transformed, feature_cols)
    return final_df

def fit_and_transform_data(train_df: DataFrame, val_df: DataFrame, test_df: DataFrame):
    """
    1. Prepare the training set (clean + log normalize), fit the pipeline on training.
    2. Apply the fitted pipeline to train/val/test.
    3. Return the three processed DataFrames.
    """
    pca_features = [f"v{i}" for i in range(1, 29)]
    train_cleaned = clean_data(train_df)
    train_cleaned = log_normalize(train_cleaned, "amount")
    pipeline = build_preprocessing_pipeline(pca_features)
    pipeline_model = pipeline.fit(train_cleaned)

    datasets = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }
    preprocessed = {}
    for name, df in datasets.items():
        preprocessed[name] = preprocess_dataset(df, pipeline_model, pca_features, name)

    return preprocessed["train"], preprocessed["val"], preprocessed["test"]

# --------------------------------------------------------------------------------
# 6. Saving Features & Labels Separately
# --------------------------------------------------------------------------------
def save_features_and_labels_to_s3(
    df: DataFrame,
    folder: str,
    label_col: str = "class"):
    """
    Splits the provided DataFrame into:
      - A 'features' CSV (everything except the label column)
      - A 'labels' CSV   (only the label column)
    and writes each to the same S3 folder.
    """
    features_output_path = f"{PROCESSED_DATA_PATH}{folder}/features.csv"
    labels_output_path   = f"{PROCESSED_DATA_PATH}{folder}/labels.csv"

    # Separate the label
    labels_df = df.select(label_col)
    # Separate the features (drop the label column)
    features_df = df.drop(label_col)

    # Write each to S3
    features_df.write.csv(features_output_path, header=True, mode="overwrite")
    labels_df.write.csv(labels_output_path, header=True, mode="overwrite")

    print(f"Saved features to {features_output_path}")
    print(f"Saved labels to {labels_output_path}")

# --------------------------------------------------------------------------------
# 7. Putting It All Together
# --------------------------------------------------------------------------------

# Preprocess
train_preprocessed, val_preprocessed, test_preprocessed = fit_and_transform_data(train_set, val_set, test_set)

# Save each split's features & labels in separate CSV files under processed_data/<split>/
save_features_and_labels_to_s3(train_preprocessed, folder="train", label_col="class")
save_features_and_labels_to_s3(val_preprocessed, folder="validation", label_col="class")
save_features_and_labels_to_s3(test_preprocessed, folder="test", label_col="class")
