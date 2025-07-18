#!/usr/bin/env bash
# Automated deployment of Glue, SageMaker Notebook, and Inference scripts with status checks
set -euo pipefail

# Configurable Parameters
export AWS_REGION="ap-southeast-2"
BUCKET_NAME="fraud-detection-012725"

# Script and data file names (assumed to be in the current directory)
GLUE_SCRIPT="optimized_fraud_detection_glue_job.py"
NOTEBOOK_SCRIPT="separated_sagemaker_notebook_code.ipynb"   # Your SageMaker Notebook code file
INFERENCE_SCRIPT="separatedinference.py"                     # Your separate inference script
RAW_DATA_FILE="test_cloudPipeline_raw_data.csv"

# S3 folder structure
SCRIPTS_FOLDER="scripts"
RAW_DATA_FOLDER="raw_data"

# Wait times
MAX_CRAWLER_WAIT=600    # 10 minutes
MAX_JOB_WAIT=1800       # 30 minutes

# ---------------------------------------------------------------------
# Function: wait_for_crawler
# Waits until the Glue crawler reports a READY state.
wait_for_crawler() {
    local timeout=$1
    local elapsed=0
    while (( elapsed < timeout )); do
        status=$(aws glue get-crawler --name raw-data-crawler --query 'Crawler.State' --output text)
        if [[ "$status" == "READY" ]]; then
            return 0
        fi
        sleep 30
        (( elapsed += 30 ))
        echo "Crawler status: $status (${elapsed}s elapsed)"
    done
    echo "Crawler timed out after ${timeout}s"
    return 1
}

# ---------------------------------------------------------------------
# Function: wait_for_job
# Waits for the Glue job to complete and checks if it succeeded.
wait_for_job() {
    local run_id=$1
    local timeout=$2
    local elapsed=0
    while (( elapsed < timeout )); do
        status=$(aws glue get-job-run --job-name fraud-detection-job --run-id "$run_id" --query 'JobRun.JobRunState' --output text)
        if [[ "$status" =~ ^(SUCCEEDED|FAILED|STOPPED)$ ]]; then
            break
        fi
        sleep 60
        (( elapsed += 60 ))
        echo "Job status: $status (${elapsed}s elapsed)"
    done
    if [[ "$status" == "SUCCEEDED" ]]; then
        return 0
    else
        return 1
    fi
}

# ---------------------------------------------------------------------
# Function: create_glue_role
# Creates the GlueServiceRole if it doesn't already exist.
create_glue_role() {
    if ! aws iam get-role --role-name GlueServiceRole &>/dev/null; then
        echo "Creating GlueServiceRole..."
        aws iam create-role --role-name GlueServiceRole \
            --assume-role-policy-document file://glue-trust-policy.json
        aws iam attach-role-policy --role-name GlueServiceRole \
            --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
        aws iam attach-role-policy --role-name GlueServiceRole \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole
    else
        echo "GlueServiceRole already exists."
    fi
}

# ---------------------------------------------------------------------
# Function: create_sagemaker_role
# Creates the SageMakerExecutionRole if it doesn't already exist.
create_sagemaker_role() {
    if ! aws iam get-role --role-name SageMakerExecutionRole &>/dev/null; then
        echo "Creating SageMakerExecutionRole..."
        aws iam create-role --role-name SageMakerExecutionRole \
            --assume-role-policy-document file://sagemaker-trust-policy.json
        aws iam attach-role-policy --role-name SageMakerExecutionRole \
            --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
        aws iam attach-role-policy --role-name SageMakerExecutionRole \
            --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
    else
        echo "SageMakerExecutionRole already exists."
    fi
}

# ---------------------------------------------------------------------
# Main function: orchestrates the entire process
main() {
    echo "=== Starting full system deployment ==="
    
    # 1. Provision S3: Upload raw data
    echo "Uploading raw data file to S3..."
    aws s3 cp "$RAW_DATA_FILE" "s3://$BUCKET_NAME/$RAW_DATA_FOLDER/"

    # 2. IAM Setup: Create roles for Glue and SageMaker
    echo "Setting up IAM roles..."
    create_glue_role
    create_sagemaker_role

    # --- Add delay to ensure IAM roles are ready/propagated ---
    echo "Waiting 10 seconds for IAM role propagation..."
    sleep 10

    # 3. Upload Scripts: Glue ETL, SageMaker Notebook, and Inference script
    echo "Uploading scripts to S3..."
    aws s3 cp "$GLUE_SCRIPT" "s3://$BUCKET_NAME/$SCRIPTS_FOLDER/"
    aws s3 cp "$NOTEBOOK_SCRIPT" "s3://$BUCKET_NAME/$SCRIPTS_FOLDER/"
    aws s3 cp "$INFERENCE_SCRIPT" "s3://$BUCKET_NAME/$SCRIPTS_FOLDER/"
    echo "Scripts uploaded to s3://$BUCKET_NAME/$SCRIPTS_FOLDER/"

    # 4. Crawler Management: Create (if needed) and start the Glue crawler
    if ! aws glue get-crawler --name raw-data-crawler &>/dev/null; then
        echo "Creating Glue crawler 'raw-data-crawler'..."
        aws glue create-crawler --name raw-data-crawler \
            --role GlueServiceRole \
            --database-name fraud_detection_db \
            --targets "S3Targets=[{Path=s3://$BUCKET_NAME/$RAW_DATA_FOLDER}]"
    else
        echo "Glue crawler 'raw-data-crawler' already exists."
    fi

    echo "Starting Glue crawler..."
    aws glue start-crawler --name raw-data-crawler
    wait_for_crawler $MAX_CRAWLER_WAIT || { echo "Crawler did not complete successfully"; exit 1; }

    # 5. Glue Job Deployment: Create (if needed) and start the Glue job
    echo "Creating (if needed) Glue job 'fraud-detection-job'..."
    aws glue create-job --name fraud-detection-job \
        --role GlueServiceRole \
        --command "Name=glueetl,ScriptLocation=s3://$BUCKET_NAME/$SCRIPTS_FOLDER/$GLUE_SCRIPT,PythonVersion=3" \
        --glue-version "3.0" \
        --worker-type "G.1X" \
        --number-of-workers 5 &>/dev/null || echo "Glue job 'fraud-detection-job' already exists."

    echo "Starting Glue job..."
    run_id=$(aws glue start-job-run --job-name fraud-detection-job --query 'JobRunId' --output text)
    echo "Glue job started with Run ID: $run_id"

    if wait_for_job "$run_id" $MAX_JOB_WAIT; then
        echo "Glue job completed successfully."
    else
        echo "Glue job failed or timed out."
        exit 1
    fi

    # 6. Final Notification for SageMaker assets:
    echo "SageMaker Notebook code and Inference script have been uploaded to s3://$BUCKET_NAME/$SCRIPTS_FOLDER/"
    echo "They are ready to be used for model training, evaluation, and endpoint deployment."
    
    echo "=== Full system deployment completed successfully ==="
}

# Execute main
main

