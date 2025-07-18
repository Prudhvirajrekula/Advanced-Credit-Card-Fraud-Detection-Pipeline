#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# This script provisions an S3 bucket and creates folder prefixes needed
# for the fraud detection pipeline.
#
# Defaults:
#   AWS_REGION="ap-southeast-2"
#   BUCKET_NAME="fraud-detection-012725"
#
# To override defaults, export environment variables before running:
#   export AWS_REGION="us-east-1"
#   export BUCKET_NAME="my-other-bucket"
# -----------------------------------------------------------------------------

set -euo pipefail

# 1. Defaults (override via environment variables, if desired)
: "${AWS_REGION:=ap-southeast-2}"
: "${BUCKET_NAME:=fraud-detection-012725}"

# 2. Subfolder structure for the pipeline
folders=(
    "raw_data"
    "processed_data/train"
    "processed_data/validation"
    "processed_data/test"
    "scripts"
    "predictions"
    "model_artifacts"
)

# -----------------------------------------------------------------------------
# Function to create an S3 bucket
# -----------------------------------------------------------------------------
create_s3_bucket() {
    local bucket_name="$1"
    local region="$2"

    echo "Creating S3 bucket: $bucket_name in region: $region"

    # For non-us-east-1, specify create-bucket-configuration:
    aws s3api create-bucket \
        --bucket "$bucket_name" \
        --region "$region" \
        --create-bucket-configuration LocationConstraint="$region"

    echo "Bucket '$bucket_name' created successfully in region '$region'."
}

# -----------------------------------------------------------------------------
# Function to enable encryption and block public access on the bucket
# -----------------------------------------------------------------------------
enable_bucket_encryption_and_public_access_block() {
    local bucket_name="$1"

    echo "Enabling AES-256 encryption on bucket: $bucket_name"
    aws s3api put-bucket-encryption \
        --bucket "$bucket_name" \
        --server-side-encryption-configuration '{
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "AES256"
                    }
                }
            ]
        }'

    echo "Blocking all public access for bucket: $bucket_name"
    aws s3api put-public-access-block \
        --bucket "$bucket_name" \
        --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

    echo "Encryption and public access block configured for bucket: $bucket_name."
}

# -----------------------------------------------------------------------------
# Function to create "folders" in the S3 bucket
# -----------------------------------------------------------------------------
create_s3_folders() {
    local bucket_name="$1"
    shift
    local folders=("$@")

    for folder in "${folders[@]}"; do
        # Ensure the folder path ends with a forward slash
        local folder_key="${folder%/}/"
        echo "Creating folder prefix: $folder_key in bucket: $bucket_name"

        # Using put-object to create a zero-byte object to represent the folder
        aws s3api put-object \
            --bucket "$bucket_name" \
            --key "$folder_key" \
            --region "$AWS_REGION"

        echo "Folder prefix '$folder_key' created."
    done
}

# -----------------------------------------------------------------------------
# Main script execution
# -----------------------------------------------------------------------------
echo "Starting S3 bucket and folder creation..."
echo "Using AWS_REGION=$AWS_REGION, BUCKET_NAME=$BUCKET_NAME"

# 1. Check if bucket exists
if aws s3api head-bucket --bucket "$BUCKET_NAME" --region "$AWS_REGION" 2>/dev/null; then
    echo "Bucket '$BUCKET_NAME' already exists (and is accessible). Skipping creation."
else
    # 2. If it doesn't exist, create it
    create_s3_bucket "$BUCKET_NAME" "$AWS_REGION"
fi

# 3. Enable encryption & public access block (safe to re-run)
enable_bucket_encryption_and_public_access_block "$BUCKET_NAME"

# 4. Create "folders" (prefixes)
create_s3_folders "$BUCKET_NAME" "${folders[@]}"

echo "S3 setup complete."
