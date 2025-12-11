#!/bin/bash

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Install requirements if needed
# pip install -r requirements.txt

# SQL Dump Path
SQL_DUMP="dataset_sqldump/report_db_photos_2025-12-03_200313.sql"

# Media Root (Relative to this script)
# Go up two levels to project root, then into backend/src/assets
MEDIA_ROOT="../../backend/src/assets"

# Checkpoints Dir
SAVE_DIR="checkpoints_$(date +%Y%m%d_%H%M%S)"

echo "Starting Fine-tuning..."
echo "SQL Dump: $SQL_DUMP"
echo "Media Root: $MEDIA_ROOT"
echo "Save Dir: $SAVE_DIR"

python train.py \
    --sql_dump_path "$SQL_DUMP" \
    --media_root "$MEDIA_ROOT" \
    --save_dir "$SAVE_DIR" \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0005 \
    --embedding_dim 128

echo "Done. Check $SAVE_DIR for results."

