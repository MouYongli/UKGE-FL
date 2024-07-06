#!/bin/bash

# Check if MODEL and DATASET are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 MODEL DATASET"
    exit 1
fi

MODEL=$1
DATASET=$2

# Hyperparameters
LEARNING_RATES=(0.001 0.05 0.01 0.05 0.1)
HIDDEN_DIMS=(128 256 512)
THRESHOLDS=$(seq 0.0 0.05 0.95)

# Loop over each combination of hyperparameters
for LR in "${LEARNING_RATES[@]}"; do
    for HD in "${HIDDEN_DIMS[@]}"; do
        for THRESHOLD in $THRESHOLDS; do
            echo "Running $MODEL on $DATASET with LR=$LR, HD=$HD, THRESHOLD=$THRESHOLD"
            python src/train_det_baseline.py --model $MODEL --dataset $DATASET --lr $LR --hidden_dim $HD --threshold $THRESHOLD
        done
    done
done
