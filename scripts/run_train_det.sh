#!/bin/bash

# Check if MODEL and DATASET are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 MODEL DATASET"
    exit 1
fi

MODEL=$1
DATASET=$2

# Navigate to the project directory
cd $HOME/Projects/UKGE-FL

# Create logs folder if it doesn't exist
LOG_DIR="$HOME/Projects/UKGE-FL/logs/det"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p $LOG_DIR
fi

# Hyperparameters
LEARNING_RATES=(0.001 0.05 0.01 0.05 0.1)
HIDDEN_DIMS=(128 256 512)
THRESHOLDS=$(seq 0.0 0.05 0.95)

# Loop over each combination of hyperparameters
for LR in "${LEARNING_RATES[@]}"; do
    for HD in "${HIDDEN_DIMS[@]}"; do
        for THRESHOLD in $THRESHOLDS; do
            LOG_FILE="$LOG_DIR/${MODEL}_${DATASET}_LR${LR}_HD${HD}_THRESH${THRESHOLD}.log"
            echo "Running $MODEL on $DATASET with LR=$LR, HD=$HD, THRESHOLD=$THRESHOLD"
            nohup python -u src/ukge/train_det_baseline.py --model $MODEL --dataset $DATASET --lr $LR --hidden_dim $HD --threshold $THRESHOLD &> $LOG_FILE
        done
    done
done
# Wait for all background jobs to finish
wait