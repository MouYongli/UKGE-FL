#!/bin/bash

# Define arrays for models and datasets
MODELS=("transe" "distmult" "complex" "rotate")
DATASETS=("cn15k" "nl27k" "ppi5k")

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

# Export variables for parallel
export LOG_DIR

# Create a function to run the job
run_job() {
    MODEL=$1
    DATASET=$2
    LR=$3
    HD=$4
    THRESHOLD=$5
    LOG_FILE="$LOG_DIR/${MODEL}_${DATASET}_LR${LR}_HD${HD}_THRESH${THRESHOLD}.log"
    echo "Running $MODEL on $DATASET with LR=$LR, HD=$HD, THRESHOLD=$THRESHOLD"
    nohup python -u src/ukge/train_det_baseline.py --model $MODEL --dataset $DATASET --lr $LR --hidden_dim $HD --threshold $THRESHOLD &> $LOG_FILE
}

export -f run_job

# Generate combinations of models, datasets, and hyperparameters and run them in parallel
parallel run_job ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${LEARNING_RATES[@]}" ::: "${HIDDEN_DIMS[@]}" ::: $THRESHOLDS

# Wait for all background jobs to finish
wait