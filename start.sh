#!/bin/bash
set -e

echo "Installing pip packages from repo requirements.txt..."
pip install -r requirements.txt

echo "Downloading dataset..."
# Run the dataset download script
if [ -z "$DATASET_DIR" ]; then
    DATASET_DIR=$(realpath ..)
fi
python script/download_dataset.py $DATASET_DIR danbooru2023/train-70k/
python script/download_dataset.py $DATASET_DIR danbooru2023/train/
python script/download_dataset.py $DATASET_DIR danbooru2023/valid/

echo "Starting checkpoint monitor..."
# Start the checkpoint monitor in the background
if [ -z "$EXP_DIR" ]; then
    EXP_DIR=exp
fi
mkdir -p $EXP_DIR
python script/checkpoint_monitor.py $EXP_DIR &
MONITOR_PID=$!

echo "Starting TensorBoard..."
# Start TensorBoard (adjust --logdir if needed, here assumed to be repo/logs)
tensorboard --logdir=$EXP_DIR --host=0.0.0.0 --port=6006 &
TB_PID=$!

echo "Starting training..."
# Run the training script (assumed to be repo/train.py)
python main.py fit --config config/train.yaml
TRAIN_EXIT_CODE=$?

echo "Training finished with exit code $TRAIN_EXIT_CODE. Stopping pod..."
# Stop the pod using runpodctl command
runpodctl stop pod "$RUNPOD_POD_ID"

echo "Terminating background processes..."
kill $MONITOR_PID $TB_PID

exit $TRAIN_EXIT_CODE
