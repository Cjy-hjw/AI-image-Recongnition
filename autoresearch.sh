#!/bin/bash
set -e

SERVER="root@182.91.14.5"
PORT=22038
SERVER_DIR="/root/AI_image_Recognition"
SERVER_PY="/root/miniconda3/envs/thyroid_env/bin/python"
SSH="ssh -p $PORT -o StrictHostKeyChecking=no"
SCP="scp -P $PORT -o StrictHostKeyChecking=no"

# 1. Deploy latest code to server
echo "Deploying code to server..."
D:/program/Python/anaconda/envs/Demo_D/python.exe deploy.py

# 2. Train on server (A800 80GB)
echo "Starting training on server (A800)..."
$SSH $SERVER "cd $SERVER_DIR && $SERVER_PY -u train.py --config configs/thyroid_config.yaml"
echo "Training completed."

# 3. Run evaluation on server, capture output
echo "Running evaluation on server..."
OUTPUT=$($SSH $SERVER "cd $SERVER_DIR && $SERVER_PY -u evaluate.py --config configs/thyroid_config.yaml 2>&1")
echo "$OUTPUT"

# 4. Extract metrics
DICE=$(echo "$OUTPUT" | grep -i "Mean Nodule Segmentation Dice" | awk -F': ' '{print $2}' | awk -F'%' '{print $1}')
AUC=$(echo "$OUTPUT"  | grep -i "Malignancy Classification AUC"  | awk -F': ' '{print $2}' | awk -F'%' '{print $1}')
F1=$(echo "$OUTPUT"   | grep -i "Average TI-RADS F1-Score"       | awk -F': ' '{print $2}' | awk -F'%' '{print $1}')

if [ -z "$DICE" ] || [ -z "$AUC" ] || [ -z "$F1" ] || [ "$DICE" = "N/A" ] || [ "$AUC" = "N/A" ] || [ "$F1" = "N/A" ]; then
    echo "Error: Failed to parse metrics."
    exit 1
fi

COMPOSITE=$(D:/program/Python/anaconda/envs/Demo_D/python.exe -c "print(round(0.4 * $DICE + 0.4 * $AUC + 0.2 * $F1, 2))")

echo "METRIC composite_score=$COMPOSITE"
echo "METRIC segmentation_dice=$DICE"
echo "METRIC classification_auc=$AUC"
echo "METRIC average_tirads_f1=$F1"
exit 0
