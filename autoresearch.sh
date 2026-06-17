#!/bin/bash
set -e

# Run evaluation and capture output
OUTPUT=$(D:/program/Python/anaconda/envs/Demo_D/python.exe evaluate.py --config configs/thyroid_config.yaml)
echo "$OUTPUT"

# Extract metrics using grep and awk
DICE=$(echo "$OUTPUT" | grep -i "Mean Nodule Segmentation Dice" | awk -F': ' '{print $2}' | awk -F'%' '{print $1}')
AUC=$(echo "$OUTPUT" | grep -i "Malignancy Classification AUC" | awk -F': ' '{print $2}' | awk -F'%' '{print $1}')
F1=$(echo "$OUTPUT" | grep -i "Average TI-RADS F1-Score" | awk -F': ' '{print $2}' | awk -F'%' '{print $1}')

# Check if any metric is empty
if [ -z "$DICE" ] || [ -z "$AUC" ] || [ -z "$F1" ] || [ "$DICE" == "N/A" ] || [ "$AUC" == "N/A" ] || [ "$F1" == "N/A" ]; then
    echo "Error: Failed to parse metrics from evaluation output."
    exit 1
fi

# Calculate composite score using Python
COMPOSITE=$(D:/program/Python/anaconda/envs/Demo_D/python.exe -c "print(round(0.4 * $DICE + 0.4 * $AUC + 0.2 * $F1, 2))")

echo "METRIC composite_score=$COMPOSITE"
echo "METRIC segmentation_dice=$DICE"
echo "METRIC classification_auc=$AUC"
echo "METRIC average_tirads_f1=$F1"
exit 0
