#!/bin/bash

# Set the output file
OUTPUT_FILE="gpu.log"

# Create header for the CSV file
echo "timestamp,gpu_id,memory_used,memory_total" > $OUTPUT_FILE

# Loop and collect data every 0.1 seconds
while true; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S.%3N")
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | while read line; do
        echo "$timestamp,$line" >> $OUTPUT_FILE
    done
    sleep 0.01
done
