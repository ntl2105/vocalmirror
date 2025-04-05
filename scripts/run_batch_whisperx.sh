#!/bin/bash

# === CONFIGURATION ===
AUDIO_DIR="/home/ubuntu/data/vidm/audio"
OUTPUT_DIR="/home/ubuntu/data/vidm/whisper_output"
LOG_FILE="process_log.txt"

mkdir -p "$OUTPUT_DIR"
echo "📝 Starting batch WhisperX processing at $(date)" > "$LOG_FILE"

# Initialize counters
total_files=0
success_count=0
fail_count=0

# Process each audio file
for audio_file in "$AUDIO_DIR"/*.wav; do
    total_files=$((total_files + 1))
    filename=$(basename -- "$audio_file")
    filename_noext="${filename%.*}"

    echo "🚀 [$filename] Processing started at $(date)" | tee -a "$LOG_FILE"

    # Timer start
    start_time=$(date +%s)

    # Run WhisperX
    if python -m whisperx "$audio_file" \
        --output_dir "$OUTPUT_DIR" \
        --output_format json \
        --language vi \
        --device cuda >> "$LOG_FILE" 2>&1; then
        echo "✅ [$filename] Successfully processed at $(date)" | tee -a "$LOG_FILE"
        success_count=$((success_count + 1))
    else
        echo "❌ [$filename] Failed to process at $(date)" | tee -a "$LOG_FILE"
        fail_count=$((fail_count + 1))
    fi

    # Timer end
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "⏱️ [$filename] Time taken: ${elapsed}s" | tee -a "$LOG_FILE"
    echo "--------------------------" | tee -a "$LOG_FILE"

done

# === SUMMARY ===
echo "🎉 Processing complete at $(date)" | tee -a "$LOG_FILE"
echo "Total files: $total_files" | tee -a "$LOG_FILE"
echo "Successful: $success_count" | tee -a "$LOG_FILE"
echo "Failed: $fail_count" | tee -a "$LOG_FILE"

echo "📝 Log written to $LOG_FILE"