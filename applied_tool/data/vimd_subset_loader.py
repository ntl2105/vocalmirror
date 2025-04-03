# vimd_subset_loader.py (Fixed Early Stop Logic)

from datasets import load_dataset
from collections import defaultdict
import torchaudio
import os
import csv
import torch
from ..utility import log_step

log_step("STEP A1: Start ViMD dataset subset loading with streaming")

# Constants
UTTERANCES_PER_PROVINCE = 5
TARGET_REGIONS = {"North", "South"}
output_dir = "./applied_tool/data/audio"
os.makedirs(output_dir, exist_ok=True)

metadata_csv_path = "./applied_tool/data/vimd_subset_metadata.csv"

# Step A2: Stream dataset (no full download)
log_step("STEP A2: Streaming ViMD dataset")
streamed_dataset = load_dataset("nguyendv02/ViMD_Dataset", split="train", streaming=True)

# Track progress
province_utterance_counts = defaultdict(int)
province_speakers = defaultdict(set)
collected_examples = []

# Track how many provinces we've filled per region
region_province_completion = defaultdict(set)

for i, example in enumerate(streamed_dataset):
    region = example.get("region")
    province = example.get("province_name")
    speaker = example.get("speakerID")

    if region not in TARGET_REGIONS:
        continue

    # Only allow one utterance per speaker
    if speaker in province_speakers[province]:
        continue

    if province_utterance_counts[province] >= UTTERANCES_PER_PROVINCE:
        continue

    try:
        # Save audio
        audio_array = example["audio"]["array"]
        sample_rate = example["audio"]["sampling_rate"]
        tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)

        filename = f"{region}_{province}_{speaker}_{example['filename']}"
        filepath = os.path.join(output_dir, filename)

        torchaudio.save(filepath, tensor, sample_rate)

        log_step(f"✅ Saved {filename} [{province_utterance_counts[province]+1}/{UTTERANCES_PER_PROVINCE}] for {province}")

        # Track and store metadata
        province_utterance_counts[province] += 1
        province_speakers[province].add(speaker)

        collected_examples.append({
            "region": region,
            "province": province,
            "speaker": speaker,
            "gender": example.get("gender"),
            "filename": filename,
            "text": example.get("text")
        })

        if province_utterance_counts[province] >= UTTERANCES_PER_PROVINCE:
            region_province_completion[region].add(province)

        # ✅ Stop if we've filled ALL provinces in BOTH regions
        if len(region_province_completion["North"].union(region_province_completion["South"])) >= 40:  # You can adjust this
            log_step("🎉 All target provinces filled. Stopping early.")
            break

    except Exception as e:
        log_step(f"❌ Failed to save audio from {province}/{speaker}: {e}")

# Save metadata to CSV
log_step("STEP A8: Saving metadata CSV")
with open(metadata_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["region", "province", "speaker", "gender", "filename", "text"])
    writer.writeheader()
    writer.writerows(collected_examples)

log_step(f"✅ Done. Saved {len(collected_examples)} utterances to {output_dir} and metadata to {metadata_csv_path}.")
