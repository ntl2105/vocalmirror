import os
import traceback
from tqdm.contrib.concurrent import thread_map
from datetime import datetime
import whisperx

# === CONFIGURATION ===
audio_dir = "/home/ubuntu/data/vidm/audio"
reference_transcript_dir = "/home/ubuntu/data/vidm/align_text"
output_dir = "/home/ubuntu/data/vidm/whisper_output"
device = "cuda"
language = "vi"

log_file_path = "log_file_v3.txt"

# Initialize log file
with open(log_file_path, "w") as log_file:
    log_file.write(f"📝 Batch forced alignment started at {datetime.utcnow()} UTC\n")
    log_file.write(f"Processing audio files in: {audio_dir}\n\n")

# === LOAD MODELS ONCE ===
model = whisperx.load_model("large-v2", device, compute_type="float16")
model_a, metadata = whisperx.load_align_model(language_code=language, device=device)

# === PROCESSING FUNCTION ===
def process_audio_file(audio_file):
    base_name = os.path.splitext(audio_file)[0]
    audio_path = os.path.join(audio_dir, audio_file)
    reference_path = os.path.join(reference_transcript_dir, base_name + ".txt")
    output_path = os.path.join(output_dir, base_name)

    try:
        # Check if reference transcript exists
        if not os.path.exists(reference_path):
            raise FileNotFoundError(f"Reference transcript not found: {reference_path}")

        # Read reference transcript
        with open(reference_path, "r", encoding="utf-8") as ref_file:
            reference_text = ref_file.read().strip()

        # Transcribe (IMPORTANT: pass path, not array!)
        result = model.transcribe(audio_path, batch_size=16, language=language)

        # Load audio as array for alignment
        audio = whisperx.load_audio(audio_path)

        # Forced alignment
        result_aligned = whisperx.align(
            reference_text,
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )

        # Save output
        os.makedirs(output_path, exist_ok=True)
        whisperx.utils.write_json(result_aligned, os.path.join(output_path, "alignment.json"))

        log_line = f"✅ [{audio_file}] Successfully processed at {datetime.utcnow()} UTC\n"
        return ("success", log_line)

    except Exception as e:
        error_details = traceback.format_exc()
        log_line = f"❌ [{audio_file}] Failed: {e}\n{error_details}\n"
        return ("fail", log_line)

# === MAIN EXECUTION ===

# Collect all .wav files
audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

# Process files with parallel threads and tqdm progress bar!
results = thread_map(process_audio_file, audio_files, max_workers=os.cpu_count(), desc="Processing files")

# Summarize results
success_count = sum(1 for status, _ in results if status == "success")
fail_count = sum(1 for status, _ in results if status == "fail")
failures = [log_line for status, log_line in results if status == "fail"]

with open(log_file_path, "a") as log_file:
    log_file.write("\n🎉 Batch processing completed!\n")
    log_file.write(f"✅ Successes: {success_count}\n")
    log_file.write(f"❌ Failures: {fail_count}\n")
    if failures:
        log_file.write("❗ Failed files:\n")
        for fail in failures:
            log_file.write(f"{fail}\n")

print(f"🎉 Done! Successes: {success_count}, Failures: {fail_count}")
print(f"📄 See detailed log here: {log_file_path}")