# extract_pitch_f0.py
# Extract pitch contour (F0) using librosa's pYIN

import librosa
import numpy as np
import torchaudio
import os
from applied_tool.utility import log_step

log_step("[F0] Starting pitch extraction")

AUDIO_DIR = "./applied_tool/data/audio"
PITCH_SAVE_PATH = "./applied_tool/data/f0_features.npy"

f0_data = {}

for fname in os.listdir(AUDIO_DIR):
    if fname.endswith(".wav"):
        path = os.path.join(AUDIO_DIR, fname)
        try:
            audio, sr = torchaudio.load(path)
            audio = audio.squeeze().numpy()

            f0 = librosa.pyin(
                y=audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )[0]
            f0 = librosa.util.fix_length(np.array(f0), size=len(audio))

            f0_data[fname] = f0
            log_step(f"[F0] ✅ Extracted pitch for {fname}")
        except Exception as e:
            log_step(f"[F0] ❌ Failed for {fname}: {str(e)}")

np.save(PITCH_SAVE_PATH, f0_data)
log_step(f"[F0] ✅ Saved pitch features to {PITCH_SAVE_PATH}")
