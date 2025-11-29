import os
import numpy as np
import librosa
import soundfile as sf

INPUT_DIR = "data/dataset_prepared/audio"
OUTPUT_DIR = "data/features/stft"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🎛 extraindo STFT (com padding)...")

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".wav"):
        continue

    path = os.path.join(INPUT_DIR, fname)

    y, sr = librosa.load(path, sr=16000)

    # STFT 1025×N
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))

    # padding horizontal para largura fixa (300)
    if stft.shape[1] < 300:
        pad_width = 300 - stft.shape[1]
        stft = np.pad(stft, ((0,0),(0,pad_width)))
    else:
        stft = stft[:, :300]

    out_path = os.path.join(OUTPUT_DIR, fname.replace(".wav", ".npy"))
    np.save(out_path, stft)

print("STFT extraído!")
