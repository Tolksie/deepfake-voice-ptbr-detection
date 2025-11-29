import librosa
import numpy as np
import os

INPUT_DIR = "data/dataset_prepared/audio"
OUTPUT_DIR = "data/features/mel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_MELS = 128
TARGET_FRAMES = 300  # padronização real para usar no PyTorch

print("🎛 extraindo mel-spectrogramas (com padding fixo)...")

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".wav"):
        continue

    path = os.path.join(INPUT_DIR, fname)

    y, sr = librosa.load(path, sr=16000)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=256
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # padding / crop
    if mel_db.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0), (0,pad_width)))
    else:
        mel_db = mel_db[:, :TARGET_FRAMES]

    np.save(f"{OUTPUT_DIR}/{fname.replace('.wav','.npy')}", mel_db)

print("features salvas com tamanho igual!")
