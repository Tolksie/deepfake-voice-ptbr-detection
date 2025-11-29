import librosa
import numpy as np
import os
from scipy.io.wavfile import write

INPUT_DIRS = ["data/natural_15", "data/fake"]
OUTPUT_DIR = "data/dataset_prepared/audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_DURATION = 3.0  # segundos
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

def preprocess_audio(path):
    # carregar áudio original
    y, sr = librosa.load(path, sr=None)

    # resample para 16 kHz
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    # remover silêncio
    y, _ = librosa.effects.trim(y, top_db=25)

    # normalizar amplitude
    y = y / np.max(np.abs(y))

    # padronizar duração
    if len(y) > TARGET_SAMPLES:
        y = y[:TARGET_SAMPLES]
    else:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode="constant")

    return y


def process_dataset():
    for base in INPUT_DIRS:
        for speaker in os.listdir(base):
            speaker_dir = os.path.join(base, speaker)
            if not os.path.isdir(speaker_dir):
                continue

            for filename in os.listdir(speaker_dir):
                if not filename.endswith(".wav"):
                    continue

                in_path = os.path.join(speaker_dir, filename)

                try:
                    audio = preprocess_audio(in_path)
                except Exception as e:
                    print("erro ao processar:", in_path, e)
                    continue

                out_name = f"{speaker}_{filename.replace('.wav', '.npy')}"
                out_path = os.path.join(OUTPUT_DIR, out_name)

                np.save(out_path, audio)

                print("OK →", out_name)


if __name__ == "__main__":
    print("Processando dataset...")
    process_dataset()
    print("FINALIZADO!")
