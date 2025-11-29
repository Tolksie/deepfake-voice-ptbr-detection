import os
import shutil
import csv

# caminhos
BASE_DIR = "data"
NATURAL_DIR = os.path.join(BASE_DIR, "natural")
FAKE_DIR = os.path.join(BASE_DIR, "fake")
OUT_DIR = os.path.join(BASE_DIR, "dataset_prepared")
OUT_AUDIO = os.path.join(OUT_DIR, "audio")
OUT_CSV = os.path.join(OUT_DIR, "labels.csv")

# garantir pastas
os.makedirs(OUT_AUDIO, exist_ok=True)

def copy_speaker(src_folder, speaker_id, label):
    """copia 15 áudios de um speaker"""
    files = [f for f in os.listdir(src_folder) if f.endswith(".wav")]
    files = sorted(files)[:15]      # só os 15 primeiros

    for i, fname in enumerate(files):
        src = os.path.join(src_folder, fname)
        new_name = f"speaker_{speaker_id:03d}_{i:03d}.wav"
        dst = os.path.join(OUT_AUDIO, new_name)
        shutil.copy(src, dst)

        # adiciona ao CSV
        rows.append([new_name, label])


# ---------------------------------------------------------
# iniciar CSV
rows = [["filename", "label"]]

print("Processando falas naturais...")

speaker_id = 1
if os.path.exists(NATURAL_DIR):
    for item in sorted(os.listdir(NATURAL_DIR)):
        spath = os.path.join(NATURAL_DIR, item)
        if os.path.isdir(spath):
            print(f" copiando {item} → speaker {speaker_id:03d}")
            copy_speaker(spath, speaker_id, 0)   # 0 = natural
            speaker_id += 1
else:
    print("ERRO: pasta data/natural_15 não existe!")
    exit()


print("Processando falas fake...")

if os.path.exists(FAKE_DIR):
    for item in sorted(os.listdir(FAKE_DIR)):
        spath = os.path.join(FAKE_DIR, item)
        if os.path.isdir(spath):
            print(f" copiando {item} → speaker {speaker_id:03d}")
            copy_speaker(spath, speaker_id, 1)   # 1 = fake
            speaker_id += 1
else:
    print("ERRO: pasta data/fake não existe!")
    exit()


# ---------------------------------------------------------
# salvar CSV
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("\n dataset_prepared gerado com sucesso!")
print(f"→ total de arquivos: {len(rows)-1}")
print(f"→ labels salvos em: {OUT_CSV}")
