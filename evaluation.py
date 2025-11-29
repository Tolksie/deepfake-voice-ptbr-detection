import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix
)

from dataset import AudioDataset
from cnn import CNNClassifier
from rnn import RNNClassifier

# ==========================
# CONFIGURAÇÃO
# ==========================

# "cnn" ou "rnn"
USE_MODEL = "rnn"

# "mel" ou "stft"
USE_FEATURE = "stft"

# nome do arquivo de pesos (estão na RAIZ do projeto)
WEIGHTS = f"model_{USE_MODEL}_{USE_FEATURE}.pt"

FEATURE_DIR = f"data/features/{USE_FEATURE}"
LABELS_CSV = "data/dataset_prepared/labels.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("usando device:", DEVICE)

# pasta onde vão ficar os resultados
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================
# CARREGAR DATASET
# ==========================

dataset = AudioDataset(FEATURE_DIR, LABELS_CSV)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# descobrir shape de uma amostra (H, W)
sample, _ = dataset[0]
H, W = sample.shape
print("shape do feature:", sample.shape)

# ==========================
# CRIAR MODELO
# ==========================

if USE_MODEL == "cnn":
    model = CNNClassifier(num_classes=2)
elif USE_MODEL == "rnn":
    model = RNNClassifier(input_size=H)
else:
    raise ValueError("USE_MODEL deve ser 'cnn' ou 'rnn'.")

# carregar pesos
if not os.path.exists(WEIGHTS):
    raise FileNotFoundError(f"arquivo de pesos não encontrado: {WEIGHTS}")

model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==========================
# AVALIAÇÃO
# ==========================

all_labels = []
all_preds = []
all_probs = []  # probabilidade da classe 1 (fake, por ex.)

with torch.no_grad():
    for X, y in loader:
        # X vem como (batch, H, W) do dataset
        # deixamos no formato (batch, 1, H, W) igual no train
        X = X.unsqueeze(1).float().to(DEVICE)
        y = y.to(DEVICE)

        logits = model(X)
        probs = torch.softmax(logits, dim=1)[:, 1]  # prob. classe 1
        preds = torch.argmax(logits, dim=1)

        all_labels.extend(y.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# ==========================
# MÉTRICAS
# ==========================

acc = accuracy_score(all_labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="binary", zero_division=0
)

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

cm = confusion_matrix(all_labels, all_preds)

print("\n===== RESULTADOS =====")
print(f"Modelo:  {USE_MODEL.upper()}  |  Feature: {USE_FEATURE.upper()}")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC-ROC  : {roc_auc:.4f}")
print("Matriz de confusão:")
print(cm)

# salvar métricas em txt
metrics_path = os.path.join(
    RESULTS_DIR, f"metrics_{USE_MODEL}_{USE_FEATURE}.txt"
)
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"Modelo: {USE_MODEL}\n")
    f.write(f"Feature: {USE_FEATURE}\n\n")
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall   : {rec:.4f}\n")
    f.write(f"F1-score : {f1:.4f}\n")
    f.write(f"AUC-ROC  : {roc_auc:.4f}\n\n")
    f.write("Matriz de confusão:\n")
    f.write(str(cm) + "\n")

print(f"\nMétricas salvas em: {metrics_path}")

# ==========================
# GRÁFICOS
# ==========================

# 1) Matriz de confusão
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"])
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title(f"Confusion Matrix - {USE_MODEL.upper()} + {USE_FEATURE.upper()}")
cm_path = os.path.join(RESULTS_DIR, f"cm_{USE_MODEL}_{USE_FEATURE}.png")
plt.tight_layout()
plt.savefig(cm_path, dpi=200)
plt.close()
print(f"Matriz de confusão salva em: {cm_path}")

# 2) ROC curve
plt.figure(figsize=(4, 3))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", label="Aleatório")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"ROC - {USE_MODEL.upper()} + {USE_FEATURE.upper()}")
plt.legend(loc="lower right")
roc_path = os.path.join(RESULTS_DIR, f"roc_{USE_MODEL}_{USE_FEATURE}.png")
plt.tight_layout()
plt.savefig(roc_path, dpi=200)
plt.close()
print(f"ROC curve salva em: {roc_path}")

# 3) gráfico de barras (Accuracy, Precision, Recall, F1)
plt.figure(figsize=(4, 3))
names = ["Acc", "Prec", "Rec", "F1"]
values = [acc, prec, rec, f1]
sns.barplot(x=names, y=values)
plt.ylim(0, 1)
plt.title(f"Métricas - {USE_MODEL.upper()} + {USE_FEATURE.upper()}")
metrics_img_path = os.path.join(
    RESULTS_DIR, f"metrics_bar_{USE_MODEL}_{USE_FEATURE}.png"
)
plt.tight_layout()
plt.savefig(metrics_img_path, dpi=200)
plt.close()
print(f"Gráfico de métricas salvo em: {metrics_img_path}")
