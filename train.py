import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from cnn import CNNClassifier
from rnn import RNNClassifier
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

USE_MODEL = "rnn"     # cnn ou rnn
USE_FEATURE = "stft"  # mel ou stft

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("usando device:", DEVICE)

feature_dir = f"data/features/{USE_FEATURE}"
labels_csv = "data/dataset_prepared/labels.csv"

dataset = AudioDataset(feature_dir, labels_csv)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# pegar shape automaticamente
sample = np.load(os.path.join(feature_dir, os.listdir(feature_dir)[0]))
H, W = sample.shape
print("shape do feature:", (H, W))

# criar modelo correto
if USE_MODEL == "cnn":
    model = CNNClassifier()
else:
    model = RNNClassifier(input_size=H)

model = model.to(DEVICE)

crit = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10):
    total, correct, loss_sum = 0, 0, 0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        X = X.unsqueeze(1)

        out = model(X)
        loss = crit(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_sum += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    print(f"Epoch {epoch+1}, Acc: {correct/total:.4f}, Loss: {loss_sum:.4f}")

torch.save(model.state_dict(), f"model_{USE_MODEL}_{USE_FEATURE}.pt")
