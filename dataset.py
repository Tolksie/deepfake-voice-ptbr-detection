import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, feature_dir, labels_csv):
        self.feature_dir = feature_dir
        self.labels = pd.read_csv(labels_csv)
        self.filenames = self.labels['filename'].tolist()
        self.targets = self.labels['label'].tolist()

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        label = self.targets[idx]

        path = os.path.join(self.feature_dir, fname.replace(".wav", ".npy"))
        feat = np.load(path)

        return feat.astype("float32"), label

    def __len__(self):
        return len(self.filenames)
