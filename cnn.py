import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # transforma QUALQUER shape para (128, 4, 4)
        self.adapt = nn.AdaptiveAvgPool2d((4,4))

        self.fc = nn.Sequential(
            nn.Linear(128*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
