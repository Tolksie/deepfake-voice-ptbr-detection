import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, 128, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = x.squeeze(1)       
        x = x.permute(0, 2, 1) # (batch, seq=300, features=input_size)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)
