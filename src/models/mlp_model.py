import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=4):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512), # BatchNorm
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout with 50% probability

            nn.Linear(512, 256),
            nn.BatchNorm1d(256), # BatchNorm
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4), # Dropout with 40% probability

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)