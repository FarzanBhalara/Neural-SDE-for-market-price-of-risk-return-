import torch
import torch.nn as nn

class DiffusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, t, r):
        return self.net(torch.cat([t, r], dim=1))
