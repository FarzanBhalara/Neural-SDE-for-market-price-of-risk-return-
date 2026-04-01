import torch
import torch.nn as nn

class DiffusionNet(nn.Module):
    def __init__(self, state_dim=60, hidden_dim=64, min_sigma=1e-4):
        super().__init__()
        input_dim = state_dim + 1  # rolling state window + normalized time
        self.min_sigma = min_sigma
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, t, state):
        return self.net(torch.cat([t, state], dim=1)) + self.min_sigma
