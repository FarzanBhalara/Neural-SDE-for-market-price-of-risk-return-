import torch
import torch.nn as nn


class LambdaNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=24, max_abs_lambda=0.25):
        super().__init__()
        self.max_abs_lambda = max_abs_lambda
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        raw = self.net(state)
        return self.max_abs_lambda * torch.tanh(raw)
