import torch
import torch.nn as nn


class VolatilityNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, min_log_var=-12.0, max_log_var=4.0):
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        log_var = self.net(state)
        return torch.clamp(log_var, min=self.min_log_var, max=self.max_log_var)

    def sigma(self, state):
        return torch.exp(0.5 * self.forward(state))
