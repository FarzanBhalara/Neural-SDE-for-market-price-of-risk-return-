import torch
import torch.nn as nn


class LambdaNet(nn.Module):
    """
    Neural market Sharpe ratio estimator.

    Produces a scalar lambda_t per time step:
        lambda_t = prior + max_abs_lambda * tanh(MLP(state_t))
    where lambda_t is interpreted as a daily price of risk per unit
    volatility, so asset-level excess return is mu_i,t = sigma_i,t * lambda_t.

    Key design choices:
    - `max_abs_lambda` is calibrated to daily Sharpe-ratio scale.
    - A learnable `prior` parameter captures the unconditional Sharpe level.
    - Dropout layers prevent the MLP from memorising target noise.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 32,
        max_abs_lambda: float = 0.06,
    ):
        super().__init__()
        self.max_abs_lambda = float(max_abs_lambda)
        # Learnable unconditional daily Sharpe level.
        self.prior = nn.Parameter(torch.tensor([0.003], dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.net(state)
        return torch.clamp(
            self.prior + self.max_abs_lambda * torch.tanh(raw),
            min=-0.04,
            max=0.06,
        )
