import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from models.drift import DriftNet
from models.diffusion import DiffusionNet


WINDOWS_FILE = "outputs/step1_windows_W60.npz"
CHECKPOINT_FILE = "outputs/step2_neural_sde_weights.pt"
LOSS_PLOT_FILE = "outputs/loss_plot.png"

DT = 1.0
EPOCHS = 600
BATCH_SIZE = 256
HIDDEN_DIM = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SEED = 42
VOL_FEATURE_WINDOW = 20
VOL_LOSS_WEIGHT = 0.5


def build_state_features(X, vol_window):
    recent = X[:, -vol_window:]
    recent_mean = recent.mean(dim=1, keepdim=True)
    recent_std = recent.std(dim=1, keepdim=True, unbiased=False)
    long_std = X.std(dim=1, keepdim=True, unbiased=False)
    abs_current = X[:, -1:].abs()

    state = torch.cat([X, recent_mean, recent_std, long_std, abs_current], dim=1)
    return state, recent_std.clamp_min(1e-4)


def load_training_tensors(device):
    data = np.load(WINDOWS_FILE, allow_pickle=True)

    X = torch.tensor(data["X"], dtype=torch.float32, device=device)
    y_next = torch.tensor(data["y"], dtype=torch.float32, device=device).unsqueeze(1)
    current = X[:, -1].unsqueeze(1)

    vol_window = min(VOL_FEATURE_WINDOW, X.shape[1])
    state, sigma_target = build_state_features(X, vol_window)

    num_samples = X.shape[0]
    denom = max(num_samples - 1, 1)
    t = torch.arange(num_samples, dtype=torch.float32, device=device).unsqueeze(1) / denom

    return data, state, y_next, current, t, sigma_target, vol_window


def gaussian_increment_nll(y_next, current, drift, sigma, dt):
    delta = y_next - current
    mean_delta = drift * dt
    variance = sigma.pow(2) * dt + 1e-6

    return 0.5 * ((delta - mean_delta).pow(2) / variance + torch.log(variance))


def plot_loss(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Average Gaussian NLL")
    plt.title("Step 2 Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_FILE, dpi=300)
    plt.close()


def main():
    os.makedirs("outputs", exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    data, state, y_next, current, t, sigma_target, vol_window = load_training_tensors(device)

    window = int(np.asarray(data["window"]).item())
    scaler_scale = float(np.asarray(data["scaler_scale"]).reshape(-1)[0])
    state_dim = state.shape[1]

    drift_net = DriftNet(state_dim=state_dim, hidden_dim=HIDDEN_DIM).to(device)
    diffusion_net = DiffusionNet(state_dim=state_dim, hidden_dim=HIDDEN_DIM).to(device)

    optimizer = optim.Adam(
        list(drift_net.parameters()) + list(diffusion_net.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    num_samples = state.shape[0]
    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0

        for start in range(0, num_samples, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]

            drift = drift_net(t[idx], state[idx])
            sigma = diffusion_net(t[idx], state[idx])

            nll_loss = gaussian_increment_nll(y_next[idx], current[idx], drift, sigma, DT).mean()
            vol_loss = (
                torch.log(sigma) - torch.log(sigma_target[idx])
            ).pow(2).mean()
            loss = nll_loss + VOL_LOSS_WEIGHT * vol_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(drift_net.parameters()) + list(diffusion_net.parameters()),
                max_norm=GRAD_CLIP,
            )
            optimizer.step()

            epoch_loss += loss.item() * idx.numel()

        epoch_loss /= num_samples
        loss_history.append(epoch_loss)

        if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:4d}/{EPOCHS} | avg_nll = {epoch_loss:.6f}")

    plot_loss(loss_history)
    print(f"Loss plot saved to {LOSS_PLOT_FILE}")

    with torch.no_grad():
        drift_fit = drift_net(t, state)
        sigma_fit = diffusion_net(t, state)
        mu_fit = current + drift_fit * DT

    torch.save(
        {
            "drift_state_dict": drift_net.state_dict(),
            "diffusion_state_dict": diffusion_net.state_dict(),
            "window": window,
            "state_dim": state_dim,
            "hidden_dim": HIDDEN_DIM,
            "dt": DT,
            "vol_feature_window": vol_window,
            "scaler_scale": scaler_scale,
            "mu_fit_mean_z": float(mu_fit.mean().item()),
            "sigma_fit_mean_z": float(sigma_fit.mean().item()),
        },
        CHECKPOINT_FILE,
    )
    print(f"Saved trained weights to {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
