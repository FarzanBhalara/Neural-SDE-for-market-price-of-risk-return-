import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from models.drift import DriftNet
from models.diffusion import DiffusionNet


WINDOWS_FILE = "outputs/step1_windows_W60.npz"
CHECKPOINT_FILE = "outputs/step2_neural_sde_weights.pt"
PROCESSED_CSV = "outputs/nifty50_step1_processed.csv"

NPZ_OUT = "outputs/step3_mu_sigma.npz"
CSV_OUT = "outputs/step3_mu_sigma.csv"
MU_PLOT_FILE = "outputs/mu_plot.png"
SIGMA_PLOT_FILE = "outputs/sigma_plot.png"

REALIZED_VOL_WINDOW = 20


def build_state_features(X, vol_window):
    recent = X[:, -vol_window:]
    recent_mean = recent.mean(dim=1, keepdim=True)
    recent_std = recent.std(dim=1, keepdim=True, unbiased=False)
    long_std = X.std(dim=1, keepdim=True, unbiased=False)
    abs_current = X[:, -1:].abs()

    return torch.cat([X, recent_mean, recent_std, long_std, abs_current], dim=1)


def load_artifacts(device):
    data = np.load(WINDOWS_FILE, allow_pickle=True)
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)

    X = torch.tensor(data["X"], dtype=torch.float32, device=device)
    num_samples = X.shape[0]
    denom = max(num_samples - 1, 1)
    t = torch.arange(num_samples, dtype=torch.float32, device=device).unsqueeze(1) / denom

    end_dates = pd.to_datetime(data["end_dates"])
    scaler_mean = float(np.asarray(data["scaler_mean"]).reshape(-1)[0])
    scaler_scale = float(np.asarray(data["scaler_scale"]).reshape(-1)[0])

    processed = pd.read_csv(PROCESSED_CSV, parse_dates=["Date"]).rename(columns={"Date": "date"})
    processed = processed.set_index("date")

    return data, checkpoint, X, t, end_dates, scaler_mean, scaler_scale, processed


def plot_mu(result_df):
    plt.figure(figsize=(11, 4.5))
    plt.plot(result_df["date"], result_df["mu"], linewidth=1.5)
    plt.title("Estimated Conditional Mean Return mu(t)")
    plt.xlabel("Date")
    plt.ylabel("Log return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MU_PLOT_FILE, dpi=300)
    plt.close()


def plot_sigma(result_df):
    plt.figure(figsize=(11, 4.5))
    plt.plot(result_df["date"], result_df["sigma"], label="Estimated sigma(t)", linewidth=1.5)

    realized = result_df["realized_sigma_20d"]
    if realized.notna().any():
        plt.plot(
            result_df["date"],
            realized,
            label=f"{REALIZED_VOL_WINDOW}-day realized sigma",
            linewidth=1.2,
            alpha=0.8,
        )
        plt.legend()

    plt.title("Estimated Volatility sigma(t)")
    plt.xlabel("Date")
    plt.ylabel("Log return volatility")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SIGMA_PLOT_FILE, dpi=300)
    plt.close()


def main():
    os.makedirs("outputs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, checkpoint, X, t, end_dates, scaler_mean, scaler_scale, processed = load_artifacts(device)

    state_dim = int(checkpoint["state_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    dt = float(checkpoint["dt"])
    vol_window = int(checkpoint["vol_feature_window"])
    state = build_state_features(X, vol_window)

    drift_net = DriftNet(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    diffusion_net = DiffusionNet(state_dim=state_dim, hidden_dim=hidden_dim).to(device)

    drift_net.load_state_dict(checkpoint["drift_state_dict"])
    diffusion_net.load_state_dict(checkpoint["diffusion_state_dict"])
    drift_net.eval()
    diffusion_net.eval()

    with torch.no_grad():
        drift_z = drift_net(t, state).squeeze(1).cpu().numpy()
        sigma_z = diffusion_net(t, state).squeeze(1).cpu().numpy()
        current_z = X[:, -1].cpu().numpy()

    mu_z = current_z + drift_z * dt
    mu_raw = scaler_mean + scaler_scale * mu_z
    sigma_model_raw = scaler_scale * sigma_z
    drift_raw = scaler_scale * drift_z

    current_logret = processed.reindex(end_dates)["logret"].to_numpy()
    realized_sigma = processed["logret"].rolling(REALIZED_VOL_WINDOW).std().reindex(end_dates).to_numpy()

    valid = np.isfinite(realized_sigma) & np.isfinite(sigma_model_raw) & (sigma_model_raw > 0)
    sigma_scale = float(
        np.dot(sigma_model_raw[valid], realized_sigma[valid]) /
        np.dot(sigma_model_raw[valid], sigma_model_raw[valid])
    )
    sigma_raw = sigma_model_raw * sigma_scale

    result_df = pd.DataFrame(
        {
            "date": end_dates,
            "current_logret": current_logret,
            "mu": mu_raw,
            "sigma": sigma_raw,
            "sigma_model": sigma_model_raw,
            "drift": drift_raw,
            f"realized_sigma_{REALIZED_VOL_WINDOW}d": realized_sigma,
        }
    )
    result_df["realized_sigma_20d"] = result_df[f"realized_sigma_{REALIZED_VOL_WINDOW}d"]
    result_df.to_csv(CSV_OUT, index=False)

    np.savez(
        NPZ_OUT,
        mu=mu_raw,
        sigma=sigma_raw,
        sigma_model=sigma_model_raw,
        drift=drift_raw,
        dates=result_df["date"].astype(str).to_numpy(),
    )

    plot_mu(result_df)
    plot_sigma(result_df)

    sigma_corr = pd.Series(result_df["sigma"]).corr(pd.Series(result_df["realized_sigma_20d"]))

    print("Step 3 complete: mu(t) and sigma(t) estimated.")
    print(f"Saved estimates to {CSV_OUT} and {NPZ_OUT}")
    print(f"Saved plots to {MU_PLOT_FILE} and {SIGMA_PLOT_FILE}")
    print(f"Sigma calibration factor: {sigma_scale:.4f}")
    print(f"Correlation with {REALIZED_VOL_WINDOW}-day realized sigma: {sigma_corr:.4f}")


if __name__ == "__main__":
    main()
