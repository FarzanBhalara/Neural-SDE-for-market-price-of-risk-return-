import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_covariance_predictions, load_panel_artifact


PANEL_FILE = "outputs/panel_step1.npz"
COV_FILE = "outputs/step2_covariance_predictions.npz"
LAMBDA_SERIES_CSV = "outputs/step5_lambda_series.csv"

OUT_CSV = "outputs/step6_market_params_panel.csv"
OUT_NPZ = "outputs/step6_market_params_panel.npz"
MU_PLOT = "outputs/step6_mu_panel_plot.png"

TARGET_HORIZON = 60


def masked_row_mean(values, valid_mask):
    values = np.asarray(values, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    masked = np.where(mask, values, 0.0)
    counts = mask.sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=float)
    nonzero = counts > 0
    out[nonzero] = masked[nonzero].sum(axis=1) / counts[nonzero]
    return out


def plot_mu_panel(dates, valid_mask, mu_excess, realized_target, date_mask):
    mean_pred = masked_row_mean(mu_excess, valid_mask)
    mean_realized = masked_row_mean(realized_target, valid_mask)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dates[date_mask], mean_pred[date_mask], label="predicted mean mu_excess", linewidth=1.4)
    ax.plot(dates[date_mask], mean_realized[date_mask], label="future mean excess return", linewidth=1.2, alpha=0.85)
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.9)
    ax.set_title("Held-Out Mu: Cross-Sectional Mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily excess return")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(MU_PLOT, dpi=300)
    plt.close(fig)


def main():
    os.makedirs("outputs", exist_ok=True)

    panel = load_panel_artifact(PANEL_FILE)
    covariance = load_covariance_predictions(COV_FILE)
    sigma = covariance["sigma_marginal"]
    beta = covariance["beta_market"]
    beta_valid_mask = covariance["valid_covariance_mask"].astype(bool)
    factor_sigma = covariance["factor_sigma"]
    idio_sigma = covariance["idio_sigma"]
    factor_var = covariance["factor_var"]
    idio_var = covariance["idio_var"]
    lambda_df = pd.read_csv(LAMBDA_SERIES_CSV, parse_dates=["date"])

    dates = pd.to_datetime(panel["dates"])
    lambda_series = lambda_df.set_index("date").reindex(dates)["lambda_pred"].to_numpy(dtype=np.float32)
    mu_excess = (sigma * lambda_series[:, None]).astype(np.float32)
    mu_total = (panel["risk_free"][:, None] + mu_excess).astype(np.float32)
    final_valid_mask = (
        panel[f"valid_lambda_{TARGET_HORIZON}d_mask"].astype(bool)
        & beta_valid_mask
        & np.isfinite(sigma)
        & (sigma > 0)
    )

    plot_mu_panel(
        dates=dates,
        valid_mask=final_valid_mask & panel["test_date_mask"].astype(bool)[:, None],
        mu_excess=mu_excess,
        realized_target=panel[f"future_excess_mean_{TARGET_HORIZON}d"],
        date_mask=panel["test_date_mask"].astype(bool),
    )

    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), len(panel["asset_ids"])),
            "asset_id": np.tile(panel["asset_ids"].astype(str), len(dates)),
            "sigma": sigma.reshape(-1),
            "beta_market": beta.reshape(-1),
            "lambda_t": np.repeat(lambda_series, len(panel["asset_ids"])),
            "mu_excess": mu_excess.reshape(-1),
            "mu_total": mu_total.reshape(-1),
            "future_excess_mean_60d": panel[f"future_excess_mean_{TARGET_HORIZON}d"].reshape(-1),
            "final_valid": final_valid_mask.reshape(-1),
        }
    )
    frame.to_csv(OUT_CSV, index=False)

    np.savez(
        OUT_NPZ,
        dates=panel["dates"],
        asset_ids=panel["asset_ids"],
        sigma=sigma.astype(np.float32),
        factor_sigma=factor_sigma.astype(np.float32),
        factor_var=factor_var.astype(np.float32),
        idio_sigma=idio_sigma.astype(np.float32),
        idio_var=idio_var.astype(np.float32),
        beta_market=beta.astype(np.float32),
        lambda_t=lambda_series.astype(np.float32),
        mu_excess=mu_excess.astype(np.float32),
        mu_total=mu_total.astype(np.float32),
        final_valid_mask=final_valid_mask.astype(bool),
    )

    print("Step 6 complete: market parameter panel exported.")
    print(f"Saved panel CSV to {OUT_CSV}")
    print(f"Saved panel NPZ to {OUT_NPZ}")
    print(f"Saved mu plot to {MU_PLOT}")


if __name__ == "__main__":
    main()
