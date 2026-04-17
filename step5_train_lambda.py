import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import load_covariance_predictions, load_panel_artifact
from models.lambda_pipeline import evaluate_lambda_panel, predict_lambda_series, train_lambda_model


PANEL_FILE = "outputs/panel_step1.npz"
COV_FILE = "outputs/step2_covariance_predictions.npz"

CHECKPOINT_FILE = "outputs/step5_lambda_model.pt"
SERIES_CSV = "outputs/step5_lambda_series.csv"
METRICS_CSV = "outputs/step5_lambda_metrics.csv"
LAMBDA_PLOT = "outputs/step5_lambda_plot.png"

TARGET_HORIZON = 60
EPOCHS = 500
HIDDEN_DIM = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 40
SEED = 42

MAX_ABS_LAMBDA = 0.035
SHRINK_WEIGHT = 0.05
SMOOTH_WEIGHT = 0.01
LAMBDA_TARGET_WEIGHT = 2.0
CROSS_SECTION_WEIGHT = 0.0
LAMBDA_SMOOTH_HALFLIFE = 10
LAMBDA_CLIP_ZSCORE = 3.0

EPS = 1e-6


def plot_lambda_series(series_df):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(series_df["date"], series_df["lambda_pred"], linewidth=1.3)
    if "lambda_target" in series_df.columns:
        ax.plot(
            series_df["date"],
            series_df["lambda_target"],
            linewidth=1.0,
            alpha=0.7,
            label="implied lambda target",
        )
        ax.legend()
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.9)
    ax.set_title("Estimated Common Lambda")
    ax.set_xlabel("Date")
    ax.set_ylabel("Lambda")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(LAMBDA_PLOT, dpi=300)
    plt.close(fig)


def main():
    os.makedirs("outputs", exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    panel = load_panel_artifact(PANEL_FILE)
    covariance = load_covariance_predictions(COV_FILE)
    sigma = covariance["sigma_marginal"]
    beta = covariance["beta_market"]
    beta_valid_mask = covariance["valid_covariance_mask"].astype(bool)
    factor_sigma = covariance["factor_sigma"]
    idio_sigma = covariance["idio_sigma"]

    config = {
        "epochs": EPOCHS,
        "hidden_dim": HIDDEN_DIM,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "patience": PATIENCE,
        "shrink_weight": SHRINK_WEIGHT,
        "smooth_weight": SMOOTH_WEIGHT,
        "eps": EPS,
        "target_horizon": TARGET_HORIZON,
        "max_abs_lambda": MAX_ABS_LAMBDA,
        "lambda_target_weight": LAMBDA_TARGET_WEIGHT,
        "market_lambda_weight": 2.0,
        "market_lambda_halflife": 60,
        "market_lambda_min_periods": 30,
        "cross_section_weight": CROSS_SECTION_WEIGHT,
        "lambda_smooth_halflife": LAMBDA_SMOOTH_HALFLIFE,
        "lambda_clip_zscore": LAMBDA_CLIP_ZSCORE,
        "min_assets": 10,
    }

    model, _, summary = train_lambda_model(
        panel=panel,
        sigma_panel=sigma,
        beta_panel=beta,
        beta_valid_mask=beta_valid_mask,
        config=config,
        factor_sigma=factor_sigma,
        idio_sigma=idio_sigma,
    )
    pred = predict_lambda_series(
        model=model,
        panel=panel,
        sigma_panel=sigma,
        beta_panel=beta,
        config=config,
        feature_mean=summary["feature_mean"],
        feature_std=summary["feature_std"],
        feature_panel=summary["feature_panel"],
        factor_sigma=factor_sigma,
        idio_sigma=idio_sigma,
    )

    dates = pd.to_datetime(panel["dates"])
    series_df = pd.DataFrame(
        {
            "date": dates,
            "lambda_pred": pred["lambda_t"],
            "lambda_target": summary["lambda_target"],
            "train_flag": panel["train_date_mask"].astype(bool),
            "val_flag": panel["val_date_mask"].astype(bool),
            "test_flag": panel["test_date_mask"].astype(bool),
        }
    )
    series_df.to_csv(SERIES_CSV, index=False)
    plot_lambda_series(series_df)

    target_panel = panel[f"future_excess_mean_{TARGET_HORIZON}d"]
    valid_rows = summary["valid_rows"]
    metrics_rows = []
    split_to_flag = {
        "train": "train_flag",
        "val": "val_flag",
        "test": "test_flag",
    }
    split_to_mask = {
        "train": "train_date_mask",
        "val": "val_date_mask",
        "test": "test_date_mask",
    }
    for split_name, date_mask_key in split_to_mask.items():
        split_mask = valid_rows & panel[date_mask_key].astype(bool)[:, None]
        metrics = evaluate_lambda_panel(pred["lambda_t"], sigma, target_panel, split_mask)
        flag_col = split_to_flag[split_name]
        metrics_rows.append(
            {
                "split": split_name,
                **metrics,
                "lambda_mean": float(series_df.loc[series_df[flag_col], "lambda_pred"].mean()),
                "lambda_std": float(series_df.loc[series_df[flag_col], "lambda_pred"].std(ddof=0)),
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": int(summary["feature_panel"].shape[1]),
            "hidden_dim": HIDDEN_DIM,
            "summary": {
                "best_val_mse": summary["best_val_mse"],
                "best_epoch": summary["best_epoch"],
            },
            "feature_names": np.asarray(summary["feature_names"], dtype=object),
            "feature_mean": summary["feature_mean"],
            "feature_std": summary["feature_std"],
            "config": config,
        },
        CHECKPOINT_FILE,
    )

    print("Step 5 complete: common lambda trained.")
    print(f"Best validation loss: {summary['best_val_mse']:.8f}")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Saved lambda checkpoint to {CHECKPOINT_FILE}")
    print(f"Saved lambda series to {SERIES_CSV}")
    print(f"Saved lambda metrics to {METRICS_CSV}")
    print(f"Saved lambda plot to {LAMBDA_PLOT}")


if __name__ == "__main__":
    main()
