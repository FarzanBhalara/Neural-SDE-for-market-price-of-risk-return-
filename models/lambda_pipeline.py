from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.features import build_lambda_date_features
from models.lambda_model import LambdaNet


def build_lambda_targets(panel, horizon):
    key = f"future_excess_mean_{horizon}d"
    mask_key = f"valid_lambda_{horizon}d_mask"
    return panel[key], panel[mask_key].astype(bool)


def build_lambda_target_series(beta_panel, target_mu, sigma_panel, valid_mask, eps=1e-6, min_assets=10):
    beta = np.asarray(beta_panel, dtype=float)
    target = np.asarray(target_mu, dtype=float)
    sigma = np.asarray(sigma_panel, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)

    lambda_target = np.full(beta.shape[0], np.nan, dtype=np.float32)
    for t in range(beta.shape[0]):
        mask = valid[t] & np.isfinite(beta[t]) & np.isfinite(target[t]) & np.isfinite(sigma[t]) & (sigma[t] > 0)
        if int(mask.sum()) < min_assets:
            continue
        weights = 1.0 / np.clip(np.square(sigma[t, mask]), eps, None)
        numer = np.sum(weights * beta[t, mask] * target[t, mask])
        denom = np.sum(weights * np.square(beta[t, mask]))
        if denom > 0:
            lambda_target[t] = float(numer / denom)
    return lambda_target


def masked_cross_sectional_mu_loss(lambda_t, beta_panel, target_mu, sigma_panel, valid_mask, eps=1e-6):
    safe_beta = torch.nan_to_num(beta_panel, nan=0.0, posinf=0.0, neginf=0.0)
    safe_target = torch.nan_to_num(target_mu, nan=0.0, posinf=0.0, neginf=0.0)
    safe_sigma = torch.nan_to_num(sigma_panel, nan=1.0, posinf=1.0, neginf=1.0)

    mu_pred = safe_beta * lambda_t.unsqueeze(1)
    weights = torch.reciprocal(torch.clamp(safe_sigma.pow(2), min=eps))
    errors = F.smooth_l1_loss(mu_pred, safe_target, reduction="none")
    masked_weights = weights[valid_mask]
    masked_errors = errors[valid_mask]
    if masked_errors.numel() == 0:
        return torch.zeros((), device=lambda_t.device), mu_pred
    masked_weights = masked_weights / masked_weights.mean().clamp_min(eps)
    masked_weights = torch.clamp(masked_weights, min=0.25, max=4.0)
    loss = torch.mean(masked_weights * masked_errors)
    return loss, mu_pred


def _scale_date_features(features, train_mask, feature_mean=None, feature_std=None):
    if feature_mean is None or feature_std is None:
        train_rows = features[np.asarray(train_mask, dtype=bool)]
        feature_mean = np.nanmean(train_rows, axis=0)
        feature_std = np.nanstd(train_rows, axis=0)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    scaled = (features - feature_mean) / feature_std
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return scaled, feature_mean.astype(np.float32), feature_std.astype(np.float32)


def _compute_lambda_losses(model, feature_tensor, beta_tensor, sigma_tensor, target_tensor, row_mask, date_mask, config):
    lambda_pred = model(feature_tensor).squeeze(1)
    mse, mu_pred = masked_cross_sectional_mu_loss(
        lambda_t=lambda_pred,
        beta_panel=beta_tensor,
        target_mu=target_tensor,
        sigma_panel=sigma_tensor,
        valid_mask=row_mask,
        eps=config["eps"],
    )

    if date_mask.any():
        shrink = torch.mean(lambda_pred[date_mask].pow(2))
    else:
        shrink = torch.zeros((), device=lambda_pred.device)

    pair_mask = date_mask[1:] & date_mask[:-1]
    if pair_mask.any():
        smooth = torch.mean((lambda_pred[1:] - lambda_pred[:-1])[pair_mask].pow(2))
    else:
        smooth = torch.zeros((), device=lambda_pred.device)

    total = mse + config["shrink_weight"] * shrink + config["smooth_weight"] * smooth
    return {
        "total": total,
        "mse": mse,
        "shrink": shrink,
        "smooth": smooth,
        "lambda_pred": lambda_pred,
        "mu_pred": mu_pred,
    }


def train_lambda_model(
    panel,
    sigma_panel,
    beta_panel,
    beta_valid_mask,
    config,
    train_date_mask=None,
    val_date_mask=None,
    factor_sigma=None,
    idio_sigma=None,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    target_panel, valid_target_mask = build_lambda_targets(panel, horizon=int(config["target_horizon"]))
    valid_rows = valid_target_mask & np.asarray(beta_valid_mask, dtype=bool) & np.isfinite(sigma_panel) & (sigma_panel > 0)
    lambda_target = build_lambda_target_series(
        beta_panel=beta_panel,
        target_mu=target_panel,
        sigma_panel=sigma_panel,
        valid_mask=valid_rows,
        eps=config["eps"],
        min_assets=int(config.get("min_assets", 10)),
    )

    if train_date_mask is None:
        train_date_mask = panel["train_date_mask"].astype(bool)
    else:
        train_date_mask = np.asarray(train_date_mask, dtype=bool)
    if val_date_mask is None:
        val_date_mask = panel["val_date_mask"].astype(bool)
    else:
        val_date_mask = np.asarray(val_date_mask, dtype=bool)

    train_rows = valid_rows & train_date_mask[:, None]
    val_rows = valid_rows & val_date_mask[:, None]
    train_feature_mask = train_rows.any(axis=1)
    train_lambda_dates = train_feature_mask & np.isfinite(lambda_target)
    val_lambda_dates = val_rows.any(axis=1) & np.isfinite(lambda_target)

    feature_panel, feature_names = build_lambda_date_features(
        panel,
        sigma_panel=sigma_panel,
        beta_panel=beta_panel,
        factor_sigma=factor_sigma,
        idio_sigma=idio_sigma,
    )
    scaled_features, feature_mean, feature_std = _scale_date_features(
        feature_panel,
        train_mask=train_feature_mask,
        feature_mean=config.get("feature_mean"),
        feature_std=config.get("feature_std"),
    )

    feature_tensor = torch.tensor(scaled_features, dtype=torch.float32, device=device)
    beta_tensor = torch.tensor(beta_panel, dtype=torch.float32, device=device)
    sigma_tensor = torch.tensor(sigma_panel, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target_panel, dtype=torch.float32, device=device)
    train_row_mask = torch.tensor(train_rows, dtype=torch.bool, device=device)
    val_row_mask = torch.tensor(val_rows, dtype=torch.bool, device=device)
    train_date_tensor = torch.tensor(train_feature_mask, dtype=torch.bool, device=device)
    val_date_tensor = torch.tensor(val_rows.any(axis=1), dtype=torch.bool, device=device)
    lambda_target_tensor = torch.tensor(np.nan_to_num(lambda_target, nan=0.0), dtype=torch.float32, device=device)
    train_lambda_tensor = torch.tensor(train_lambda_dates, dtype=torch.bool, device=device)
    val_lambda_tensor = torch.tensor(val_lambda_dates, dtype=torch.bool, device=device)

    model = LambdaNet(state_dim=feature_tensor.shape[1], hidden_dim=config["hidden_dim"], max_abs_lambda=config["max_abs_lambda"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    patience_left = int(config["patience"])

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        train_metrics = _compute_lambda_losses(
            model=model,
            feature_tensor=feature_tensor,
            beta_tensor=beta_tensor,
            sigma_tensor=sigma_tensor,
            target_tensor=target_tensor,
            row_mask=train_row_mask,
            date_mask=train_date_tensor,
            config=config,
        )
        if train_lambda_tensor.any():
            lambda_target_loss = torch.mean((train_metrics["lambda_pred"][train_lambda_tensor] - lambda_target_tensor[train_lambda_tensor]).pow(2))
        else:
            lambda_target_loss = torch.zeros((), device=device)
        train_metrics["lambda_target_loss"] = lambda_target_loss
        train_metrics["total"] = (
            config.get("lambda_target_weight", 1.0) * lambda_target_loss
            + config.get("cross_section_weight", 0.25) * train_metrics["mse"]
            + config["shrink_weight"] * train_metrics["shrink"]
            + config["smooth_weight"] * train_metrics["smooth"]
        )
        train_metrics["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip"])
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_metrics = _compute_lambda_losses(
                model=model,
                feature_tensor=feature_tensor,
                beta_tensor=beta_tensor,
                sigma_tensor=sigma_tensor,
                target_tensor=target_tensor,
                row_mask=val_row_mask,
                date_mask=val_date_tensor,
                config=config,
            )
            if val_lambda_tensor.any():
                val_lambda_loss = torch.mean((val_metrics["lambda_pred"][val_lambda_tensor] - lambda_target_tensor[val_lambda_tensor]).pow(2))
            else:
                val_lambda_loss = torch.zeros((), device=device)
            val_metrics["lambda_target_loss"] = val_lambda_loss
            val_metrics["total"] = (
                config.get("lambda_target_weight", 1.0) * val_lambda_loss
                + config.get("cross_section_weight", 0.25) * val_metrics["mse"]
                + config["shrink_weight"] * val_metrics["shrink"]
                + config["smooth_weight"] * val_metrics["smooth"]
            )

        history.append(
            {
                "epoch": epoch,
                "train_total": float(train_metrics["total"].item()),
                "train_mse": float(train_metrics["mse"].item()),
                "train_lambda_target": float(train_metrics["lambda_target_loss"].item()),
                "train_shrink": float(train_metrics["shrink"].item()),
                "train_smooth": float(train_metrics["smooth"].item()),
                "val_total": float(val_metrics["total"].item()),
                "val_mse": float(val_metrics["mse"].item()),
                "val_lambda_target": float(val_metrics["lambda_target_loss"].item()),
                "val_shrink": float(val_metrics["shrink"].item()),
                "val_smooth": float(val_metrics["smooth"].item()),
            }
        )

        if val_metrics["total"].item() < best_val:
            best_val = float(val_metrics["total"].item())
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_left = int(config["patience"])
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    model.load_state_dict(best_state)
    return model, history, {
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "feature_panel": feature_panel,
        "feature_names": feature_names,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "lambda_target": lambda_target,
        "valid_rows": valid_rows,
        "train_rows": train_rows,
        "val_rows": val_rows,
    }


def predict_lambda_series(
    model,
    panel,
    sigma_panel,
    beta_panel,
    config,
    feature_mean,
    feature_std,
    feature_panel=None,
    factor_sigma=None,
    idio_sigma=None,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if feature_panel is None:
        feature_panel, _ = build_lambda_date_features(
            panel,
            sigma_panel=sigma_panel,
            beta_panel=beta_panel,
            factor_sigma=factor_sigma,
            idio_sigma=idio_sigma,
        )

    scaled_features, _, _ = _scale_date_features(
        feature_panel,
        train_mask=np.ones(feature_panel.shape[0], dtype=bool),
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    feature_tensor = torch.tensor(scaled_features, dtype=torch.float32, device=device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        lambda_series = model(feature_tensor).squeeze(1).cpu().numpy().astype(np.float32)

    mu_excess = (beta_panel * lambda_series[:, None]).astype(np.float32)
    return {
        "lambda_t": lambda_series,
        "mu_excess": mu_excess,
        "feature_panel": feature_panel,
        "scaled_features": scaled_features,
    }


def evaluate_lambda_panel(lambda_t, beta_panel, target_mu, valid_mask):
    pred = beta_panel * lambda_t[:, None]
    valid = np.asarray(valid_mask, dtype=bool)
    actual = np.asarray(target_mu, dtype=float)[valid]
    predicted = np.asarray(pred, dtype=float)[valid]

    if actual.size == 0:
        return {
            "count": 0,
            "corr": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
        }

    errors = predicted - actual
    return {
        "count": int(actual.size),
        "corr": float(pd.Series(actual).corr(pd.Series(predicted))),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }
