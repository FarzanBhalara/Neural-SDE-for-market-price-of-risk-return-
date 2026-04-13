import numpy as np
import pandas as pd


def rolling_volatility(series, window):
    return series.rolling(window).std(ddof=0)


def ewma_volatility(series, decay=0.94):
    squared = series.pow(2)
    ewma_var = squared.ewm(alpha=1.0 - decay, adjust=False).mean()
    return np.sqrt(ewma_var)


def rolling_rms_volatility(series, window):
    return np.sqrt(series.pow(2).rolling(window, min_periods=window).mean())


def rolling_panel_volatility(excess_panel, window):
    frame = pd.DataFrame(excess_panel)
    return np.sqrt(frame.pow(2).rolling(window, min_periods=window).mean()).to_numpy(dtype=np.float32)


def ewma_panel_volatility(excess_panel, decay=0.94):
    frame = pd.DataFrame(excess_panel)
    ewma_var = frame.pow(2).ewm(alpha=1.0 - decay, adjust=False).mean()
    return np.sqrt(ewma_var).to_numpy(dtype=np.float32)


def qlike_loss(realized_vol, predicted_vol, eps=1e-8):
    realized_var = np.square(np.asarray(realized_vol, dtype=float))
    predicted_var = np.square(np.asarray(predicted_vol, dtype=float)).clip(min=eps)
    return np.mean(np.log(predicted_var) + realized_var / predicted_var)


def sigma_metrics(realized_vol, predicted_vol):
    realized = pd.Series(realized_vol, dtype=float)
    predicted = pd.Series(predicted_vol, dtype=float)
    valid = realized.notna() & predicted.notna() & (predicted > 0)
    realized = realized[valid]
    predicted = predicted[valid]

    if realized.empty:
        return {
            "count": 0,
            "corr": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "qlike": np.nan,
        }

    errors = predicted - realized
    return {
        "count": int(valid.sum()),
        "corr": float(realized.corr(predicted)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "qlike": float(qlike_loss(realized, predicted)),
    }


def standardized_residual_diagnostics(next_returns, sigma):
    next_returns = pd.Series(next_returns, dtype=float)
    sigma = pd.Series(sigma, dtype=float)
    valid = next_returns.notna() & sigma.notna() & (sigma > 0)
    residual = next_returns[valid] / sigma[valid]
    squared = residual.pow(2)
    lagged_squared = squared.shift(1)

    if residual.empty:
        return {
            "count": 0,
            "resid_mean": np.nan,
            "resid_std": np.nan,
            "sq_lag1_corr": np.nan,
        }

    return {
        "count": int(valid.sum()),
        "resid_mean": float(residual.mean()),
        "resid_std": float(residual.std(ddof=0)),
        "sq_lag1_corr": float(squared.corr(lagged_squared)),
    }


def pooled_sigma_metrics(realized_panel, predicted_panel, valid_mask):
    valid = np.asarray(valid_mask, dtype=bool)
    return sigma_metrics(
        np.asarray(realized_panel, dtype=float)[valid],
        np.asarray(predicted_panel, dtype=float)[valid],
    )


def pooled_residual_diagnostics(next_return_panel, sigma_panel, valid_mask):
    valid = np.asarray(valid_mask, dtype=bool)
    return standardized_residual_diagnostics(
        np.asarray(next_return_panel, dtype=float)[valid],
        np.asarray(sigma_panel, dtype=float)[valid],
    )


def per_asset_sigma_metrics(realized_panel, predicted_panel, next_return_panel, valid_mask, asset_ids):
    rows = []
    realized = np.asarray(realized_panel, dtype=float)
    predicted = np.asarray(predicted_panel, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)

    for idx, asset in enumerate(asset_ids):
        asset_mask = mask[:, idx]
        base = sigma_metrics(realized[:, idx][asset_mask], predicted[:, idx][asset_mask])
        diag = standardized_residual_diagnostics(next_return_panel[:, idx][asset_mask], predicted[:, idx][asset_mask])
        rows.append({"asset_id": str(asset), **base, **diag})
    return pd.DataFrame(rows)


def panel_baselines(excess_return_panel, valid_sigma_mask, sigma_target_panel, next_return_panel):
    rolling20 = rolling_panel_volatility(excess_return_panel, 20)
    ewma = ewma_panel_volatility(excess_return_panel, decay=0.94)

    outputs = {}
    for name, values in {"rolling20": rolling20, "ewma": ewma}.items():
        outputs[name] = {
            "sigma": values,
            "metrics": pooled_sigma_metrics(sigma_target_panel, values, valid_sigma_mask),
            "diagnostics": pooled_residual_diagnostics(next_return_panel, values, valid_sigma_mask),
        }
    return outputs
