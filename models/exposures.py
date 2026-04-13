import numpy as np
import pandas as pd


def compute_rolling_market_beta(excess_return, market_return, valid_mask, window=60, eps=1e-8):
    asset_df = pd.DataFrame(excess_return)
    market = pd.Series(market_return, index=asset_df.index, dtype=float)
    valid_df = pd.DataFrame(valid_mask, index=asset_df.index, columns=asset_df.columns).astype(bool)

    beta = pd.DataFrame(np.nan, index=asset_df.index, columns=asset_df.columns, dtype=float)
    market_var = market.rolling(window, min_periods=window).var(ddof=0)

    for column in asset_df.columns:
        asset = asset_df[column].where(valid_df[column])
        cov = asset.rolling(window, min_periods=window).cov(market)
        beta[column] = cov / market_var.replace(0.0, np.nan)

    beta_valid = beta.notna().to_numpy(dtype=bool)
    return beta.to_numpy(dtype=np.float32), beta_valid


def smooth_market_beta(beta, valid_mask, halflife=20, shrink_target=1.0, shrink_weight=0.15):
    beta_df = pd.DataFrame(np.asarray(beta, dtype=float))
    valid_df = pd.DataFrame(np.asarray(valid_mask, dtype=bool), index=beta_df.index, columns=beta_df.columns)
    beta_df = beta_df.where(valid_df)

    smoothed = beta_df.ewm(halflife=halflife, adjust=False, ignore_na=True).mean()
    if shrink_weight > 0.0:
        smoothed = (1.0 - shrink_weight) * smoothed + shrink_weight * float(shrink_target)

    return smoothed.where(valid_df).to_numpy(dtype=np.float32)


def clip_beta(beta, lower=-5.0, upper=5.0):
    return np.clip(np.asarray(beta, dtype=np.float32), lower, upper)


def build_beta_mask(beta, valid_mask):
    return np.asarray(valid_mask, dtype=bool) & np.isfinite(beta)
