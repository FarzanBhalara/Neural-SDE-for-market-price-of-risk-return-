from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-8


def load_member_price_panel(path):
    return pd.read_csv(path, parse_dates=["date", "snapshot_date_used", "snapshot_date"])


def _latest_asset_metadata(df):
    meta = (
        df.sort_values(["date", "nse_symbol"])
        .drop_duplicates("nse_symbol", keep="last")
        .set_index("nse_symbol")
    )
    return meta


def build_panel_matrices(df, calendar_dates=None):
    frame = df.copy()
    frame["member_value"] = 1.0

    if calendar_dates is None:
        dates = np.sort(frame["date"].dropna().unique())
    else:
        dates = pd.to_datetime(pd.Index(calendar_dates).unique()).sort_values().to_numpy()
    assets = np.sort(frame["nse_symbol"].dropna().unique())

    adj_close = (
        frame.pivot_table(index="date", columns="nse_symbol", values="adj_close", aggfunc="last")
        .reindex(index=dates, columns=assets)
        .astype(float)
    )
    member_mask = (
        frame.pivot_table(index="date", columns="nse_symbol", values="member_value", aggfunc="max", fill_value=0.0)
        .reindex(index=dates, columns=assets, fill_value=0.0)
        .astype(bool)
    )
    price_mask = adj_close.notna()

    meta = _latest_asset_metadata(frame)
    security_names = pd.Index(assets).map(meta["security_name"]).fillna("").to_numpy(dtype=object)
    industries = pd.Index(assets).map(meta["industry"]).fillna("").to_numpy(dtype=object)

    return {
        "dates": pd.to_datetime(dates),
        "asset_ids": assets.astype(object),
        "security_names": security_names,
        "industries": industries,
        "adj_close_df": adj_close,
        "member_mask_df": member_mask,
        "price_mask_df": price_mask,
    }


def compute_log_returns(adj_close_df):
    log_price = np.log(adj_close_df)
    log_return = log_price.diff()
    valid = adj_close_df.notna() & adj_close_df.shift(1).notna()
    log_return = log_return.where(valid)
    return log_return


def compute_excess_returns(log_return_df, risk_free_series):
    aligned_rf = risk_free_series.reindex(log_return_df.index).astype(float)
    excess = log_return_df.sub(aligned_rf, axis=0)
    return excess


def _rolling_full_mask(valid_df, window):
    counts = valid_df.astype(float).rolling(window, min_periods=window).sum()
    return counts.eq(float(window))


def _future_full_mask(valid_df, horizon):
    future = valid_df.shift(-1)
    future_counts = future.iloc[::-1].astype(float).rolling(horizon, min_periods=horizon).sum().iloc[::-1]
    return future_counts.eq(float(horizon))


def build_membership_masks(member_mask_df, price_mask_df, return_valid_df, lookback, horizons):
    lookback_mask = _rolling_full_mask(return_valid_df, lookback) & member_mask_df & price_mask_df
    masks = {
        f"lookback_{lookback}_mask": lookback_mask,
    }

    next_mask = return_valid_df.shift(-1)
    next_mask = next_mask.where(next_mask.notna(), False).astype(bool) & member_mask_df
    masks["next_return_mask"] = next_mask

    for horizon in horizons:
        forward_mask = _future_full_mask(return_valid_df, horizon) & member_mask_df
        masks[f"forward_{horizon}d_mask"] = forward_mask

    return masks


def _forward_window_frame(frame, horizon, reducer):
    future = frame.shift(-1)
    rolled = reducer(future.iloc[::-1], horizon).iloc[::-1]
    return rolled


def build_forward_vol_targets(excess_return_df, horizons):
    targets = {}
    squared = excess_return_df.pow(2)
    for horizon in horizons:
        mean_sq = _forward_window_frame(
            squared,
            horizon,
            lambda x, h: x.rolling(h, min_periods=h).mean(),
        )
        targets[f"sigma_target_{horizon}d"] = np.sqrt(mean_sq.clip(lower=0.0))
    return targets


def build_forward_mean_targets(excess_return_df, horizons):
    targets = {}
    for horizon in horizons:
        future_mean = _forward_window_frame(
            excess_return_df,
            horizon,
            lambda x, h: x.rolling(h, min_periods=h).mean(),
        )
        targets[f"future_excess_mean_{horizon}d"] = future_mean
    return targets


def build_date_splits(dates, train_end, val_end):
    dates = pd.to_datetime(dates)
    train_mask = dates <= pd.Timestamp(train_end)
    val_mask = (dates > pd.Timestamp(train_end)) & (dates <= pd.Timestamp(val_end))
    test_mask = dates > pd.Timestamp(val_end)
    return {
        "train_date_mask": np.asarray(train_mask, dtype=bool),
        "val_date_mask": np.asarray(val_mask, dtype=bool),
        "test_date_mask": np.asarray(test_mask, dtype=bool),
    }


def build_date_splits_by_ratio(dates, train_ratio=0.70, val_ratio=0.15):
    dates = pd.to_datetime(dates)
    n_dates = len(dates)
    train_end_idx = max(int(np.floor(train_ratio * n_dates)) - 1, 0)
    val_end_idx = max(int(np.floor((train_ratio + val_ratio) * n_dates)) - 1, train_end_idx + 1)
    train_end = dates[train_end_idx]
    val_end = dates[min(val_end_idx, n_dates - 1)]
    splits = build_date_splits(dates, train_end=train_end, val_end=val_end)
    splits["train_end_date"] = np.asarray([str(pd.Timestamp(train_end).date())], dtype="<U16")
    splits["val_end_date"] = np.asarray([str(pd.Timestamp(val_end).date())], dtype="<U16")
    return splits


def save_panel_artifact(path, artifact_dict):
    serializable = {}
    for key, value in artifact_dict.items():
        if isinstance(value, pd.DataFrame):
            serializable[key] = value.to_numpy()
        elif isinstance(value, pd.Series):
            serializable[key] = value.to_numpy()
        elif isinstance(value, pd.Index):
            serializable[key] = value.to_numpy()
        else:
            serializable[key] = value
    np.savez(path, **serializable)


def summarize_panel_artifact(artifact):
    dates = pd.to_datetime(artifact["dates"])
    member_mask = artifact["member_mask"].astype(bool)
    price_mask = artifact["price_mask"].astype(bool)
    valid_sigma = artifact["valid_sigma_mask"].astype(bool)
    valid_lambda_20 = artifact["valid_lambda_20d_mask"].astype(bool)
    valid_lambda_60 = artifact["valid_lambda_60d_mask"].astype(bool)
    train = artifact["train_date_mask"].astype(bool)
    val = artifact["val_date_mask"].astype(bool)
    test = artifact["test_date_mask"].astype(bool)

    split = np.full(len(dates), "test", dtype=object)
    split[train] = "train"
    split[val] = "val"

    return pd.DataFrame(
        {
            "date": dates,
            "split": split,
            "member_count": member_mask.sum(axis=1),
            "price_count": price_mask.sum(axis=1),
            "valid_sigma_count": valid_sigma.sum(axis=1),
            "valid_lambda_20d_count": valid_lambda_20.sum(axis=1),
            "valid_lambda_60d_count": valid_lambda_60.sum(axis=1),
        }
    )


def load_panel_npz(path):
    npz = np.load(path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}
