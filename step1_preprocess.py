import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIG (edit these)
# -----------------------------
TICKER = "^NSEI"                 # NIFTY 50 index on Yahoo
START_DATE = "2010-01-01"
END_DATE = "2020-01-01"

PRICE_COL = "Adj Close"          # use Adjusted Close for returns
WINDOW = 60                      # rolling window length (state length)
TRAIN_SPLIT = 0.80               # for fitting scaler on train only

DATA_DIR = "data"
OUT_DIR = "outputs"

RAW_CSV = os.path.join(DATA_DIR, "nifty50_raw.csv")
PROC_CSV = os.path.join(OUT_DIR, "nifty50_step1_processed.csv")
NPZ_FILE = os.path.join(OUT_DIR, f"step1_windows_W{WINDOW}.npz")


# -----------------------------
# Step 1.1: Download data
# -----------------------------
def download_nifty50():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=False)

    if df is None or df.empty:
        raise RuntimeError(
            "Download returned empty data. "
            "Check internet, ticker '^NSEI', and yfinance version."
        )

    # Keep standard OHLCV columns if present; save raw
    df.to_csv(RAW_CSV)
    return df


# -----------------------------
# Step 1.2: Clean + select price series
# -----------------------------
def clean_and_select_price(df: pd.DataFrame) -> pd.Series:
    # Ensure Date index is sorted, unique
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found in columns: {list(df.columns)}")

    price = df[PRICE_COL].astype(float)
    if isinstance(price, pd.DataFrame):
        if price.shape[1] != 1:
            raise ValueError(
                f"Expected a single '{PRICE_COL}' column, got columns: {list(price.columns)}"
            )
        # yfinance can return a 2D frame for a single ticker; squeeze to Series
        price = price.iloc[:, 0]
        price.name = PRICE_COL

    # Handle missing values:
    # For market data, forward-fill is standard; then drop remaining NaNs
    price = price.ffill().dropna()

    return price


# -----------------------------
# Step 1.3: Compute log-returns
#   r_t = log(P_t / P_{t-1})
# -----------------------------
def compute_log_returns(price: pd.Series) -> pd.Series:
    logret = np.log(price / price.shift(1))
    logret = logret.replace([np.inf, -np.inf], np.nan).dropna()
    return logret


# -----------------------------
# Step 1.4: Normalize inputs (z-score)
# IMPORTANT: fit scaler on TRAIN only to avoid look-ahead bias
# -----------------------------
def normalize_series(logret: pd.Series, train_split: float):
    values = logret.values.reshape(-1, 1)

    n = len(values)
    n_train = int(train_split * n)

    train_vals = values[:n_train]
    full_vals = values

    scaler = StandardScaler()
    scaler.fit(train_vals)  # fit only on train

    norm_full = scaler.transform(full_vals).reshape(-1)
    norm = pd.Series(norm_full, index=logret.index, name="logret_z")

    # also return train cutoff index for later use
    return norm, scaler, n_train


# -----------------------------
# Step 1.5: Build rolling windows (state inputs)
# State at time t: [x_{t-W+1}, ..., x_t]
# Target (optional): x_{t+1}
# -----------------------------
def make_rolling_windows(norm_series: pd.Series, window: int):
    x = norm_series.values.astype(np.float32)
    idx = norm_series.index

    if len(x) <= window:
        raise ValueError(f"Not enough points ({len(x)}) for window={window}")

    X = []
    y = []
    end_dates = []

    for t in range(window - 1, len(x) - 1):
        X.append(x[t - window + 1: t + 1])   # length = window
        y.append(x[t + 1])                   # next-step prediction target
        end_dates.append(idx[t])             # window end date

    X = np.stack(X)  # shape: (N_windows, window)
    y = np.array(y, dtype=np.float32)  # shape: (N_windows,)
    end_dates = np.array(end_dates)

    return X, y, end_dates


# -----------------------------
# Step 1.6: Save artifacts
# -----------------------------
def save_outputs(price, logret, norm, scaler, X, y, end_dates, n_train):
    os.makedirs(OUT_DIR, exist_ok=True)

    # FORCE correct types & shapes
    aligned_price = pd.Series(
        price.loc[logret.index].to_numpy().ravel(),
        index=logret.index,
        name="price_adj_close"
    )

    logret_s = pd.Series(
        logret.values,
        index=logret.index,
        name="logret"
    )

    norm_s = pd.Series(
        np.asarray(norm).ravel(),   # <-- THIS is the key line
        index=logret.index,
        name="logret_z"
    )

    out_df = pd.concat([aligned_price, logret_s, norm_s], axis=1)

    out_df.to_csv(PROC_CSV, index=True)

    np.savez(
        NPZ_FILE,
        X=X,
        y=y,
        end_dates=end_dates,
        window=WINDOW,
        train_size=n_train,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )


# -----------------------------
# MAIN: Run Step-1
# -----------------------------
def main():
    print("Step 1: Downloading NIFTY 50 data...")
    df = download_nifty50()
    print(f"Downloaded rows: {len(df)}")
    print(f"Saved raw CSV to: {RAW_CSV}")

    print("\nStep 1: Cleaning & selecting price series...")
    price = clean_and_select_price(df)
    print(f"Price points after cleaning: {len(price)}")

    print("\nStep 1: Computing log-returns...")
    logret = compute_log_returns(price)
    print(f"Log-return points: {len(logret)}")

    print("\nStep 1: Normalizing (z-score) using TRAIN-only fit...")
    norm, scaler, n_train = normalize_series(logret, TRAIN_SPLIT)
    print(f"Train points used for scaler fit: {n_train}")
    print(f"Scaler mean (train): {float(scaler.mean_[0]):.6f}, std (train): {float(scaler.scale_[0]):.6f}")

    print(f"\nStep 1: Creating rolling windows with WINDOW={WINDOW}...")
    X, y, end_dates = make_rolling_windows(norm, WINDOW)
    print(f"X shape: {X.shape}  (num_windows, window)")
    print(f"y shape: {y.shape}  (num_windows,)")

    print("\nStep 1: Saving processed outputs...")
    save_outputs(price, logret, norm, scaler, X, y, end_dates, n_train)
    print(f"Saved processed CSV to: {PROC_CSV}")
    print(f"Saved windows NPZ to: {NPZ_FILE}")

    print("\n✅ Step-1 DONE perfectly.")
    print("Next step: feed X (state windows) to your neural model / Neural SDE.")


if __name__ == "__main__":
    main()
