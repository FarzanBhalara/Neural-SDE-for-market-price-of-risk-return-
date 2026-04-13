from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
INDEX_CSV = DATA_DIR / "nifty50_index_yahoo_2010_2020.csv"
RISK_FREE_OUT = DATA_DIR / "risk_free_daily.csv"

FRED_SERIES_ID = "INTDSRINM193N"
FRED_CSV_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_SERIES_ID}"
START_DATE = "2010-01-01"
END_DATE = "2020-01-01"


def normalize_saved_columns(frame):
    rename_map = {}
    for col in frame.columns:
        text = str(col).strip()
        if text.startswith("('") and "',_'" in text:
            rename_map[col] = text.split("',_'", 1)[0][2:]
        else:
            rename_map[col] = text
    return frame.rename(columns=rename_map)


def load_trading_dates():
    frame = pd.read_csv(INDEX_CSV, low_memory=False)
    frame = normalize_saved_columns(frame)
    frame["date"] = pd.to_datetime(frame["date"])
    dates = frame["date"].drop_duplicates().sort_values()
    return dates[(dates >= pd.Timestamp(START_DATE)) & (dates < pd.Timestamp(END_DATE))]


def download_monthly_discount_rate():
    monthly = pd.read_csv(FRED_CSV_URL, parse_dates=["observation_date"])
    monthly = monthly.rename(columns={"observation_date": "date", FRED_SERIES_ID: "annual_rate_percent"})
    monthly = monthly[(monthly["date"] >= START_DATE) & (monthly["date"] < END_DATE)].copy()
    monthly["annual_rate_percent"] = pd.to_numeric(monthly["annual_rate_percent"], errors="coerce")
    monthly = monthly.dropna(subset=["annual_rate_percent"]).sort_values("date")
    if monthly.empty:
        raise RuntimeError("FRED monthly discount-rate series returned no rows for the requested range.")
    monthly["annual_rate"] = monthly["annual_rate_percent"] / 100.0
    return monthly


def build_daily_proxy(trading_dates, monthly):
    daily = pd.DataFrame({"date": trading_dates})
    monthly = monthly[["date", "annual_rate", "annual_rate_percent"]].sort_values("date")
    daily = pd.merge_asof(
        daily.sort_values("date"),
        monthly,
        on="date",
        direction="backward",
    )
    daily["annual_rate"] = daily["annual_rate"].ffill().bfill()
    daily["annual_rate_percent"] = daily["annual_rate_percent"].ffill().bfill()
    daily["daily_rate"] = np.log1p(daily["annual_rate"]) / 252.0
    daily["source"] = "FRED_INTDSRINM193N_monthly_discount_rate_forward_filled"
    daily["source_url"] = FRED_CSV_URL
    return daily


def main():
    DATA_DIR.mkdir(exist_ok=True)
    trading_dates = load_trading_dates()
    monthly = download_monthly_discount_rate()
    daily = build_daily_proxy(trading_dates, monthly)
    daily.to_csv(RISK_FREE_OUT, index=False)

    print(f"Saved risk-free proxy to {RISK_FREE_OUT.as_posix()}")
    print(f"Rows: {len(daily)}")
    print(f"Start: {daily['date'].min().date()}")
    print(f"End: {daily['date'].max().date()}")
    print(f"Annual rate mean: {daily['annual_rate'].mean():.4f}")
    print(f"Annual rate min/max: {daily['annual_rate'].min():.4f} / {daily['annual_rate'].max():.4f}")


if __name__ == "__main__":
    main()
