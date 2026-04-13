import calendar
import csv
import io
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import yfinance as yf


START_DATE = "2010-01-01"
END_DATE = "2020-01-01"
MONTHLY_START = pd.Timestamp("2010-01-01")
MONTHLY_END = pd.Timestamp("2019-12-01")

BASE_URL = (
    "https://www.niftyindices.com/"
    "Market_Capitalisation_Weightage_Beta_for_NIFTY_50_And_NIFTY_Next_50"
)
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "nifty50_historical_membership_raw"
MEMBERSHIP_OUT = DATA_DIR / "nifty50_historical_membership_monthly.csv"
SYMBOL_MAP_OUT = DATA_DIR / "nifty50_symbol_map.csv"
STATUS_OUT = DATA_DIR / "nifty50_yahoo_download_status.csv"
PRICES_OUT = DATA_DIR / "nifty50_prices_yahoo_2010_2020.csv"
INDEX_OUT = DATA_DIR / "nifty50_index_yahoo_2010_2020.csv"

MONTH_ABBR = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dec",
}


def month_end(ts):
    return pd.Timestamp(ts.year, ts.month, calendar.monthrange(ts.year, ts.month)[1])


def month_zip_url(ts):
    return f"{BASE_URL}/mcwb_{MONTH_ABBR[ts.month]}{ts.strftime('%y')}.zip"


def download_bytes(url, retries=3, sleep_s=1.0):
    headers = {"User-Agent": "Mozilla/5.0"}
    last_error = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_s * attempt)
    raise last_error


def parse_nifty50_csv(csv_bytes, snapshot_date):
    text = csv_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if len(rows) < 1:
        raise ValueError(f"Unexpected CSV format for snapshot {snapshot_date:%Y-%m-%d}")

    header_row_idx = None
    for idx, row in enumerate(rows[:10]):
        labels = {cell.strip().lower() for cell in row}
        if "security symbol" in labels:
            header_row_idx = idx
            break
    if header_row_idx is None:
        raise KeyError(f"Could not locate CSV header for snapshot {snapshot_date:%Y-%m-%d}")

    header = []
    seen = {}
    for col in rows[header_row_idx]:
        col = col.strip()
        count = seen.get(col, 0)
        seen[col] = count + 1
        header.append(col if count == 0 else f"{col}_{count}")
    data_rows = []
    for row in rows[header_row_idx + 1:]:
        if not row or not row[0].strip():
            continue
        if row[0].strip().lower().startswith("source"):
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        data_rows.append(row[: len(header)])

    frame = pd.DataFrame(data_rows, columns=header)
    columns_lower = {col.lower(): col for col in frame.columns}
    symbol_col = next((col for key, col in columns_lower.items() if key == "security symbol"), None)
    name_col = next((col for key, col in columns_lower.items() if key == "security name"), None)
    industry_col = next((col for key, col in columns_lower.items() if key == "industry"), None)

    if symbol_col is None:
        raise KeyError(f"Required constituent columns not found for snapshot {snapshot_date:%Y-%m-%d}")

    rename_map = {symbol_col: "Security Symbol"}
    if name_col is not None:
        rename_map[name_col] = "Security Name"
    if industry_col is not None:
        rename_map[industry_col] = "Industry"

    frame = frame.rename(columns=rename_map)
    if "Security Name" not in frame.columns:
        frame["Security Name"] = frame["Security Symbol"]
    keep = ["Security Symbol", "Security Name"]
    if "Industry" in frame.columns:
        keep.append("Industry")
    frame = frame[keep].copy()
    frame["snapshot_date"] = snapshot_date
    frame["source_url"] = month_zip_url(pd.Timestamp(snapshot_date))
    return frame


def find_nifty50_member_file(namelist):
    candidates = []
    for name in namelist:
        lower = name.lower()
        if lower.endswith("/"):
            continue
        if lower.endswith("nifty50_mcwb.csv"):
            candidates.append(name)
        elif "niftymcwb" in lower and lower.endswith(".csv") and "jrnifty" not in lower:
            candidates.append(name)

    if not candidates:
        raise KeyError("No NIFTY 50 member CSV found inside archive.")
    return sorted(candidates, key=len)[0]


def fetch_monthly_membership():
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    snapshots = []
    failures = []
    for ts in pd.date_range(MONTHLY_START, MONTHLY_END, freq="MS"):
        url = month_zip_url(ts)
        zip_path = RAW_DIR / f"mcwb_{ts:%Y_%m}.zip"
        try:
            if zip_path.exists():
                payload = zip_path.read_bytes()
            else:
                payload = download_bytes(url)
                zip_path.write_bytes(payload)
            archive = zipfile.ZipFile(io.BytesIO(payload))
            member_file = find_nifty50_member_file(archive.namelist())
            frame = parse_nifty50_csv(archive.read(member_file), month_end(ts))
            snapshots.append(frame)
            print(f"Downloaded membership snapshot {ts:%Y-%m}")
        except Exception as exc:  # noqa: BLE001
            failures.append({"month": ts.strftime("%Y-%m"), "url": url, "error": str(exc)})
            print(f"Failed membership snapshot {ts:%Y-%m}: {exc}")

    if not snapshots:
        raise RuntimeError("No monthly NIFTY 50 membership snapshots were downloaded.")

    membership = pd.concat(snapshots, ignore_index=True)
    membership = membership.rename(columns=str.strip)
    membership["Security Symbol"] = membership["Security Symbol"].astype(str).str.strip()
    membership["Security Name"] = membership["Security Name"].astype(str).str.strip()
    membership = membership[membership["Security Symbol"].notna()]
    membership = membership[~membership["Security Symbol"].isin(["", "nan", "NaN"])]
    membership = membership.reset_index(drop=True)
    membership.to_csv(MEMBERSHIP_OUT, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(RAW_DIR / "download_failures.csv", index=False)

    return membership


def build_symbol_map(membership):
    frame = (
        membership[["Security Symbol", "Security Name"]]
        .drop_duplicates()
        .sort_values(["Security Symbol", "Security Name"])
        .reset_index(drop=True)
        .rename(columns={"Security Symbol": "nse_symbol", "Security Name": "security_name"})
    )
    frame["nse_symbol"] = frame["nse_symbol"].fillna("").str.strip()
    frame["security_name"] = frame["security_name"].fillna("").str.strip()
    frame = frame[frame["nse_symbol"] != ""].reset_index(drop=True)
    frame["yahoo_ticker"] = frame["nse_symbol"] + ".NS"
    frame.to_csv(SYMBOL_MAP_OUT, index=False)
    return frame


def download_price_panel(symbol_map):
    tickers = symbol_map["yahoo_ticker"].drop_duplicates().tolist()
    chunks = [tickers[i : i + 20] for i in range(0, len(tickers), 20)]

    status_rows = []
    price_frames = []
    for chunk in chunks:
        print(f"Downloading Yahoo prices for chunk of {len(chunk)} tickers")
        data = yf.download(
            tickers=chunk,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=False,
            actions=True,
            repair=True,
            threads=True,
            group_by="ticker",
            progress=False,
        )

        if data.empty:
            for ticker in chunk:
                status_rows.append({"yahoo_ticker": ticker, "status": "empty_chunk", "rows": 0})
            continue

        for ticker in chunk:
            if len(chunk) == 1:
                ticker_frame = data.copy()
            elif ticker in data.columns.get_level_values(0):
                ticker_frame = data[ticker].copy()
            else:
                status_rows.append({"yahoo_ticker": ticker, "status": "missing_ticker", "rows": 0})
                continue

            ticker_frame = ticker_frame.reset_index()
            ticker_frame.columns = [str(col).lower().replace(" ", "_") for col in ticker_frame.columns]
            numeric_cols = [col for col in ticker_frame.columns if col != "date"]
            all_null = ticker_frame[numeric_cols].isna().all().all()
            row_count = int(ticker_frame["date"].notna().sum())
            status_rows.append(
                {
                    "yahoo_ticker": ticker,
                    "status": "ok" if not all_null else "all_null",
                    "rows": row_count,
                }
            )
            if all_null:
                continue

            ticker_frame["yahoo_ticker"] = ticker
            price_frames.append(ticker_frame)

    status = pd.DataFrame(status_rows).sort_values(["status", "yahoo_ticker"]).reset_index(drop=True)
    status.to_csv(STATUS_OUT, index=False)

    if not price_frames:
        raise RuntimeError("Yahoo download returned no price frames.")

    prices = pd.concat(price_frames, ignore_index=True)
    prices = prices.sort_values(["yahoo_ticker", "date"]).reset_index(drop=True)
    prices.to_csv(PRICES_OUT, index=False)
    return prices, status


def download_index_series():
    index_df = yf.download(
        "^NSEI",
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        actions=True,
        repair=True,
        progress=False,
    )
    if index_df.empty:
        raise RuntimeError("Yahoo download returned empty NIFTY index data.")

    index_df = index_df.reset_index()
    index_df.columns = [str(col).lower().replace(" ", "_") for col in index_df.columns]
    index_df.to_csv(INDEX_OUT, index=False)
    return index_df


def main():
    membership = fetch_monthly_membership()
    symbol_map = build_symbol_map(membership)
    prices, status = download_price_panel(symbol_map)
    index_df = download_index_series()

    print(f"Saved membership snapshots to {MEMBERSHIP_OUT}")
    print(f"Saved symbol map to {SYMBOL_MAP_OUT}")
    print(f"Saved Yahoo ticker status to {STATUS_OUT}")
    print(f"Saved Yahoo price panel to {PRICES_OUT}")
    print(f"Saved NIFTY index series to {INDEX_OUT}")
    print(f"Monthly snapshots: {membership['snapshot_date'].nunique()}")
    print(f"Unique historical symbols: {symbol_map['nse_symbol'].nunique()}")
    print(f"Yahoo tickers with price rows: {(status['status'] == 'ok').sum()}")
    print(f"Yahoo price rows: {len(prices)}")
    print(f"NIFTY index rows: {len(index_df)}")


if __name__ == "__main__":
    main()
