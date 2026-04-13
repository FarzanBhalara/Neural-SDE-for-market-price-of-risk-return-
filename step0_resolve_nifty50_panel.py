import calendar
import re
from pathlib import Path

import pandas as pd
import yfinance as yf


START_DATE = "2010-01-01"
END_DATE = "2020-01-01"

DATA_DIR = Path("data")
MEMBERSHIP_IN = DATA_DIR / "nifty50_historical_membership_monthly.csv"
SYMBOL_MAP_IN = DATA_DIR / "nifty50_symbol_map.csv"
STATUS_IN = DATA_DIR / "nifty50_yahoo_download_status.csv"
PRICES_IN = DATA_DIR / "nifty50_prices_yahoo_2010_2020.csv"
INDEX_IN = DATA_DIR / "nifty50_index_yahoo_2010_2020.csv"

RESOLUTION_OUT = DATA_DIR / "nifty50_symbol_resolution.csv"
RESOLVED_PRICES_OUT = DATA_DIR / "nifty50_prices_yahoo_resolved_2010_2020.csv"
DAILY_PANEL_OUT = DATA_DIR / "nifty50_member_price_panel_2010_2020.csv"
COVERAGE_OUT = DATA_DIR / "nifty50_member_price_coverage_by_date.csv"

MANUAL_RESOLUTIONS = {
    "HEROHONDA": {
        "resolved_ticker": "HEROMOTOCO.NS",
        "resolution_status": "alias_same_entity",
        "notes": "Hero Honda Motors was renamed Hero MotoCorp; Yahoo exposes the current symbol history.",
    },
    "IBULHSGFIN": {
        "resolved_ticker": "SAMMAANCAP.NS",
        "resolution_status": "alias_same_entity",
        "notes": "Indiabulls Housing Finance was renamed Sammaan Capital; Yahoo exposes the current symbol history.",
    },
    "INFOSYSTCH": {
        "resolved_ticker": "INFY.NS",
        "resolution_status": "alias_same_entity",
        "notes": "Legacy Infosys symbol maps to the current INFY ticker on Yahoo.",
    },
    "INFRATEL": {
        "resolved_ticker": "INDUSTOWER.NS",
        "resolution_status": "alias_same_entity",
        "notes": "Bharti Infratel history is exposed under the current Indus Towers Yahoo ticker.",
    },
    "MCDOWELL-N": {
        "resolved_ticker": "UNITDSPR.BO",
        "resolution_status": "alias_same_entity_bse",
        "notes": "United Spirits history was only recoverable from Yahoo's BSE ticker.",
    },
    "SSLT": {
        "resolved_ticker": "VEDL.NS",
        "resolution_status": "alias_same_entity",
        "notes": "Sesa Sterlite was renamed Vedanta Ltd.; Yahoo exposes the current ticker history.",
    },
}

UNRESOLVED_NOTES = {
    "CAIRN": "Cairn India later merged into Vedanta; Yahoo does not expose a clean standalone current-history ticker.",
    "HDFC": "HDFC Ltd. later merged into HDFC Bank; Yahoo no longer exposes the legacy standalone line cleanly.",
    "IDFC": "IDFC Ltd. later restructured/merged; Yahoo does not expose a clean legacy same-entity ticker.",
    "RANBAXY": "Ranbaxy was acquired by Sun Pharma; there is no clean standalone continuation ticker on Yahoo.",
    "SESAGOA": "Sesa Goa is a pre-merger predecessor of Sesa Sterlite/Vedanta and cannot be mapped one-to-one.",
    "STER": "Sterlite Industries is a pre-merger predecessor of Sesa Sterlite/Vedanta and cannot be mapped one-to-one.",
    "TATAMOTORS": "Yahoo quote pages exist, but the chart API did not return a usable historical series during this fetch.",
    "TATAMTRDVR": "Tata Motors DVR class shares are no longer available as a clean Yahoo historical series.",
}


def month_end(ts):
    return pd.Timestamp(ts.year, ts.month, calendar.monthrange(ts.year, ts.month)[1])


def normalize_saved_columns(frame):
    rename_map = {}
    for col in frame.columns:
        text = str(col).strip()
        match = re.match(r"^\('([^']+)',_'.*'\)$", text)
        if match:
            rename_map[col] = match.group(1)
        else:
            rename_map[col] = text
    frame = frame.rename(columns=rename_map)
    return frame


def format_price_frame(frame, ticker):
    frame = frame.reset_index()
    frame.columns = [str(col).lower().replace(" ", "_") for col in frame.columns]
    frame["yahoo_ticker"] = ticker
    return frame


def download_yahoo_tickers(tickers):
    tickers = sorted(set(tickers))
    if not tickers:
        empty_prices = pd.DataFrame(columns=["date", "yahoo_ticker"])
        empty_status = pd.DataFrame(columns=["yahoo_ticker", "status", "rows"])
        return empty_prices, empty_status

    data = yf.download(
        tickers=tickers,
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

    status_rows = []
    price_frames = []
    if data.empty:
        for ticker in tickers:
            status_rows.append({"yahoo_ticker": ticker, "status": "empty_download", "rows": 0})
        empty_prices = pd.DataFrame(columns=["date", "yahoo_ticker"])
        return empty_prices, pd.DataFrame(status_rows)

    if isinstance(data.columns, pd.MultiIndex) and set(tickers).issubset(set(data.columns.get_level_values(0))):
        for ticker in tickers:
            ticker_frame = data[ticker].copy()
            ticker_frame = format_price_frame(ticker_frame, ticker)
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
            if not all_null:
                price_frames.append(ticker_frame)
    else:
        ticker = tickers[0]
        ticker_frame = format_price_frame(data.copy(), ticker)
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
        if not all_null:
            price_frames.append(ticker_frame)

    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    status = pd.DataFrame(status_rows).sort_values(["status", "yahoo_ticker"]).reset_index(drop=True)
    return prices, status


def load_inputs():
    membership = pd.read_csv(MEMBERSHIP_IN, parse_dates=["snapshot_date"])
    membership = membership.rename(
        columns={
            "Security Symbol": "nse_symbol",
            "Security Name": "security_name",
            "Industry": "industry",
        }
    )
    membership["nse_symbol"] = membership["nse_symbol"].astype(str).str.strip()
    membership["security_name"] = membership["security_name"].astype(str).str.strip()

    symbol_map = pd.read_csv(SYMBOL_MAP_IN)
    status = pd.read_csv(STATUS_IN)
    prices = pd.read_csv(PRICES_IN, low_memory=False)
    prices = normalize_saved_columns(prices)
    prices["date"] = pd.to_datetime(prices["date"])

    index_df = pd.read_csv(INDEX_IN, low_memory=False)
    index_df = normalize_saved_columns(index_df)
    index_df["date"] = pd.to_datetime(index_df["date"])
    return membership, symbol_map, status, prices, index_df


def build_symbol_meta(membership):
    return (
        membership.groupby("nse_symbol", as_index=False)
        .agg(
            security_names=("security_name", lambda s: " | ".join(dict.fromkeys(s))),
            first_snapshot=("snapshot_date", "min"),
            last_snapshot=("snapshot_date", "max"),
            snapshot_count=("snapshot_date", "nunique"),
        )
        .sort_values("nse_symbol")
        .reset_index(drop=True)
    )


def build_resolution_table(symbol_meta, status):
    status_map = dict(zip(status["yahoo_ticker"], status["status"]))
    rows = []
    for record in symbol_meta.to_dict("records"):
        symbol = record["nse_symbol"]
        original_ticker = f"{symbol}.NS"
        base_status = status_map.get(original_ticker, "missing")

        if base_status == "ok":
            resolved_ticker = original_ticker
            resolution_status = "direct"
            notes = "Direct Yahoo NSE ticker downloaded successfully."
        elif symbol in MANUAL_RESOLUTIONS:
            resolved_ticker = MANUAL_RESOLUTIONS[symbol]["resolved_ticker"]
            resolution_status = MANUAL_RESOLUTIONS[symbol]["resolution_status"]
            notes = MANUAL_RESOLUTIONS[symbol]["notes"]
        else:
            resolved_ticker = ""
            resolution_status = "unresolved"
            notes = UNRESOLVED_NOTES.get(symbol, "No clean Yahoo continuation ticker was identified.")

        rows.append(
            {
                "nse_symbol": symbol,
                "security_names": record["security_names"],
                "first_snapshot": record["first_snapshot"],
                "last_snapshot": record["last_snapshot"],
                "snapshot_count": record["snapshot_count"],
                "original_yahoo_ticker": original_ticker,
                "original_yahoo_status": base_status,
                "resolved_ticker": resolved_ticker,
                "resolution_status": resolution_status,
                "notes": notes,
            }
        )

    return pd.DataFrame(rows)


def merge_price_sources(base_prices, alias_prices):
    all_prices = pd.concat([base_prices, alias_prices], ignore_index=True)
    all_prices = all_prices.sort_values(["yahoo_ticker", "date"]).drop_duplicates(["yahoo_ticker", "date"], keep="first")
    return all_prices.reset_index(drop=True)


def build_daily_panel(membership, resolution, all_prices, index_df):
    calendar_df = index_df[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    calendar_df["month_anchor"] = calendar_df["date"].map(month_end)

    snapshot_lookup = membership[["snapshot_date"]].drop_duplicates().sort_values("snapshot_date")
    anchor_lookup = pd.DataFrame({"month_anchor": sorted(calendar_df["month_anchor"].unique())})
    anchor_lookup = pd.merge_asof(
        anchor_lookup.sort_values("month_anchor"),
        snapshot_lookup.rename(columns={"snapshot_date": "snapshot_date_used"}).sort_values("snapshot_date_used"),
        left_on="month_anchor",
        right_on="snapshot_date_used",
        direction="backward",
    )

    calendar_df = calendar_df.merge(anchor_lookup, on="month_anchor", how="left")

    membership_daily = calendar_df.merge(
        membership,
        left_on="snapshot_date_used",
        right_on="snapshot_date",
        how="left",
    )

    panel = membership_daily.merge(
        resolution[
            [
                "nse_symbol",
                "resolved_ticker",
                "resolution_status",
                "original_yahoo_ticker",
                "original_yahoo_status",
                "notes",
            ]
        ],
        on="nse_symbol",
        how="left",
    )

    price_cols = [col for col in all_prices.columns if col not in {"capital_gains"}]
    panel = panel.merge(
        all_prices[price_cols],
        left_on=["date", "resolved_ticker"],
        right_on=["date", "yahoo_ticker"],
        how="left",
    )
    panel["resolved_available"] = panel["resolution_status"].ne("unresolved")
    panel["has_price"] = panel["adj_close"].notna()
    panel["coverage_status"] = "ok"
    panel.loc[~panel["resolved_available"], "coverage_status"] = "unresolved_symbol"
    panel.loc[panel["resolved_available"] & ~panel["has_price"], "coverage_status"] = "missing_price_on_date"

    ordered_cols = [
        "date",
        "month_anchor",
        "snapshot_date_used",
        "snapshot_date",
        "nse_symbol",
        "security_name",
        "industry",
        "original_yahoo_ticker",
        "original_yahoo_status",
        "resolved_ticker",
        "resolution_status",
        "coverage_status",
        "notes",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
        "repaired?",
    ]
    keep_cols = [col for col in ordered_cols if col in panel.columns]
    panel = panel[keep_cols].sort_values(["date", "nse_symbol"]).reset_index(drop=True)
    return panel


def joined_symbols(series):
    values = sorted({value for value in series if isinstance(value, str) and value})
    return "|".join(values)


def build_coverage_report(panel):
    grouped = panel.groupby("date", sort=True)
    coverage = grouped.agg(
        snapshot_date_used=("snapshot_date_used", "first"),
        member_count=("nse_symbol", "size"),
        resolved_count=("resolved_ticker", lambda s: int(s.fillna("").astype(str).str.len().gt(0).sum())),
        price_count=("adj_close", lambda s: int(s.notna().sum())),
        unresolved_count=("coverage_status", lambda s: int((s == "unresolved_symbol").sum())),
        missing_price_count=("coverage_status", lambda s: int((s == "missing_price_on_date").sum())),
    )
    coverage["resolved_ratio"] = coverage["resolved_count"] / coverage["member_count"]
    coverage["price_ratio"] = coverage["price_count"] / coverage["member_count"]
    unresolved_symbols = (
        panel.loc[panel["coverage_status"] == "unresolved_symbol"]
        .groupby("date")["nse_symbol"]
        .apply(joined_symbols)
        .rename("unresolved_symbols")
    )
    missing_price_symbols = (
        panel.loc[panel["coverage_status"] == "missing_price_on_date"]
        .groupby("date")["nse_symbol"]
        .apply(joined_symbols)
        .rename("missing_price_symbols")
    )
    coverage = coverage.reset_index()
    coverage = coverage.merge(unresolved_symbols, on="date", how="left")
    coverage = coverage.merge(missing_price_symbols, on="date", how="left")
    coverage["unresolved_symbols"] = coverage["unresolved_symbols"].fillna("")
    coverage["missing_price_symbols"] = coverage["missing_price_symbols"].fillna("")
    return coverage


def main():
    membership, symbol_map, status, prices, index_df = load_inputs()
    _ = symbol_map

    symbol_meta = build_symbol_meta(membership)
    resolution = build_resolution_table(symbol_meta, status)

    existing_tickers = set(prices["yahoo_ticker"].dropna().unique())
    alias_tickers = set(resolution.loc[resolution["resolution_status"].str.startswith("alias"), "resolved_ticker"])
    missing_alias_tickers = sorted(ticker for ticker in alias_tickers if ticker and ticker not in existing_tickers)

    alias_prices, alias_status = download_yahoo_tickers(missing_alias_tickers)
    alias_status_map = dict(zip(alias_status["yahoo_ticker"], alias_status["status"]))

    resolution["resolved_download_status"] = resolution["resolved_ticker"].map(alias_status_map).fillna(
        resolution["resolved_ticker"].map(lambda t: "ok" if t in existing_tickers else "")
    )
    resolution["resolved_price_rows"] = resolution["resolved_ticker"].map(
        lambda ticker: int((prices["yahoo_ticker"] == ticker).sum()) if ticker in existing_tickers else int((alias_prices["yahoo_ticker"] == ticker).sum())
    )

    failed_aliases = set(
        alias_status.loc[alias_status["status"] != "ok", "yahoo_ticker"].dropna().tolist()
    )
    if failed_aliases:
        resolution.loc[resolution["resolved_ticker"].isin(failed_aliases), "resolution_status"] = "unresolved"
        resolution.loc[resolution["resolved_ticker"].isin(failed_aliases), "notes"] = (
            "Alias mapping was identified, but Yahoo did not return usable price history during resolution."
        )
        resolution.loc[resolution["resolved_ticker"].isin(failed_aliases), "resolved_ticker"] = ""

    all_prices = merge_price_sources(prices, alias_prices)
    resolved_prices = resolution.loc[
        resolution["resolution_status"] != "unresolved",
        ["nse_symbol", "resolved_ticker", "resolution_status"],
    ].merge(
        all_prices,
        left_on="resolved_ticker",
        right_on="yahoo_ticker",
        how="left",
    )
    resolved_prices = resolved_prices.sort_values(["nse_symbol", "date"]).reset_index(drop=True)

    panel = build_daily_panel(membership, resolution, all_prices, index_df)
    coverage = build_coverage_report(panel)

    resolution = resolution.sort_values(["resolution_status", "nse_symbol"]).reset_index(drop=True)
    resolution.to_csv(RESOLUTION_OUT, index=False)
    resolved_prices.to_csv(RESOLVED_PRICES_OUT, index=False)
    panel.to_csv(DAILY_PANEL_OUT, index=False)
    coverage.to_csv(COVERAGE_OUT, index=False)

    print(f"Saved symbol resolution table to {RESOLUTION_OUT.as_posix()}")
    print(f"Saved resolved price panel to {RESOLVED_PRICES_OUT.as_posix()}")
    print(f"Saved daily member panel to {DAILY_PANEL_OUT.as_posix()}")
    print(f"Saved date-wise coverage report to {COVERAGE_OUT.as_posix()}")
    print(f"Direct symbols: {(resolution['resolution_status'] == 'direct').sum()}")
    print(f"Alias-resolved symbols: {resolution['resolution_status'].str.startswith('alias').sum()}")
    print(f"Unresolved symbols: {(resolution['resolution_status'] == 'unresolved').sum()}")
    print(f"Additional alias tickers downloaded: {(alias_status['status'] == 'ok').sum()}")
    print(f"Mean priced members per date: {coverage['price_count'].mean():.2f}")
    print(f"Min priced members on a date: {coverage['price_count'].min()}")
    print(f"Max priced members on a date: {coverage['price_count'].max()}")


if __name__ == "__main__":
    main()
