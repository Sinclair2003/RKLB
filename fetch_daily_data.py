"""
Universal Daily Data Fetcher (Yahoo Finance)
===========================================
Usage examples:
    pip install yfinance pandas
    python fetch_daily_data.py --symbols AAPL
    python fetch_daily_data.py --symbols AAPL,MSFT,NVDA --start 2020-01-01
    python fetch_daily_data.py --symbols TSLA SPY QQQ --output-dir ./my_data

Features:
  1) Accept any symbol(s) from CLI
  2) Download daily OHLCV from Yahoo Finance
  3) Clean data:
     - flatten multi-index columns
     - remove duplicate dates
     - sort ascending by date
     - replace volume == 0 with missing flag
     - fill NaN with missing flag
  4) Save each symbol to CSV: <symbol>_daily.csv
"""

import argparse
import os
from datetime import datetime

import pandas as pd
import yfinance as yf

PRICE_COLS = ["open", "high", "low", "close", "volume"]


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch daily data for arbitrary symbols")
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Ticker symbols. Supports space-separated or comma-separated forms, e.g. AAPL MSFT or AAPL,MSFT",
    )
    parser.add_argument("--start", default="2017-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD), exclusive in yfinance",
    )
    parser.add_argument("--output-dir", default="./market_data", help="Output folder")
    parser.add_argument("--missing-flag", type=float, default=-999, help="Missing value fill flag")
    return parser.parse_args()


def normalize_symbols(raw_symbols):
    symbols = []
    for token in raw_symbols:
        parts = [s.strip() for s in token.split(",") if s.strip()]
        symbols.extend(parts)
    # keep order, remove duplicates
    seen = set()
    unique = []
    for s in symbols:
        us = s.upper()
        if us not in seen:
            seen.add(us)
            unique.append(us)
    return unique


def clean(df, missing_flag):
    raw_len = len(df)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = PRICE_COLS

    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    df.loc[df["volume"] == 0, "volume"] = None

    missing_count = df[PRICE_COLS].isna().sum()
    total_missing = int(missing_count.sum())

    df[PRICE_COLS] = df[PRICE_COLS].fillna(missing_flag)

    cleaned_len = len(df)
    dropped = raw_len - cleaned_len
    print(f"  Cleaned: {raw_len:,} rows -> {cleaned_len:,} rows ({dropped} dropped)")
    if total_missing > 0:
        print(f"  Missing values (filled with {missing_flag}):")
        for col, cnt in missing_count.items():
            if cnt > 0:
                print(f"    {col}: {int(cnt)}")
    else:
        print("  No missing values")

    return df


def symbol_to_filename(symbol):
    safe = symbol.replace("^", "").replace("=", "_").replace("/", "_").replace("-", "_")
    return safe.lower()


def main():
    args = parse_args()
    symbols = normalize_symbols(args.symbols)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Daily Data Fetch: {args.start} ~ {args.end}")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Output Dir: {os.path.abspath(args.output_dir)}")
    print(f"  Missing value flag: {args.missing_flag}")
    print("=" * 60)

    for symbol in symbols:
        print(f"\n> {symbol}")

        df = yf.download(
            symbol,
            start=args.start,
            end=args.end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            print("  No data retrieved, please check network or ticker")
            continue

        df = clean(df, args.missing_flag)

        filename = f"{symbol_to_filename(symbol)}_daily.csv"
        filepath = os.path.join(args.output_dir, filename)
        df.to_csv(filepath)

        size_kb = os.path.getsize(filepath) / 1024
        print(f"  Date range: {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"  Saved: {filepath} ({size_kb:.0f} KB)")

        close_min = df["close"].min()
        close_max = df["close"].max()
        print(f"  Close range: {close_min:.2f} ~ {close_max:.2f}")
        flag_count = int((df[PRICE_COLS] == args.missing_flag).sum().sum())
        print(f"  Missing flag count: {flag_count}")

    print(f"\n{'=' * 60}")
    print("All done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
