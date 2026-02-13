"""
Fetch daily OHLCV data for arbitrary Yahoo Finance symbols.

Usage:
  pip install yfinance pandas
  python fetch_any_tickers.py
  python fetch_any_tickers.py --symbols AAPL
  python fetch_any_tickers.py --symbols AAPL,MSFT,NVDA --start 2020-01-01
  python fetch_any_tickers.py --symbols TSLA SPY QQQ --start-map TSLA:2010-06-29,SPY:1993-01-29
"""

import argparse
import os
from datetime import datetime

import pandas as pd
import yfinance as yf

PRICE_COLS = ["open", "high", "low", "close", "volume"]

# ==== You can change default args here ====
# Default symbols are same as your old script.
DEFAULT_SYMBOLS = [
    "BTC-USD",
    "^NDX",
    "^VIX",
    "DX-Y.NYB",
    "^TNX",
    "GC=F",
    "CL=F",
    "^GSPC",
    "NVDA",
    "AAPL",
    "MSFT",
]
DEFAULT_OUTPUT_DIR = "./market_data"
DEFAULT_MISSING_FLAG = -999.0
# If --start is omitted, script uses max available history for each symbol.
DEFAULT_START = None


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch daily data for arbitrary symbols")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=DEFAULT_SYMBOLS,
        help="Symbols, supports both 'AAPL MSFT' and 'AAPL,MSFT'",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="Global start date: YYYY-MM-DD. If omitted, each symbol uses max available history.",
    )
    parser.add_argument(
        "--start-map",
        default="",
        help="Per-symbol start date map, e.g. 'AAPL:1980-12-12,NVDA:1999-01-22'",
    )
    parser.add_argument(
        "--end",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="End date: YYYY-MM-DD (exclusive in yfinance)",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="CSV output directory")
    parser.add_argument("--missing-flag", type=float, default=DEFAULT_MISSING_FLAG, help="Fill value for missing data")
    return parser.parse_args()


def normalize_symbols(raw_symbols):
    symbols = []
    for token in raw_symbols:
        symbols.extend([x.strip().upper() for x in token.split(",") if x.strip()])

    seen = set()
    unique = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def parse_start_map(start_map_arg):
    if not start_map_arg.strip():
        return {}

    start_map = {}
    items = [item.strip() for item in start_map_arg.split(",") if item.strip()]
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid --start-map item: '{item}', expected SYMBOL:YYYY-MM-DD")
        symbol, start = item.split(":", 1)
        start_map[symbol.strip().upper()] = start.strip()
    return start_map


def clean_data(df, missing_flag):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = PRICE_COLS
    df.index.name = "date"

    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.loc[df["volume"] == 0, "volume"] = None
    df[PRICE_COLS] = df[PRICE_COLS].fillna(missing_flag)
    return df


def symbol_to_filename(symbol):
    return (
        symbol.replace("^", "")
        .replace("=", "_")
        .replace("/", "_")
        .replace("-", "_")
        .lower()
    )


def main():
    args = parse_args()
    symbols = normalize_symbols(args.symbols)
    start_map = parse_start_map(args.start_map)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    if args.start:
        print(f"Fetch range: {args.start} ~ {args.end}")
    else:
        print(f"Fetch range: auto(max history per symbol) ~ {args.end}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Output: {os.path.abspath(args.output_dir)}")
    print("=" * 60)

    for symbol in symbols:
        print(f"\n> {symbol}")
        symbol_start = args.start or start_map.get(symbol)
        if symbol_start:
            df = yf.download(
                symbol,
                start=symbol_start,
                end=args.end,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        else:
            # Use max period to get the earliest available history for this symbol.
            df = yf.download(
                symbol,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            end_ts = pd.to_datetime(args.end)
            df = df[df.index < end_ts]

        if df.empty:
            print("  No data returned (check symbol/network).")
            continue

        df = clean_data(df, args.missing_flag)
        out = os.path.join(args.output_dir, f"{symbol_to_filename(symbol)}_daily.csv")
        df.to_csv(out)

        print(f"  Start used: {symbol_start if symbol_start else 'max-history(auto)'}")
        print(f"  Rows: {len(df):,}")
        print(f"  Date range: {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"  Saved: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
