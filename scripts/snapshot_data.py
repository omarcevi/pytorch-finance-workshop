"""Refreshes the snapshot data for the 5 default tickers.

Usage:
    python3 scripts/snapshot_data.py                # downloads from Yahoo Finance (if there is a network)
    python3 scripts/snapshot_data.py --offline      # writes the calibrated offline series without a network
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from workshop_utils import (
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_TICKERS,
    _generate_calibrated_offline_df,
    load_prices,
)


def refresh_snapshots(out_dir: Path, *, offline_only: bool = False) -> None:
    """Saves the daily OHLCV data of the default tickers as CSV files in the data/ folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ticker in DEFAULT_TICKERS:
        if offline_only:
            df = _generate_calibrated_offline_df(ticker, DEFAULT_START, DEFAULT_END)
            source = "offline-calibrated"
        else:
            df = load_prices(ticker, start=DEFAULT_START, end=DEFAULT_END, force_fallback=False)
            source = "yfinance/fallback"
        target = out_dir / f"{ticker}.csv"
        df.to_csv(target)
        print(f"Saved: {target} ({len(df)} rows, source={source})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refreshes the snapshot price data.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data",
        help="Folder for the CSV files (default: data/).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Writes the calibrated offline series without using the network.",
    )
    args = parser.parse_args()
    refresh_snapshots(args.out_dir, offline_only=args.offline)


if __name__ == "__main__":
    main()
