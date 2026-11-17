"""Varsayılan 5 sembol için yedek (snapshot) verileri günceller.

Kullanım:
    python3 scripts/snapshot_data.py                # Yahoo Finance'ten indirir (ağ varsa)
    python3 scripts/snapshot_data.py --offline      # Çevrimdışı kalibre edilmiş yedekleri yazar
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
    """Varsayılan sembollerin günlük OHLCV verilerini data/ klasörüne CSV olarak kaydeder."""
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
        print(f"Kaydedildi: {target} ({len(df)} satır, kaynak={source})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Yedek (snapshot) fiyat verilerini yeniler.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data",
        help="CSV dosyalarının kaydedileceği klasör (varsayılan: data/).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Ağ kullanmadan kalibre edilmiş çevrimdışı yedek dosyaları üretir.",
    )
    args = parser.parse_args()
    refresh_snapshots(args.out_dir, offline_only=args.offline)


if __name__ == "__main__":
    main()
