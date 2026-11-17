# Snapshot Data Folder

Under the Yahoo Finance Terms of Service, the raw market data (`*.csv`) is not included in the public git repository (it is excluded in `.gitignore`).

## Creating and Serving the Snapshot

1. **Before the workshop (instructor):**
   ```bash
   python3 scripts/snapshot_data.py
   ```
   This command downloads daily OHLCV data for `THYAO.IS`, `ASELS.IS`, `XU100.IS`, `BTC-USD` and `AAPL` from `2015-01-01` to today as `data/*.csv`. The instructor uploads these files only to the cloud location shared with participants (the `SNAPSHOT_BASE_URL` constant in `src/workshop_utils.py`).

2. **Offline / CI environment:**
   When `yfinance` access fails, the `load_prices` function falls back, in this order, to:
   - the local `data/{ticker}.csv` file,
   - the CSV file at `SNAPSHOT_BASE_URL`,
   - if neither is available, the calibrated deterministic offline generator in `workshop_utils.py` (for the five default tickers).
   This way every cell runs without problems, even in a clean, fully offline Colab runtime or in CI tests.
