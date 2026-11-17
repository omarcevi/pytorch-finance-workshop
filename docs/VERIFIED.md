# Verified Environment and API Behaviour (`VERIFIED.md`)

This document records the package versions, the `yfinance` API behaviour and the data fallback decisions that were tested and verified first-hand in the runtime while building the **"Bringing AI to Financial Data"** PyTorch workshop.

---

## 1. Google Colab Runtime Versions (17 November 2026)

| Package | Verified version | Notes |
|---|---|---|
| `torch` | `2.14.1+cu130` | Every training cell finishes in < 15 s on CPU. |
| `yfinance` | `0.2.66` | Uses a `curl_cffi`-based `requests.Session(impersonate="chrome")`. |
| `pandas` | `2.2.3` | `DatetimeIndex` and time-series operations verified. |
| `numpy` | `2.4.6` | Fixed seeding (`np.random.seed`) verified. |
| `scikit-learn` | `1.6.1` | Used for `StandardScaler` and the classification metrics. |
| `matplotlib` | `3.10.0` | All visualisations produced with no extra libraries. |

---

## 2. `yfinance` (`0.2.66`) API Verification

### 2.1. The `yf.download` Signature and Default Arguments

```python
yf.download(
    tickers, start=None, end=None, actions=False, threads=True,
    ignore_tz=None, group_by='column', auto_adjust=None, back_adjust=False,
    repair=False, keepna=False, progress=True, period=None, interval='1d',
    prepost=False, proxy=_SENTINEL_, rounding=False, timeout=10,
    session=None, multi_level_index=True
)
```

Verified critical behaviour:

1. **The `auto_adjust` default (`auto_adjust=None` → `True`):**
   - The function signature defines `auto_adjust=None`. If the argument is not passed, it emits `FutureWarning: YF.download() has changed argument auto_adjust default to True` and uses `auto_adjust=True` internally.
   - With `auto_adjust=True` the returned columns are `['Close', 'High', 'Low', 'Open', 'Volume']`; there is **no** separate `'Adj Close'` column (the close and OHLC prices are already adjusted for dividends and splits).
   - We pass `auto_adjust=True` explicitly to silence the warning and pin the behaviour.
2. **`MultiIndex` columns even for a single ticker (`multi_level_index=True`):**
   - Even when `yf.download("THYAO.IS", ...)` is called with a single ticker, the default `multi_level_index=True` returns the columns as a `pd.MultiIndex`: `[('Close', 'THYAO.IS'), ('High', 'THYAO.IS'), ...]`.
   - To get single-level columns, pass `multi_level_index=False` or check `isinstance(df.columns, pd.MultiIndex)` and drop a level (`df.columns.get_level_values(0)`). `load_prices` safely flattens both cases.
3. **Time zone (`ignore_tz`):**
   - For daily data (`interval="1d"`), `yf.download` uses `ignore_tz=True` by default and strips the time zone from the `DatetimeIndex` (`tz_localize(None)`).

### 2.2. Error Classes (`yfinance.exceptions`) and Rate-Limit Behaviour

Classes in the `yfinance.exceptions` module:
- `YFException` (base class)
- `YFRateLimitError` (`"Too Many Requests. Rate limited. Try after a while."`)
- `YFPricesMissingError`
- `YFTzMissingError`
- `YFTickerMissingError`
- `YFDataException`, `YFEarningsDateMissing`, `YFInvalidPeriodError`, `YFNotImplementedError`

### 2.3. Key Differences Between `yf.download` and `Ticker.history`

| Feature | `yf.download(ticker, ...)` | `yf.Ticker(ticker).history(...)` |
|---|---|---|
| **Rate-limit (`YFRateLimitError`) behaviour** | `yfinance.multi._download_one` **catches** every `Exception` with `try/except`, writes it to `yfinance.shared._ERRORS[ticker]` and returns an **empty `DataFrame`** (it does not raise!). | `PriceHistory.history` contains an `except YFRateLimitError: raise` block; even with `raise_errors=False` (the default) it **raises** `YFRateLimitError` **directly**. |
| **Other errors (invalid ticker, network error)** | Catches them, records them in `shared._ERRORS` and returns an empty `DataFrame`. | With `raise_errors=False` (the default) it returns an empty `DataFrame`; with `raise_errors=True` it raises `YFTzMissingError` / `YFPricesMissingError`. |
| **Column structure** | `MultiIndex` by default (`multi_level_index=True`), 5 columns (`Open, High, Low, Close, Volume`). | Always a single-level `Index`; because of the default `actions=True` it also includes `Dividends` and `Stock Splits` columns. |
| **Time zone (`DatetimeIndex.tz`)** | For daily data `ignore_tz=True` → `tz=None`. | Returns a `DatetimeIndex` in the exchange's local time zone (`Europe/Istanbul`, `America/New_York`, etc.). |

**Conclusion:** the `load_prices` function in `src/workshop_utils.py`:
- catches both `YFRateLimitError` and general `Exception`s,
- checks both the `yfinance.shared._ERRORS` record and whether the returned `DataFrame` is empty,
- standardises `MultiIndex` columns and time-zone information, and always returns a clean `DataFrame` with single-level `['Open', 'High', 'Low', 'Close', 'Volume']` columns and a `tz=None` `DatetimeIndex`.

---

## 3. Yahoo Finance Terms of Service and the Snapshot Decision

- **Terms:** The Yahoo Terms of Service do not allow Yahoo Finance market data to be redistributed in a public repository (a public GitHub repo).
- **Solution:**
  1. The `data/*.csv` files are listed in `.gitignore` and kept out of git history.
  2. `src/workshop_utils.py` defines a single constant (`SNAPSHOT_BASE_URL = "https://gist.githubusercontent.com/omarcevi/ac2e894a14dd4d068a3baa9935631ec5/raw"`); instead of being added to the public repository, the data is hosted on a secret (unlisted) GitHub Gist (`https://gist.github.com/omarcevi/ac2e894a14dd4d068a3baa9935631ec5`) that can only be reached through the link.
  3. `load_prices`, in this order:
     - first tries a live `yfinance` download,
     - on an error, an empty table or a rate limit, prints a warning (for example `⚠️ Could not get THYAO.IS from Yahoo Finance (...). Loading the snapshot data...`) and falls back first to the local `data/{ticker}.csv` cache, then to the `SNAPSHOT_BASE_URL` address, and finally to the calibrated deterministic offline generator, which guarantees that it works even in a completely offline, clean Colab runtime.
