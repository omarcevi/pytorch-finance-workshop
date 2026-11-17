"""Bringing AI to Financial Data — PyTorch workshop helper module.

This module holds data download (with offline/snapshot fallbacks), feature engineering,
the chronological split, sliding-window datasets, the PyTorch models, plotting helpers
and the 'catch up' functions run at the start of every part.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import random
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Default tickers and date range
DEFAULT_TICKERS: tuple[str, ...] = ("THYAO.IS", "ASELS.IS", "XU100.IS", "BTC-USD", "AAPL")
DEFAULT_START: str = "2015-01-01"
DEFAULT_END: str = "2026-11-17"
MIN_HISTORY_ROWS: int = 120

# Under the Yahoo Finance terms of use the raw CSVs are not added to the public git repo.
# Participants read the snapshot from one fixed URL or from a local data/ folder.
SNAPSHOT_BASE_URL: str = os.environ.get(
    "WORKSHOP_SNAPSHOT_URL",
    "https://gist.githubusercontent.com/omarcevi/ac2e894a14dd4d068a3baa9935631ec5/raw",
)

FEATURE_COLS: list[str] = ["log_return", "roll_mean_5", "roll_std_10", "rsi_14"]
REQUIRED_OHLCV_COLS: list[str] = ["Open", "High", "Low", "Close", "Volume"]

# Calibrated parameters for a completely offline runtime (no network and no local CSV)
_OFFLINE_CALIBRATION: dict[str, dict[str, float]] = {
    "THYAO.IS": {"start_price": 6.50, "drift": 0.00132, "vol": 0.0245, "base_vol": 4.5e7, "seed": 101},
    "ASELS.IS": {"start_price": 1.45, "drift": 0.00185, "vol": 0.0235, "base_vol": 3.2e7, "seed": 202},
    "XU100.IS": {"start_price": 855.0, "drift": 0.00095, "vol": 0.0155, "base_vol": 2.5e9, "seed": 303},
    "BTC-USD": {"start_price": 314.0, "drift": 0.00135, "vol": 0.0360, "base_vol": 1.8e10, "seed": 404},
    "AAPL": {"start_price": 24.50, "drift": 0.00088, "vol": 0.0178, "base_vol": 8.5e7, "seed": 505},
}


def set_seed(seed: int = 42) -> None:
    """Fixes every source of randomness (for reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _candidate_data_dirs(custom_dir: Path | str | None = None) -> list[Path]:
    """Returns the folders searched for local snapshot CSV files."""
    dirs: list[Path] = []
    if custom_dir is not None:
        dirs.append(Path(custom_dir))
    env_dir = os.environ.get("WORKSHOP_DATA_DIR")
    if env_dir:
        dirs.append(Path(env_dir))
    here = Path(__file__).resolve().parent
    dirs.extend(
        [
            Path.cwd() / "data",
            here.parent / "data",
            here / "data",
            Path("/content/pytorch-finance-workshop/data"),
            Path("/content/data"),
        ]
    )
    unique_dirs: list[Path] = []
    for d in dirs:
        if d not in unique_dirs:
            unique_dirs.append(d)
    return unique_dirs


def _generate_calibrated_offline_df(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Generates a calibrated synthetic series for the default tickers when there is no network and no local CSV."""
    cfg = _OFFLINE_CALIBRATION[ticker]
    rng = np.random.default_rng(int(cfg["seed"]))
    freq = "D" if ticker == "BTC-USD" else "B"
    dates = pd.date_range(start=start, end=end, freq=freq)
    n = len(dates)
    # Realistic volatility clustering (GARCH-like) and slight negative autocorrelation
    rets = np.zeros(n, dtype=np.float64)
    sigma = cfg["vol"]
    for i in range(1, n):
        z = rng.standard_normal()
        sigma = 0.92 * sigma + 0.08 * cfg["vol"] * (0.5 + abs(z))
        rets[i] = cfg["drift"] - 0.04 * rets[i - 1] + sigma * z
    close = cfg["start_price"] * np.exp(np.cumsum(rets))
    open_p = np.empty_like(close)
    open_p[0] = close[0]
    open_p[1:] = close[:-1] * np.exp(rng.normal(0.0, 0.003, size=n - 1))
    spread = np.abs(rng.normal(0.012, 0.006, size=n))
    high = np.maximum(open_p, close) * (1.0 + spread)
    low = np.minimum(open_p, close) * (1.0 - spread)
    volume = (cfg["base_vol"] * np.exp(rng.normal(0.0, 0.45, size=n))).astype(np.int64)
    df = pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df


def _normalize_ohlcv_df(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Brings the DataFrame's columns and date index into the standard OHLCV format."""
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_OHLCV_COLS)

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out.sort_index()

    missing_cols = [c for c in REQUIRED_OHLCV_COLS if c not in out.columns]
    if missing_cols:
        raise ValueError(f"The data is missing expected OHLCV columns: {missing_cols}")

    out = out[REQUIRED_OHLCV_COLS].dropna(how="all")
    if start:
        out = out.loc[out.index >= pd.to_datetime(start)]
    if end:
        out = out.loc[out.index <= pd.to_datetime(end)]
    out.index.name = "Date"
    return out


def _load_snapshot_fallback(
    ticker: str,
    start: str | None,
    end: str | None,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Loads data from a local data/ folder, from SNAPSHOT_BASE_URL, or from the calibrated offline generator."""
    # 1. Check local CSV files
    for d in _candidate_data_dirs(data_dir):
        csv_path = d / f"{ticker}.csv"
        if csv_path.is_file():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            return _normalize_ohlcv_df(df, start, end)

    # 2. Try the remote SNAPSHOT_BASE_URL (unless offline mode is forced)
    if os.environ.get("WORKSHOP_OFFLINE", "0") != "1" and SNAPSHOT_BASE_URL.startswith("http"):
        url = f"{SNAPSHOT_BASE_URL.rstrip('/')}/{ticker}.csv"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                csv_bytes = resp.read()
            df = pd.read_csv(io.BytesIO(csv_bytes), index_col=0, parse_dates=True)
            return _normalize_ohlcv_df(df, start, end)
        except Exception:  # noqa: BLE001, S110
            pass

    # 3. For one of the 5 default tickers, generate the calibrated offline series
    if ticker in _OFFLINE_CALIBRATION:
        df = _generate_calibrated_offline_df(
            ticker,
            start or DEFAULT_START,
            end or DEFAULT_END,
        )
        return _normalize_ohlcv_df(df, start, end)

    raise ValueError(
        f"Could not get data for '{ticker}' from Yahoo Finance or from the snapshot. "
        "Please enter a valid ticker (e.g. 'THYAO.IS', 'ASELS.IS', 'AAPL')."
    )


def load_prices(
    ticker: str = "THYAO.IS",
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    *,
    force_fallback: bool = False,
    data_dir: Path | str | None = None,
    min_rows: int = MIN_HISTORY_ROWS,
) -> pd.DataFrame:
    """Downloads prices with yfinance first; on an error or a rate limit it loads the snapshot instead.

    Args:
        ticker: Ticker symbol (e.g. 'THYAO.IS', 'ASELS.IS', 'XU100.IS', 'BTC-USD', 'AAPL').
        start: Start date ('YYYY-MM-DD').
        end: End date ('YYYY-MM-DD').
        force_fallback: If True, use the snapshot directly.
        data_dir: A custom folder to search for snapshot CSV files.
        min_rows: The minimum number of trading days accepted.
    """
    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Invalid ticker: please enter a non-empty ticker symbol.")
    clean_ticker = ticker.strip().upper()

    use_fallback = force_fallback or (os.environ.get("WORKSHOP_FORCE_FALLBACK", "0") == "1")
    df: pd.DataFrame | None = None

    if not use_fallback:
        try:
            if hasattr(yf, "shared") and hasattr(yf.shared, "_ERRORS"):
                yf.shared._ERRORS.clear()
            raw = yf.download(
                clean_ticker,
                start=start,
                end=end,
                auto_adjust=True,
                multi_level_index=False,
                progress=False,
                threads=False,
            )
            yf_err = ""
            if hasattr(yf, "shared") and hasattr(yf.shared, "_ERRORS"):
                yf_err = yf.shared._ERRORS.get(clean_ticker, "")
            if yf_err:
                raise RuntimeError(f"yfinance error: {yf_err}")
            if raw is None or raw.empty:
                raise RuntimeError("Yahoo Finance returned an empty table")
            df = _normalize_ohlcv_df(raw, start, end)
            if df.empty:
                raise RuntimeError("No data in the selected date range")
        except Exception as exc:  # noqa: BLE001
            print(
                f"⚠️ Could not get {clean_ticker} from Yahoo Finance ({type(exc).__name__}: {exc}). "
                "Loading the snapshot data..."
            )
            df = _load_snapshot_fallback(clean_ticker, start, end, data_dir=data_dir)
    else:
        print(f"ℹ️ Using the snapshot data for {clean_ticker}.")
        df = _load_snapshot_fallback(clean_ticker, start, end, data_dir=data_dir)

    if len(df) < min_rows:
        raise ValueError(
            f"Not enough history for '{clean_ticker}': found {len(df)} days, "
            f"need at least {min_rows} trading days. Please pick a ticker with a longer history."
        )
    return df


# --- Part 2: cleaning, features and the chronological split ------------------------------


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes the Relative Strength Index (RSI, 0-100) from close prices."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans missing values and builds the log return, moving average, volatility and RSI features."""
    out = df[REQUIRED_OHLCV_COLS].copy()
    # Fill holidays or missing days with past values only (never leak information from the future!)
    out = out.ffill().dropna()

    # Log return instead of price: r_t = ln(Close_t / Close_{t-1})
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out["roll_mean_5"] = out["log_return"].rolling(window=5).mean()
    out["roll_std_10"] = out["log_return"].rolling(window=10).std()
    out["rsi_14"] = compute_rsi(out["Close"], period=14) / 100.0
    out["volume_change"] = np.log((out["Volume"] + 1.0) / (out["Volume"].shift(1) + 1.0)).clip(-3.0, 3.0)

    # The day's direction: 1.0 (up), 0.0 (down or flat)
    out["up_day"] = (out["log_return"] > 0.0).astype(np.float32)

    return out.dropna().copy()


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the data into train, validation and test without breaking time order (chronologically)."""
    if not (0.0 < train_ratio < 1.0) or not (0.0 < val_ratio < 1.0) or (train_ratio + val_ratio >= 1.0):
        raise ValueError("train_ratio and val_ratio must each be in (0, 1) and sum to less than 1.")
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def fit_scaler_on_train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the StandardScaler on the training data ONLY (prevents data leakage)."""
    cols = feature_cols or FEATURE_COLS
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[cols].values.astype(np.float32)).astype(np.float32)
    x_val = scaler.transform(val_df[cols].values.astype(np.float32)).astype(np.float32)
    x_test = scaler.transform(test_df[cols].values.astype(np.float32)).astype(np.float32)
    return scaler, x_train, x_val, x_test


# --- Parts 3, 4 and 5: PyTorch datasets and models ---------------------------------------


class SlidingWindowDataset(Dataset):
    """PyTorch Dataset that builds sliding windows over feature and target arrays.

    Window i:
      - Input (x): features[i : i + window_size]  (days t - W ... t - 1)
      - Target (y): targets[i + window_size]       (day t, the day right after the window)
    So the target day (i + window_size) is never inside the input window.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, window_size: int = 30) -> None:
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same length.")
        if len(features) <= window_size:
            raise ValueError(f"The data length ({len(features)}) must be greater than the window size ({window_size}).")
        self.features = np.asarray(features, dtype=np.float32)
        if self.features.ndim == 1:
            self.features = self.features[:, None]
        self.targets = np.asarray(targets, dtype=np.float32)
        self.window_size = int(window_size)

    def __len__(self) -> int:
        return len(self.features) - self.window_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class PriceWindowDataset(Dataset):
    """Dataset of window-normalised prices for the Part 4 price prediction.

    Note: for a stock like THYAO, which went from 5 TL to 340 TL over the years, a single global
    scaler fitted on the whole training set puts test prices at z > 10, which saturates the LSTM's
    tanh/sigmoid gates. To keep the story clean, each 30-day window is expressed only relative to
    its own first day (w / w[0] - 1). The target is the price of the day after the window
    (idx + window_size), relative to the same base.
    """

    def __init__(self, prices: np.ndarray, window_size: int = 30) -> None:
        if len(prices) <= window_size:
            raise ValueError("The price array must be longer than the window size.")
        self.prices = np.asarray(prices, dtype=np.float32).reshape(-1)
        self.window_size = int(window_size)

    def __len__(self) -> int:
        return len(self.prices) - self.window_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self.prices[idx : idx + self.window_size]
        target_price = self.prices[idx + self.window_size]
        base_price = window[0]
        last_price = window[-1]
        x_rel = ((window / base_price) - 1.0).reshape(self.window_size, 1)
        y_rel = (target_price / base_price) - 1.0
        return (
            torch.from_numpy(x_rel.astype(np.float32)),
            torch.tensor(y_rel, dtype=torch.float32),
            torch.tensor(base_price, dtype=torch.float32),
            torch.tensor(last_price, dtype=torch.float32),
        )


class LSTMModel(nn.Module):
    """LSTM model for regression (Parts 3-4) and for binary classification logits (Part 5)."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take only the hidden state of the last time step: (batch, hidden_dim)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


class GRUModel(nn.Module):
    """GRU-based alternative model for the Part 6 challenge."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class TinyTransformerModel(nn.Module):
    """Small Transformer encoder model for the Part 6 bonus challenge."""

    def __init__(self, input_dim: int = 4, d_model: int = 16, nhead: int = 2, max_len: int = 128) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=32,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        h = self.input_proj(x) + self.pos_emb[:, :seq_len, :]
        h = self.encoder(h)
        return self.fc(h[:, -1, :]).squeeze(-1)


# --- Training and evaluation helpers -----------------------------------------------------


def train_price_lstm(
    train_prices: np.ndarray,
    val_prices: np.ndarray,
    test_prices: np.ndarray,
    *,
    window_size: int = 30,
    hidden_dim: int = 32,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.003,
    seed: int = 42,
) -> dict[str, Any]:
    """Trains the Part 4 price model and compares it with the naive baseline."""
    set_seed(seed)
    train_ds = PriceWindowDataset(train_prices, window_size=window_size)
    val_ds = PriceWindowDataset(val_prices, window_size=window_size)
    test_ds = PriceWindowDataset(test_prices, window_size=window_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_dim=1, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        batch_losses: list[float] = []
        for bx, by, _, _ in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        v_losses: list[float] = []
        with torch.no_grad():
            for bx, by, _, _ in val_loader:
                v_losses.append(float(criterion(model(bx), by).item()))
        val_losses.append(float(np.mean(v_losses)))

    model.eval()
    preds_tl: list[np.ndarray] = []
    actual_tl: list[np.ndarray] = []
    naive_tl: list[np.ndarray] = []
    with torch.no_grad():
        for bx, by, base_p, last_p in test_loader:
            pred_rel = model(bx)
            preds_tl.append(((pred_rel + 1.0) * base_p).numpy())
            actual_tl.append(((by + 1.0) * base_p).numpy())
            naive_tl.append(last_p.numpy())

    pred_arr = np.concatenate(preds_tl)
    actual_arr = np.concatenate(actual_tl)
    naive_arr = np.concatenate(naive_tl)

    lstm_mae = float(np.mean(np.abs(actual_arr - pred_arr)))
    naive_mae = float(np.mean(np.abs(actual_arr - naive_arr)))
    lstm_rmse = float(np.sqrt(np.mean((actual_arr - pred_arr) ** 2)))
    naive_rmse = float(np.sqrt(np.mean((actual_arr - naive_arr) ** 2)))

    corr_lag0 = float(np.corrcoef(pred_arr[1:], actual_arr[1:])[0, 1])
    corr_lag1 = float(np.corrcoef(pred_arr[1:], actual_arr[:-1])[0, 1])

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "actual_prices": actual_arr,
        "predicted_prices": pred_arr,
        "naive_prices": naive_arr,
        "lstm_mae": lstm_mae,
        "naive_mae": naive_mae,
        "lstm_rmse": lstm_rmse,
        "naive_rmse": naive_rmse,
        "corr_lag0": corr_lag0,
        "corr_lag1": corr_lag1,
    }


def train_direction_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    window_size: int = 30,
    hidden_dim: int = 32,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    seed: int = 42,
    model_cls: type[nn.Module] = LSTMModel,
) -> dict[str, Any]:
    """Trains and evaluates the Part 5 return-based direction (up/down) classifier."""
    set_seed(seed)
    train_ds = SlidingWindowDataset(x_train, y_train, window_size=window_size)
    val_ds = SlidingWindowDataset(x_val, y_val, window_size=window_size)
    test_ds = SlidingWindowDataset(x_test, y_test, window_size=window_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = x_train.shape[1]
    model = model_cls(input_dim=input_dim, hidden_dim=hidden_dim) if model_cls != TinyTransformerModel else model_cls(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        b_losses: list[float] = []
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            b_losses.append(float(loss.item()))
        train_losses.append(float(np.mean(b_losses)))

        model.eval()
        v_losses: list[float] = []
        with torch.no_grad():
            for bx, by in val_loader:
                v_losses.append(float(criterion(model(bx), by).item()))
        val_losses.append(float(np.mean(v_losses)))

    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for bx, by in test_loader:
            all_logits.append(model(bx).numpy())
            all_targets.append(by.numpy())

    logits_arr = np.concatenate(all_logits)
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))
    preds_arr = (probs_arr >= 0.5).astype(np.float32)
    targets_arr = np.concatenate(all_targets)

    # Majority-class baseline (computed from the training set only)
    train_targets = y_train[window_size:]
    majority_class = 1.0 if float(np.mean(train_targets)) >= 0.5 else 0.0
    majority_preds = np.full_like(targets_arr, majority_class)

    test_acc = float(accuracy_score(targets_arr, preds_arr))
    test_bal_acc = float(balanced_accuracy_score(targets_arr, preds_arr))
    majority_acc = float(accuracy_score(targets_arr, majority_preds))
    cm = confusion_matrix(targets_arr, preds_arr, labels=[0, 1])

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "y_true": targets_arr,
        "y_pred": preds_arr,
        "y_prob": probs_arr,
        "test_acc": test_acc,
        "test_bal_acc": test_bal_acc,
        "majority_class": majority_class,
        "majority_acc": majority_acc,
        "confusion_matrix": cm,
    }


def walk_forward_evaluation(
    feat_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    *,
    n_splits: int = 4,
    window_size: int = 30,
    epochs: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Honest validation over several time periods with an expanding (walk-forward) window."""
    cols = feature_cols or FEATURE_COLS
    n = len(feat_df)
    initial_train = int(n * 0.50)
    step = (n - initial_train) // n_splits

    results: list[dict[str, Any]] = []
    for fold in range(n_splits):
        tr_end = initial_train + fold * step
        te_end = min(n, tr_end + step)
        if te_end - tr_end <= window_size + 10:
            continue

        tr_slice = feat_df.iloc[:tr_end]
        te_slice = feat_df.iloc[tr_end:te_end]

        scaler = StandardScaler()
        x_tr = scaler.fit_transform(tr_slice[cols].values.astype(np.float32)).astype(np.float32)
        x_te = scaler.transform(te_slice[cols].values.astype(np.float32)).astype(np.float32)
        y_tr = tr_slice["up_day"].values.astype(np.float32)
        y_te = te_slice["up_day"].values.astype(np.float32)

        res = train_direction_classifier(
            x_tr,
            y_tr,
            x_te,
            y_te,
            x_te,
            y_te,
            window_size=window_size,
            epochs=epochs,
            seed=seed + fold,
        )
        results.append(
            {
                "fold": fold + 1,
                "train_end_date": str(tr_slice.index[-1].date()),
                "test_end_date": str(te_slice.index[-1].date()),
                "accuracy": res["test_acc"],
                "balanced_accuracy": res["test_bal_acc"],
                "majority_acc": res["majority_acc"],
            }
        )
    return results


def backtest_with_costs(
    test_returns: np.ndarray,
    signals: np.ndarray,
    cost_bps: float = 15.0,
) -> dict[str, Any]:
    """Part 6 bonus: a simple long/flat strategy test with trading costs (commission + spread)."""
    rets = np.asarray(test_returns, dtype=np.float64).reshape(-1)
    sigs = np.asarray(signals, dtype=np.float64).reshape(-1)
    min_len = min(len(rets), len(sigs))
    rets = rets[:min_len]
    sigs = sigs[:min_len]

    # A trading cost is charged on every position change (buy or sell)
    trades = np.abs(np.diff(sigs, prepend=0.0))
    cost_rate = cost_bps / 10000.0
    gross_strat_rets = sigs * rets
    net_strat_rets = gross_strat_rets - trades * cost_rate

    cum_buy_hold = np.exp(np.cumsum(rets))
    cum_gross = np.exp(np.cumsum(gross_strat_rets))
    cum_net = np.exp(np.cumsum(net_strat_rets))

    return {
        "cum_buy_hold": cum_buy_hold,
        "cum_gross": cum_gross,
        "cum_net": cum_net,
        "n_trades": int(np.sum(trades > 0)),
        "final_buy_hold": float(cum_buy_hold[-1]),
        "final_gross": float(cum_gross[-1]),
        "final_net": float(cum_net[-1]),
    }


# --- Plotting helpers (matplotlib only) --------------------------------------------------


def plot_ohlcv(df: pd.DataFrame, ticker: str = "THYAO.IS") -> plt.Figure:
    """Part 1: plots the close price and the trading volume in two charts, one above the other."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(df.index, df["Close"], color="#1f77b4", linewidth=1.4, label="Adjusted close")
    ax1.set_title(f"{ticker} — Daily Close Price and Volume")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.bar(df.index, df["Volume"], color="#7f7f7f", width=2.0, alpha=0.6, label="Volume")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_chronological_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ticker: str = "THYAO.IS",
) -> plt.Figure:
    """Part 2: shows the chronological train (70%), validation (15%) and test (15%) regions."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(train_df.index, train_df["Close"], label="Train (70%)", color="#1f77b4", linewidth=1.4)
    ax.plot(val_df.index, val_df["Close"], label="Validation (15%)", color="#ff7f0e", linewidth=1.4)
    ax.plot(test_df.index, test_df["Close"], label="Test (15%)", color="#2ca02c", linewidth=1.4)
    ax.axvline(val_df.index[0], color="#444444", linestyle="--", alpha=0.7)
    ax.axvline(test_df.index[0], color="#444444", linestyle="--", alpha=0.7)
    ax.set_title(f"{ticker} — Chronological Split (No Leakage from the Future)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close price")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Training and Validation Loss",
) -> plt.Figure:
    """Plots the training and validation loss curves."""
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(epochs, train_losses, marker="o", label="Train loss", color="#1f77b4")
    ax.plot(epochs, val_losses, marker="s", label="Validation loss", color="#d62728")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_price_deception(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    ticker: str = "THYAO.IS",
) -> plt.Figure:
    """Part 0 (hook) and Part 4: the price prediction chart that looks 'perfect' from afar."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(dates, actual, label="Actual price", color="#1f77b4", linewidth=1.6)
    ax.plot(dates, predicted, label="LSTM prediction", color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_title(f"{ticker} Test Set — Actual Price vs. LSTM Prediction (the 'Perfect' Look)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_zoom_and_lag(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    naive: np.ndarray,
    ticker: str = "THYAO.IS",
    zoom_days: int = 35,
) -> plt.Figure:
    """Part 4 surprise: zoom into the last few weeks (a one-day shift) and the lag correlation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.6, 1.0]})

    z_dates = dates[-zoom_days:]
    z_act = actual[-zoom_days:]
    z_pred = predicted[-zoom_days:]
    z_naive = naive[-zoom_days:]

    ax1.plot(z_dates, z_act, marker="o", markersize=4, label="Actual price (P_t)", color="#1f77b4", linewidth=1.6)
    ax1.plot(
        z_dates,
        z_pred,
        marker="x",
        markersize=4,
        label="LSTM prediction",
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
    )
    ax1.plot(
        z_dates,
        z_naive,
        marker=".",
        markersize=4,
        label="Naive baseline (tomorrow = today)",
        color="#2ca02c",
        linestyle=":",
        linewidth=1.5,
    )
    ax1.set_title(f"{ticker} — Zoom on the Last {zoom_days} Days (a One-Day Lag!)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(loc="best", fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    # Lag correlation bar chart: Lag 0 (today) vs Lag 1 (yesterday)
    lags = [-2, -1, 0, 1, 2]
    corrs = []
    for lag in lags:
        if lag == 0:
            c = np.corrcoef(predicted, actual)[0, 1]
        elif lag > 0:
            c = np.corrcoef(predicted[lag:], actual[:-lag])[0, 1]
        else:
            k = -lag
            c = np.corrcoef(predicted[:-k], actual[k:])[0, 1]
        corrs.append(float(c))

    colors = ["#7f7f7f" if l != 1 else "#d62728" for l in lags]
    ax2.bar([f"Lag {l}" for l in lags], corrs, color=colors, alpha=0.85)
    ax2.set_ylim(min(0.85, min(corrs) - 0.02), 1.005)
    ax2.set_title("Correlation of Prediction and Actual Price\n(Peaks at Lag 1 = Yesterday's Price!)")
    ax2.set_ylabel("Pearson correlation (r)")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_direction_results(
    cm: np.ndarray,
    wf_results: list[dict[str, Any]],
    ticker: str = "THYAO.IS",
) -> plt.Figure:
    """Part 5: the confusion matrix and the walk-forward accuracy chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1], labels=["Predicted: down (0)", "Predicted: up (1)"])
    ax1.set_yticks([0, 1], labels=["Actual: down (0)", "Actual: up (1)"])
    ax1.set_title(f"{ticker} — Confusion Matrix (Test)")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontweight="bold")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    folds = [f"Fold {r['fold']}" for r in wf_results]
    accs = [r["accuracy"] for r in wf_results]
    maj_accs = [r["majority_acc"] for r in wf_results]
    x = np.arange(len(folds))
    width = 0.35
    ax2.bar(x - width / 2, accs, width, label="LSTM accuracy", color="#1f77b4")
    ax2.bar(x + width / 2, maj_accs, width, label="Majority-class baseline", color="#ff7f0e")
    ax2.axhline(0.50, color="#d62728", linestyle="--", linewidth=1.2, label="50% chance level")
    ax2.set_xticks(x, labels=folds)
    ax2.set_ylim(0.35, 0.70)
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{ticker} — Walk-Forward Validation Results")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# --- Notebook helpers: setup and friendly messages ---------------------------------------


class WorkshopCheckError(AssertionError):
    """Raised by a TODO self-check: says what is wrong and gives a hint."""

    def __init__(self, todo: int, problem: str, hint: str) -> None:
        super().__init__(f"TODO {todo}: {problem}")
        self.todo = todo
        self.problem = problem
        self.hint = hint


def friendly_lines(value: BaseException) -> list[str] | None:
    """Short learner-facing lines for an unfilled TODO or a failed check; None for any other error."""
    if isinstance(value, WorkshopCheckError):
        return [f"❌ TODO {value.todo}: {value.problem}", f"💡 Hint: {value.hint}"]
    if isinstance(value, NotImplementedError) and str(value).startswith("TODO"):
        return [
            f"✏️ {value} isn't done yet.",
            "Replace the raise NotImplementedError(...) line with your code, then run the cell again.",
        ]
    return None


def setup_notebook() -> None:
    """Projector-friendly charts, quiet yfinance logs and, in IPython, friendly TODO/check messages."""
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10})
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover - IPython ships with Jupyter and Colab
        return
    shell = get_ipython()
    if shell is None:
        return
    with contextlib.suppress(Exception):
        from matplotlib_inline.backend_inline import set_matplotlib_formats

        set_matplotlib_formats("retina")

    # IPython (7.x in Colab, 9.x locally) ignores the handler's return value, so the handler shows the output itself.
    def _handler(self, etype, value, tb, tb_offset=None):
        lines = friendly_lines(value)
        if lines is not None:
            self._showtraceback(etype, value, lines)
        else:
            self.showtraceback((etype, value, tb), tb_offset=tb_offset)

    shell.set_custom_exc((NotImplementedError, WorkshopCheckError), _handler)


# --- 'Catch up' functions ----------------------------------------------------------------


def catch_up_part_1(ticker: str = "THYAO.IS") -> dict[str, Any]:
    """For the start of Part 1: downloads/loads the raw OHLCV data and returns it."""
    set_seed(42)
    raw_df = load_prices(ticker)
    print(f"✅ Part 1 ready: loaded {len(raw_df)} days of OHLCV data for {ticker}.")
    return {"ticker": ticker, "raw_df": raw_df}


def catch_up_part_2(ticker: str = "THYAO.IS") -> dict[str, Any]:
    """For the end of Part 2 / start of Part 3: prepares the cleaned features, the split and the scaler."""
    state = catch_up_part_1(ticker)
    feat_df = build_features(state["raw_df"])
    train_df, val_df, test_df = chronological_split(feat_df)
    scaler, x_train, x_val, x_test = fit_scaler_on_train(train_df, val_df, test_df, FEATURE_COLS)
    state.update(
        {
            "feat_df": feat_df,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "feature_cols": list(FEATURE_COLS),
            "scaler": scaler,
            "x_train": x_train,
            "x_val": x_val,
            "x_test": x_test,
        }
    )
    print(
        f"✅ Part 2 ready: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} rows "
        f"and {len(FEATURE_COLS)} features scaled."
    )
    return state


def catch_up_part_3(
    ticker: str = "THYAO.IS",
    window_size: int = 30,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    """For the end of Part 3 / start of Part 4: prepares the PyTorch Dataset, DataLoader and base LSTM objects."""
    state = catch_up_part_2(ticker)
    set_seed(seed)
    y_train = state["train_df"]["log_return"].values.astype(np.float32)
    y_val = state["val_df"]["log_return"].values.astype(np.float32)
    y_test = state["test_df"]["log_return"].values.astype(np.float32)

    train_ds = SlidingWindowDataset(state["x_train"], y_train, window_size=window_size)
    val_ds = SlidingWindowDataset(state["x_val"], y_val, window_size=window_size)
    test_ds = SlidingWindowDataset(state["x_test"], y_test, window_size=window_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = LSTMModel(input_dim=len(FEATURE_COLS), hidden_dim=32)

    state.update(
        {
            "window_size": window_size,
            "batch_size": batch_size,
            "train_ds": train_ds,
            "val_ds": val_ds,
            "test_ds": test_ds,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "demo_model": model,
        }
    )
    print(f"✅ Part 3 ready: window size={window_size}, training samples={len(train_ds)}.")
    return state


def catch_up_part_4(
    ticker: str = "THYAO.IS",
    window_size: int = 30,
    epochs: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    """Prepares the Part 4 price model and the naive-baseline comparison."""
    state = catch_up_part_2(ticker)
    price_res = train_price_lstm(
        state["train_df"]["Close"].values,
        state["val_df"]["Close"].values,
        state["test_df"]["Close"].values,
        window_size=window_size,
        epochs=epochs,
        seed=seed,
    )
    test_dates = state["test_df"].index[window_size:]
    state.update({"price_res": price_res, "test_dates": test_dates, "window_size": window_size})
    print(
        f"✅ Part 4 ready: LSTM MAE={price_res['lstm_mae']:.2f} TL | "
        f"naive baseline (tomorrow = today) MAE={price_res['naive_mae']:.2f} TL."
    )
    return state


def catch_up_part_5(
    ticker: str = "THYAO.IS",
    window_size: int = 30,
    epochs: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    """Prepares the Part 5 direction classifier and the walk-forward evaluation."""
    state = catch_up_part_2(ticker)
    y_train = state["train_df"]["up_day"].values.astype(np.float32)
    y_val = state["val_df"]["up_day"].values.astype(np.float32)
    y_test = state["test_df"]["up_day"].values.astype(np.float32)

    cls_res = train_direction_classifier(
        state["x_train"],
        y_train,
        state["x_val"],
        y_val,
        state["x_test"],
        y_test,
        window_size=window_size,
        epochs=epochs,
        seed=seed,
    )
    wf_res = walk_forward_evaluation(
        state["feat_df"],
        FEATURE_COLS,
        n_splits=4,
        window_size=window_size,
        epochs=10,
        seed=seed,
    )
    state.update({"cls_res": cls_res, "wf_res": wf_res, "window_size": window_size})
    print(
        f"✅ Part 5 ready: test accuracy={cls_res['test_acc']:.3f} | "
        f"balanced accuracy={cls_res['test_bal_acc']:.3f} | "
        f"majority class={cls_res['majority_acc']:.3f}."
    )
    return state


def catch_up_part_6(
    ticker: str = "THYAO.IS",
    window_size: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Prepares the full dataset and the classification results for the Part 6 challenges."""
    state = catch_up_part_5(ticker=ticker, window_size=window_size, epochs=15, seed=seed)
    print("✅ Part 6 (challenges): every variable is ready!")
    return state


# Earlier (Turkish) names, kept so notebooks opened before the rename still run.
yetistir_bolum_1 = catch_up_part_1
yetistir_bolum_2 = catch_up_part_2
yetistir_bolum_3 = catch_up_part_3
yetistir_bolum_4 = catch_up_part_4
yetistir_bolum_5 = catch_up_part_5
yetistir_bolum_6 = catch_up_part_6
