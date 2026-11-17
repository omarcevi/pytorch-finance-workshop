"""Tests for the TODO self-checks: each passes on a correct answer (including correct alternatives)
and explains the mistake on a wrong one."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from workshop_utils import (
    FEATURE_COLS,
    REQUIRED_OHLCV_COLS,
    SlidingWindowDataset,
    WorkshopCheckError,
    catch_up_part_2,
    check_todo_1,
    check_todo_2,
    check_todo_3,
    check_todo_4,
    check_todo_5,
    check_todo_6,
    check_todo_7,
    check_todo_8,
    check_todo_9,
    chronological_split,
    fit_scaler_on_train,
    set_seed,
    train_direction_classifier,
)


@pytest.fixture(scope="module")
def state() -> dict:
    return catch_up_part_2("THYAO.IS")


@pytest.fixture(scope="module")
def feat_raw(state: dict) -> pd.DataFrame:
    df = state["raw_df"][REQUIRED_OHLCV_COLS].ffill().dropna().copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["roll_mean_5"] = df["log_return"].rolling(window=5).mean()
    df["roll_std_10"] = df["log_return"].rolling(window=10).std()
    return df


# --- TODO 1 ------------------------------------------------------------------------------


def test_check_todo_1(state: dict, capsys: pytest.CaptureFixture[str]) -> None:
    check_todo_1(state["raw_df"])
    out = capsys.readouterr().out
    assert out.startswith("✅ TODO 1 looks right")
    assert str(len(state["raw_df"])) in out

    with pytest.raises(WorkshopCheckError, match="OHLCV"):
        check_todo_1(state["raw_df"][["Close"]])
    with pytest.raises(WorkshopCheckError, match="DataFrame"):
        check_todo_1(None)


# --- TODO 2 ------------------------------------------------------------------------------


def test_check_todo_2(feat_raw: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    check_todo_2(feat_raw)
    assert capsys.readouterr().out.startswith("✅ TODO 2 looks right")

    alt = feat_raw.copy()
    alt["log_return"] = np.log(alt["Close"]).diff()
    check_todo_2(alt)

    wrong_mean = feat_raw.copy()
    wrong_mean["roll_mean_5"] = wrong_mean["log_return"].rolling(window=10).mean()
    with pytest.raises(WorkshopCheckError, match="roll_mean_5"):
        check_todo_2(wrong_mean)

    wrong_ret = feat_raw.copy()
    wrong_ret["log_return"] = wrong_ret["Close"].pct_change()
    with pytest.raises(WorkshopCheckError, match="log_return"):
        check_todo_2(wrong_ret)


# --- TODO 3 ------------------------------------------------------------------------------


def test_check_todo_3(state: dict, capsys: pytest.CaptureFixture[str]) -> None:
    feat_df = state["feat_df"]
    check_todo_3(*chronological_split(feat_df), feat_df)
    assert capsys.readouterr().out.startswith("✅ TODO 3 looks right")

    shuffled = feat_df.sample(frac=1, random_state=0)
    with pytest.raises(WorkshopCheckError, match="chronological"):
        check_todo_3(*chronological_split(shuffled), feat_df)

    with pytest.raises(WorkshopCheckError, match="70%"):
        check_todo_3(*chronological_split(feat_df, train_ratio=0.8, val_ratio=0.1), feat_df)


# --- TODO 4 ------------------------------------------------------------------------------


def test_check_todo_4(state: dict, capsys: pytest.CaptureFixture[str]) -> None:
    train_df, val_df, test_df = state["train_df"], state["val_df"], state["test_df"]
    _, x_train, x_val, x_test = fit_scaler_on_train(train_df, val_df, test_df, FEATURE_COLS)
    check_todo_4(x_train, x_val, x_test, train_df, val_df, test_df)
    assert capsys.readouterr().out.startswith("✅ TODO 4 looks right")

    scaler = StandardScaler().fit(train_df[FEATURE_COLS].values)
    alt = [scaler.transform(d[FEATURE_COLS].values) for d in (train_df, val_df, test_df)]
    check_todo_4(*alt, train_df, val_df, test_df)

    leaky = StandardScaler().fit(pd.concat([train_df, val_df, test_df])[FEATURE_COLS].values)
    leaked = [leaky.transform(d[FEATURE_COLS].values) for d in (train_df, val_df, test_df)]
    with pytest.raises(WorkshopCheckError, match="all the data"):
        check_todo_4(*leaked, train_df, val_df, test_df)


# --- TODO 5 ------------------------------------------------------------------------------


def _datasets(state: dict, rows: int | None = None) -> tuple[SlidingWindowDataset, SlidingWindowDataset, np.ndarray, np.ndarray]:
    x_train, x_val = state["x_train"][:rows], state["x_val"]
    y_train = state["train_df"]["log_return"].values.astype(np.float32)[:rows]
    y_val = state["val_df"]["log_return"].values.astype(np.float32)
    return SlidingWindowDataset(x_train, y_train, 30), SlidingWindowDataset(x_val, y_val, 30), x_train, x_val


def test_check_todo_5(state: dict, capsys: pytest.CaptureFixture[str]) -> None:
    train_ds, val_ds, x_train, x_val = _datasets(state)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    check_todo_5(train_ds, val_ds, DataLoader(train_ds, batch_size=64, shuffle=True), val_loader, x_train, x_val)
    assert capsys.readouterr().out.startswith("✅ TODO 5 looks right")

    with pytest.raises(WorkshopCheckError, match="shuffle=True"):
        check_todo_5(train_ds, val_ds, DataLoader(train_ds, batch_size=64, shuffle=False), val_loader, x_train, x_val)


# --- TODO 6 ------------------------------------------------------------------------------


class _LSTMBase(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)


class GoodLSTM(_LSTMBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class NoSqueezeLSTM(_LSTMBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class FirstStepLSTM(_LSTMBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, 0, :]).squeeze(-1)


def test_check_todo_6(capsys: pytest.CaptureFixture[str]) -> None:
    check_todo_6(GoodLSTM)
    assert capsys.readouterr().out.startswith("✅ TODO 6 looks right")
    with pytest.raises(WorkshopCheckError, match="squeeze"):
        check_todo_6(NoSqueezeLSTM)
    with pytest.raises(WorkshopCheckError, match="last time step"):
        check_todo_6(FirstStepLSTM)


# --- TODO 7 ------------------------------------------------------------------------------


def _train(loader: DataLoader, epochs: int, variant: str, val_loader: DataLoader | None = None) -> tuple[nn.Module, list[float]]:
    set_seed(42)
    model = GoodLSTM(input_dim=len(FEATURE_COLS), hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        model.train()
        batch_losses = []
        for bx, by in loader:
            if variant not in ("zero_grad_after_step", "no_zero_grad"):
                optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            if variant != "no_step":
                optimizer.step()
            if variant == "zero_grad_after_step":
                optimizer.zero_grad()
            batch_losses.append(loss.item())
        losses.append(float(np.mean(batch_losses)))
        if val_loader is not None:  # like the notebook: a validation pass after every epoch
            model.eval()
            with torch.no_grad():
                for vx, vy in val_loader:
                    criterion(model(vx), vy)
    return model, losses


def test_check_todo_7(state: dict, capsys: pytest.CaptureFixture[str]) -> None:
    train_ds, *_ = _datasets(state, rows=200)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model, losses = _train(loader, 2, "correct")
    rng_before = torch.get_rng_state()
    check_todo_7(model, loader, losses, epochs=2)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert capsys.readouterr().out.startswith("✅ TODO 7 looks right")

    model, losses = _train(loader, 2, "zero_grad_after_step")
    check_todo_7(model, loader, losses, epochs=2)

    # The notebook runs a validation pass after every epoch; creating that loader iterator draws from torch's RNG.
    _, val_ds, *_ = _datasets(state)
    model, losses = _train(loader, 2, "correct", val_loader=DataLoader(val_ds, batch_size=64, shuffle=False))
    check_todo_7(model, loader, losses, epochs=2)

    model, losses = _train(loader, 2, "no_zero_grad")
    with pytest.raises(WorkshopCheckError, match="don't match"):
        check_todo_7(model, loader, losses, epochs=2)

    model, losses = _train(loader, 2, "no_step")
    with pytest.raises(WorkshopCheckError, match="optimizer.step"):
        check_todo_7(model, loader, losses, epochs=2)


# --- TODO 8 ------------------------------------------------------------------------------


def test_check_todo_8(capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(0)
    actual = 100 + np.cumsum(rng.normal(0, 1, 300))
    naive = np.r_[actual[0], actual[:-1]]
    predicted = naive + rng.normal(0, 0.5, 300)
    lstm_mae = float(np.mean(np.abs(actual - predicted)))
    naive_mae = float(np.mean(np.abs(actual - naive)))
    lstm_rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    naive_rmse = float(np.sqrt(np.mean((actual - naive) ** 2)))
    arrays = (actual, predicted, naive)

    check_todo_8(lstm_mae, naive_mae, lstm_rmse, naive_rmse, *arrays)
    assert capsys.readouterr().out.startswith("✅ TODO 8 looks right")

    with pytest.raises(WorkshopCheckError, match="sqrt"):
        check_todo_8(lstm_mae, naive_mae, float(np.mean((actual - predicted) ** 2)), naive_rmse, *arrays)
    with pytest.raises(WorkshopCheckError, match="np.abs"):
        check_todo_8(float(np.mean(actual - predicted)), naive_mae, lstm_rmse, naive_rmse, *arrays)


# --- TODO 9 ------------------------------------------------------------------------------


def test_check_todo_9(state: dict, capsys: pytest.CaptureFixture[str]) -> None:
    y = {k: state[f"{k}_df"]["up_day"].values.astype(np.float32) for k in ("train", "val", "test")}
    common = {"window_size": 30, "epochs": 1, "seed": 42}
    cls_res = train_direction_classifier(state["x_train"], y["train"], state["x_val"], y["val"], state["x_test"], y["test"], **common)
    check_todo_9(cls_res, state["test_df"])
    assert capsys.readouterr().out.startswith("✅ TODO 9 looks right")

    # Validation data passed where the test data belongs
    wrong = train_direction_classifier(state["x_train"], y["train"], state["x_val"], y["val"], state["x_val"], y["val"], **common)
    with pytest.raises(WorkshopCheckError, match="up_day"):
        check_todo_9(wrong, state["test_df"])
