"""Tests for the TODO self-checks: each passes on a correct answer (including correct alternatives)
and explains the mistake on a wrong one."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from workshop_utils import (
    FEATURE_COLS,
    REQUIRED_OHLCV_COLS,
    WorkshopCheckError,
    catch_up_part_2,
    check_todo_1,
    check_todo_2,
    check_todo_3,
    check_todo_4,
    chronological_split,
    fit_scaler_on_train,
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
