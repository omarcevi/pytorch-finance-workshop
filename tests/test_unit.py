"""Unit tests (no network needed):
- load_prices fallback behaviour on errors, empty tables and rate limits (YFRateLimitError)
- SlidingWindowDataset and PriceWindowDataset shapes and target leakage
- The chronological split and a StandardScaler fitted on train only
- make_starter.py solution-block removal
- Every 'catch up' cell works on its own in a fresh process, and so does the rest of its part after it
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_starter import convert_notebook, strip_solution_blocks

from workshop_utils import (
    FEATURE_COLS,
    PriceWindowDataset,
    SlidingWindowDataset,
    build_features,
    chronological_split,
    fit_scaler_on_train,
    load_prices,
)

# --- 1. load_prices fallback tests (no network / mocked) ---------------------------------


def test_load_prices_fallback_on_exception(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When yfinance raises a generic exception, load_prices falls back to the snapshot."""
    monkeypatch.delenv("WORKSHOP_FORCE_FALLBACK", raising=False)
    with patch("workshop_utils.yf.download", side_effect=RuntimeError("Network error")):
        df = load_prices("THYAO.IS", force_fallback=False)
    captured = capsys.readouterr().out
    assert "Loading the snapshot data" in captured
    assert not df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_load_prices_fallback_on_empty_dataframe(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When yfinance returns an empty DataFrame, load_prices falls back to the snapshot."""
    monkeypatch.delenv("WORKSHOP_FORCE_FALLBACK", raising=False)
    with patch("workshop_utils.yf.download", return_value=pd.DataFrame()):
        df = load_prices("ASELS.IS", force_fallback=False)
    captured = capsys.readouterr().out
    assert "Loading the snapshot data" in captured
    assert len(df) > 1000


def test_load_prices_fallback_on_rate_limit_error(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On YFRateLimitError, or when yfinance leaves an entry in shared._ERRORS, load_prices falls back to the snapshot."""
    monkeypatch.delenv("WORKSHOP_FORCE_FALLBACK", raising=False)
    rate_exc = yf.exceptions.YFRateLimitError()
    with patch("workshop_utils.yf.download", side_effect=rate_exc):
        df = load_prices("XU100.IS", force_fallback=False)
    captured = capsys.readouterr().out
    assert "YFRateLimitError" in captured
    assert len(df) > 1000

    # Also test yf.download's real behaviour of swallowing the error and writing it to shared._ERRORS
    def _fake_swallowed_rate_limit(*args, **kwargs):
        yf.shared._ERRORS["AAPL"] = "YFRateLimitError('Too Many Requests.')"
        return pd.DataFrame()

    with patch("workshop_utils.yf.download", side_effect=_fake_swallowed_rate_limit):
        df_aapl = load_prices("AAPL", force_fallback=False)
    assert len(df_aapl) > 1000


def test_load_prices_invalid_ticker_and_short_history() -> None:
    """An invalid ticker and a very short history raise a clear ValueError."""
    with pytest.raises(ValueError, match="Could not get data for"):
        load_prices("NO_SUCH_TICKER_999", force_fallback=True)

    with pytest.raises(ValueError, match="Not enough history"):
        load_prices("THYAO.IS", start="2026-10-01", end="2026-11-15", force_fallback=True, min_rows=120)


# --- 2. SlidingWindowDataset shape and target-leakage tests ------------------------------


def test_sliding_window_shapes_and_no_target_leakage() -> None:
    """SlidingWindowDataset produces the right shapes and the target day never leaks into the input window."""
    # Each day's value is its own time index (0, 1, 2, ..., 99)
    n, f, w = 100, 4, 30
    timeline = np.arange(n, dtype=np.float32)
    features = np.stack([timeline + k * 1000 for k in range(f)], axis=1)
    targets = timeline.copy()

    ds = SlidingWindowDataset(features, targets, window_size=w)
    assert len(ds) == n - w

    for idx in range(len(ds)):
        x_win, y_target = ds[idx]
        assert x_win.shape == (w, f)
        assert y_target.shape == ()
        # The first feature column of the input window holds days [idx, idx+1, ..., idx+w-1]
        window_days = x_win[:, 0].numpy()
        target_day = float(y_target.item())
        assert np.allclose(window_days, np.arange(idx, idx + w, dtype=np.float32))
        assert target_day == float(idx + w)
        # The target day must be later than every day in the window (no leakage from the future)
        assert target_day > float(window_days.max())
        assert target_day not in set(window_days.tolist())


def test_price_window_dataset_no_leakage() -> None:
    """PriceWindowDataset takes the target price from the day outside the window (idx + W)."""
    prices = np.arange(10.0, 110.0, dtype=np.float32)
    w = 30
    ds = PriceWindowDataset(prices, window_size=w)
    assert len(ds) == len(prices) - w

    x_rel, y_rel, base_p, last_p = ds[5]
    assert x_rel.shape == (w, 1)
    assert np.isclose(float(base_p), prices[5])
    assert np.isclose(float(last_p), prices[5 + w - 1])
    reconstructed_target = (float(y_rel) + 1.0) * float(base_p)
    assert np.isclose(reconstructed_target, prices[5 + w], atol=1e-4)


# --- 3. Chronological split and leak-free scaler tests -----------------------------------


def test_chronological_split_and_train_only_scaler() -> None:
    """The split is strictly chronological and the StandardScaler is fitted on train only."""
    df = load_prices("THYAO.IS", force_fallback=True)
    feat_df = build_features(df)
    train_df, val_df, test_df = chronological_split(feat_df, train_ratio=0.70, val_ratio=0.15)

    assert len(train_df) + len(val_df) + len(test_df) == len(feat_df)
    assert train_df.index.is_monotonic_increasing
    assert val_df.index.is_monotonic_increasing
    assert test_df.index.is_monotonic_increasing
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()

    # Add a huge value to the test set on purpose; if the scaler was fitted on train only,
    # scaler.mean_ and scaler.scale_ must NOT be affected by it at all!
    test_tampered = test_df.copy()
    test_tampered[FEATURE_COLS] = test_tampered[FEATURE_COLS] + 99999.0

    scaler, x_train, x_val, _ = fit_scaler_on_train(train_df, val_df, test_tampered, FEATURE_COLS)
    expected_mean = train_df[FEATURE_COLS].values.astype(np.float32).mean(axis=0)
    expected_std = train_df[FEATURE_COLS].values.astype(np.float32).std(axis=0, ddof=0)

    assert np.allclose(scaler.mean_, expected_mean, atol=1e-5)
    assert np.allclose(scaler.scale_, expected_std, atol=1e-5)
    assert np.allclose(x_train.mean(axis=0), 0.0, atol=1e-5)
    assert not np.allclose(x_val.mean(axis=0), 0.0, atol=1e-3)


# --- 4. make_starter.py solution-block removal tests -------------------------------------


def test_make_starter_strips_all_solution_blocks() -> None:
    """make_starter.py removes every # BEGIN SOLUTION block and leaves no solution code behind."""
    sample_code = (
        "x = 1\n"
        "if x > 0:\n"
        "    # BEGIN SOLUTION: compute the y value\n"
        "    secret_solution_var = x + 41\n"
        "    # END SOLUTION\n"
        "print(x)"
    )
    stripped, count = strip_solution_blocks(sample_code)
    assert count == 1
    assert "secret_solution_var" not in stripped
    assert "# BEGIN SOLUTION" not in stripped
    assert "# END SOLUTION" not in stripped
    assert "    # TODO 1: compute the y value" in stripped
    assert '    raise NotImplementedError("TODO 1: compute the y value")' in stripped

    sol_path = ROOT / "notebooks" / "workshop_solution.ipynb"
    sta_path = ROOT / "notebooks" / "workshop_starter.ipynb"
    sol_nb = json.loads(sol_path.read_text(encoding="utf-8"))
    sta_nb = json.loads(sta_path.read_text(encoding="utf-8"))

    converted_nb, total_blocks = convert_notebook(sol_nb)
    assert total_blocks >= 8
    assert converted_nb == sta_nb

    sta_raw = sta_path.read_text(encoding="utf-8")
    assert "# BEGIN SOLUTION" not in sta_raw
    assert "# END SOLUTION" not in sta_raw

    # Every BEGIN SOLUTION block in the solution notebook maps to a TODO cell in the starter notebook
    sol_todo_cells = [
        c["id"] for c in sol_nb["cells"] if c["cell_type"] == "code" and "# BEGIN SOLUTION" in "".join(c["source"])
    ]
    sta_todo_cells = [
        c["id"] for c in sta_nb["cells"] if c["cell_type"] == "code" and re.search(r'NotImplementedError\("TODO \d+:', "".join(c["source"]))
    ]
    assert sol_todo_cells == sta_todo_cells


def test_every_todo_cell_has_a_check_and_a_hint() -> None:
    """Every solution cell with a TODO block calls its self-check and carries a hint for the starter."""
    sol_nb = json.loads((ROOT / "notebooks" / "workshop_solution.ipynb").read_text(encoding="utf-8"))
    todo_cells = [c for c in sol_nb["cells"] if c["cell_type"] == "code" and "# BEGIN SOLUTION" in "".join(c["source"])]
    numbers = []
    for cell in todo_cells:
        source = "".join(cell["source"])
        found = re.findall(r"check_todo_(\d+)\(", source)
        assert found, f"{cell['id']} has no check_todo_ call"
        numbers.extend(int(n) for n in found)
        hint = cell.get("metadata", {}).get("workshop", {}).get("hint")
        assert isinstance(hint, str) and hint.strip(), f"{cell['id']} has no hint"
    assert numbers == list(range(1, 10))


def test_make_starter_adds_help_cells() -> None:
    """Each TODO cell is followed by a collapsed hint (when the solution cell has one) and a collapsed solution."""

    def code(cell_id: str, source: str, hint: str | None = None) -> dict:
        meta = {"workshop": {"hint": hint}} if hint else {}
        return {"cell_type": "code", "id": cell_id, "metadata": meta, "outputs": [], "execution_count": None, "source": source}

    solution_nb = {
        "cells": [
            {"cell_type": "markdown", "id": "intro", "metadata": {}, "source": "# Title"},
            code("first", "x = 1\n# BEGIN SOLUTION: compute the y value\nsecret_solution_var = x + 41\n# END SOLUTION\nprint(x)", hint="use x + 41"),
            code("second", "# BEGIN SOLUTION: double it\nz = 2 * secret_solution_var\n# END SOLUTION"),
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    starter, count = convert_notebook(solution_nb)
    assert count == 2
    ids = [c["id"] for c in starter["cells"]]
    assert ids == ["intro", "first", "first-help", "second", "second-help"]

    help_1 = "".join(starter["cells"][2]["source"])
    assert starter["cells"][2]["cell_type"] == "markdown"
    for text in ("💡 Hint", "use x + 41", "👀 Show the solution", "secret_solution_var = x + 41"):
        assert text in help_1
    assert "\n```python\n" in help_1

    assert 'raise NotImplementedError("TODO 2: double it")' in "".join(starter["cells"][3]["source"])
    help_2 = "".join(starter["cells"][4]["source"])
    assert "💡 Hint" not in help_2
    assert "z = 2 * secret_solution_var" in help_2


# --- 5. 'Catch up' cells in a fresh kernel -----------------------------------------------


def test_every_catch_up_cell_works_in_fresh_process() -> None:
    """Every 'catch up' cell in the notebook runs without errors in a brand-new Python process."""
    sol_path = ROOT / "notebooks" / "workshop_solution.ipynb"
    sol_nb = json.loads(sol_path.read_text(encoding="utf-8"))
    catchup_cells = [
        c for c in sol_nb["cells"] if c["cell_type"] == "code" and "Catch up:" in "".join(c["source"])
    ]
    assert len(catchup_cells) == 6, f"Expected 6 catch-up cells, found {len(catchup_cells)}."

    env = {**subprocess.os.environ, "WORKSHOP_FORCE_FALLBACK": "1"}
    for cell in catchup_cells:
        code = "".join(cell["source"])
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, f"Catch-up cell {cell['id']} failed:\n{proc.stderr}"
        assert "✅" in proc.stdout


def test_every_catch_up_cell_restores_its_section() -> None:
    """In a reset runtime, the rest of a part's solution cells must also run after its catch-up cell.

    (If a catch-up cell runs on its own but does not load names such as np/plt/torch, the next cell raises NameError.)
    """
    sol_nb = json.loads((ROOT / "notebooks" / "workshop_solution.ipynb").read_text(encoding="utf-8"))
    code_cells = [c for c in sol_nb["cells"] if c["cell_type"] == "code"]
    starts = [i for i, c in enumerate(code_cells) if c["id"].endswith("-catchup")]
    assert len(starts) == 6

    env = {**subprocess.os.environ, "WORKSHOP_FORCE_FALLBACK": "1", "MPLBACKEND": "Agg"}
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(code_cells)
        section = code_cells[start:end]
        code = "display = print\n" + "\n".join("".join(c["source"]) for c in section)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        ids = [c["id"] for c in section]
        assert proc.returncode == 0, f"The part did not run after its catch-up cell ({ids}):\n{proc.stderr[-2000:]}"
