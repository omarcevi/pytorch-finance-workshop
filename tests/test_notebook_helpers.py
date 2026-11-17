"""Tests for the learner-facing notebook helpers: friendly TODO/check display and setup_notebook()."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import pytest
from IPython.core.interactiveshell import InteractiveShell

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from workshop_utils import WorkshopCheckError, friendly_lines, setup_notebook


def test_friendly_lines_for_check_error() -> None:
    err = WorkshopCheckError(4, "scaler was fitted on all the data.", "fit on train_df only.")
    assert str(err) == "TODO 4: scaler was fitted on all the data."
    assert (err.todo, err.problem, err.hint) == (4, "scaler was fitted on all the data.", "fit on train_df only.")
    assert friendly_lines(err) == ["❌ TODO 4: scaler was fitted on all the data.", "💡 Hint: fit on train_df only."]


def test_friendly_lines_for_todo_placeholder() -> None:
    lines = friendly_lines(NotImplementedError("TODO 1: load the data"))
    assert lines[0] == "✏️ TODO 1: load the data isn't done yet."
    assert lines[1] == "Replace the raise NotImplementedError(...) line with your code, then run the cell again."


def test_friendly_lines_ignores_other_errors() -> None:
    assert friendly_lines(ValueError("x")) is None
    assert friendly_lines(NotImplementedError("abstract")) is None


def test_setup_notebook_in_ipython_shows_friendly_block_but_keeps_real_tracebacks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    shell = InteractiveShell.instance()
    setup_notebook()

    result = shell.run_cell('raise NotImplementedError("TODO 1: load the data")')
    out = capsys.readouterr()
    text = out.out + out.err
    assert "✏️ TODO 1: load the data isn't done yet." in text
    assert "Traceback" not in text
    assert result.error_in_exec is not None

    shell.run_cell("{}['missing']")
    out = capsys.readouterr()
    text = out.out + out.err
    assert "KeyError" in text
    assert "Traceback" in text


def test_setup_notebook_quiets_yfinance_and_sets_fonts() -> None:
    setup_notebook()
    assert logging.getLogger("yfinance").level == logging.CRITICAL
    assert matplotlib.rcParams["font.size"] == 12
