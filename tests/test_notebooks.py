"""End-to-end notebook execution and validation tests:
- workshop_solution.ipynb finishes in under 5 minutes (< 300 s) on CPU with the fallback data.
- Part 4: naive baseline MAE <= LSTM MAE * 1.1.
- Part 5: test direction accuracy between 0.45 and 0.60.
- workshop_starter.ipynb runs without errors up to the first TODO cell and raises NotImplementedError there.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import papermill as pm
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_NB = ROOT / "notebooks" / "workshop_solution.ipynb"
STARTER_NB = ROOT / "notebooks" / "workshop_starter.ipynb"


def _cell_output_text(nb_dict: dict, cell_id: str) -> str:
    """Joins the stdout text of the cell with the given ID."""
    for cell in nb_dict.get("cells", []):
        if cell.get("id") == cell_id:
            chunks = []
            for out in cell.get("outputs", []):
                if "text" in out:
                    t = out["text"]
                    chunks.append("".join(t) if isinstance(t, list) else str(t))
            return "\n".join(chunks)
    raise KeyError(f"Cell not found: {cell_id}")


def test_solution_notebook_executes_under_5_min_and_meets_story_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """workshop_solution.ipynb runs end to end in < 300 s and meets the Part 4 and Part 5 metrics."""
    monkeypatch.setenv("WORKSHOP_FORCE_FALLBACK", "1")
    out_nb = tmp_path / "executed_solution.ipynb"

    t0 = time.perf_counter()
    pm.execute_notebook(
        str(SOLUTION_NB),
        str(out_nb),
        cwd=str(ROOT),
        kernel_name="python3",
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 300.0, f"The solution notebook took longer than 5 minutes: {elapsed:.1f} s"

    nb_data = json.loads(out_nb.read_text(encoding="utf-8"))

    # Part 4 metric check: naive_mae <= lstm_mae * 1.1
    sec4_out = _cell_output_text(nb_data, "sec4-naive-code")
    m_lstm = re.search(r"LSTM price model\s*->\s*MAE:\s*([0-9.]+)", sec4_out)
    m_naive = re.search(r"Naive baseline \(P_t\)\s*->\s*MAE:\s*([0-9.]+)", sec4_out)
    assert m_lstm and m_naive, f"Could not read the Part 4 MAE output:\n{sec4_out}"
    lstm_mae = float(m_lstm.group(1))
    naive_mae = float(m_naive.group(1))
    assert naive_mae <= lstm_mae * 1.1, f"Naive baseline ({naive_mae}) <= LSTM ({lstm_mae}) * 1.1 does not hold!"

    # Part 5 metric check: 0.45 <= test_acc <= 0.60
    sec5_out = _cell_output_text(nb_data, "sec5-cls-code")
    m_acc = re.search(r"LSTM test accuracy\s*:\s*([0-9.]+)%", sec5_out)
    assert m_acc, f"Could not read the Part 5 accuracy output:\n{sec5_out}"
    test_acc = float(m_acc.group(1)) / 100.0
    assert 0.45 <= test_acc <= 0.60, f"Part 5 test accuracy outside [0.45, 0.60]: {test_acc:.4f}"


def test_starter_notebook_runs_up_to_first_todo(monkeypatch: pytest.MonkeyPatch) -> None:
    """The starter notebook runs without errors up to its first TODO cell and raises NotImplementedError there."""
    monkeypatch.setenv("WORKSHOP_FORCE_FALLBACK", "1")
    monkeypatch.chdir(ROOT)

    nb = json.loads(STARTER_NB.read_text(encoding="utf-8"))
    ns: dict = {}

    hit_first_todo = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # A simple stand-in for IPython's display()
        ns.setdefault("display", lambda *a, **kw: None)
        if "NotImplementedError(\"TODO:" in source:
            with pytest.raises(NotImplementedError, match="TODO:"):
                exec(compile(source, f"<{cell['id']}>", "exec"), ns)  # noqa: S102
            hit_first_todo = True
            break
        else:
            exec(compile(source, f"<{cell['id']}>", "exec"), ns)  # noqa: S102

    assert hit_first_todo, "No TODO cell found in the starter notebook!"
