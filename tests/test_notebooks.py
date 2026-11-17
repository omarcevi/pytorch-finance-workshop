"""Defter (Notebook) uçtan uca yürütme ve doğrulama testleri:
- workshop_solution.ipynb yedek (fallback) veriyle CPU üzerinde 5 dakikanın (< 300 sn) altında tamamlanmalıdır.
- Bölüm 4: Saf referans MAE <= LSTM MAE * 1.1 olmalıdır.
- Bölüm 5: Test yön doğruluğu (accuracy) 0.45 ile 0.60 arasında olmalıdır.
- workshop_starter.ipynb ilk TODO hücresine kadar hatasız çalışmalı ve ilk TODO'da NotImplementedError üretmelidir.
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
    """Belirtilen hücre ID'sinin standart çıktı (stdout) metnini birleştirir."""
    for cell in nb_dict.get("cells", []):
        if cell.get("id") == cell_id:
            chunks = []
            for out in cell.get("outputs", []):
                if "text" in out:
                    t = out["text"]
                    chunks.append("".join(t) if isinstance(t, list) else str(t))
            return "\n".join(chunks)
    raise KeyError(f"Hücre bulunamadı: {cell_id}")


def test_solution_notebook_executes_under_5_min_and_meets_story_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """workshop_solution.ipynb uçtan uca < 300 sn içinde çalışmalı ve Bölüm 4 & 5 metriklerini sağlamalıdır."""
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
    assert elapsed < 300.0, f"Çözüm defteri 5 dakikadan uzun sürdü: {elapsed:.1f} sn"

    nb_data = json.loads(out_nb.read_text(encoding="utf-8"))

    # Bölüm 4 metrik kontrolü: naive_mae <= lstm_mae * 1.1
    sec4_out = _cell_output_text(nb_data, "sec4-naive-code")
    m_lstm = re.search(r"LSTM Fiyat Modeli\s*->\s*MAE:\s*([0-9.]+)", sec4_out)
    m_naive = re.search(r"Saf Referans \(P_t\)\s*->\s*MAE:\s*([0-9.]+)", sec4_out)
    assert m_lstm and m_naive, f"Bölüm 4 MAE çıktısı okunamadı:\n{sec4_out}"
    lstm_mae = float(m_lstm.group(1))
    naive_mae = float(m_naive.group(1))
    assert naive_mae <= lstm_mae * 1.1, f"Saf referans ({naive_mae}) <= LSTM ({lstm_mae}) * 1.1 koşulu sağlanmadı!"

    # Bölüm 5 metrik kontrolü: 0.45 <= test_acc <= 0.60
    sec5_out = _cell_output_text(nb_data, "sec5-cls-code")
    m_acc = re.search(r"LSTM Test Doğruluğu \(Accuracy\)\s*:\s*%([0-9.]+)", sec5_out)
    assert m_acc, f"Bölüm 5 doğruluk çıktısı okunamadı:\n{sec5_out}"
    test_acc = float(m_acc.group(1)) / 100.0
    assert 0.45 <= test_acc <= 0.60, f"Bölüm 5 test doğruluğu [0.45, 0.60] aralığı dışında: {test_acc:.4f}"


def test_starter_notebook_runs_up_to_first_todo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Başlangıç defteri ilk TODO hücresine kadar hatasız çalışmalı ve ilk TODO'da NotImplementedError vermelidir."""
    monkeypatch.setenv("WORKSHOP_FORCE_FALLBACK", "1")
    monkeypatch.chdir(ROOT)

    nb = json.loads(STARTER_NB.read_text(encoding="utf-8"))
    ns: dict = {}

    hit_first_todo = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # IPython display() çağrısı için basit sarmalayıcı
        ns.setdefault("display", lambda *a, **kw: None)
        if "NotImplementedError(\"TODO:" in source:
            with pytest.raises(NotImplementedError, match="TODO:"):
                exec(compile(source, f"<{cell['id']}>", "exec"), ns)  # noqa: S102
            hit_first_todo = True
            break
        else:
            exec(compile(source, f"<{cell['id']}>", "exec"), ns)  # noqa: S102

    assert hit_first_todo, "Başlangıç defterinde hiç TODO hücresine rastlanmadı!"
