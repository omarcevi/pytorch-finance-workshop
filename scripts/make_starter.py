"""Çözüm defterinden (workshop_solution.ipynb) başlangıç defterini (workshop_starter.ipynb) üretir.

Her '# BEGIN SOLUTION' ... '# END SOLUTION' bloğunu girintiyi koruyarak bir '# TODO:' ipucu
ve 'raise NotImplementedError("TODO")' satırıyla değiştirir; hücre çıktılarını temizler.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOLUTION = ROOT / "notebooks" / "workshop_solution.ipynb"
DEFAULT_STARTER = ROOT / "notebooks" / "workshop_starter.ipynb"

SOLUTION_BLOCK_RE = re.compile(
    r"^(?P<indent>[ \t]*)# BEGIN SOLUTION(?::[ \t]*(?P<hint>[^\n]*))?\n"
    r"(?P<body>.*?)"
    r"^[ \t]*# END SOLUTION[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def strip_solution_blocks(source_text: str) -> tuple[str, int]:
    """Kod metni içindeki tüm # BEGIN SOLUTION ... # END SOLUTION bloklarını TODO ile değiştirir."""
    count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        indent = match.group("indent") or ""
        hint = (match.group("hint") or "Bu kısmı tamamlayın").strip()
        return (
            f"{indent}# TODO: {hint}\n"
            f'{indent}raise NotImplementedError("TODO: {hint}")'
        )

    new_text = SOLUTION_BLOCK_RE.sub(_replace, source_text)
    if "# BEGIN SOLUTION" in new_text or "# END SOLUTION" in new_text:
        raise ValueError("Eşleşmeyen # BEGIN SOLUTION veya # END SOLUTION etiketi bulundu!")
    return new_text, count


def _to_source_lines(text: str) -> list[str]:
    """Metni Jupyter notebook 'source' satır listesi biçimine dönüştürür."""
    if not text:
        return []
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def convert_notebook(solution_nb: dict) -> tuple[dict, int]:
    """Çözüm notebook sözlüğünü starter notebook sözlüğüne dönüştürür."""
    starter_nb = copy.deepcopy(solution_nb)
    total_blocks = 0

    for cell in starter_nb.get("cells", []):
        raw_source = cell.get("source", [])
        text = "".join(raw_source) if isinstance(raw_source, list) else str(raw_source)

        if cell.get("cell_type") == "markdown":
            text = text.replace("workshop_solution.ipynb", "workshop_starter.ipynb")
            text = text.replace("(Çözüm Defteri)", "(Başlangıç Defteri — Starter)")
            cell["source"] = _to_source_lines(text)
        elif cell.get("cell_type") == "code":
            new_text, n_blocks = strip_solution_blocks(text)
            total_blocks += n_blocks
            cell["source"] = _to_source_lines(new_text)
            cell["outputs"] = []
            cell["execution_count"] = None

    if total_blocks == 0:
        raise ValueError("Çözüm defterinde hiç '# BEGIN SOLUTION' bloğu bulunamadı!")
    return starter_nb, total_blocks


def build_starter(
    solution_path: Path = DEFAULT_SOLUTION,
    starter_path: Path = DEFAULT_STARTER,
) -> int:
    """Dosyadan çözüm defterini okur, starter defterini yazar ve değiştirilen blok sayısını döndürür."""
    solution_nb = json.loads(solution_path.read_text(encoding="utf-8"))
    starter_nb, count = convert_notebook(solution_nb)
    starter_path.parent.mkdir(parents=True, exist_ok=True)
    starter_path.write_text(
        json.dumps(starter_nb, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return count


def main() -> None:
    sol = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOLUTION
    sta = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_STARTER
    n = build_starter(sol, sta)
    print(f"Başlangıç defteri üretildi: {sta} ({n} çözüm bloğu TODO'ya dönüştürüldü).")


if __name__ == "__main__":
    main()
