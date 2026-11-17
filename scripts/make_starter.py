"""Generates the starter notebook (workshop_starter.ipynb) from the solution notebook (workshop_solution.ipynb).

Replaces every '# BEGIN SOLUTION' ... '# END SOLUTION' block, keeping its indentation, with a '# TODO:' hint
and a 'raise NotImplementedError("TODO")' line, and clears the cell outputs.
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
    """Replaces every # BEGIN SOLUTION ... # END SOLUTION block in the code with a TODO."""
    count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        indent = match.group("indent") or ""
        hint = (match.group("hint") or "Complete this part").strip()
        return (
            f"{indent}# TODO: {hint}\n"
            f'{indent}raise NotImplementedError("TODO: {hint}")'
        )

    new_text = SOLUTION_BLOCK_RE.sub(_replace, source_text)
    if "# BEGIN SOLUTION" in new_text or "# END SOLUTION" in new_text:
        raise ValueError("Found an unmatched # BEGIN SOLUTION or # END SOLUTION marker!")
    return new_text, count


def _to_source_lines(text: str) -> list[str]:
    """Turns text into the Jupyter notebook 'source' list-of-lines format."""
    if not text:
        return []
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def convert_notebook(solution_nb: dict) -> tuple[dict, int]:
    """Turns a solution notebook dict into a starter notebook dict."""
    starter_nb = copy.deepcopy(solution_nb)
    total_blocks = 0

    for cell in starter_nb.get("cells", []):
        raw_source = cell.get("source", [])
        text = "".join(raw_source) if isinstance(raw_source, list) else str(raw_source)

        if cell.get("cell_type") == "markdown":
            text = text.replace("workshop_solution.ipynb", "workshop_starter.ipynb")
            text = text.replace("(Solution Notebook)", "(Starter Notebook)")
            cell["source"] = _to_source_lines(text)
        elif cell.get("cell_type") == "code":
            new_text, n_blocks = strip_solution_blocks(text)
            total_blocks += n_blocks
            cell["source"] = _to_source_lines(new_text)
            cell["outputs"] = []
            cell["execution_count"] = None

    if total_blocks == 0:
        raise ValueError("No '# BEGIN SOLUTION' block found in the solution notebook!")
    return starter_nb, total_blocks


def build_starter(
    solution_path: Path = DEFAULT_SOLUTION,
    starter_path: Path = DEFAULT_STARTER,
) -> int:
    """Reads the solution notebook, writes the starter notebook and returns the number of replaced blocks."""
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
    print(f"Generated the starter notebook: {sta} ({n} solution blocks turned into TODOs).")


if __name__ == "__main__":
    main()
