# Notebook learner experience: design

**Status:** approved by Omar, 18 November 2026.

## Intent

Make the starter and solution notebooks cleaner and friendlier for two audiences at once: the students following the live session (Işık IEEE CS, 18 November 2026, 16:30, Colab on a projector) and people working through the repo on their own afterwards. For the room: less noise and readable charts, without moving any cell the instructor presents. For self-learners: things that stand in for the instructor, such as feedback on each TODO and hints in the notebook.

## Goals

- Unfilled TODOs and wrong answers show one friendly line, not a Python traceback.
- Each of the 9 TODOs tells the learner whether their code is right, and what is wrong if not.
- Hints and solutions are available inside the starter notebook, collapsed.
- No plumbing code (sys.path, downloads, try/except) in the learner's face at the start.
- No red yfinance log output when Yahoo fails; the friendly ⚠️ fallback line stays.
- Charts are sharp and readable on a projector.
- The table of contents reads cleanly without clock times; the live schedule is still visible.

## Non-goals

- No change to the teaching content, the models, the data or the numbers.
- No new dependencies and no Colab-only widgets.

## Anti-goals

- Hiding real errors: only TODO placeholders and failed checks get the short display; any other exception keeps its full traceback.
- Breaking today's session: notebook cell IDs referenced by `docs/INSTRUCTOR.md` and the guide keep their IDs; catch-up cells keep working in a reset runtime; a notebook opened before the change keeps working with the new `workshop_utils.py`.

## Approaches

1. **Helpers in `workshop_utils.py` plus generator changes (chosen).** A `setup_notebook()` helper (chart style, quiet yfinance, friendly error display through IPython's custom exception hook) and `check_todo_1..9()` functions; `build_notebooks.py` adds the check calls, collapsed setup/hook cells and layout; `make_starter.py` adds collapsed hint/solution cells. Works in Colab, Jupyter and on GitHub with no installs, and fits the existing single-source generator.
2. **A grading tool (nbgrader, otter-grader).** Rejected: extra installs and setup in Colab for 40 students, and heavier than nine checks need.
3. **Inline `assert`s in the notebook cells.** Rejected: failures print exactly the tracebacks this change removes, and the messages can't carry hints.

## Design

1. **`setup_notebook()`** in `workshop_utils.py`, called from the setup cell and from every catch-up bootstrap:
   - matplotlib: retina figures in IPython, larger font sizes for a projector;
   - silences the `yfinance` logger and its warnings;
   - registers an IPython custom exception handler that renders `NotImplementedError("TODO: …")` and `WorkshopCheckError` as one friendly line; every other exception falls through to IPython's normal traceback. No-op outside IPython.
2. **`check_todo_1..9(...)`** in `workshop_utils.py`: each validates one TODO against an independent recomputation and prints `✅ TODO N looks right: …`, or raises `WorkshopCheckError` with what is wrong and a hint (for example: scaler fitted on all the data, loader not shuffled, output shape (64, 1) needs `squeeze(-1)`, optimizer never stepped). One call per TODO cell, outside the solution block.
3. **Starter only:** `make_starter.py` inserts, after each TODO cell, a markdown cell with a collapsed 💡 Hint (text stored in the solution cell's metadata by the builder) and a collapsed 👀 Solution (the stripped solution block).
4. **Collapsed cells:** the setup cell becomes "⚙️ Setup: run this first" and the Part 0 hook "🎣 The chart we'll explain in Part 4", both form cells with hidden code.
5. **Headings:** clock times move from the `##` headings to a line under each, `⏱️ Live session: 16:40–16:55`.
6. **Top of the notebook:** a short "How this notebook works" box (✏️ TODO, ✅ check, 💡 hint, 🔄 catch up). Displayed price tables are rounded to 2 decimals.

## Left to the builder

- Exact wording of hints, check messages and the "how it works" box.
- Check tolerances, verified against the real snapshot data and the offline CI data.
- Guide updates: mention the checks and hints; keep expected outputs in sync with the notebooks.
- Verification: unit tests for every check (passing and failing input), the full test suite with CI settings, a Colab-like run from an empty directory, a starter run-through with the guide's solutions, and the old notebook against the new module.
