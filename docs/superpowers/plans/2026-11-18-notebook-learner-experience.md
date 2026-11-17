# Notebook Learner Experience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Friendlier, cleaner starter and solution notebooks: self-checks on every TODO, friendly one-line messages instead of tracebacks, in-notebook hints and solutions, hidden plumbing, quiet yfinance, projector-ready charts.

**Architecture:** New learner-facing helpers (`setup_notebook`, `WorkshopCheckError`, `check_todo_1..9`) live in `src/workshop_utils.py`, which notebooks download as one file. `scripts/build_notebooks.py` lays out the solution notebook and calls the helpers; `scripts/make_starter.py` numbers the TODOs and adds collapsed hint/solution cells to the starter. Both notebooks are regenerated, and the solution notebook is re-executed.

**Tech Stack:** Python 3.12, PyTorch, pandas, scikit-learn, matplotlib, IPython/ipykernel (already installed with Jupyter/Colab), pytest, nbformat/nbclient, papermill.

**Spec:** `docs/superpowers/specs/2026-11-18-notebook-learner-experience-design.md`

## Global Constraints

- `src/workshop_utils.py` stays a single module: notebooks (including ones opened before this change) download it by URL. Only add to it; keep every existing name, including the `yetistir_bolum_1..6` aliases.
- No new dependencies; `requirements.txt` unchanged.
- Existing notebook cell IDs stay unchanged. The only new cells are `<todo-cell-id>-help` markdown cells in the starter.
- Unfilled-TODO and failed-check display: one friendly block, no Python traceback. Every other exception keeps IPython's normal traceback.
- Copy markers: ✏️ unfilled TODO, ✅ check passed, ❌ check failed, 💡 hint, 👀 solution, 🔄 catch up, ⚙️ setup, 🎣 hook, ⏱️ live session.
- Live-session times, one per part: Part 0 16:30–16:40, Part 1 16:40–16:55, Part 2 16:55–17:15, Part 3 17:15–17:35, Break 17:35–17:45, Part 4 17:45–18:05, Part 5 18:05–18:25, Part 6 18:25–18:40, Part 7 18:40–18:45.
- Test commands use `PY=/private/tmp/claude-501/-Users-omar6-Documents-Programming/256197a7-ce17-466c-88e5-316894c78fd1/scratchpad/venv/bin/python` and CI settings: `$PY scripts/snapshot_data.py --offline` once, then `WORKSHOP_FORCE_FALLBACK=1 WORKSHOP_OFFLINE=1 $PY -m pytest ...`. Delete `data/*.csv` before committing (gitignored anyway).
- Commits have no Claude co-author trailer (user's CLAUDE.md).

## Review Focus

1. A real mistake in learner code (KeyError from a typo, a shape error) must still show the full traceback: pinned in Task 1.
2. Correct but differently written answers must pass, for example `np.log(close).diff()`, `scaler.fit(train).transform(...)`, `zero_grad()` placed after `step()`: pinned in Tasks 2 and 3.
3. Re-running a TODO cell, or running a check, must not change later outputs: check 7's reference run must leave torch's global RNG state untouched. Pinned in Task 3.
4. Checks run in a reset runtime after a catch-up cell, so `CATCHUP_BOOTSTRAP` must import them. Pinned in Task 5 by the existing `test_every_catch_up_cell_restores_its_section`.
5. Tolerances must hold on both the offline CI data and the real snapshot data: tests use the offline data, and Task 6 runs the solution notebook on the real data.

---

### Task 1: Friendly display and notebook setup

**Files:**
- Modify: `src/workshop_utils.py` (new section before the catch-up functions)
- Test: `tests/test_notebook_helpers.py` (create)

**Interfaces:**
- Produces: `class WorkshopCheckError(AssertionError)`, constructed as `WorkshopCheckError(todo: int, problem: str, hint: str)`; `str(e) == f"TODO {todo}: {problem}"`; attributes `.todo`, `.problem`, `.hint`.
- Produces: `friendly_lines(value: BaseException) -> list[str] | None`. For `WorkshopCheckError` it returns `["❌ TODO {n}: {problem}", "💡 Hint: {hint}"]`. For `NotImplementedError` whose message starts with `"TODO"` it returns `["✏️ {message} isn't done yet.", "Replace the raise NotImplementedError(...) line with your code, then run the cell again."]`. Otherwise `None`.
- Produces: `setup_notebook() -> None`. It always sets matplotlib rcParams (`font.size` 12, `axes.titlesize` 13, `axes.labelsize` 12, `legend.fontsize` 10) and sets the `yfinance` logger to `CRITICAL`. Only in IPython (`IPython.get_ipython()` is not None), it also sets the retina figure format (wrapped in try/except) and registers `shell.set_custom_exc((NotImplementedError, WorkshopCheckError), handler)`. The handler returns `friendly_lines(value)` when that is not None, else `shell.InteractiveTB.structured_traceback(etype, value, tb, tb_offset=tb_offset)`. It never calls `showtraceback` itself, to avoid a duplicate error output.

- [ ] **Step 1: Write the failing tests** in `tests/test_notebook_helpers.py`:
  - `test_friendly_lines_for_check_error`: `friendly_lines(WorkshopCheckError(4, "scaler was fitted on all the data.", "fit on train_df only."))` → `["❌ TODO 4: scaler was fitted on all the data.", "💡 Hint: fit on train_df only."]`.
  - `test_friendly_lines_for_todo_placeholder`: `friendly_lines(NotImplementedError("TODO 1: load the data"))[0] == "✏️ TODO 1: load the data isn't done yet."`.
  - `test_friendly_lines_ignores_other_errors`: `friendly_lines(ValueError("x")) is None` and `friendly_lines(NotImplementedError("abstract")) is None`.
  - `test_setup_notebook_in_ipython_shows_friendly_block_but_keeps_real_tracebacks(capsys)`: `shell = InteractiveShell.instance()`, `setup_notebook()`, then:
    - `shell.run_cell('raise NotImplementedError("TODO 1: load the data")')`: output contains `"✏️ TODO 1: load the data isn't done yet."` and not `"Traceback"`, and `result.error_in_exec` is set, so "Run all" still stops.
    - `shell.run_cell("{}['missing']")`: output contains `"KeyError"` and `"Traceback"`.
  - `test_setup_notebook_quiets_yfinance_and_sets_fonts`: after `setup_notebook()`, `logging.getLogger("yfinance").level == logging.CRITICAL` and `matplotlib.rcParams["font.size"] == 12`.
- [ ] **Step 2: Run** `$PY -m pytest tests/test_notebook_helpers.py -q`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** `WorkshopCheckError`, `friendly_lines` and `setup_notebook` with the interfaces above.
- [ ] **Step 4: Run** the same command. Expected: 5 passed. Then run `ruff check src tests`: clean.
- [ ] **Step 5: Commit** with message `Add setup_notebook and friendly TODO/check display`.

### Task 2: Checks for the data TODOs (1–4)

**Files:**
- Modify: `src/workshop_utils.py` (new "Self-checks" section after Task 1's code)
- Test: `tests/test_checks.py` (create)

**Interfaces:**
- Consumes: `WorkshopCheckError(todo, problem, hint)` from Task 1.
- Produces, each returning `None`, printing one line starting `✅ TODO n looks right` on success, and raising `WorkshopCheckError` otherwise:
  - `check_todo_1(raw_df: pd.DataFrame) -> None`: is a DataFrame, has the five OHLCV columns, at least `MIN_HISTORY_ROWS` rows, DatetimeIndex sorted ascending. Success line includes the row count.
  - `check_todo_2(feat_df: pd.DataFrame) -> None`: is called before the RSI and `dropna()` lines. Checks `log_return`, `roll_mean_5` and `roll_std_10` against recomputation from `feat_df["Close"]` (NaN-aware, `rtol=1e-6`). The message names the first wrong column.
  - `check_todo_3(train_df, val_df, test_df, feat_df) -> None`: lengths equal `int(n*0.70)`, `int(n*0.85) - int(n*0.70)` and the rest. Each part's index is increasing, `train.index.max() < val.index.min()` and `val.index.max() < test.index.min()`, and the concatenation equals `feat_df.index`.
  - `check_todo_4(x_train, x_val, x_test, train_df, val_df, test_df) -> None`: compares with `StandardScaler` fitted on `train_df[FEATURE_COLS]` (`atol=1e-4`). If the arrays instead match a scaler fitted on all three parts, it raises with the problem "the scaler was fitted on all the data, so the test set leaked into it."
- Test fixture: module-scoped `state = catch_up_part_2("THYAO.IS")` (offline data under CI settings), plus `feat_raw`: `state["raw_df"][OHLCV].ffill().dropna()` with the three TODO-2 columns added the way the notebook does.

- [ ] **Step 1: Write the failing tests** (one pass test and the listed failing cases each; failing cases use `pytest.raises(WorkshopCheckError, match=...)`):
  - `test_check_todo_1`: passes on `state["raw_df"]`. `match="OHLCV"` with only the `Close` column. `match="DataFrame"` with `None`.
  - `test_check_todo_2`: passes on `feat_raw`. Also passes when `log_return = np.log(close).diff()`, a correct alternative. `match="roll_mean_5"` with `rolling(10)`. `match="log_return"` with `pct_change()`.
  - `test_check_todo_3`: passes on `chronological_split(feat_df)`. `match="chronological"` with `feat_df.sample(frac=1, random_state=0)` split by position. `match="70%"` with `train_ratio=0.8`.
  - `test_check_todo_4`: passes on `fit_scaler_on_train` output, and also with `scaler.fit(train).transform(...)`. `match="all the data"` when fitted on `pd.concat([train, val, test])`.
- [ ] **Step 2: Run** `$PY -m pytest tests/test_checks.py -q` with CI settings. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** `check_todo_1..4`. Hints name the function or argument to use, for example "use chronological_split(feat_df, train_ratio=0.70, val_ratio=0.15)".
- [ ] **Step 4: Run** the same command. Expected: all pass. `ruff check` clean.
- [ ] **Step 5: Commit** with message `Add self-checks for TODOs 1-4`.

### Task 3: Checks for the PyTorch and metric TODOs (5–9)

**Files:**
- Modify: `src/workshop_utils.py` (same section)
- Test: `tests/test_checks.py`

**Interfaces:**
- Produces (same success/failure contract as Task 2):
  - `check_todo_5(train_ds, val_ds, train_loader, val_loader, x_train, x_val, window_size: int = 30) -> None`: `len(train_ds) == len(x_train) - window_size` and the same for val; first batch shape `(64, window_size, x_train.shape[1])`; `train_loader.sampler` is a `RandomSampler`, otherwise `match="shuffle=True"`; `val_loader.sampler` is a `SequentialSampler`.
  - `check_todo_6(model_cls: type[nn.Module]) -> None`: inside `torch.random.fork_rng()`, build `model_cls(input_dim=4, hidden_dim=8)` and run it on a `(3, 30, 4)` tensor. The output must equal `model.fc(model.lstm(x)[0][:, -1, :]).squeeze(-1)` with shape `(3,)`. Shape `(3, 1)` raises "add squeeze(-1)". Matching the first time step raises "use the last time step out[:, -1, :]".
  - `check_todo_7(model: nn.Module, train_loader: DataLoader, train_losses: list[float], epochs: int = 10, lr: float = 0.001, seed: int = 42) -> None`: inside `torch.random.fork_rng()`, run `set_seed(seed)`, build `type(model)(input_dim=model.lstm.input_size, hidden_dim=model.lstm.hidden_size)`, and run the reference loop (Adam, `MSELoss`, `epochs` epochs over `train_loader`). Compare `train_losses` to the reference with `rtol=1e-4`. If `model`'s weights still equal a fresh init, raise "the weights never changed: call optimizer.step()". Any other mismatch raises "zero_grad → forward → loss → backward → step".
  - `check_todo_8(lstm_mae, naive_mae, lstm_rmse, naive_rmse, actual_prices, predicted_prices, naive_prices) -> None`: recompute all four values (`rtol=1e-6`). An RMSE equal to the MSE raises "take np.sqrt". An MAE equal to the signed mean raises "use np.abs".
  - `check_todo_9(cls_res: dict, test_df: pd.DataFrame, window_size: int = 30) -> None`: has the keys `test_acc`, `test_bal_acc`, `majority_acc`, `y_true`, `y_pred`. `y_true` must equal `test_df["up_day"].values[window_size:]`, otherwise "use the up_day column as the target". `0 <= test_acc <= 1`.

- [ ] **Step 1: Write the failing tests:**
  - `test_check_todo_5`: passes with the Part 3 objects. `match="shuffle=True"` with a train loader built with `shuffle=False`.
  - `test_check_todo_6`: passes with a correct `SimpleLSTM` defined in the test. `match="squeeze"` without `squeeze`. `match="last time step"` with `out[:, 0, :]`.
  - `test_check_todo_7`: passes with a correct loop. Also passes with a loop that calls `zero_grad()` after `step()`. `match="optimizer.step"` with a loop missing `step()`. Also asserts `torch.get_rng_state()` is identical before and after the check (Review Focus 3).
  - `test_check_todo_8`: passes with correct values. `match="sqrt"` with RMSE given as MSE. `match="np.abs"` with a signed-mean MAE.
  - `test_check_todo_9`: passes with `train_direction_classifier` output. `match="up_day"` when the classifier was trained and tested on `log_return` targets.
  - Loops and models in the tests use `epochs=2` and a 200-row slice to keep the suite fast. Pass `epochs=2` to `check_todo_7`.
- [ ] **Step 2: Run** `$PY -m pytest tests/test_checks.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** `check_todo_5..9`.
- [ ] **Step 4: Run** the same command. Expected: all pass, in under 60 s. `ruff check` clean.
- [ ] **Step 5: Commit** with message `Add self-checks for TODOs 5-9`.

### Task 4: Numbered TODOs and help cells in the starter

**Files:**
- Modify: `scripts/make_starter.py`
- Test: `tests/test_unit.py` (update `test_make_starter_strips_all_solution_blocks`)

**Interfaces:**
- Consumes: a solution code cell's `metadata["workshop"]["hint"]` (a string set by Task 5's builder; it may be absent).
- Produces: `strip_solution_blocks(source_text: str, start: int = 1) -> tuple[str, int]`, which numbers the blocks `start, start+1, ...`. Each block becomes `# TODO {n}: {hint}` and `raise NotImplementedError("TODO {n}: {hint}")`. `convert_notebook` numbers TODOs across the whole notebook in order. After each code cell with blocks it inserts a markdown cell with `id = f"{cell_id}-help"` holding a collapsed hint (`<details><summary>💡 Hint</summary>` … `</details>`, only when a metadata hint exists) and a collapsed solution (`<details><summary>👀 Show the solution</summary>` with the dedented solution body in a fenced `python` block). Blank lines go around the fenced block so Markdown renders it.

- [ ] **Step 1: Update the test.** The sample-code assertions become `"    # TODO 1: compute the y value"` and `'    raise NotImplementedError("TODO 1: compute the y value")'`. Add `test_make_starter_adds_help_cells`: a two-cell solution notebook (one code cell with a block and `metadata.workshop.hint = "use x + 41"`) converts to three cells. The third has id `"<id>-help"` and contains `"💡 Hint"`, `"use x + 41"`, `"👀 Show the solution"` and `"secret_solution_var = x + 41"`. Also assert a second block in a later cell is numbered `TODO 2`.
- [ ] **Step 2: Run** `$PY -m pytest tests/test_unit.py -q -k make_starter`. Expected: FAIL.
- [ ] **Step 3: Implement** the changes in `scripts/make_starter.py`.
- [ ] **Step 4: Run** the same command. Expected: pass. (The full-notebook equality part of the old test passes again after Task 5 regenerates the notebooks.)
- [ ] **Step 5: Commit** with message `Number starter TODOs and add collapsed hint/solution cells`.

### Task 5: Notebook layout, checks wired in, notebooks regenerated

**Files:**
- Modify: `scripts/build_notebooks.py`
- Regenerate: `notebooks/workshop_solution.ipynb` (executed), `notebooks/workshop_starter.ipynb`
- Test: `tests/test_unit.py` (new `test_every_todo_cell_has_a_check_and_a_hint`)

**Interfaces:**
- Consumes: `setup_notebook`, `check_todo_1..9` (Tasks 1–3); `metadata.workshop.hint` read by Task 4.
- Produces: `code_cell(cell_id, code, *, collapsed=False, hint: str | None = None)`, which stores `hint` in `metadata["workshop"]["hint"]`.

- [ ] **Step 1: Write the failing test** `test_every_todo_cell_has_a_check_and_a_hint`. For each solution code cell containing `# BEGIN SOLUTION`: the source contains `check_todo_` and `metadata.workshop.hint` is a non-empty string. The nine check numbers, in cell order, are 1..9.
- [ ] **Step 2: Run** `$PY -m pytest tests/test_unit.py -q -k todo_cell_has`. Expected: FAIL.
- [ ] **Step 3: Change the builder:**
  - `intro-header`: add a "How this notebook works" box with one line each for ✏️ TODO cells, ✅ checks, 💡 hints and 👀 solutions (starter), and 🔄 catch-up cells.
  - `setup-code`: a form cell titled `⚙️ Setup: run this first` with `collapsed=True`. It calls `setup_notebook()` after the imports and imports `check_todo_1..9` and `setup_notebook`. Keep `print(f"PyTorch version: {torch.__version__} | Device: CPU")`. Rewrite `setup-md` to say to run it first.
  - `sec0-hook-plot`: a form cell titled `🎣 The chart we'll explain in Part 4` with `collapsed=True`.
  - `CATCHUP_BOOTSTRAP`: import `setup_notebook` and `check_todo_1..9` too, and call `setup_notebook()`.
  - Headings: remove the `(hh:mm–hh:mm)` from every `##` heading and add `⏱️ Live session: hh:mm–hh:mm` as the line under it (times in Global Constraints).
  - Add a check call right after each `# END SOLUTION` (TODO 6: right after the class; TODO 7: after the epoch loop, before the print), with the Task 2/3 signatures and the notebook's variable names. Add a `hint=` to each of the 9 TODO cells, one sentence each that points at the API without giving the answer (for example TODO 4: "fit_transform learns the mean and std; transform only applies them.").
  - `sec1-load-code`: `display(raw_df.tail(5).round(2))`.
- [ ] **Step 4: Regenerate.** Run `$PY scripts/build_notebooks.py`. Execute the solution notebook with nbclient (`record_timing=False`, cwd = repo root, `WORKSHOP_FORCE_FALLBACK=1`, `WORKSHOP_DATA_DIR=<scratchpad>/gist`) and write it back with `nbformat.write`. Then run `$PY scripts/make_starter.py`. Expected: no error outputs; the stored outputs contain nine `✅ TODO` lines.
- [ ] **Step 5: Run** the full suite with CI settings: `$PY -m pytest -q tests/`. Expected: all pass. `ruff check src scripts tests` clean.
- [ ] **Step 6: Commit** with message `Lay out notebooks for learners: checks, hints, hidden setup, quieter output`.

### Task 6: Guide, verification on real data, PRs

**Files:**
- Modify: `guide/pytorch-finance-workshop.md` (Notebooks section: one paragraph on ✏️/✅/💡/👀/🔄), `README.md` (one line in the quick-start section)
- Site: `../omarcevi.github.io` on a new branch: run `python3 scripts/build_pytorch_finance_guide.py`.

- [ ] **Step 1:** Update the guide and README text. Rebuild the site pages, and check the rebuilt pages contain no Turkish other than Işık/Şile.
- [ ] **Step 2: Colab-like run on real data.** Run the scratchpad `sim_colab.py` on the new solution notebook from an empty directory with a copy of the new `src/workshop_utils.py`. Expected: hook OK, all 6 catch-up paths OK, full run OK.
- [ ] **Step 3: Starter walk-through.** In one fresh IPython `InteractiveShell` with `setup_notebook()`, run the starter's code cells in order, replacing each `raise NotImplementedError("TODO n: …")` line with that TODO's solution body. Expected: all nine `✅ TODO n looks right` lines and no errors. Then run the unmodified starter's first TODO cell. Expected: the ✏️ block with no `Traceback`.
- [ ] **Step 4: Old notebook against the new module.** Run `sim_colab.py` on `git show main:notebooks/workshop_solution.ipynb`. Expected: all OK.
- [ ] **Step 5:** Commit; push branches `notebook-learner-experience` (workshop repo) and the site branch; open both PRs. Wait for CI on the workshop PR. Ask the user before merging.
