# Rehearsal and Measurement Report (`DRY_RUN.md`)

This document records the end-to-end rehearsal of the **"Bringing AI to Financial Data"** PyTorch workshop, done from a student's point of view in a live Google Colab runtime: the timings per part, the Part 4 and Part 5 metrics verified on the 5 default tickers × 3 random seeds (`seed`), and the rough edges found and fixed during the rehearsal.

---

## 1. Live Google Colab Rehearsal (Dry Run) and Part Timings

The starter notebook (`notebooks/workshop_starter.ipynb`) was opened in a live Google Colab runtime, the **9 TODO cells** containing `NotImplementedError("TODO: ...")` were filled in using only the code blocks from the written guide (`guide/pytorch-finance-workshop.md`), and every cell was run from top to bottom.

### 1.1. Runtime per Part (CPU)

| Part | Cell IDs (`id`) | Runtime (CPU) | Student coding + explanation time | Notes |
|---|---|---|---|---|
| **Setup & Part 0 (Hook)** | `setup-code`, `sec0-hook-plot` | **2.4 s** | 10 min (16:30–16:40) | `workshop_utils.py` is loaded and the pre-generated prediction chart is shown. |
| **Part 1: Setup and Data** | `sec1-load-code`, `sec1-plot-code` | **0.8 s** | 15 min (16:40–16:55) | **TODO 1** is filled in; the `(3023, 5)` OHLCV table and the chart are drawn. |
| **Part 2: Cleaning and Features** | `sec2-feat-code`, `sec2-split-code`, `sec2-scaler-code` | **0.6 s** | 20 min (16:55–17:15) | **TODO 2, 3, 4** are filled in; the chronological split (70%/15%/15%) and the leak-free `StandardScaler` are checked. |
| **Part 3: PyTorch Basics** | `sec3-tensor-code`, `sec3-dataset-code`, `sec3-model-code`, `sec3-train-code` | **2.9 s** | 20 min (17:15–17:35) | **TODO 5, 6, 7** are filled in; the `(64, 30, 4)` tensor shapes and the 10-epoch training loop run. |
| **Part 4: Price Prediction and the Surprise** | `sec4-train-code`, `sec4-naive-code`, `sec4-zoom-code` | **4.6 s** | 20 min (17:45–18:05) | **TODO 8** is filled in; the price LSTM trains for 15 epochs, and the MAE/RMSE comparison with the naive baseline (*"tomorrow = today"*) and the one-day lag (`Lag 1`) chart are produced. |
| **Part 5: The Right Setup (Direction)** | `sec5-cls-code`, `sec5-wf-code` | **11.4 s** | 20 min (18:05–18:25) | **TODO 9** is filled in; the 15-epoch `BCEWithLogitsLoss` classifier and 4-fold walk-forward validation run. |
| **Part 6: Challenges** | `sec6-ch1-sol` .. `sec6-ch4-sol` | **12.8 s** | 15 min (18:25–18:40) | `ASELS.IS`, the 5-feature model, `GRUModel`, `TinyTransformerModel` and the backtest with 15 bps trading costs run. |
| **Catch-up cells (6)** | `sec1-catchup` .. `sec6-catchup` | **18.5 s** | — | Each one was tested on its own in a fresh kernel. |
| **Total ("Run All")** | **All 55 cells** | **~54.0 s (< 1 min)** | **2 h 15 min** | Far below the 5-minute (300 s) limit. |

---

## 2. Robustness Check of the Part 4 Surprise (5 Tickers × 3 Seeds)

### 2.1. A Known Trap (Non-Stationary Global Scaler vs. Per-Window Relative Scaling) and the Setup We Chose

- **The second trap we found:** The `THYAO.IS` close price is in the **4.52 TL – 151.03 TL** range in the training period (`2015–2023`) and in the **242.20 TL – 348.25 TL** range in the test period (`2025–2026`) (for `ASELS.IS`, train: `1.33–34.38 TL`, test: `78.00–434.00 TL`). If a single global `StandardScaler` is fitted on the raw training prices, the THYAO test prices land at $z \approx 8$–$11$. Because the `tanh` and `sigmoid` gates inside `nn.LSTM` saturate for $z > 3$ (`tanh(z) ≈ 1.0`), the model **locks into a flat ceiling line** in the test region. That muddies the "one-day lag" story, because the model then fails for a completely different reason (activation saturation).
- **The fix:**
  1. In `PriceWindowDataset`, each 30-day price window is scaled as a relative change **against its own first day only** (`w / w[0] - 1.0`); the target is defined against the same base (`P_{t+1} / w[0] - 1.0`), and at the end the prediction is multiplied by `w[0]` to turn it back into TL. The window's first day `w[0]` is in the past, so nothing leaks, and the input always stays stationary (`[-0.3, +0.5]`).
  2. The `tanh` saturation you get with a global scaler is also explained openly at the start of Part 4 in the notebook and in the guide, as **"the second trap — activation saturation"**.

### 2.2. Measured Part 4 Results (the Naive Baseline Beats the LSTM in 15/15 Runs)

Acceptance criterion: `Naive MAE <= LSTM MAE * 1.1` (in fact `Naive MAE < LSTM MAE` in all 15 runs, so the ratio is always `< 1.00`). All values were measured with the notebook code on the real snapshot data.

| Ticker | Seed | LSTM MAE | Naive (`P_t`) MAE | Ratio (`Naive / LSTM`) | LSTM RMSE | Naive RMSE | `corr( Pred_t, Actual_t )` (`Lag 0`) | `corr( Pred_t, Actual_{t-1} )` (`Lag 1`) |
|---|---|---|---|---|---|---|---|---|
| **THYAO.IS** | `42` | **5.07 TL** | **4.55 TL** | **0.898** | 6.86 TL | 6.26 TL | 0.9451 | **0.9915** |
| **THYAO.IS** | `123` | **5.70 TL** | **4.55 TL** | **0.799** | 7.55 TL | 6.26 TL | 0.9403 | **0.9858** |
| **THYAO.IS** | `2026` | **4.87 TL** | **4.55 TL** | **0.935** | 6.69 TL | 6.26 TL | 0.9498 | **0.9953** |
| **ASELS.IS** | `42` | **6.59 TL** | **6.23 TL** | **0.945** | 9.19 TL | 8.87 TL | 0.9958 | **0.9994** |
| **ASELS.IS** | `123` | **7.23 TL** | **6.23 TL** | **0.862** | 10.11 TL | 8.87 TL | 0.9953 | **0.9987** |
| **ASELS.IS** | `2026` | **6.46 TL** | **6.23 TL** | **0.965** | 9.07 TL | 8.87 TL | 0.9960 | **0.9995** |
| **XU100.IS** | `42` | **184.45** | **138.25** | **0.750** | 240.66 | 193.20 | 0.9929 | **0.9985** |
| **XU100.IS** | `123` | **161.36** | **138.25** | **0.857** | 223.95 | 193.20 | 0.9926 | **0.9982** |
| **XU100.IS** | `2026` | **168.07** | **138.25** | **0.823** | 226.10 | 193.20 | 0.9935 | **0.9989** |
| **BTC-USD** | `42` | **1489.72 $** | **1336.03 $** | **0.897** | 2040.62 $ | 1879.39 $ | 0.9944 | **0.9999** |
| **BTC-USD** | `123` | **1545.53 $** | **1336.03 $** | **0.864** | 2065.66 $ | 1879.39 $ | 0.9939 | **0.9992** |
| **BTC-USD** | `2026` | **1370.20 $** | **1336.03 $** | **0.975** | 1917.39 $ | 1879.39 $ | 0.9943 | **0.9998** |
| **AAPL** | `42` | **3.98 $** | **3.18 $** | **0.798** | 5.68 $ | 4.64 $ | 0.9928 | **0.9983** |
| **AAPL** | `123` | **3.91 $** | **3.18 $** | **0.811** | 5.61 $ | 4.64 $ | 0.9924 | **0.9980** |
| **AAPL** | `2026` | **5.96 $** | **3.18 $** | **0.533** | 7.28 $ | 4.64 $ | 0.9932 | **0.9988** |

Units: TL for `THYAO.IS` and `ASELS.IS`, index points for `XU100.IS`, US dollars ($) for `BTC-USD` and `AAPL`.

---

## 3. Part 5 Direction Classification Check (5 Tickers × 3 Seeds)

Acceptance criterion: test direction accuracy (`Test Accuracy`) between `0.45` and `0.60` (`45% – 60%`). In all 15 runs accuracy came out between `0.4688` and `0.5521`, inside the acceptance band:

| Ticker | Seed | Test accuracy (`Accuracy`) | Balanced accuracy (`Balanced Acc`) | Majority-class baseline (`Majority Acc`) |
|---|---|---|---|---|
| **THYAO.IS** | `42` | **0.5474** (54.74%) | 0.5351 | 0.5190 |
| **THYAO.IS** | `123` | **0.5427** (54.27%) | 0.5309 | 0.5190 |
| **THYAO.IS** | `2026` | **0.5521** (55.21%) | 0.5408 | 0.5190 |
| **ASELS.IS** | `42` | **0.5213** (52.13%) | 0.5249 | 0.4858 |
| **ASELS.IS** | `123` | **0.5213** (52.13%) | 0.5244 | 0.4858 |
| **ASELS.IS** | `2026` | **0.5190** (51.90%) | 0.5224 | 0.4858 |
| **XU100.IS** | `42` | **0.5024** (50.24%) | 0.4984 | 0.5048 |
| **XU100.IS** | `123` | **0.4879** (48.79%) | 0.4835 | 0.5048 |
| **XU100.IS** | `2026` | **0.5048** (50.48%) | 0.5006 | 0.5048 |
| **BTC-USD** | `42` | **0.4976** (49.76%) | 0.5035 | 0.4911 |
| **BTC-USD** | `123` | **0.5024** (50.24%) | 0.5078 | 0.4911 |
| **BTC-USD** | `2026` | **0.5024** (50.24%) | 0.5090 | 0.4911 |
| **AAPL** | `42` | **0.4688** (46.88%) | 0.4509 | 0.5337 |
| **AAPL** | `123` | **0.5072** (50.72%) | 0.4811 | 0.5337 |
| **AAPL** | `2026` | **0.4952** (49.52%) | 0.4705 | 0.5337 |

---

## 4. Rough Edges Found and Fixed During the Rehearsal

1. **Colab working folder (`/content`) and the Part 0 hook image (`NameError` fix):**
   - **Finding:** When the starter notebook was opened in a live Colab session (working folder `/content`), `docs/hook_preview.png` was not yet under the local `/content/docs/`, so the `sec0-hook-plot` cell fell into its `else:` block and raised a `NameError`, because the Part 4 catch-up function (`catch_up_part_4`, then named `yetistir_bolum_4`) was not imported in `setup-code`.
   - **Fix:** Imports of `catch_up_part_4` and `plot_price_deception` were added to both `setup-code` and `sec0-hook-plot`, and the path `/content/pytorch-finance-workshop/docs/hook_preview.png` was added to the list of candidate paths. Rerunning it in live Colab confirmed that it runs without errors (`0 error`).
2. **Self-contained "🔄 Catch up" cells:**
   - **Finding:** A student whose runtime was reset (runtime restart) may forget to run the first setup cell (`setup-code`) and go straight to the **"🔄 Catch up"** cell of the part they are on.
   - **Fix:** Each of the 6 catch-up cells got its own `sys.path` / `urllib.request` bootstrap code at the top, and `tests/test_unit.py::test_every_catch_up_cell_works_in_fresh_process` checks that each one runs on its own in a fresh Python process.
3. **`yfinance` single-ticker `MultiIndex` and `FutureWarning` behaviour:**
   - **Finding:** `yf.download("THYAO.IS")` uses `multi_level_index=True` and `auto_adjust=None` by default, so it returned `MultiIndex` columns even for a single ticker and printed a `FutureWarning`.
   - **Fix:** `load_prices` now passes `auto_adjust=True, multi_level_index=False` explicitly, with an extra `isinstance(out.columns, pd.MultiIndex)` check as a safeguard.
4. **Catch-up cells left the next cell of their part broken after a runtime reset (`NameError`):**
   - **Finding:** The catch-up cells ran on their own but did not import `np`, `plt` or `torch`, so after a runtime reset the next cell of the part raised `NameError` (for example `NameError: name 'plt' is not defined`), which is exactly the case these cells exist for.
   - **Fix:** All six catch-up cells now start with a shared catch-up bootstrap (`CATCHUP_BOOTSTRAP` in `scripts/build_notebooks.py`) that imports these names. `tests/test_unit.py::test_every_catch_up_cell_restores_its_section` runs each catch-up cell together with the rest of its part in a fresh process.
5. **The Part 0 hook gave away the Part 4 answer in Colab:**
   - **Finding:** In Colab `docs/hook_preview.png` is not on disk, so the Part 0 hook trained the price model live and printed the Part 4 answer (LSTM MAE vs. naive MAE) before the surprise.
   - **Fix:** The hook now shows the pre-generated image from GitHub; only if that download also fails does it train the model, with its output hidden.
