# Slide Outline (`SLIDES_OUTLINE.md`)

**Event:** Bringing AI to Financial Data (PyTorch Workshop)  
**Community:** Işık IEEE Computer Society · Işık University Şile Campus (DMF-214)  
**Date / time:** Wednesday 18 November 2026, 16:30–18:45  

This outline is a **9-slide** plan for the 10-minute opening (Slides 1–5) at the start of the hands-on Colab session and the 5-minute wrap-up (Slides 6–9) at the end.

---

## Slide 1 · Cover and Quick Start (Opening — 16:30)
- **Title:** Bringing AI to Financial Data: Time Series with PyTorch
- **Subtitle:** Işık IEEE Computer Society Hands-on Workshop · Omar Elcircevi
- **Bullets:**
  - 💻 **Starter notebook (Colab):** `https://colab.research.google.com/github/omarcevi/pytorch-finance-workshop/blob/main/notebooks/workshop_starter.ipynb` *(large QR code + short link on the slide: `omarcevi.dev/w/pytorch-finans`)*
  - 📘 **Step-by-step guide:** `https://omarcevi.dev/workshops/pytorch-finance-workshop/`
  - ⚠️ **Note:** For education only; not investment advice.
- **Visual:** The Colab badge and a large QR code to the starter notebook.

---

## Slide 2 · Hook: "Could We Make Money with This Chart?" (16:34)
- **Title:** A "Perfect" Price Prediction with 15 Lines of PyTorch?
- **Bullets:**
  - Blue line: actual `THYAO.IS` closing prices in the test period; red line: our LSTM model's prediction.
  - You will find this chart in hundreds of blog posts and videos online, under headlines like "AI has cracked the stock market".
  - **Question:** If we hooked this model up live tomorrow morning, would we make money? We'll find the answer together at 17:50.
- **Notebook chart:** `docs/hook_preview.png` (`Part 0 / Part 4 — plot_price_deception` output).

---

## Slide 3 · Today's Roadmap (16:37)
- **Title:** What Will We Build in 2 Hours 15 Minutes?
- **Bullets:**
  - **16:40 – 17:15 | Data and Feature Engineering:** Borsa Istanbul (`THYAO.IS`) OHLCV data, log returns, volatility, RSI and a leak-free chronological split.
  - **17:15 – 17:35 | PyTorch from Scratch:** `Tensor`, `SlidingWindowDataset` (30 days), the `nn.LSTM` architecture and the five-step training loop.
  - **17:45 – 18:25 | The Big Surprise and the Right Setup:** the one-day-lag illusion behind price prediction, the naive baseline, and direction classification + walk-forward validation.
- **Visual:** A timeline flowchart (`Part 1 -> Part 6`, with the **🔄 Catch up** safety net at the start of each part).

---

## Slide 4 · The Golden Rule: Don't Leak the Future into Time Series (16:55 transition)
- **Title:** Where Does Data Leakage Hide?
- **Bullets:**
  - **Rule 1:** Never split train/test with a random `shuffle=True`; train on the past, test on the future (70% train / 15% val / 15% test).
  - **Rule 2:** Call `StandardScaler.fit()` on the **training** region only; fitting on all the data leaks the future mean and volatility into the past.
  - **Rule 3:** Fill missing days only with past values (`ffill`); never fill backwards from the future (`bfill`).
- **Notebook chart:** `guide/images/chronological_split.svg` (`Part 2 — plot_chronological_split` output).

---

## Slide 5 · PyTorch LSTM Architecture and Shape Tracking (17:15 transition)
- **Title:** From a 30-Day Window to One Prediction: Tensor Shapes
- **Bullets:**
  - Input tensor `bx`: **`(batch=64, seq_len=30, features=4)`** — 64 samples, 30 trading days, 4 features.
  - `nn.LSTM(4, 32, batch_first=True)` output: **`(64, 30, 32)`** — a 32-dimensional hidden state for each day.
  - Summary of the last day, `out[:, -1, :]`: **`(64, 32)`** $\rightarrow$ `nn.Linear(32, 1)` $\rightarrow$ **`(64,)`** prediction vector.
- **Notebook chart:** `Part 3 — plot_training_curves` (training and validation MSE loss curves).

---

## Slide 6 · The Twist: Why Did "Tomorrow = Today" Beat the LSTM? (17:55)
- **Title:** An Optical Illusion: Not a Prediction, a One-Day Lag!
- **Bullets:**
  - `THYAO.IS` test set: **LSTM MAE = 5.07 TL**, while the **naive baseline ($\hat{P}_{t+1} = P_t$) MAE = 4.55 TL**, and the naive baseline learns nothing at all!
  - Zooming into the last 35 days shows that the red prediction line is a **copy of the blue actual price series shifted one day to the right** (`Lag 1` correlation = **0.9915**).
  - Why? Prices are very close to a random walk ($P_{t+1} = P_t + \epsilon$); the easiest way to minimise the MSE loss is to repeat yesterday's price.
- **Notebook chart:** `guide/images/zoom_and_lag.svg` (`Part 4 — plot_zoom_and_lag` two-panel chart).

---

## Slide 7 · The Right Setup: Direction Classification and the 50% Reality (18:15)
- **Title:** Return Direction Instead of Price Level (`BCEWithLogitsLoss`)
- **Bullets:**
  - We classify tomorrow's direction (`Up = 1 / Down = 0`) from stationary features (`log_return`, moving average, volatility, RSI).
  - Single test-window accuracy **54.74%** against a majority-class baseline of **51.90%**. With 422 test predictions the 95% confidence interval is about **±4.8 points (≈50%–59.5%)**, so this cannot be told apart from chance. The 4-fold **walk-forward** mean is **51.66%**.
  - Why is daily direction prediction in the 50–55% band in an honest setup? **Efficient markets**, **high noise** and **changing market regimes**.
- **Notebook chart:** `guide/images/direction_results.svg` (`Part 5 — plot_direction_results`: confusion matrix and walk-forward bar chart).

---

## Slide 8 · Is That an Edge? Trading Costs and Uncertainty (18:35)
- **Title:** One Good Backtest Is Not an Edge
- **Bullets:**
  - On the `THYAO.IS` test window buy & hold ends at **0.99x**, the strategy at **1.24x** gross and **1.18x** net after **15 basis points (0.15%)** per position change. With only **29** position changes, costs take little off; the same holds across 5 seeds (net **1.13x–1.24x**).
  - Why we don't believe it yet: the model is in the market only about 1 day in 5 (it mostly says "down") during a flat market, and the 95% confidence interval on its 54.74% accuracy (±4.8 points, ≈50%–59.5%) includes both 50% and the 51.90% baseline. This is one lucky window, not evidence of an edge; the honest next step is to rerun it across seeds, tickers and walk-forward folds.
  - Real quant teams don't predict the daily price of a single stock: they rank hundreds of assets together, use alternative data and build trading costs directly into the optimisation.
- **Notebook chart:** The `Part 6 Challenge 4 (Bonus) — Backtest with trading costs` chart (buy & hold vs. gross strategy vs. strategy net of 15 bps).

---

## Slide 9 · 3 Takeaways and Wrap-up (18:40)
- **Title:** Summary: 3 Rules for Being a Good Data Scientist
- **Bullets:**
  1. **Always race against a naive baseline:** A model that cannot beat "tomorrow = today" or the majority class has learned nothing.
  2. **Respect time order:** High scores obtained without a chronological split and walk-forward validation are misleading.
  3. **Hunt for leakage:** If you see a chart that is too good to be true (90% accuracy, or price lines sitting on top of each other), check for data leakage and the one-day lag first.
  - 🔗 **Code and guide:** `github.com/omarcevi/pytorch-finance-workshop` · `omarcevi.dev/workshops/pytorch-finance-workshop`
  - ⚠️ *Nothing in this workshop is investment advice.*
