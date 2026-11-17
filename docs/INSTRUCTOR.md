# Instructor Guide (`INSTRUCTOR.md`)

**Event:** Bringing AI to Financial Data (PyTorch Workshop)  
**Community:** Işık IEEE Computer Society  
**Place and time:** Işık University Şile Campus, DMF-214 · Wednesday 18 November 2026, **16:30–18:45** (2 hours 15 minutes)  
**Audience:** Beginner to intermediate university students who know basic Python and `pandas` but have never written a PyTorch training loop from scratch (~40 people, Google Colab CPU on their own laptops).

> ⚠️ **Instructor reminder:** At the opening, at the Part 4 surprise and at the wrap-up, say clearly that this work is **not investment advice** and that these models must not be used as real trading tools.

---

## 1. Pre-Event Checklist

### Morning (18 November 2026, 09:00–12:00)
- [ ] **Test `yfinance` from Colab:** Open a clean Google Colab CPU runtime and test `load_prices("THYAO.IS")` both through live `yfinance` and with `force_fallback=True`.
- [ ] **Check the snapshot (`SNAPSHOT_BASE_URL`):** Confirm that the `SNAPSHOT_BASE_URL` address in `src/workshop_utils.py` (`https://gist.githubusercontent.com/omarcevi/ac2e894a14dd4d068a3baa9935631ec5/raw` — secret GitHub Gist: `https://gist.github.com/omarcevi/ac2e894a14dd4d068a3baa9935631ec5`) is reachable; when 40 students in DMF-214 connect to Yahoo Finance at the same time, hitting the rate limit (`YFRateLimitError`) is very likely.
- [ ] **Short link and QR code:** Check the short link (e.g. `omarcevi.dev/w/pytorch-finans` / `bit.ly/isik-pytorch-2026`) for the starter notebook (`workshop_starter.ipynb`) Colab link and the guide page (`https://omarcevi.dev/workshops/pytorch-finance-workshop/`), and the large QR code for the first slide.
- [ ] **Backup tab:** Keep the `workshop_solution.ipynb` notebook, **already run with all outputs ready**, open in a second browser tab.
- [ ] **Offline local backup (`requirements.txt`):** In case of a general outage of Google Colab or the campus internet, have a local Jupyter environment installed with `pip install -r requirements.txt` and the `data/*.csv` files ready on your laptop.

### In the room (16:00–16:25, DMF-214)
- [ ] Check the projector resolution; set the Colab zoom to `125–135%` and choose the light theme (the dark theme is hard to read on a projector).
- [ ] Write the Wi-Fi details, the guide address (`omarcevi.dev/workshops/pytorch-finance-workshop`) and the note *"If you fall behind, run the 🔄 Catch up cell at the start of the part"* on the board.

---

## 2. Minute-by-Minute Run Sheet (16:30–18:45)

| Time | Length | Part | Cell to have open (`id`) | What the instructor does | What students do |
|---|---|---|---|---|---|
| **16:30–16:35** | 5 min | **Opening & Setup** | `intro-header`, `setup-code` | Short introduction, the disclaimer (not investment advice), gets everyone to open the Colab link and run the first setup cell. | Open the starter notebook in Colab and run the `setup-code` cell (`PyTorch version: 2.14.1 \| Device: CPU`). |
| **16:35–16:40** | 5 min | **Part 0: Intro (Hook)** | `sec0-md`, `sec0-hook-plot` | Puts the pre-generated THYAO price prediction chart on the screen. Asks *"Could we make money with this chart?"*, takes a show of hands, and keeps the answer for Part 4. | Look at the chart and make a guess. |
| **16:40–16:48** | 8 min | **Part 1: Downloading Data** | `sec1-load-code` | Writes the `load_prices("THYAO.IS")` TODO live. If Yahoo hits a rate limit, shows how the snapshot takes over (`⚠️ Could not get THYAO.IS from Yahoo Finance (...). Loading the snapshot data...`). Explains OHLCV and dividend/split adjustment (`auto_adjust=True`) in 3 sentences. | **TODO 1:** Complete the `raw_df = load_prices(TICKER, ...)` line and look at the last 5 rows. |
| **16:48–16:55** | 7 min | **Part 1: Visualisation** | `sec1-plot-code` | Plots the close price and the volume; points out the difference in scale between single-digit lira in 2015–2016 and 300+ TL in 2026. | Run the chart cell and look at the chart. |
| **16:55–17:03** | 8 min | **Part 2: Features** | `sec2-feat-code` | Explains why we use only `ffill` and never `bfill`, and why we compute the log return ($\ln(P_t/P_{t-1})$) instead of using the raw price. | **TODO 2:** Write the `log_return`, `roll_mean_5` and `roll_std_10` columns. |
| **17:03–17:09** | 6 min | **Part 2: Chronological Split** | `sec2-split-code` | Explains why shuffling a time series at random (`shuffle=True`) means copying the future; plots the 70%/15%/15% chart. | **TODO 3:** Complete the `chronological_split` call and plot the three-region chart. |
| **17:09–17:15** | 6 min | **Part 2: Scaler and Leakage** | `sec2-scaler-code` | Shows that `StandardScaler.fit_transform` is run on **`train_df` only**; stresses that fitting on all the data is data leakage. | **TODO 4:** Write the `scaler.fit_transform(train)` and `scaler.transform(val/test)` lines. |
| **17:15–17:22** | 7 min | **Part 3: Tensor & Dataset** | `sec3-tensor-code`, `sec3-dataset-code` | Shows the `SlidingWindowDataset` (30 days). Explains that the split stays chronological, but shuffling the windows inside `train_loader` (`shuffle=True`) is safe. | **TODO 5:** Create the `SlidingWindowDataset` and `DataLoader` objects; see `torch.Size([64, 30, 4])` on the screen. |
| **17:22–17:28** | 6 min | **Part 3: `nn.Module` & Shapes** | `sec3-model-code` | Writes the `SimpleLSTM` class; shows the shape changes `bx -> lstm_out -> last_step -> pred_out` line by line on the screen. | **TODO 6:** Complete the `forward` method and check the four-step shape output. |
| **17:28–17:35** | 7 min | **Part 3: Training Loop** | `sec3-train-code` | Writes the five-step PyTorch loop (`zero_grad`, `forward`, `loss`, `backward`, `step`) live and plots the loss curve. | **TODO 7:** Complete the training loop and run 10 epochs (~3 s); the last line reads `Last epoch (10) -> Train MSE: 0.000646 \| Val MSE: 0.000509`. |
| **17:35–17:45** | 10 min | **☕ Break** | `break-md` | Goes to students who have questions; shows anyone who has fallen behind the `sec4-catchup` (**🔄 Catch up: Part 4**) cell. | Take a break. |
| **17:45–17:52** | 7 min | **Part 4: Price Prediction** | `sec4-catchup`, `sec4-train-code` | After the break, has everyone run `sec4-catchup`. Trains the price model (`train_price_lstm`) and plots the test chart that looks "perfect" from afar. Also briefly mentions what `tanh` saturation (the second trap) would do if a global scaler were used. | Run the `sec4-train-code` cell and see the "perfect" chart on their own screen. |
| **17:52–17:58** | 6 min | **Part 4: Surprise 1 (Naive Baseline)** | `sec4-naive-code` | **Key teaching moment:** Has students compute the naive baseline (*"tomorrow = today"*, $\hat{P}_{t+1}=P_t$). The naive baseline's MAE (**4.55 TL**) beats the LSTM's MAE (**5.07 TL**), and its RMSE does too (6.26 TL vs. 6.86 TL)! | **TODO 8:** Write the `lstm_mae`, `naive_mae`, `lstm_rmse`, `naive_rmse` calculations and see the surprise. |
| **17:58–18:05** | 7 min | **Part 4: Surprise 2 (One-Day Lag)** | `sec4-zoom-code` | Zooms into the last 35 days (`plot_zoom_and_lag`). Shows that the red prediction line is a **copy of the blue actual line shifted one day to the right**, and that the `Lag 1` correlation comes out at **0.9915** (against 0.9451 for `Lag 0`). | Run the zoomed-in chart and the lag-correlation bars. |
| **18:05–18:15** | 10 min | **Part 5: Direction Classification** | `sec5-catchup`, `sec5-cls-code` | Reframes the problem as tomorrow's direction (`up_day`, `BCEWithLogitsLoss`) on return features. Compares the LSTM accuracy (`54.74%`) and balanced accuracy (`53.51%`) with the majority-class baseline (`51.90%`). | **TODO 9:** Complete the `train_direction_classifier` call and print the metrics. |
| **18:15–18:25** | 10 min | **Part 5: Walk-Forward & the 50% Reality** | `sec5-wf-code` | Runs 4-fold walk-forward validation (`mean 51.66%`; folds 50.58%, 49.42%, 54.34%, 52.31%). Points out that with 422 test predictions the 95% confidence interval on 54.74% is about ±4.8 points (≈50%–59.5%), so the single test score cannot be told apart from chance. Explains why the honest result comes out close to 50%: efficient markets, noise and regime changes. | Produce the walk-forward table and the confusion matrix. |
| **18:25–18:40** | 15 min | **Part 6: Challenges** | `sec6-catchup`, `sec6-ch1..4` | Walks around the room helping students; in the last 4 minutes puts the Challenge 4 (Bonus) backtest chart on the screen: on the THYAO test window buy & hold ends at 0.99x, the strategy at 1.24x gross and 1.18x net after 15 bps costs. With only 29 position changes, costs take little off. Stresses that this is one window, from a model that is in the market only about 1 day in 5, with an accuracy inside the ±4.8-point band around 50%: not evidence of an edge. | Try the challenges: their own ticker, the volume feature, `GRUModel` or `TinyTransformerModel`. |
| **18:40–18:45** | 5 min | **Part 7: Wrap-up & Questions** | `sec7-md` | Sums up the 3 key lessons, explains what real quant teams do differently, repeats the disclaimer and takes questions. | Save the notes and links. |

---

## 3. What Do We Cut If We Fall Behind? (Contingency Plan)

- **If we are 10 minutes behind:**
  - In Part 2, instead of writing the `sec2-feat-code` and `sec2-split-code` cells live, have everyone run `sec3-catchup` (**🔄 Catch up: Part 3**) directly and explain from the chart (saves 5 min).
  - Cut Part 6 (Challenges) from 15 minutes to 10; tell students to pick only Challenge 1 (their own ticker) or Challenge 3 (`GRUModel`) (saves 5 min).
  - **Never cut:** Part 4 (Price Prediction and the Surprise) and Part 5 (Direction Classification).
- **If we are 20 minutes behind:**
  - In Part 3, skip the tensor/Dataset setup with `sec3-catchup` and have students write only `SimpleLSTM.forward` and the 5-line training loop (`sec3-train-code`) live (saves 8 min).
  - In Part 5, run the `sec5-wf-code` (walk-forward) cell as it is and interpret the result from the chart in 2 minutes (saves 4 min).
  - In Part 6, instead of individual coding, run the **Challenge 4 (Bonus): backtest with trading costs** solution cell together and go straight to the wrap-up (saves 8 min).

---

## 4. Script Lines

### 4.1. Part 0 — Intro (Hook) Script (`16:35`)
> *"Everyone, look closely at the chart on the screen. The blue line is the actual closing price of Turkish Airlines stock in the test period; the red dashed line is the prediction of the LSTM model we'll train in a moment with 15 lines of PyTorch code. They sit almost exactly on top of each other! Now let's be honest and raise our hands: if we hooked this model up to live trading tomorrow morning, would we make money? Who says 'yes, we would'? Who says 'there's a catch here'? Good — I'm not giving you the answer. First we'll build this model together from scratch, and then in Part 4 we'll look at this chart through a magnifying glass."*

### 4.2. Part 4 — The Twist Script (`17:52`)
> *"Our model is trained, and you've produced the beautiful chart we saw in Part 0 on your own screen. Now let's ask ourselves: if we used no AI, no PyTorch, no graphics card at all, and wrote the laziest forecasting rule in the world, what would it say? 'Tomorrow's price is the same as today's close ($\hat{P}_{t+1} = P_t$).' Now run the `sec4-naive-code` cell and look at the numbers: our LSTM's average error is **5.07 TL**, and the error of the 'tomorrow = today' rule, which learns nothing, is **4.55 TL**! So how can a model that looks so perfect on the chart come out worse than a naive rule that says 'tomorrow is the same as today'? Let's run the next cell (`sec4-zoom-code`) and zoom into the last 35 days... Look: the red line is a **copy of the blue line shifted exactly one day to the right**! Our model didn't learn to predict the future; it learned that the easiest way to shrink the squared error (MSE) is to repeat yesterday's price. From far away we squeeze hundreds of days into one chart, so that one-day horizontal shift is invisible!"*

---

## 5. The 5 Questions Students Ask Most, with Answers

1. **"Can I use this model for real trading on the stock market?"**
   - **Answer:** No. This workshop is for education only. The Part 6 backtest looks good on paper (on the THYAO test window the strategy ends at 1.18x net of costs, against 0.99x for buy & hold), but that comes from one 22-month window, an accuracy that cannot be told apart from 50%, and a strategy that sat out of the market about 4 days in 5 during a flat market. Nothing shows it would hold on the next window. On top of that, unhedged strategies built on a single stock can lead to serious losses.
2. **"Why don't we add more LSTM layers (3–4 layers) or more neurons?"**
   - **Answer:** In daily financial data we have a limited number of samples (~2,000–3,000 days) and a very high level of noise. Adding layers and parameters does not add signal; it makes the model memorise the random noise in the training set (**overfitting**) and get even worse on the test set.
3. **"Would adding news, KAP disclosures or social-media sentiment change the result?"**
   - **Answer:** Yes. In the real world, **alternative data** (KAP disclosures from Turkey's Public Disclosure Platform, balance-sheet items, interest-rate and exchange-rate expectations, sector news flow) is far more valuable than price history alone. But there too you have to watch out for timestamp leakage (whether the news came out during the trading session or after the close).
4. **"Why did we use an LSTM instead of a Transformer?"**
   - **Answer:** On short, low-dimensional time series (30 steps × 4 features), a small `nn.LSTM` or `nn.GRU` trains very fast and is stable on little data, thanks to its built-in sequential bias (inductive bias). Transformer architectures make a difference mainly with very long contexts or large multivariate datasets (and in the Part 6 bonus we try a small Transformer encoder ourselves!).
5. **"Is 52–54% accuracy useful in real life?"**
   - **Answer:** Not on what we have shown today. On the THYAO test window, costs were not the problem: the strategy changed position only 29 times, so 15 bps costs took it from 1.24x gross to 1.18x net (buy & hold: 0.99x), and across 5 seeds the net result stayed between 1.13x and 1.24x. The problem is the evidence. With 422 test predictions the 95% confidence interval on 54.74% is about ±4.8 points (≈50%–59.5%), which includes both 50% and the 51.90% majority-class baseline, and the model was in the market only about 1 day in 5 (it mostly said "down") during a flat market. That is one lucky window, not evidence of an edge; the honest next step is to rerun it across seeds, tickers and walk-forward folds. If a small, stable edge of 52–53% did hold up, then on low-cost institutional infrastructure, across hundreds of independent stocks and with strict risk management, the law of large numbers could turn it into a meaningful Sharpe ratio. That is also why real quant teams account for trading costs in every test.
