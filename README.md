# Machine Learning Final Project

This is the final project for the Machine Learning course in Spring 2026 at NYU Shanghai. The competition repository is available at https://github.com/NYUSH-ML/ml-competition-sp26.

This repository contains my final code package for the CSI 500 machine-learning competition. The project builds a long-only portfolio from model-generated stock-selection signals, validates the final CSV against the competition rules, and reports a chronological no-leakage self-test.

> This repository is not investment advice and is used only for education.

## Final Submission

The submitted Phase 2 portfolio is:

```text
submissions/Yuxin_He_week2_orig66_xgb34_k34.csv
```

It contains 34 CSI 500 constituents, equal weighted at `1/34 = 2.941%`. The final construction is a fixed portfolio-level ensemble:

```text
66% updated LightGBM-style multi-horizon model portfolio
34% official-style XGBoost baseline portfolio
select top 34 by blended portfolio weight
equal weight selected names
```

The key idea is to combine two complete model portfolios instead of averaging raw model scores. This avoids score-scale mismatch between LightGBM and XGBoost and makes the ensemble weight directly interpretable.

## Repository Contents

Core files:

```text
generate_submission.py          # rebuilds the final submitted CSV
self_test.py                    # rolling no-leakage validation/test evaluation
phase2_baseline_xgb.py          # self-contained official-style XGBoost component
score_submission.py             # portfolio/benchmark/excess-return scorer
validate_submission.py          # competition-rule validator
download_data.py                # data snapshot downloader
requirements.txt                # Python dependencies
```

LightGBM component support:

```text
factor_experiments_slim_asof20260430/features.py
factor_experiments_slim_asof20260430/train_ensemble.py
factor_experiments_slim_asof20260430/self_test_multihorizon_lgb.py
```

Archived outputs used by the final reproduction and self-test:

```text
submissions/Yuxin_He_week2_orig66_xgb34_k34.csv
submissions/original_submission1_model_asof20260508.csv
submissions/phase2_baseline_xgb_asof20260508.csv
submissions/rolling_3d_compare/*.csv
submissions/rolling_3d_baseline_xgb_phase2/*.csv
logs/self_test_*.csv
logs/self_test_summary.json
```

The rolling component CSV files are intentionally kept because `self_test.py` uses them to reconstruct historical as-of portfolios without lookahead. Raw market data and model binaries are not committed.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Regenerate the final CSV from the included component portfolios:

```bash
python generate_submission.py --out submissions/Yuxin_He_week2_orig66_xgb34_k34.csv
```

Validate the final portfolio:

```bash
python validate_submission.py submissions/Yuxin_He_week2_orig66_xgb34_k34.csv
```

Run the no-leakage self-test:

```bash
python self_test.py
```

`self_test.py` requires the competition data files under `data/`:

```text
data/constituents.csv
data/prices.parquet
data/index.parquet
```

If the data files are missing, download them with:

```bash
python download_data.py --start 20250101 --end 20260508
```

## Data

The final submitted model uses only the data provided by the competition workflow:

- `prices.parquet`: daily stock OHLCV data
- `index.parquet`: CSI 500 benchmark index data
- `constituents.csv`: eligible CSI 500 stock universe

I intentionally excluded external datasets from the final package. During experimentation, I tested or considered industry labels, size/style controls, margin-financing-style variables, and northbound-flow-style variables. They were not included in the final model because their rolling validation gains were inconsistent and some introduced timestamp or reporting-lag risk.

The repository excludes local market-data snapshots through `.gitignore`; only `data/.gitkeep` is tracked.

## Signal Design

The final model focuses on short-horizon OHLCV signals rather than long-term firm quality. The prediction horizon is short, so the feature families are designed to capture near-term cross-sectional price pressure, liquidity, volatility, and continuation/reversal behavior.

Main signal families:

| Signal family | Examples | Purpose |
|---|---|---|
| Short-term momentum | 1/3/5/10/20-day returns and ranks | Captures recent winners and losers over short horizons. |
| Reversal and acceleration | Return acceleration and short-window reversal terms | Separates persistent trends from overextended moves. |
| Liquidity and volume pressure | Turnover averages, volume z-scores, amount ratios, Amihud-style illiquidity | Checks whether moves are supported by trading activity. |
| Volatility and risk | Rolling volatility, volatility ratios, range, skewness, kurtosis | Captures short-term risk and instability. |
| Price location and technical state | Moving-average ratios, RSI, MACD-style terms, distance to recent highs/lows | Measures breakout, consolidation, and support/resistance location. |
| Intraday and overnight behavior | Overnight return, VWAP bias, close-strength-style features | Captures gap behavior and intraday buying/selling pressure. |

The LightGBM-style component uses a richer 52-feature OHLCV set and a multi-horizon target. The XGBoost baseline component uses a smaller 14-feature set modeled after the official quick-start baseline, which makes it a useful regularizing component.

## Model Components

### 1. LightGBM-Style Multi-Horizon Component

The first component is an updated version of my Phase 1 LightGBM-style model. It is trained on the provided OHLCV panel and predicts three forward-return horizons:

```text
TARGET_HORIZONS = (3, 5, 10)
```

For each horizon, the training procedure uses chronological walk-forward validation:

- 180 trading days for training
- 20 trading days for validation
- 5 trading days of embargo
- 8 chronological folds
- 60-trading-day recency half-life in sample weights
- best-2 validation folds selected by validation IC

The three horizon scores are blended with fixed weights:

```text
s_LGB = 0.50 * s_3d + 0.30 * s_5d + 0.20 * s_10d
```

The final current LightGBM-style component portfolio is:

```text
submissions/original_submission1_model_asof20260508.csv
```

### 2. Official-Style XGBoost Baseline Component

The second component is a conservative XGBoost model trained from scratch with the smaller official-style feature set:

```text
ret_1d, ret_5d, ret_10d, ret_20d, ret_60d,
vol_20d, volume_z_20d, turnover_ma_20d,
close_over_ma20, close_over_ma60, rsi_14,
ret_5d_rank, ret_20d_rank, vol_20d_rank
```

It predicts the 5-day forward return target and uses a chronological train/validation split with a 5-day embargo. The final current XGBoost component portfolio is:

```text
submissions/phase2_baseline_xgb_asof20260508.csv
```

### 3. Portfolio-Level Ensemble

The final submission blends the component portfolio weights:

```text
blended_weight_signal_i = 0.66 * LGB_weight_i + 0.34 * XGB_weight_i
```

Then it selects the top 34 names by the blended weight signal and assigns equal weights. No manual stock pool, blacklist, target list, or hand-picked overlay is used.

Why portfolio-level blending?

- It avoids raw-score scale mismatch across model families.
- It requires a stock to receive support from model-generated portfolios.
- It reduces sensitivity to noisy rank differences near the top of the list.
- Equal weighting keeps the final portfolio diversified and simple.

## Self-Test Methodology

The competition asked for a training/validation/test split with sound methodology and no leakage. The self-test uses rolling chronological splits. For each historical as-of date:

1. Load the LightGBM-style and XGBoost component portfolios generated at that same as-of date.
2. Apply the same fixed 66/34 top-34 equal-weight recipe.
3. Score the resulting portfolio on the subsequent 5-trading-day out-of-sample window.
4. Compare against the CSI 500 benchmark and the official-style XGBoost component.
5. Compute rank IC between model-generated portfolio scores and future stock returns.

The as-of date is always strictly before the evaluation window.

Validation windows:

| As-of date | Evaluation window | Role |
|---|---|---|
| 2026-03-13 | 2026-03-16 to 2026-03-20 | Model comparison |
| 2026-03-20 | 2026-03-23 to 2026-03-27 | Model comparison |
| 2026-03-27 | 2026-03-30 to 2026-04-03 | Model comparison |
| 2026-04-03 | 2026-04-07 to 2026-04-13 | Model comparison |

Test windows:

| As-of date | Evaluation window | Role |
|---|---|---|
| 2026-04-10 | 2026-04-13 to 2026-04-17 | Held-out evaluation |
| 2026-04-17 | 2026-04-20 to 2026-04-24 | Held-out evaluation |
| 2026-04-24 | 2026-04-27 to 2026-05-06 | Held-out evaluation |

## Results

Return summary from `logs/self_test_return_summary.csv`:

| Split | Windows | Mean portfolio | Mean benchmark | Mean excess | Worst excess | Hit rate | Cumulative excess |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 4 | -0.022% | -0.747% | +0.725% | -0.957% | 50.0% | +2.856% |
| Test | 3 | +6.891% | +2.537% | +4.354% | +3.539% | 100.0% | +13.634% |
| All | 7 | +2.941% | +0.661% | +2.280% | -0.957% | 71.4% | +16.879% |

Baseline comparison from `logs/self_test_baseline_comparison.csv`:

| Split | Mean excess advantage vs XGB | Cumulative advantage vs XGB | Hit rate vs XGB | Final-selection IC advantage |
|---|---:|---:|---:|---:|
| Validation | -0.298% | -1.293% | 25.0% | -0.0435 |
| Test | +2.161% | +6.921% | 100.0% | +0.0993 |
| All | +0.756% | +5.739% | 57.1% | +0.0177 |

The model does not dominate the XGBoost baseline in the earlier validation period, but it outperforms meaningfully in the later held-out test windows. This is consistent with the report's interpretation that March to early April 2026 was a difficult regime for short-horizon signals, while the later April test regime was more favorable for momentum/liquidity-based OHLCV models.

IC summary from `logs/self_test_ic_summary.csv`:

| Split | Signal | Mean IC | ICIR | Positive rate |
|---|---|---:|---:|---:|
| Validation | Raw blended score | +0.0537 | +0.28 | 50.0% |
| Validation | Final top-34 selection | +0.0177 | +0.12 | 25.0% |
| Test | Raw blended score | +0.1935 | +3.84 | 100.0% |
| Test | Final top-34 selection | +0.2118 | +5.79 | 100.0% |
| All | Raw blended score | +0.1136 | +0.73 | 71.4% |
| All | Final top-34 selection | +0.1009 | +0.68 | 57.1% |

## Reproducibility Notes

- The submitted CSV can be regenerated from `generate_submission.py` and the included current component portfolios.
- `self_test.py` reconstructs historical portfolios from as-of-specific component CSV files.
- Evaluation windows begin after their corresponding as-of dates.
- The final blend weights and top-34 equal-weight rule are fixed across windows.
- No external data, private data, other students' submissions, or manual stock-picking overlay is used.
- Raw data files are excluded from Git; download them locally before running the full self-test.
