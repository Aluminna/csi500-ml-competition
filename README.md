# CSI 500 Machine Learning Competition

An end-to-end portfolio modeling project for a CSI 500 machine-learning competition. The repository contains OHLCV feature engineering, LightGBM and XGBoost components, walk-forward evaluation, portfolio construction, and the exact Phase 2 submission recipe.

> Educational and research use only. This repository is not investment advice and does not represent a live trading strategy.

The final Phase 2 portfolio uses:

```text
66% updated original submission-1 model portfolio
34% official-style XGB baseline portfolio
top 34 stocks
equal weight
```

The final CSV is:

```text
submissions/Yuxin_He_week2_orig66_xgb34_k34.csv
```

## Method

The submitted portfolio is a fixed-weight portfolio-level ensemble of two model-generated portfolios:

1. `original_submission1_model_asof20260508.csv`
   - Updated version of my Phase 1 LGB-style model portfolio.
   - The rolling self-test uses historical `pure_lgb_k40_sqrt_<asof>.csv` component portfolios generated at the corresponding as-of dates.

2. `phase2_baseline_xgb_asof20260508.csv`
   - Official-style XGBoost baseline component trained from scratch with self-contained OHLCV features and a 5-day target.
   - The rolling self-test uses historical `baseline_xgb_<asof>.csv` portfolios generated at the corresponding as-of dates.

The final construction blends the two portfolio-weight signals with fixed weights:

```text
blended_score = 0.66 * original_weight_signal + 0.34 * baseline_xgb_weight_signal
```

Then it selects the top 34 stocks by `blended_score` and assigns equal weights. No manual stock pool, blacklist, target list, or hand-picked overlay is used.

## Reproduce the Submission

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the submitted portfolio:

```bash
python generate_submission.py --out submissions/Yuxin_He_week2_orig66_xgb34_k34.csv
```

Validate the portfolio:

```bash
python validate_submission.py submissions/Yuxin_He_week2_orig66_xgb34_k34.csv
```

These two commands use the included component portfolio CSV files and do not require the full market-data snapshot.

Run the no-leakage self-test with return and IC metrics:

```bash
python self_test.py
```

The self-test writes:

```text
logs/self_test_detail.csv
logs/self_test_return_summary.csv
logs/self_test_ic_summary.csv
logs/self_test_baseline_comparison.csv
logs/self_test_summary.json
```

## Self-Test Design

The self-test uses rolling chronological splits. For each historical as-of date, it loads the component portfolios generated for that same as-of date, applies the same fixed 66/34 top34-equal recipe, and evaluates the subsequent 5-trading-day out-of-sample window.

Validation windows:

| As-of date | Evaluation window |
|---|---|
| 2026-03-13 | 2026-03-16 to 2026-03-20 |
| 2026-03-20 | 2026-03-23 to 2026-03-27 |
| 2026-03-27 | 2026-03-30 to 2026-04-03 |
| 2026-04-03 | 2026-04-07 to 2026-04-13 |

Test windows:

| As-of date | Evaluation window |
|---|---|
| 2026-04-10 | 2026-04-13 to 2026-04-17 |
| 2026-04-17 | 2026-04-20 to 2026-04-24 |
| 2026-04-24 | 2026-04-27 to 2026-05-06 |

For every row, the as-of date is strictly before the evaluation window, and the component portfolios are historical as-of files rather than the final 2026-05-08 portfolio.

## IC Reporting

Because the final output is a portfolio-level ensemble, IC is computed from model-generated portfolio scores:

| Metric | Meaning |
|---|---|
| `blend_raw_rank_ic_full` | Spearman rank IC between the pre-top34 66/34 blended portfolio score and future 5-day stock returns over the CSI500 universe. |
| `candidate_final_rank_ic_full` | Spearman rank IC between the final top34 equal-weight selection signal and future 5-day stock returns. Unselected stocks receive score 0. |
| `baseline_raw_rank_ic_full` | Spearman rank IC of the official-style XGB component weight signal. |
| `baseline_same_k_rank_ic_full` | Spearman rank IC of a top34 equal-weight version of the baseline signal, used as an apples-to-apples construction comparison. |

The return self-test also compares against the official-style XGB baseline component on the same rolling windows.

## Baseline Comparison Highlights

`logs/self_test_baseline_comparison.csv` summarizes the no-leakage metrics that are directly comparable to the baseline:

- mean excess-return advantage versus the official-style XGB baseline
- cumulative excess-return advantage versus the official-style XGB baseline
- share of windows where the submitted portfolio beats the baseline
- IC advantage of the final top34 selection signal versus a top34 equal-weight baseline signal
- IC advantage of the raw blended score versus the raw baseline score

The archived seven-window walk-forward summary is intentionally reported in full:

| Split | Windows | Mean excess-return advantage vs. XGB | Hit rate vs. XGB | Final-selection IC advantage |
|---|---:|---:|---:|---:|
| Validation | 4 | -0.30% | 25.0% | -0.0435 |
| Test | 3 | +2.16% | 100.0% | +0.0993 |
| All | 7 | +0.76% | 57.1% | +0.0177 |

These are historical competition-window results, not estimates of future performance.

## Included Code

Core reproduction files:

```text
generate_submission.py
self_test.py
score_submission.py
validate_submission.py
phase2_baseline_xgb.py
run_rolling_3d_baseline_xgb_phase2.py
run_rolling_3d_two_versions.py
search_original_baseline_blend.py
```

The package also includes the Phase 1 feature/model support files used by the LGB-style component under `factor_experiments_slim_asof20260430/`.

## Data

The GitHub repository intentionally excludes the local market-data snapshot. Download a fresh CSI 500 constituent list, stock OHLCV panel, and benchmark series with:

```bash
python download_data.py --start 20250101 --end 20260421
```

This creates `data/constituents.csv`, `data/prices.parquet`, and `data/index.parquet`. The downloader uses public market-data interfaces exposed by AkShare; availability and field names may change over time.

No private data, future data after an as-of date, other students' submissions, or manually selected stock lists are used by the documented pipeline.

## Repository Layout

```text
generate_submission.py                # exact 66/34 Top-34 recipe
phase2_baseline_xgb.py                # self-contained XGBoost baseline
self_test.py                          # chronological no-leakage evaluation
validate_submission.py                # competition-rule checks
score_submission.py                   # realized return/excess-return scoring
factor_experiments_slim_asof20260430/ # Phase 1 LightGBM experiments
submissions/                          # small component/output CSV files
logs/                                 # archived self-test summaries
```

## Reproducibility Notes

- Every rolling evaluation uses component portfolios generated at that historical as-of date.
- Evaluation windows begin strictly after their corresponding as-of dates.
- The final blend weights and Top-34 equal-weight rule are fixed across windows.
- Raw market data, trained model binaries, caches, and local credentials are excluded from version control.
