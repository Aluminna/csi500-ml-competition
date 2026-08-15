"""
Self-test for the exact slim LGB multi-horizon submission pipeline.

This matches:
  submissions/slim_ohlcv_top40/
    sub1_best_h3_5_10_r60_k40_best2_sqrt_rank_tradable1.csv

Key configuration:
  - slim OHLCV feature set from this directory's features.py
  - LightGBM only
  - target horizons 3/5/10 with weights 0.50/0.30/0.20
  - recency training sample halflife = 60 trading days
  - walk-forward CV inside each historical as-of date
  - fold selection = best2 among recent folds
  - portfolio = top40 sqrt_rank, recent tradable lookback = 1 day

The test is a no-leakage historical "as-of" simulation. For each test as-of
date, the model is retrained exactly as the production generator would have
done using only labels available before that as-of date. The generated
portfolio is then scored over the next 3/5/10 trading-day windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import train_ensemble as te
from features import (
    FEATURE_COLUMNS,
    TARGET_HORIZONS,
    build_features,
    prediction_frame,
    target_column_for_horizon,
    training_frame,
)


ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

TARGET_HORIZON_WEIGHTS = {3: 0.50, 5: 0.30, 10: 0.20}
RECENCY_HALFLIFE_DAYS = 60.0
TOP_K = 40
FOLD_MODE = "best2"
PORTFOLIO_MODE = "sqrt_rank"
RECENT_TRADABLE_DAYS = 1
DEFAULT_TEST_START = "2026-02-25"
DEFAULT_REBALANCE_STEP = 5


def _parse_date(raw: str | None) -> pd.Timestamp | None:
    return None if raw is None else pd.Timestamp(str(raw))


def _rank_ic(y_true: pd.Series, score: pd.Series) -> float:
    aligned = pd.concat([y_true, score], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(aligned) < 20:
        return float("nan")
    rho, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(rho) if not np.isnan(rho) else float("nan")


def _window_return(
    weights: pd.Series,
    close_pivot: pd.DataFrame,
    open_pivot: pd.DataFrame,
    idx_close: pd.Series,
    idx_open: pd.Series,
    as_of: pd.Timestamp,
    horizon: int,
) -> dict | None:
    """Score next horizon trading days using the official entry convention."""
    dates = close_pivot.index
    as_of_idx = int(np.searchsorted(dates, np.datetime64(as_of)))
    if as_of_idx >= len(dates) or pd.Timestamp(dates[as_of_idx]) != as_of:
        return None
    start_idx = as_of_idx + 1
    exit_idx = as_of_idx + int(horizon)
    if start_idx >= len(dates) or exit_idx >= len(dates):
        return None

    start_date = pd.Timestamp(dates[start_idx])
    exit_date = pd.Timestamp(dates[exit_idx])
    stocks = [s for s in weights.index if s in close_pivot.columns]
    if len(stocks) < 10:
        return None

    # Entry is the last close strictly before start_date, i.e. as_of close.
    entry = close_pivot.loc[as_of, stocks].copy()
    entry = entry.fillna(open_pivot.loc[start_date, stocks])

    # Exit is close on exit_date; if missing, use the last close in the window.
    window_close = close_pivot.loc[start_date:exit_date, stocks]
    exit_ = window_close.ffill().iloc[-1]
    exit_ = exit_.fillna(entry)

    rets = (exit_ / entry - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    portfolio_return = float((weights.reindex(stocks).fillna(0.0) * rets).sum())

    idx_entry = idx_close.get(as_of, np.nan)
    if pd.isna(idx_entry):
        idx_entry = idx_open.get(start_date, np.nan)
    idx_exit = idx_close.get(exit_date, np.nan)
    if pd.isna(idx_entry) or pd.isna(idx_exit) or idx_entry <= 0:
        return None
    benchmark_return = float(idx_exit / idx_entry - 1.0)

    return {
        "eval_start": start_date.date().isoformat(),
        "eval_end": exit_date.date().isoformat(),
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "excess_return": portfolio_return - benchmark_return,
    }


def _fit_predict_for_asof(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    all_dates: np.ndarray,
    as_of: pd.Timestamp,
) -> tuple[pd.Series, pd.Series, dict[str, str], dict[str, str]]:
    """Train exact horizon bundles for one as-of and return combined scores."""
    as_of_idx = int(np.searchsorted(all_dates, np.datetime64(as_of)))
    pred_df = prediction_frame(panel, as_of=as_of)
    if pred_df.empty:
        raise RuntimeError(f"No prediction rows for {as_of.date()}")

    horizon_preds = []
    selected_folds: dict[str, str] = {}
    selected_weights: dict[str, str] = {}
    total_hw = sum(TARGET_HORIZON_WEIGHTS.values())

    for horizon in TARGET_HORIZONS:
        target_col = target_column_for_horizon(horizon)
        cutoff_idx = max(0, as_of_idx - int(horizon))
        train_cutoff = pd.Timestamp(all_dates[cutoff_idx])
        train_pool = training_frame(panel, max_date=train_cutoff, target_column=target_col)

        models, cv_df = te.walk_forward_cv(
            train_pool,
            target_column=target_col,
            target_horizon=int(horizon),
            recency_halflife_days=RECENCY_HALFLIFE_DAYS,
        )
        ensemble, ensemble_weights, selected_cv = te.select_ensemble(
            models, cv_df, method=FOLD_MODE
        )
        pred_stack = np.stack([m.predict(pred_df[FEATURE_COLUMNS]) for m in ensemble])
        preds_h = np.average(pred_stack, axis=0, weights=ensemble_weights)
        horizon_preds.append(preds_h * (TARGET_HORIZON_WEIGHTS[int(horizon)] / total_hw))

        selected_folds[f"h{horizon}"] = ",".join(selected_cv["fold"].astype(int).astype(str))
        selected_weights[f"h{horizon}"] = ",".join(f"{w:.4f}" for w in ensemble_weights)

    combined_preds = np.sum(np.stack(horizon_preds), axis=0)
    scores = pd.Series(
        combined_preds,
        index=pred_df["stock_code"].astype(str).str.zfill(6),
        name="score",
    )

    base_tradable = pred_df.set_index(pred_df["stock_code"].astype(str).str.zfill(6))["is_tradable"]
    recent_ok = te.recent_tradability(
        prices, as_of, lookback_days=RECENT_TRADABLE_DAYS
    )
    is_tradable = base_tradable
    if recent_ok is not None:
        recent_ok = recent_ok.reindex(scores.index).fillna(False)
        is_tradable = base_tradable.reindex(scores.index).fillna(False) & recent_ok

    weights = te.build_portfolio(
        scores,
        top_k=TOP_K,
        weighting=PORTFOLIO_MODE,
        is_tradable=is_tradable,
    )
    return weights, scores, selected_folds, selected_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(te.DEFAULT_PRICES_PATH))
    parser.add_argument("--index", default=str(te.DEFAULT_INDEX_PATH))
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--rebalance-step", type=int, default=DEFAULT_REBALANCE_STEP)
    parser.add_argument("--max-dates", type=int, default=0)
    parser.add_argument(
        "--out-prefix",
        default=str(LOG_DIR / "self_test_multihorizon_lgb"),
    )
    args = parser.parse_args()

    # Keep repeated self-test CV artifacts out of the production model folder.
    cv_tmp = LOG_DIR / "self_test_multihorizon_cv_tmp"
    model_tmp = LOG_DIR / "self_test_multihorizon_models"
    cv_tmp.mkdir(exist_ok=True)
    model_tmp.mkdir(exist_ok=True)
    te.LOG_DIR = cv_tmp
    te.MODEL_DIR = model_tmp

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_prices = pd.read_parquet(args.index)
    index_prices["date"] = pd.to_datetime(index_prices["date"])

    print("\n>> Building slim features")
    panel = build_features(prices)
    all_dates = np.sort(panel["date"].unique())
    max_horizon = max(TARGET_HORIZONS)
    last_asof = pd.Timestamp(all_dates[len(all_dates) - 1 - max_horizon])

    test_start = _parse_date(args.test_start)
    test_end = _parse_date(args.test_end) or last_asof
    test_end = min(test_end, last_asof)
    asof_dates = [
        pd.Timestamp(d)
        for d in all_dates
        if pd.Timestamp(d) >= test_start and pd.Timestamp(d) <= test_end
    ][:: max(1, args.rebalance_step)]
    if args.max_dates and args.max_dates > 0:
        asof_dates = asof_dates[-args.max_dates :]
    if not asof_dates:
        raise RuntimeError("No as-of dates selected for self-test.")

    print("\n>> Exact matching self-test configuration")
    print(f"   features     : {len(FEATURE_COLUMNS)} slim OHLCV columns")
    print(f"   horizons     : {TARGET_HORIZON_WEIGHTS}")
    print(f"   model        : LGB only | walk-forward best2")
    print(f"   portfolio    : top{TOP_K} {PORTFOLIO_MODE}, tradable{RECENT_TRADABLE_DAYS}")
    print(f"   test as-of   : {asof_dates[0].date()} -> {asof_dates[-1].date()}"
          f"  ({len(asof_dates)} dates, step={args.rebalance_step})")

    close_pivot = prices.pivot(index="date", columns="stock_code", values="close").sort_index()
    open_pivot = prices.pivot(index="date", columns="stock_code", values="open").sort_index()
    idx_close = index_prices.set_index("date")["close"].sort_index()
    idx_open = index_prices.set_index("date")["open"].sort_index()

    records = []
    skipped = []
    for i, as_of in enumerate(asof_dates, start=1):
        print(f"\n>> As-of {i}/{len(asof_dates)}: {as_of.date()}")
        try:
            weights, scores, selected_folds, selected_weights = _fit_predict_for_asof(
                panel, prices, all_dates, as_of
            )
        except RuntimeError as exc:
            skipped.append({"as_of": as_of.date().isoformat(), "reason": str(exc)})
            print(f"   [skip] {exc}")
            continue
        pred_panel = panel[panel["date"] == as_of].copy()

        ic_by_h = {}
        pred_panel_indexed = pred_panel.set_index(
            pred_panel["stock_code"].astype(str).str.zfill(6)
        )
        for horizon in TARGET_HORIZONS:
            target_col = target_column_for_horizon(horizon)
            ic_by_h[f"rank_ic_target_{horizon}d"] = _rank_ic(
                pred_panel_indexed[target_col], scores
            )

        for horizon in TARGET_HORIZONS:
            res = _window_return(
                weights, close_pivot, open_pivot, idx_close, idx_open, as_of, int(horizon)
            )
            if res is None:
                continue
            records.append({
                "as_of": as_of.date().isoformat(),
                "horizon_days": int(horizon),
                "n_stocks": int((weights > 0).sum()),
                "max_weight": float(weights.max()),
                "selected_folds": ";".join(f"{k}:{v}" for k, v in selected_folds.items()),
                "fold_weights": ";".join(f"{k}:{v}" for k, v in selected_weights.items()),
                **ic_by_h,
                **res,
            })

    detail = pd.DataFrame(records)
    if detail.empty:
        raise RuntimeError("No self-test records produced.")

    def _hit_rate(x: pd.Series) -> float:
        return float((x > 0).mean())

    summary = (
        detail.groupby("horizon_days")
        .agg(
            n_windows=("excess_return", "size"),
            mean_excess=("excess_return", "mean"),
            median_excess=("excess_return", "median"),
            worst_excess=("excess_return", "min"),
            best_excess=("excess_return", "max"),
            std_excess=("excess_return", "std"),
            hit_rate=("excess_return", _hit_rate),
            mean_portfolio_return=("portfolio_return", "mean"),
            mean_benchmark_return=("benchmark_return", "mean"),
            n_stocks=("n_stocks", "first"),
            max_weight=("max_weight", "max"),
        )
        .reset_index()
    )
    summary["compounded_excess"] = (
        detail.groupby("horizon_days")["excess_return"]
        .apply(lambda x: float(np.prod(1.0 + x) - 1.0))
        .values
    )
    for horizon in TARGET_HORIZONS:
        col = f"rank_ic_target_{horizon}d"
        summary[f"mean_{col}"] = (
            detail.groupby("horizon_days")[col].mean().reindex(summary["horizon_days"]).values
        )

    out_prefix = Path(args.out_prefix)
    detail_path = out_prefix.with_name(out_prefix.name + "_detail.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    json_path = out_prefix.with_name(out_prefix.name + "_summary.json")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    payload = {
        "configuration": {
            "model": "LightGBM multi-horizon ensemble",
            "feature_set": "factor_experiments_slim_asof20260430/features.py",
            "n_features": len(FEATURE_COLUMNS),
            "target_horizon_weights": TARGET_HORIZON_WEIGHTS,
            "recency_halflife_days": RECENCY_HALFLIFE_DAYS,
            "fold_mode": FOLD_MODE,
            "portfolio": {
                "top_k": TOP_K,
                "weighting": PORTFOLIO_MODE,
                "recent_tradable_days": RECENT_TRADABLE_DAYS,
            },
            "test_asof_start": asof_dates[0].date().isoformat(),
            "test_asof_end": asof_dates[-1].date().isoformat(),
            "rebalance_step": args.rebalance_step,
        },
        "summary": summary.to_dict(orient="records"),
        "skipped": skipped,
        "detail_path": str(detail_path),
        "summary_path": str(summary_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n>> Wrote {detail_path}")
    print(f">> Wrote {summary_path}")
    print(f">> Wrote {json_path}")
    print("\nSelf-test summary:")
    print(summary[[
        "horizon_days", "n_windows", "mean_excess", "worst_excess",
        "hit_rate", "compounded_excess", "n_stocks", "max_weight",
    ]].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
