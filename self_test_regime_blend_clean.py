"""
self_test_regime_blend_clean.py

Matching no-lookahead self-test for submissions/regime_blend_clean_repro.csv.

This is intentionally a new file, so the original self_test.py remains
unchanged. The tested production structure is:

  70% walk-forward IC-weighted regime factor model
  30% slim LightGBM multi-horizon support portfolio

For each historical as-of date, the script:

  1. Recomputes regime IC weights using only dates available before as-of,
     with the same horizon-specific embargo as train_regime_blend.py.
  2. Retrains the slim LGB support pipeline as of that historical date.
  3. Blends support rank scores with regime scores using the production
     30% / 70% weights.
  4. Builds a top-k portfolio using the requested deterministic weighting rule
     and scores the next 3/5/10 trading-day windows with
     score_submission.score_window.

Outputs:

  logs/self_test_regime_blend_clean_detail.csv
  logs/self_test_regime_blend_clean_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).parent
FACTOR_DIR = ROOT / "factor_experiments_slim_asof20260430"
sys.path.insert(0, str(FACTOR_DIR))

import self_test_multihorizon_lgb as slim_self_test  # noqa: E402

from score_submission import score_window  # noqa: E402
from train_regime_blend import (  # noqa: E402
    DATA_DIR as REGIME_DATA_DIR,
    EXTRA_DIR,
    HORIZON_EMBARGO,
    HORIZON_WEIGHTS,
    MAX_FEATURE_IC,
    RECENCY_HALFLIFE_REGIME,
    REGIME_WINDOW_TRADING_DAYS,
    _add_industry_rank_features,
    _add_market_relative_features,
    build_features,
    compute_regime_ic,
    get_active_features,
    portfolio_from_scores,
    regime_scores_from_ic,
    source_score,
    target_column_for_horizon,
)


LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

DEFAULT_SUPPORT_WEIGHT = 0.30
DEFAULT_TOP_K = 50
DEFAULT_VAL_START = "2026-03-11"
DEFAULT_TEST_START = "2026-04-01"
DEFAULT_REBALANCE_STEP = 5


def _parse_date(raw: str | None) -> pd.Timestamp | None:
    return None if raw is None else pd.Timestamp(str(raw))


def _date_str(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def _parse_horizons(raw: str) -> list[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("at least one horizon is required")
    return out


def _rank_ic(pred_df: pd.DataFrame, scores: pd.Series, horizon: int) -> float:
    target = target_column_for_horizon(horizon)
    if target not in pred_df.columns:
        return float("nan")
    target_values = pred_df.set_index("stock_code")[target]
    common = scores.index.intersection(target_values.dropna().index)
    if len(common) < 20:
        return float("nan")
    rho, _ = spearmanr(scores.loc[common], target_values.loc[common])
    return float(rho) if not np.isnan(rho) else float("nan")


def _normalise_01(values: pd.Series) -> pd.Series:
    values = values.astype(float).copy()
    mn, mx = values.min(), values.max()
    if mx > mn:
        values = (values - mn) / (mx - mn)
    return values


def _max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(-drawdown.min())


def _round_or_none(value: float, digits: int = 6):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, digits)


def _icir(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return float("nan")
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        return float("nan")
    return float(values.mean() / std)


def _summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    ic = df["rank_ic"].dropna()
    excess = df["excess_return"].dropna()
    return {
        "n_windows": int(len(df)),
        "rank_ic": _round_or_none(ic.mean()) if not ic.empty else None,
        "icir": _round_or_none(_icir(ic)) if not ic.empty else None,
        "ic_hit_rate": _round_or_none((ic > 0).mean(), 4) if not ic.empty else None,
        "mean_excess_return": _round_or_none(excess.mean()) if not excess.empty else None,
        "median_excess_return": _round_or_none(excess.median()) if not excess.empty else None,
        "worst_excess_return": _round_or_none(excess.min()) if not excess.empty else None,
        "max_drawdown": _round_or_none(_max_drawdown(excess)) if not excess.empty else None,
        "cumulative_excess_return": (
            _round_or_none(np.prod(1.0 + excess) - 1.0) if not excess.empty else None
        ),
        "portfolio_hit_rate": (
            _round_or_none((excess > 0).mean(), 4) if not excess.empty else None
        ),
    }


def _fmt(value, spec: str, na: str = "n/a") -> str:
    if value is None:
        return na
    value = float(value)
    if not np.isfinite(value):
        return na
    return format(value, spec)


def load_regime_panel(data_dir: Path, industry_file: Path | None):
    prices = pd.read_parquet(data_dir / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)

    index_prices = pd.read_parquet(data_dir / "index.parquet")
    index_prices["date"] = pd.to_datetime(index_prices["date"])

    panel = build_features(prices)
    panel = _add_market_relative_features(panel, index_prices)

    used_industry = industry_file
    if used_industry is None:
        for candidate in [EXTRA_DIR / "industry.csv", data_dir / "industry.csv"]:
            if candidate.exists():
                used_industry = candidate
                break
    if used_industry is not None and used_industry.exists():
        industry = pd.read_csv(used_industry, dtype={"stock_code": str})
        industry["stock_code"] = industry["stock_code"].astype(str).str.zfill(6)
        panel = _add_industry_rank_features(panel, industry)

    active_features = get_active_features(panel)
    panel[active_features] = panel[active_features].replace([np.inf, -np.inf], np.nan)
    return prices, index_prices, panel, active_features, used_industry


def compute_regime_scores_for_asof(
    panel: pd.DataFrame,
    active_features: list[str],
    as_of: pd.Timestamp,
) -> tuple[pd.Series, pd.Series]:
    total_weight = float(sum(HORIZON_WEIGHTS.values()))
    blended_ic: dict[str, float] = {}

    for horizon, raw_weight in HORIZON_WEIGHTS.items():
        ic_h = compute_regime_ic(
            panel,
            active_features,
            as_of=as_of,
            horizon=int(horizon),
            window_trading_days=REGIME_WINDOW_TRADING_DAYS,
            halflife_days=RECENCY_HALFLIFE_REGIME,
            embargo_calendar_days=HORIZON_EMBARGO.get(int(horizon), 6),
        )
        frac = raw_weight / total_weight
        for feat, value in ic_h.items():
            blended_ic[feat] = blended_ic.get(feat, 0.0) + frac * float(value)

    ic_series = pd.Series(blended_ic).dropna()
    pred_df = panel[panel["date"] == as_of].copy()
    if "is_tradable" in pred_df.columns:
        pred_df = pred_df[pred_df["is_tradable"]]
    raw_scores = regime_scores_from_ic(
        pred_df,
        active_features,
        ic_series.clip(lower=-MAX_FEATURE_IC, upper=MAX_FEATURE_IC),
        positive_only=True,
    )
    return _normalise_01(raw_scores), ic_series


def blend_scores(
    support_weights: pd.Series,
    regime_scores: pd.Series,
    support_weight: float,
) -> pd.Series:
    regime_weight = 1.0 - support_weight
    support_scores = source_score(support_weights[support_weights > 0])
    combined = sorted(set(support_scores.index) | set(regime_scores.index))
    support_aligned = _normalise_01(support_scores.reindex(combined, fill_value=0.0))
    regime_aligned = _normalise_01(regime_scores.reindex(combined, fill_value=0.0))
    return support_weight * support_aligned + regime_weight * regime_aligned


def selected_asof_dates(
    all_dates: np.ndarray,
    val_start: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    step: int,
) -> list[tuple[str, pd.Timestamp]]:
    out = []
    for d in all_dates:
        ts = pd.Timestamp(d)
        if ts < val_start or ts > test_end:
            continue
        split = "test" if ts >= test_start else "validation"
        out.append((split, ts))
    return out[:: max(1, step)]


def evaluate_asof(
    split: str,
    as_of: pd.Timestamp,
    horizons: list[int],
    support_panel: pd.DataFrame,
    support_prices: pd.DataFrame,
    support_dates: np.ndarray,
    regime_panel: pd.DataFrame,
    regime_prices: pd.DataFrame,
    regime_index: pd.DataFrame,
    regime_dates: pd.DatetimeIndex,
    active_features: list[str],
    support_weight: float,
    top_k: int,
    portfolio_mode: str,
) -> list[dict]:
    support_weights, _support_raw_scores, selected_folds, selected_weights = (
        slim_self_test._fit_predict_for_asof(
            support_panel,
            support_prices,
            support_dates,
            as_of,
        )
    )
    regime_scores, ic_series = compute_regime_scores_for_asof(
        regime_panel,
        active_features,
        as_of,
    )
    final_scores = blend_scores(support_weights, regime_scores, support_weight)
    weights = portfolio_from_scores(final_scores, top_k=top_k, mode=portfolio_mode)

    pred_df = regime_panel[regime_panel["date"] == as_of].copy()
    asof_idx = int(np.searchsorted(regime_dates.values, np.datetime64(as_of)))
    records = []
    for horizon in horizons:
        if asof_idx + horizon >= len(regime_dates):
            continue
        start = regime_dates[asof_idx + 1]
        end = regime_dates[asof_idx + horizon]
        scored = score_window(weights, regime_prices, regime_index, start, end)
        records.append({
            "split": split,
            "as_of": _date_str(as_of),
            "start": _date_str(start),
            "end": _date_str(end),
            "horizon_days": int(horizon),
            "rank_ic": _rank_ic(pred_df, final_scores, horizon),
            "portfolio_return": scored["portfolio_return"],
            "benchmark_return": scored["benchmark_return"],
            "excess_return": scored["excess_return"],
            "n_stocks": int((weights > 0).sum()),
            "support_overlap": int(len(set(weights.index) & set(support_weights.index))),
            "n_positive_ic_features": int((ic_series > 0).sum()),
            "top_features": ",".join(ic_series.sort_values(ascending=False).head(8).index),
            "support_selected_folds": json.dumps(selected_folds, sort_keys=True),
            "support_selected_weights": json.dumps(selected_weights, sort_keys=True),
        })
    return records


def build_summary(detail: pd.DataFrame, args, data_info: dict) -> dict:
    results = {}
    for split, split_df in detail.groupby("split"):
        split_result = {}
        for horizon, hdf in split_df.groupby("horizon_days"):
            split_result[f"{int(horizon)}d"] = _summarize(hdf)
        split_result["all_horizons"] = _summarize(split_df)
        results[split] = split_result

    return {
        "model": {
            "type": "clean regime blend",
            "regime_weight": round(1.0 - float(args.support_weight), 6),
            "support_weight": round(float(args.support_weight), 6),
            "regime_component": "walk-forward IC-weighted linear factor model",
            "support_component": "slim LightGBM multi-horizon pipeline",
            "production_command": (
                "python train_regime_blend.py --out submissions/regime_blend_clean_repro.csv "
                "--top-k 50 --as-of 2026-04-30"
            ),
            "support_generation_command": (
                "python factor_experiments_slim_asof20260430/generate_variants.py "
                "--prices data/prices.parquet --as-of 2026-04-30 "
                "--out-dir factor_experiments_slim_asof20260430/submissions/slim_ohlcv_top40 "
                "--top-ks 40 --fold-modes best2 --portfolio-modes sqrt_rank "
                "--recent-days 1 --target-horizons 3,5,10 "
                "--horizon-weights 0.5,0.3,0.2 --recency-halflife-days 60"
            ),
            "manual_exclusions": [],
            "portfolio_mode": args.portfolio_mode,
        },
        "split": {
            "validation_start": args.val_start,
            "test_start": args.test_start,
            "test_end": args.test_end,
            "rebalance_step_trading_days": int(args.rebalance_step),
            "horizons": [int(x) for x in _parse_horizons(args.horizons)],
        },
        "data": data_info,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-prices", default=str(ROOT / "data" / "prices.parquet"))
    parser.add_argument("--support-index", default=str(ROOT / "data" / "index.parquet"))
    parser.add_argument("--regime-data-dir", default=str(REGIME_DATA_DIR))
    parser.add_argument("--industry-file", default=None)
    parser.add_argument("--support-weight", type=float, default=DEFAULT_SUPPORT_WEIGHT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--portfolio-mode",
        choices=["sqrt_rank", "equal"],
        default="sqrt_rank",
    )
    parser.add_argument("--horizons", default="3,5,10")
    parser.add_argument("--val-start", default=DEFAULT_VAL_START)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--rebalance-step", type=int, default=DEFAULT_REBALANCE_STEP)
    parser.add_argument("--max-dates", type=int, default=0)
    parser.add_argument(
        "--out-tag",
        default=None,
        help="Optional tag inserted into output filenames, e.g. dense_step1.",
    )
    args = parser.parse_args()

    horizons = _parse_horizons(args.horizons)
    max_horizon = max(horizons)

    print("\n>> Loading support data and slim feature panel")
    support_prices = pd.read_parquet(args.support_prices)
    support_prices["date"] = pd.to_datetime(support_prices["date"])
    support_prices["stock_code"] = support_prices["stock_code"].astype(str).str.zfill(6)
    support_index = pd.read_parquet(args.support_index)
    support_index["date"] = pd.to_datetime(support_index["date"])
    support_panel = slim_self_test.build_features(support_prices)
    support_dates = np.sort(support_panel["date"].unique())

    print("\n>> Loading regime data and feature panel")
    regime_data_dir = Path(args.regime_data_dir)
    industry_file = Path(args.industry_file) if args.industry_file else None
    regime_prices, regime_index, regime_panel, active_features, used_industry = load_regime_panel(
        regime_data_dir,
        industry_file,
    )
    regime_dates = pd.DatetimeIndex(sorted(regime_panel["date"].unique()))

    val_start = pd.Timestamp(args.val_start)
    test_start = pd.Timestamp(args.test_start)
    last_asof = min(
        pd.Timestamp(support_dates[-1 - max(slim_self_test.TARGET_HORIZONS)]),
        pd.Timestamp(regime_dates[-1 - max_horizon]),
    )
    test_end = _parse_date(args.test_end) or last_asof
    test_end = min(test_end, last_asof)

    asof_dates = selected_asof_dates(
        support_dates,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
        step=args.rebalance_step,
    )
    if args.max_dates and args.max_dates > 0:
        asof_dates = asof_dates[-args.max_dates:]
    if not asof_dates:
        raise RuntimeError("No as-of dates selected.")

    print("\n>> Clean regime blend self-test configuration")
    print(f"   model        : {1 - args.support_weight:.0%} regime + {args.support_weight:.0%} slim LGB support")
    print(f"   portfolio    : top{args.top_k} {args.portfolio_mode}")
    print(f"   horizons     : {horizons}")
    print(f"   as-of dates  : {asof_dates[0][1].date()} -> {asof_dates[-1][1].date()} ({len(asof_dates)} dates)")
    print(f"   splits       : validation < {test_start.date()}, test >= {test_start.date()}")
    print(f"   industry     : {used_industry}")

    records = []
    skipped = []
    for i, (split, as_of) in enumerate(asof_dates, start=1):
        print(f"\n>> As-of {i}/{len(asof_dates)} [{split}]: {as_of.date()}")
        try:
            records.extend(
                evaluate_asof(
                    split=split,
                    as_of=as_of,
                    horizons=horizons,
                    support_panel=support_panel,
                    support_prices=support_prices,
                    support_dates=support_dates,
                    regime_panel=regime_panel,
                    regime_prices=regime_prices,
                    regime_index=regime_index,
                    regime_dates=regime_dates,
                    active_features=active_features,
                    support_weight=args.support_weight,
                    top_k=args.top_k,
                    portfolio_mode=args.portfolio_mode,
                )
            )
        except Exception as exc:
            skipped.append({"as_of": _date_str(as_of), "split": split, "reason": str(exc)})
            print(f"   [skip] {exc}")

    detail = pd.DataFrame(records)
    if detail.empty:
        raise RuntimeError(f"No self-test records produced. Skipped: {skipped}")

    output_parts = []
    if args.portfolio_mode != "sqrt_rank":
        output_parts.append(args.portfolio_mode)
    if args.out_tag:
        output_parts.append(args.out_tag)
    output_suffix = "" if not output_parts else "_" + "_".join(output_parts)
    detail_path = LOG_DIR / f"self_test_regime_blend_clean{output_suffix}_detail.csv"
    detail.to_csv(detail_path, index=False)

    data_info = {
        "support_prices": str(Path(args.support_prices)),
        "support_index": str(Path(args.support_index)),
        "support_prices_start": _date_str(support_prices["date"].min()),
        "support_prices_end": _date_str(support_prices["date"].max()),
        "regime_data_dir": str(regime_data_dir),
        "regime_prices_start": _date_str(regime_prices["date"].min()),
        "regime_prices_end": _date_str(regime_prices["date"].max()),
        "industry_file": str(used_industry) if used_industry else None,
        "n_stocks": int(regime_prices["stock_code"].nunique()),
        "n_regime_features": int(len(active_features)),
        "skipped": skipped,
    }
    summary = build_summary(detail, args, data_info)
    summary_path = LOG_DIR / f"self_test_regime_blend_clean{output_suffix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n>> Clean regime blend self-test results")
    for split in ["validation", "test"]:
        if split not in summary["results"]:
            continue
        print(f"   {split}:")
        for horizon in horizons:
            metrics = summary["results"][split].get(f"{horizon}d", {})
            if not metrics:
                continue
            print(
                f"     {horizon}d | IC={_fmt(metrics['rank_ic'], '+.4f')} "
                f"ICIR={_fmt(metrics['icir'], '+.2f')} "
                f"mean excess={_fmt(metrics['mean_excess_return'], '+.3%')} "
                f"hit={_fmt(metrics['portfolio_hit_rate'], '.1%')} "
                f"worst={_fmt(metrics['worst_excess_return'], '+.3%')}"
            )

    print("\n>> Results saved")
    print(f"   {detail_path}")
    print(f"   {summary_path}")


if __name__ == "__main__":
    main()
