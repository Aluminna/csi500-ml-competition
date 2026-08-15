from __future__ import annotations

"""No-leakage self-test for the orig66/xgb34 top34 two-model ensemble.

For every historical as-of date, this script loads the component portfolios that
were generated using only information available at that as-of date, applies the
same fixed 66/34 top34-equal recipe, and evaluates the subsequent five-trading-
day out-of-sample window. It also reports rank IC for both the pre-topk blended
portfolio score and the final top34 selection signal.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generate_submission import (
    BASELINE_WEIGHT,
    CURRENT_BASELINE,
    CURRENT_ORIGINAL,
    DEFAULT_OUT,
    FINAL_AS_OF,
    ORIGINAL_WEIGHT,
    ROOT,
    TOP_K,
    blend_portfolios,
    generate,
    load_weights,
    topk_equal,
    write_submission,
)
from score_submission import score_window


DATA_DIR = ROOT / "data"
SUB_DIR = ROOT / "submissions"
LOG_DIR = ROOT / "logs"

ROLLING_ORIGINAL_TEMPLATE = SUB_DIR / "rolling_3d_compare" / "pure_lgb_k40_sqrt_{asof}.csv"
ROLLING_BASELINE_TEMPLATE = SUB_DIR / "rolling_3d_baseline_xgb_phase2" / "baseline_xgb_{asof}.csv"

WINDOWS_5D = [
    ("20260313", "20260316", "20260320", "validation"),
    ("20260320", "20260323", "20260327", "validation"),
    ("20260327", "20260330", "20260403", "validation"),
    ("20260403", "20260407", "20260413", "validation"),
    ("20260410", "20260413", "20260417", "test"),
    ("20260417", "20260420", "20260424", "test"),
    ("20260424", "20260427", "20260506", "test"),
]


def fmt_pct(value: float) -> str:
    return f"{value:+.3%}"


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.astype(float)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(-drawdown.min())


def load_market_data() -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")
    index_df["date"] = pd.to_datetime(index_df["date"])
    constituents = pd.read_csv(DATA_DIR / "constituents.csv", dtype={"stock_code": str})
    universe = set(constituents["stock_code"].astype(str).str.zfill(6))
    return prices, index_df, universe


def stock_window_returns(
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    universe: set[str],
) -> pd.Series:
    before = prices[prices["date"] < start].sort_values("date")
    entry = before.groupby("stock_code")["close"].last()
    in_window = prices[(prices["date"] >= start) & (prices["date"] <= end)].sort_values("date")
    exit_ = in_window.groupby("stock_code")["close"].last()

    missing_entry = exit_.index.difference(entry.index)
    if len(missing_entry):
        first_open = in_window.groupby("stock_code")["open"].first()
        entry = pd.concat([entry, first_open.reindex(missing_entry)]).dropna()

    aligned = pd.concat([entry.rename("entry"), exit_.rename("exit")], axis=1).dropna()
    returns = (aligned["exit"] / aligned["entry"] - 1.0).astype(float)
    return returns[returns.index.isin(universe)]


def rank_ic(scores: pd.Series, future_returns: pd.Series) -> float:
    frame = pd.concat(
        [scores.rename("score"), future_returns.rename("future_return")],
        axis=1,
    ).dropna()
    if len(frame) < 5 or frame["score"].nunique() < 2 or frame["future_return"].nunique() < 2:
        return float("nan")
    return float(frame["score"].corr(frame["future_return"], method="spearman"))


def component_path(template: Path, asof: str) -> Path:
    return Path(str(template).format(asof=asof))


def candidate_for_asof(asof: str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    original = load_weights(component_path(ROLLING_ORIGINAL_TEMPLATE, asof))
    baseline = load_weights(component_path(ROLLING_BASELINE_TEMPLATE, asof))
    blended = blend_portfolios(original, baseline)
    candidate = topk_equal(blended, TOP_K)
    baseline_same_k = topk_equal(baseline, TOP_K)
    return candidate, blended, original, baseline, baseline_same_k


def score_weights(
    weights: pd.Series,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    start: str,
    end: str,
) -> dict:
    return score_window(
        weights,
        prices,
        index_df,
        pd.Timestamp(start),
        pd.Timestamp(end),
    )


def evaluate_window(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    universe: set[str],
    asof: str,
    start: str,
    end: str,
    split: str,
) -> dict:
    candidate, blended, original, baseline, baseline_same_k = candidate_for_asof(asof)
    candidate_score = score_weights(candidate, prices, index_df, start, end)
    baseline_score = score_weights(baseline, prices, index_df, start, end)
    baseline_same_k_score = score_weights(baseline_same_k, prices, index_df, start, end)

    future_returns = stock_window_returns(prices, pd.Timestamp(start), pd.Timestamp(end), universe)
    universe_index = future_returns.index
    active_index = universe_index.intersection(blended[blended > 0].index)

    return {
        "split": split,
        "as_of": pd.Timestamp(asof).date().isoformat(),
        "eval_start": pd.Timestamp(start).date().isoformat(),
        "eval_end": pd.Timestamp(end).date().isoformat(),
        "candidate_portfolio_return": candidate_score["portfolio_return"],
        "benchmark_return": candidate_score["benchmark_return"],
        "candidate_excess_return": candidate_score["excess_return"],
        "baseline_xgb_excess_return": baseline_score["excess_return"],
        "baseline_same_k_equal_excess_return": baseline_same_k_score["excess_return"],
        "excess_vs_baseline_xgb": candidate_score["excess_return"] - baseline_score["excess_return"],
        "excess_vs_baseline_same_k_equal": (
            candidate_score["excess_return"] - baseline_same_k_score["excess_return"]
        ),
        "n_stocks": int(len(candidate)),
        "max_weight": float(candidate.max()),
        "blend_raw_rank_ic_full": rank_ic(blended.reindex(universe_index).fillna(0.0), future_returns),
        "candidate_final_rank_ic_full": rank_ic(
            candidate.reindex(universe_index).fillna(0.0),
            future_returns,
        ),
        "baseline_raw_rank_ic_full": rank_ic(baseline.reindex(universe_index).fillna(0.0), future_returns),
        "baseline_same_k_rank_ic_full": rank_ic(
            baseline_same_k.reindex(universe_index).fillna(0.0),
            future_returns,
        ),
        "original_raw_rank_ic_full": rank_ic(original.reindex(universe_index).fillna(0.0), future_returns),
        "blend_raw_rank_ic_active": rank_ic(blended.reindex(active_index), future_returns.reindex(active_index)),
        "component_files": json.dumps(
            {
                "original": str(component_path(ROLLING_ORIGINAL_TEMPLATE, asof).relative_to(ROOT)),
                "baseline_xgb": str(component_path(ROLLING_BASELINE_TEMPLATE, asof).relative_to(ROOT)),
            },
            sort_keys=True,
        ),
    }


def summarize_returns(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ["validation", "test", "all"]:
        group = detail if split == "all" else detail[detail["split"] == split]
        if group.empty:
            continue
        excess = group["candidate_excess_return"].astype(float)
        baseline_excess = group["baseline_xgb_excess_return"].astype(float)
        baseline_same_k_excess = group["baseline_same_k_equal_excess_return"].astype(float)
        diff = group["excess_vs_baseline_xgb"].astype(float)
        diff_same_k = group["excess_vs_baseline_same_k_equal"].astype(float)
        cumulative_excess = float(np.prod(1.0 + excess) - 1.0)
        cumulative_baseline_xgb_excess = float(np.prod(1.0 + baseline_excess) - 1.0)
        cumulative_baseline_same_k_excess = float(np.prod(1.0 + baseline_same_k_excess) - 1.0)
        rows.append(
            {
                "split": split,
                "n_windows": int(len(group)),
                "mean_portfolio_return": float(group["candidate_portfolio_return"].mean()),
                "mean_benchmark_return": float(group["benchmark_return"].mean()),
                "mean_excess_return": float(excess.mean()),
                "median_excess_return": float(excess.median()),
                "worst_excess_return": float(excess.min()),
                "best_excess_return": float(excess.max()),
                "hit_rate": float((excess > 0).mean()),
                "cumulative_excess_return": cumulative_excess,
                "cumulative_baseline_xgb_excess_return": cumulative_baseline_xgb_excess,
                "cumulative_baseline_same_k_equal_excess_return": cumulative_baseline_same_k_excess,
                "cumulative_excess_vs_baseline_xgb": cumulative_excess - cumulative_baseline_xgb_excess,
                "cumulative_excess_vs_baseline_same_k_equal": (
                    cumulative_excess - cumulative_baseline_same_k_excess
                ),
                "max_drawdown": max_drawdown(excess),
                "mean_baseline_xgb_excess_return": float(baseline_excess.mean()),
                "mean_baseline_same_k_equal_excess_return": float(baseline_same_k_excess.mean()),
                "mean_excess_vs_baseline_xgb": float(diff.mean()),
                "mean_excess_vs_baseline_same_k_equal": float(diff_same_k.mean()),
                "hit_rate_vs_baseline_xgb": float((diff > 0).mean()),
                "hit_rate_vs_baseline_same_k_equal": float((diff_same_k > 0).mean()),
                "max_weight": float(group["max_weight"].max()),
            }
        )
    return pd.DataFrame(rows)


def summarize_ic(detail: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "blend_raw_rank_ic_full",
        "candidate_final_rank_ic_full",
        "baseline_raw_rank_ic_full",
        "baseline_same_k_rank_ic_full",
        "original_raw_rank_ic_full",
        "blend_raw_rank_ic_active",
    ]
    rows = []
    for split in ["validation", "test", "all"]:
        group = detail if split == "all" else detail[detail["split"] == split]
        if group.empty:
            continue
        for metric in metrics:
            values = group[metric].dropna().astype(float)
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "n_windows": int(len(values)),
                    "mean_ic": float(values.mean()),
                    "std_ic": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                    "icir": (
                        float(values.mean() / values.std(ddof=1))
                        if len(values) > 1 and values.std(ddof=1) != 0
                        else float("nan")
                    ),
                    "min_ic": float(values.min()),
                    "max_ic": float(values.max()),
                    "positive_rate": float((values > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_baseline_comparison(return_summary: pd.DataFrame, ic_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ["validation", "test", "all"]:
        ret_rows = return_summary[return_summary["split"] == split]
        if ret_rows.empty:
            continue
        ret = ret_rows.iloc[0]
        ic_rows = ic_summary[ic_summary["split"] == split]

        def mean_ic(metric: str) -> float:
            values = ic_rows.loc[ic_rows["metric"] == metric, "mean_ic"]
            return float(values.iloc[0]) if len(values) else float("nan")

        final_ic = mean_ic("candidate_final_rank_ic_full")
        baseline_same_k_ic = mean_ic("baseline_same_k_rank_ic_full")
        blend_raw_ic = mean_ic("blend_raw_rank_ic_full")
        baseline_raw_ic = mean_ic("baseline_raw_rank_ic_full")
        rows.append(
            {
                "split": split,
                "n_windows": int(ret["n_windows"]),
                "mean_excess_advantage_vs_baseline_xgb": float(ret["mean_excess_vs_baseline_xgb"]),
                "cumulative_excess_advantage_vs_baseline_xgb": float(
                    ret["cumulative_excess_vs_baseline_xgb"]
                ),
                "hit_rate_vs_baseline_xgb": float(ret["hit_rate_vs_baseline_xgb"]),
                "final_selection_ic_advantage_vs_baseline_same_k": final_ic - baseline_same_k_ic,
                "raw_blend_ic_advantage_vs_baseline_raw": blend_raw_ic - baseline_raw_ic,
            }
        )
    return pd.DataFrame(rows)


def check_current_repro(out_path: Path) -> dict:
    generated = generate()
    submitted = load_weights(out_path)
    common = generated.index.union(submitted.index)
    diff = generated.reindex(common).fillna(0.0) - submitted.reindex(common).fillna(0.0)
    return {
        "same_set": set(generated.index) == set(submitted.index),
        "same_order": list(generated.index) == list(submitted.index),
        "max_abs_weight_diff": float(diff.abs().max()),
        "n_stocks": int(len(generated)),
        "sum_weight": float(generated.sum()),
        "max_weight": float(generated.max()),
    }


def print_table(df: pd.DataFrame, percent_cols: list[str]) -> None:
    if df.empty:
        print("(empty)")
        return
    formatters = {col: fmt_pct for col in percent_cols if col in df.columns}
    print(df.to_string(index=False, formatters=formatters))


def print_ic_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("(empty)")
        return
    formatters = {
        "mean_ic": lambda x: f"{x:+.4f}",
        "std_ic": lambda x: f"{x:+.4f}",
        "icir": lambda x: f"{x:+.3f}",
        "min_ic": lambda x: f"{x:+.4f}",
        "max_ic": lambda x: f"{x:+.4f}",
        "positive_rate": lambda x: f"{x:.1%}",
    }
    print(df.to_string(index=False, formatters=formatters))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    prices, index_df, universe = load_market_data()

    write_submission(generate(), out_path)
    rows = [
        evaluate_window(prices, index_df, universe, asof, start, end, split)
        for asof, start, end, split in WINDOWS_5D
    ]
    detail = pd.DataFrame(rows)
    return_summary = summarize_returns(detail)
    ic_summary = summarize_ic(detail)
    baseline_comparison = summarize_baseline_comparison(return_summary, ic_summary)
    repro = check_current_repro(out_path)

    detail_path = LOG_DIR / "self_test_detail.csv"
    return_summary_path = LOG_DIR / "self_test_return_summary.csv"
    ic_summary_path = LOG_DIR / "self_test_ic_summary.csv"
    baseline_comparison_path = LOG_DIR / "self_test_baseline_comparison.csv"
    json_path = LOG_DIR / "self_test_summary.json"
    detail.to_csv(detail_path, index=False)
    return_summary.to_csv(return_summary_path, index=False)
    ic_summary.to_csv(ic_summary_path, index=False)
    baseline_comparison.to_csv(baseline_comparison_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "model": "66% original submission-1 model + 34% official-style XGB baseline",
                "final_as_of": FINAL_AS_OF,
                "recipe": {
                    "original_weight": ORIGINAL_WEIGHT,
                    "baseline_weight": BASELINE_WEIGHT,
                    "top_k": TOP_K,
                    "reweight": "equal",
                },
                "current_sources": {
                    "original": str(CURRENT_ORIGINAL.relative_to(ROOT)),
                    "baseline_xgb": str(CURRENT_BASELINE.relative_to(ROOT)),
                },
                "rolling_sources": {
                    "original_template": str(ROLLING_ORIGINAL_TEMPLATE.relative_to(ROOT)),
                    "baseline_template": str(ROLLING_BASELINE_TEMPLATE.relative_to(ROOT)),
                },
                "reproducibility_check": repro,
                "baseline_comparison": baseline_comparison.to_dict(orient="records"),
                "notes": [
                    "For every self-test row, as_of is strictly before eval_start.",
                    "The same fixed 66/34 top34-equal recipe is applied at every as-of date.",
                    "Rank IC is Spearman correlation between model-generated portfolio score and future 5-day stock return.",
                    "The final IC uses the top34 equal-weight selection signal, coded as zero for unselected stocks.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n>> Phase 2 orig66/xgb34 top34 two-model ensemble self-test")
    print(f"   submission : {out_path.relative_to(ROOT)}")
    print(f"   recipe     : original={ORIGINAL_WEIGHT:.3f}, baseline={BASELINE_WEIGHT:.3f}")
    print(f"   portfolio  : top{TOP_K} equal")
    print(f"   detail     : {detail_path.relative_to(ROOT)}")
    print(f"   returns    : {return_summary_path.relative_to(ROOT)}")
    print(f"   IC summary : {ic_summary_path.relative_to(ROOT)}")
    print(f"   baseline   : {baseline_comparison_path.relative_to(ROOT)}")
    print(f"   json       : {json_path.relative_to(ROOT)}")
    print(
        "\nCurrent submission reproducibility\n"
        f"   same_set={repro['same_set']} same_order={repro['same_order']} "
        f"max_abs_weight_diff={repro['max_abs_weight_diff']:.12g} "
        f"n={repro['n_stocks']} sum={repro['sum_weight']:.12f} "
        f"max={repro['max_weight']:.3%}"
    )

    print("\nReturn Summary")
    print_table(
        return_summary,
        [
            "mean_portfolio_return",
            "mean_benchmark_return",
            "mean_excess_return",
            "median_excess_return",
            "worst_excess_return",
            "best_excess_return",
            "hit_rate",
            "cumulative_excess_return",
            "cumulative_baseline_xgb_excess_return",
            "cumulative_baseline_same_k_equal_excess_return",
            "cumulative_excess_vs_baseline_xgb",
            "cumulative_excess_vs_baseline_same_k_equal",
            "max_drawdown",
            "mean_baseline_xgb_excess_return",
            "mean_baseline_same_k_equal_excess_return",
            "mean_excess_vs_baseline_xgb",
            "mean_excess_vs_baseline_same_k_equal",
            "hit_rate_vs_baseline_xgb",
            "hit_rate_vs_baseline_same_k_equal",
            "max_weight",
        ],
    )

    print("\nIC Summary")
    print_ic_table(ic_summary)

    print("\nBaseline Comparison Highlights")
    print_table(
        baseline_comparison,
        [
            "mean_excess_advantage_vs_baseline_xgb",
            "cumulative_excess_advantage_vs_baseline_xgb",
            "hit_rate_vs_baseline_xgb",
            "final_selection_ic_advantage_vs_baseline_same_k",
            "raw_blend_ic_advantage_vs_baseline_raw",
        ],
    )

    print("\nPer-window Detail")
    print_table(
        detail[
            [
                "split",
                "as_of",
                "eval_start",
                "eval_end",
                "candidate_excess_return",
                "baseline_xgb_excess_return",
                "baseline_same_k_equal_excess_return",
                "excess_vs_baseline_xgb",
                "candidate_final_rank_ic_full",
                "baseline_same_k_rank_ic_full",
                "blend_raw_rank_ic_full",
                "baseline_raw_rank_ic_full",
            ]
        ],
        [
            "candidate_excess_return",
            "baseline_xgb_excess_return",
            "baseline_same_k_equal_excess_return",
            "excess_vs_baseline_xgb",
            "candidate_final_rank_ic_full",
            "baseline_same_k_rank_ic_full",
            "blend_raw_rank_ic_full",
            "baseline_raw_rank_ic_full",
        ],
    )


if __name__ == "__main__":
    main()
