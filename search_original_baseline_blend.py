from __future__ import annotations

"""Search a simple two-portfolio blend: original submission1 model + baseline XGB.

This is intentionally simple and package-friendly:

* base portfolio A: updated original submission1 LGB-style portfolio
* base portfolio B: official-style baseline XGB portfolio
* final score: fixed weighted average of portfolio weights
* construction: deterministic top-k reweighting

The rolling evaluation uses historical as-of component portfolios when available,
so each scored 5-day window is out-of-sample for the corresponding component
files.  No stock list, blacklist, or hand-picked overlay is used.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

CURRENT_ORIGINAL = ROOT / "submissions" / "original_submission1_model_asof20260508.csv"
CURRENT_BASELINE = ROOT / "submissions" / "phase2_baseline_xgb_asof20260508.csv"

ROLLING_ORIGINAL_TEMPLATE = (
    ROOT / "submissions" / "rolling_3d_compare" / "pure_lgb_k40_sqrt_{asof}.csv"
)
ROLLING_BASELINE_TEMPLATE = (
    ROOT / "submissions" / "rolling_3d_baseline_xgb_phase2" / "baseline_xgb_{asof}.csv"
)

WINDOWS_5D = [
    ("20260313", "20260316", "20260320", "validation"),
    ("20260320", "20260323", "20260327", "validation"),
    ("20260327", "20260330", "20260403", "validation"),
    ("20260403", "20260407", "20260413", "validation"),
    ("20260410", "20260413", "20260417", "test"),
    ("20260417", "20260420", "20260424", "test"),
    ("20260424", "20260427", "20260506", "test"),
]


def _parse_csv(raw: str, cast):
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _fmt_pct(x: float) -> str:
    return f"{x:+.3%}"


def _load_weights(path: Path) -> pd.Series:
    df = pd.read_csv(path, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    weights = df.set_index("stock_code")["weight"].astype(float)
    weights = weights[weights > 0]
    if weights.empty:
        raise ValueError(f"{path} has no positive weights.")
    if weights.index.duplicated().any():
        raise ValueError(f"{path} contains duplicate stock_code values.")
    return weights / weights.sum()


def _blend(original: pd.Series, baseline: pd.Series, original_weight: float) -> pd.Series:
    original_weight = float(original_weight)
    baseline_weight = 1.0 - original_weight
    out = original.mul(original_weight).add(baseline.mul(baseline_weight), fill_value=0.0)
    out = out[out > 0].sort_values(ascending=False)
    if out.empty:
        raise ValueError("Empty blended portfolio.")
    return out / out.sum()


def _reweight(weights: pd.Series, top_k: int, mode: str) -> pd.Series:
    chosen = weights.sort_values(ascending=False).head(int(top_k))
    if mode == "preserve":
        out = chosen / chosen.sum()
    elif mode == "equal":
        out = pd.Series(1.0 / len(chosen), index=chosen.index)
    elif mode == "sqrt_rank":
        ranks = np.arange(len(chosen), 0, -1, dtype=float)
        out = pd.Series(np.sqrt(ranks), index=chosen.index)
        out = out / out.sum()
    elif mode == "rank":
        ranks = np.arange(len(chosen), 0, -1, dtype=float)
        out = pd.Series(ranks, index=chosen.index)
        out = out / out.sum()
    else:
        raise ValueError(f"Unknown reweight mode: {mode}")
    return out / out.sum()


def _write_submission(weights: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "stock_code": weights.index.astype(str).str.zfill(6),
            "weight": weights.astype(float).to_numpy(),
        }
    ).to_csv(path, index=False)


def _window_returns(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    start: str,
    end: str,
) -> tuple[pd.Series, float]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    before = prices[prices["date"] < start_ts].sort_values("date")
    entry = before.groupby("stock_code")["close"].last()

    in_window = prices[(prices["date"] >= start_ts) & (prices["date"] <= end_ts)].sort_values("date")
    exit_ = in_window.groupby("stock_code")["close"].last()

    missing_entry = exit_.index.difference(entry.index)
    if len(missing_entry):
        first_open = in_window.groupby("stock_code")["open"].first()
        entry = pd.concat([entry, first_open.reindex(missing_entry)]).dropna()

    aligned = pd.concat([entry.rename("entry"), exit_.rename("exit")], axis=1).dropna()
    stock_returns = aligned["exit"] / aligned["entry"] - 1.0

    idx_window = index_df[(index_df["date"] >= start_ts) & (index_df["date"] <= end_ts)].sort_values("date")
    if idx_window.empty:
        raise RuntimeError(f"No CSI500 index data in [{start}, {end}]")
    idx_before = index_df[index_df["date"] < start_ts]
    idx_entry = idx_before["close"].iloc[-1] if not idx_before.empty else idx_window["open"].iloc[0]
    idx_exit = idx_window["close"].iloc[-1]
    benchmark_return = float(idx_exit / idx_entry - 1.0)
    return stock_returns.astype(float), benchmark_return


def _score_candidate(
    original_weight: float,
    top_k: int,
    reweight: str,
    window_returns: dict[tuple[str, str], tuple[pd.Series, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for asof, start, end, split in WINDOWS_5D:
        original_path = Path(str(ROLLING_ORIGINAL_TEMPLATE).format(asof=asof))
        baseline_path = Path(str(ROLLING_BASELINE_TEMPLATE).format(asof=asof))
        if not original_path.exists() or not baseline_path.exists():
            rows.append(
                {
                    "as_of": asof,
                    "start": start,
                    "end": end,
                    "split": split,
                    "missing": True,
                    "missing_original": not original_path.exists(),
                    "missing_baseline": not baseline_path.exists(),
                }
            )
            continue

        weights = _reweight(
            _blend(_load_weights(original_path), _load_weights(baseline_path), original_weight),
            top_k=top_k,
            mode=reweight,
        )
        stock_returns, benchmark_return = window_returns[(start, end)]
        portfolio_return = float((weights * stock_returns.reindex(weights.index).fillna(0.0)).sum())
        rows.append(
            {
                "as_of": pd.Timestamp(asof).date().isoformat(),
                "start": pd.Timestamp(start).date().isoformat(),
                "end": pd.Timestamp(end).date().isoformat(),
                "split": split,
                "missing": False,
                "original_weight": float(original_weight),
                "baseline_weight": float(1.0 - original_weight),
                "top_k": int(top_k),
                "reweight": reweight,
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "excess_return": portfolio_return - benchmark_return,
                "n_stocks": int(len(weights)),
                "max_weight": float(weights.max()),
                "top10": ",".join(weights.index[:10]),
            }
        )
    return rows


def _summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = detail[~detail["missing"].astype(bool)].copy()
    for key, group in ok.groupby(["original_weight", "baseline_weight", "top_k", "reweight"]):
        ow, bw, top_k, reweight = key
        for split in ["validation", "test", "all"]:
            df = group if split == "all" else group[group["split"] == split]
            if df.empty:
                continue
            excess = df["excess_return"].astype(float)
            rows.append(
                {
                    "variant": f"orig{int(round(ow * 100)):02d}_xgb{int(round(bw * 100)):02d}_k{int(top_k)}_{reweight}",
                    "split": split,
                    "n_windows": int(len(df)),
                    "original_weight": float(ow),
                    "baseline_weight": float(bw),
                    "top_k": int(top_k),
                    "reweight": reweight,
                    "mean_portfolio_return": float(df["portfolio_return"].mean()),
                    "mean_benchmark_return": float(df["benchmark_return"].mean()),
                    "mean_excess_return": float(excess.mean()),
                    "median_excess_return": float(excess.median()),
                    "worst_excess_return": float(excess.min()),
                    "latest_excess_return": float(excess.iloc[-1]),
                    "hit_rate": float((excess > 0).mean()),
                    "cumulative_excess_return": float(np.prod(1.0 + excess) - 1.0),
                    "max_weight": float(df["max_weight"].max()),
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["selection_score"] = (
        summary["mean_excess_return"]
        + 0.35 * summary["latest_excess_return"]
        + 0.25 * summary["worst_excess_return"].clip(upper=0.0)
    )
    return summary.sort_values(
        ["split", "selection_score", "mean_excess_return"],
        ascending=[True, False, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(ROOT / "data" / "prices.parquet"))
    parser.add_argument("--index", default=str(ROOT / "data" / "index.parquet"))
    parser.add_argument("--original-weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--top-ks", default="30,32,35,40,45,50")
    parser.add_argument("--reweights", default="equal,sqrt_rank,preserve")
    parser.add_argument("--out-dir", default=str(ROOT / "submissions" / "original_baseline_blend_current"))
    parser.add_argument("--out-prefix", default=str(ROOT / "logs" / "original_baseline_blend"))
    parser.add_argument("--write-top-n", type=int, default=12)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    original_weights = _parse_csv(args.original_weights, float)
    top_ks = _parse_csv(args.top_ks, int)
    reweights = _parse_csv(args.reweights, str)

    window_returns = {
        (start, end): _window_returns(prices, index_df, start, end)
        for _, start, end, _ in WINDOWS_5D
    }

    detail_rows: list[dict[str, object]] = []
    for ow in original_weights:
        for top_k in top_ks:
            for reweight in reweights:
                detail_rows.extend(_score_candidate(ow, top_k, reweight, window_returns))

    detail = pd.DataFrame(detail_rows)
    summary = _summarize(detail)

    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = ROOT / out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = Path(f"{out_prefix}_detail.csv")
    summary_path = Path(f"{out_prefix}_summary.csv")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    current_original = _load_weights(CURRENT_ORIGINAL)
    current_baseline = _load_weights(CURRENT_BASELINE)

    top_all = summary[summary["split"] == "all"].head(args.write_top_n)
    written = []
    for _, row in top_all.iterrows():
        weights = _reweight(
            _blend(current_original, current_baseline, row["original_weight"]),
            top_k=int(row["top_k"]),
            mode=str(row["reweight"]),
        )
        path = out_dir / f"{row['variant']}.csv"
        _write_submission(weights, path)
        written.append(str(path.relative_to(ROOT)))

    print("\n>> Simple original + baseline_xgb blend search")
    print(f"   detail : {detail_path.relative_to(ROOT)}")
    print(f"   summary: {summary_path.relative_to(ROOT)}")
    print(f"   wrote  : {len(written)} current candidate CSVs")
    for path in written[: min(8, len(written))]:
        print(f"      {path}")

    cols = [
        "variant", "split", "n_windows", "mean_excess_return",
        "worst_excess_return", "latest_excess_return", "hit_rate",
        "cumulative_excess_return", "max_weight", "selection_score",
    ]
    show = pd.concat(
        [
            summary[summary["split"] == "test"].head(10),
            summary[summary["split"] == "all"].head(10),
        ],
        ignore_index=True,
    )
    print("\n>> Top rolling candidates")
    print(show[cols].to_string(index=False, formatters={
        "mean_excess_return": _fmt_pct,
        "worst_excess_return": _fmt_pct,
        "latest_excess_return": _fmt_pct,
        "hit_rate": lambda x: f"{x:.1%}",
        "cumulative_excess_return": _fmt_pct,
        "max_weight": lambda x: f"{x:.3%}",
        "selection_score": _fmt_pct,
    }))


if __name__ == "__main__":
    main()
