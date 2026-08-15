from __future__ import annotations

"""No-lookahead rolling test for the official-style baseline XGB component."""

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2_baseline_xgb import DEFAULT_TOP_K, fit_predict_baseline_xgb
from score_submission import score_window


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SUB_DIR = ROOT / "submissions"
LOG_DIR = ROOT / "logs"

WINDOWS = [
    ("20260313", "20260316", "20260318"),
    ("20260320", "20260323", "20260325"),
    ("20260327", "20260330", "20260401"),
    ("20260403", "20260407", "20260409"),
    ("20260410", "20260413", "20260415"),
    ("20260417", "20260420", "20260422"),
    ("20260424", "20260427", "20260429"),
    ("20260427", "20260428", "20260430"),
    ("20260430", "20260506", "20260508"),
]


def _write_submission(weights: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "stock_code": weights.index.astype(str).str.zfill(6),
        "weight": weights.astype(float).values,
    }).to_csv(path, index=False)


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.astype(float)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(-drawdown.min())


def _summarize(detail: pd.DataFrame) -> pd.DataFrame:
    excess = detail["excess_return"].astype(float)
    return pd.DataFrame([{
        "variant": "official_style_baseline_xgb",
        "n_windows": int(len(detail)),
        "mean_portfolio_return": float(detail["portfolio_return"].mean()),
        "mean_benchmark_return": float(detail["benchmark_return"].mean()),
        "mean_excess_return": float(excess.mean()),
        "median_excess_return": float(excess.median()),
        "worst_excess_return": float(excess.min()),
        "best_excess_return": float(excess.max()),
        "hit_rate": float((excess > 0).mean()),
        "cumulative_excess_return": float(np.prod(1.0 + excess) - 1.0),
        "max_drawdown": _max_drawdown(excess),
        "max_weight": float(detail["max_weight"].max()),
        "mean_val_rank_ic": float(detail["val_rank_ic"].mean()),
    }])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--out-prefix", default=str(LOG_DIR / "rolling_3d_baseline_xgb_phase2"))
    parser.add_argument("--out-dir", default=str(SUB_DIR / "rolling_3d_baseline_xgb_phase2"))
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--quiet-train", action="store_true")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    windows = WINDOWS[-args.max_windows:] if args.max_windows and args.max_windows > 0 else WINDOWS
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_log = out_prefix.with_name(out_prefix.name + "_train.log")
    log_fh = train_log.open("w", encoding="utf-8") if args.quiet_train else None
    rows = []
    metadata_rows = []
    try:
        for asof_raw, start_raw, end_raw in windows:
            as_of = pd.Timestamp(asof_raw)
            start = pd.Timestamp(start_raw)
            end = pd.Timestamp(end_raw)
            print(f">> As-of {as_of.date()} | score {start.date()} -> {end.date()}")
            ctx = contextlib.redirect_stdout(log_fh) if log_fh is not None else contextlib.nullcontext()
            with ctx:
                weights, _scores, metadata = fit_predict_baseline_xgb(
                    prices,
                    as_of=as_of,
                    top_k=args.top_k,
                )

            sub_path = out_dir / f"baseline_xgb_{asof_raw}.csv"
            _write_submission(weights, sub_path)
            scored = score_window(weights, prices, index_df, start, end)
            rows.append({
                "as_of": as_of.date().isoformat(),
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "portfolio_return": scored["portfolio_return"],
                "benchmark_return": scored["benchmark_return"],
                "excess_return": scored["excess_return"],
                "n_stocks": int(len(weights)),
                "max_weight": float(weights.max()),
                "val_rank_ic": float(metadata["val_rank_ic"]),
                "train_start": metadata["train_start"],
                "train_end": metadata["train_end"],
                "val_start": metadata["val_start"],
                "val_end": metadata["val_end"],
                "submission": str(sub_path.relative_to(ROOT)),
            })
            metadata_rows.append(metadata)
            print(
                f"   excess={scored['excess_return'] * 100:+.3f}% "
                f"portfolio={scored['portfolio_return'] * 100:+.3f}% "
                f"val_ic={metadata['val_rank_ic']:+.4f}"
            )
    finally:
        if log_fh is not None:
            log_fh.close()

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("No rolling baseline XGB records produced.")
    summary = _summarize(detail)

    detail_path = out_prefix.with_name(out_prefix.name + "_detail.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    metadata_path = out_prefix.with_name(out_prefix.name + "_metadata.json")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata_path.write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")

    print("\n>> Summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.6f}"))
    print(f"\n>> Detail: {detail_path}")
    print(f">> Summary: {summary_path}")
    print(f">> Metadata: {metadata_path}")
    if args.quiet_train:
        print(f">> Training log: {train_log}")


if __name__ == "__main__":
    main()
