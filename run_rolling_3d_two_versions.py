from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import self_test_regime_blend_clean as clean
from score_submission import score_window
from train_regime_blend import portfolio_from_scores


ROOT = Path(__file__).parent

WINDOWS = [
    ("20260227", "20260302", "20260304"),
    ("20260306", "20260309", "20260311"),
    ("20260313", "20260316", "20260318"),
    ("20260320", "20260323", "20260325"),
    ("20260327", "20260330", "20260401"),
    ("20260403", "20260407", "20260409"),
    ("20260410", "20260413", "20260415"),
    ("20260417", "20260420", "20260422"),
    ("20260424", "20260427", "20260429"),
]


def _write_submission(weights: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "stock_code": weights.index.astype(str).str.zfill(6),
        "weight": weights.astype(float).values,
    })
    out.to_csv(path, index=False)


def _fmt_date(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


def _summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for version, df in detail.groupby("version", dropna=False):
        excess = df["excess_return"].astype(float)
        rows.append({
            "version": version,
            "n_windows": int(len(df)),
            "mean_portfolio_return": float(df["portfolio_return"].mean()),
            "mean_benchmark_return": float(df["benchmark_return"].mean()),
            "mean_excess_return": float(excess.mean()),
            "median_excess_return": float(excess.median()),
            "worst_excess_return": float(excess.min()),
            "best_excess_return": float(excess.max()),
            "hit_rate": float((excess > 0).mean()),
            "cumulative_excess_return": float(np.prod(1.0 + excess) - 1.0),
            "max_drawdown": _max_drawdown(excess),
        })
    return pd.DataFrame(rows).sort_values("mean_excess_return", ascending=False)


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(-drawdown.min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-prices", default=str(ROOT / "data" / "prices.parquet"))
    parser.add_argument("--support-index", default=str(ROOT / "data" / "index.parquet"))
    parser.add_argument("--regime-data-dir", default=str(clean.REGIME_DATA_DIR))
    parser.add_argument("--score-prices", default=str(ROOT / "data" / "prices.parquet"))
    parser.add_argument("--score-index", default=str(ROOT / "data" / "index.parquet"))
    parser.add_argument("--out-dir", default=str(ROOT / "submissions" / "rolling_3d_compare"))
    parser.add_argument("--out-prefix", default=str(ROOT / "logs" / "rolling_3d_two_versions"))
    parser.add_argument("--quiet-train", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(">> Loading support data")
    support_prices = pd.read_parquet(args.support_prices)
    support_prices["date"] = pd.to_datetime(support_prices["date"])
    support_prices["stock_code"] = support_prices["stock_code"].astype(str).str.zfill(6)
    support_panel = clean.slim_self_test.build_features(support_prices)
    support_dates = np.array(pd.DatetimeIndex(sorted(pd.to_datetime(support_panel["date"].unique()))).values)

    print(">> Loading regime data")
    regime_prices, regime_index, regime_panel, active_features, used_industry = clean.load_regime_panel(
        Path(args.regime_data_dir), None
    )

    print(">> Loading score data")
    score_prices = pd.read_parquet(args.score_prices)
    score_prices["date"] = pd.to_datetime(score_prices["date"])
    score_prices["stock_code"] = score_prices["stock_code"].astype(str).str.zfill(6)
    score_index = pd.read_parquet(args.score_index)
    score_index["date"] = pd.to_datetime(score_index["date"])

    records = []
    skipped = []
    train_log_path = out_prefix.with_name(out_prefix.name + "_train.log")
    log_fh = train_log_path.open("w", encoding="utf-8") if args.quiet_train else None

    try:
        for asof_raw, start_raw, end_raw in WINDOWS:
            as_of = pd.Timestamp(asof_raw)
            start = pd.Timestamp(start_raw)
            end = pd.Timestamp(end_raw)
            print(f">> As-of {as_of.date()} | score {start.date()} -> {end.date()}")

            try:
                ctx = contextlib.redirect_stdout(log_fh) if log_fh is not None else contextlib.nullcontext()
                with ctx:
                    support_weights, _support_scores, selected_folds, selected_weights = (
                        clean.slim_self_test._fit_predict_for_asof(
                            support_panel,
                            support_prices,
                            support_dates,
                            as_of,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                skipped.append({
                    "as_of": _fmt_date(as_of),
                    "component": "support_lgb",
                    "reason": str(exc),
                })
                print(f"   [skip] support LGB failed: {exc}")
                continue

            versions: list[tuple[str, pd.Series, pd.Series | None]] = []

            lgb_weights = support_weights[support_weights > 0].sort_values(ascending=False)
            versions.append(("pure_lgb_k40_sqrt", lgb_weights, None))

            try:
                regime_scores, ic_series = clean.compute_regime_scores_for_asof(
                    regime_panel,
                    active_features,
                    as_of,
                )
                final_scores = clean.blend_scores(
                    support_weights,
                    regime_scores,
                    support_weight=0.30,
                )
                regime_weights = portfolio_from_scores(
                    final_scores,
                    top_k=50,
                    mode="sqrt_rank",
                )
                versions.append(("regime_clean_70_30_k50_sqrt", regime_weights, ic_series))
            except Exception as exc:  # noqa: BLE001
                skipped.append({
                    "as_of": _fmt_date(as_of),
                    "component": "regime_blend",
                    "reason": str(exc),
                })
                print(f"   [skip] regime blend failed: {exc}")

            for version, weights, ic_series in versions:
                sub_path = out_dir / f"{version}_{asof_raw}.csv"
                _write_submission(weights, sub_path)
                scored = score_window(weights, score_prices, score_index, start, end)
                records.append({
                    "version": version,
                    "as_of": _fmt_date(as_of),
                    "start": _fmt_date(start),
                    "end": _fmt_date(end),
                    "portfolio_return": scored["portfolio_return"],
                    "benchmark_return": scored["benchmark_return"],
                    "excess_return": scored["excess_return"],
                    "n_stocks": int((weights > 0).sum()),
                    "max_weight": float(weights.max()),
                    "submission": str(sub_path),
                    "support_selected_folds": json.dumps(selected_folds, sort_keys=True),
                    "support_selected_weights": json.dumps(selected_weights, sort_keys=True),
                    "n_positive_ic_features": (
                        int((ic_series > 0).sum()) if ic_series is not None else None
                    ),
                })
                print(
                    f"   {version}: port {scored['portfolio_return']*100:+.3f}% | "
                    f"bench {scored['benchmark_return']*100:+.3f}% | "
                    f"excess {scored['excess_return']*100:+.3f}%"
                )
    finally:
        if log_fh is not None:
            log_fh.close()

    detail = pd.DataFrame(records)
    if detail.empty:
        raise RuntimeError(f"No records produced. Skipped: {skipped}")

    summary = _summarize(detail)
    detail_path = out_prefix.with_name(out_prefix.name + "_detail.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    skipped_path = out_prefix.with_name(out_prefix.name + "_skipped.json")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    skipped_path.write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    print("\n>> Summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.6f}"))
    print(f"\n>> Wrote {detail_path}")
    print(f">> Wrote {summary_path}")
    print(f">> Wrote {skipped_path}")
    if args.quiet_train:
        print(f">> Training log: {train_log_path}")


if __name__ == "__main__":
    main()
