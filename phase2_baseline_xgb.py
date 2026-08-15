from __future__ import annotations

"""Clean official-style XGBoost baseline component for phase 2.

This file intentionally keeps the original quick-start feature logic
self-contained so the component is not affected by experiments in features.py.
It trains with a no-lookahead as-of cutoff and outputs the same top-K
rank-weighted portfolio style as the official baseline.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SUB_DIR = ROOT / "submissions"
LOG_DIR = ROOT / "logs"

FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "vol_20d", "volume_z_20d", "turnover_ma_20d",
    "close_over_ma20", "close_over_ma60", "rsi_14",
    "ret_5d_rank", "ret_20d_rank", "vol_20d_rank",
]
TARGET_COLUMN = "target_5d"
FORWARD_HORIZON = 5

VAL_DAYS = 10
EMBARGO_DAYS = 5
MIN_STOCKS = 30
MAX_WEIGHT = 0.10
DEFAULT_TOP_K = 50

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    early_stopping_rounds=30,
)


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    close = df["close"]

    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["ret_60d"] = close.pct_change(60)
    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    vol = df["volume"].astype(float)
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (vol - vol_mean) / vol_std

    if "turnover" in df.columns:
        df["turnover_ma_20d"] = df["turnover"].astype(float).rolling(20).mean()
    else:
        df["turnover_ma_20d"] = np.nan

    df["close_over_ma20"] = close / close.rolling(20).mean() - 1.0
    df["close_over_ma60"] = close / close.rolling(60).mean() - 1.0

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = up / down
    df["rsi_14"] = 100 - 100 / (1 + rs)

    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return df


def build_official_features(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "stock_code", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    panel = (
        prices.groupby("stock_code", group_keys=False)
        .apply(_per_stock_features)
        .reset_index(drop=True)
    )
    for base in ["ret_5d", "ret_20d", "vol_20d"]:
        panel[f"{base}_rank"] = panel.groupby("date")[base].rank(
            method="average", pct=True
        )
    return panel


def training_frame(
    panel: pd.DataFrame,
    min_date: pd.Timestamp | None = None,
    max_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    df = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of: pd.Timestamp | str | None = None) -> pd.DataFrame:
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    return panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df[TARGET_COLUMN],
        eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
        verbose=False,
    )
    return model


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def build_portfolio(scores: pd.Series, top_k: int = DEFAULT_TOP_K) -> pd.Series:
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")

    scores = scores.copy()
    scores.index = scores.index.astype(str).str.zfill(6)
    chosen = scores.sort_values(ascending=False).head(top_k)
    ranks = np.arange(len(chosen), 0, -1, dtype=float)
    weights = pd.Series(ranks / ranks.sum(), index=chosen.index)

    for _ in range(50):
        over = weights > MAX_WEIGHT
        if not over.any():
            break
        excess = float((weights[over] - MAX_WEIGHT).sum())
        weights.loc[over] = MAX_WEIGHT
        free = ~over
        if not free.any() or weights.loc[free].sum() <= 0:
            break
        weights.loc[free] += excess * weights.loc[free] / weights.loc[free].sum()

    weights = weights / weights.sum()
    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights <= MAX_WEIGHT + 1e-9).all()
    assert (weights > 0).sum() >= MIN_STOCKS
    return weights


def fit_predict_baseline_xgb(
    prices: pd.DataFrame,
    as_of: pd.Timestamp | str | None,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[pd.Series, pd.Series, dict]:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    panel = build_official_features(prices)

    as_of_ts = pd.Timestamp(as_of) if as_of is not None else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    if as_of_idx >= len(trading_dates) or pd.Timestamp(trading_dates[as_of_idx]) != as_of_ts:
        raise RuntimeError(f"as_of={as_of_ts.date()} is not available in panel")

    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = training_frame(panel, max_date=train_cutoff)

    all_dates = np.sort(train_pool["date"].unique())
    if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 20:
        raise RuntimeError("Not enough dates to train baseline XGB.")
    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]

    model = train_model(train_df, val_df)
    val_pred = model.predict(val_df[FEATURE_COLUMNS])
    val_ic = rank_ic(
        val_df[TARGET_COLUMN].to_numpy(),
        val_pred,
        val_df["date"].to_numpy(),
    )

    pred_df = prediction_frame(panel, as_of=as_of_ts)
    if pred_df.empty:
        raise RuntimeError(f"No prediction rows for {as_of_ts.date()}")
    pred_df = pred_df.assign(score=model.predict(pred_df[FEATURE_COLUMNS]))
    scores = pred_df.set_index("stock_code")["score"]
    weights = build_portfolio(scores, top_k=top_k)

    feature_gain = model.get_booster().get_score(importance_type="gain")
    metadata = {
        "as_of": as_of_ts.date().isoformat(),
        "model": "official-style XGBoost baseline",
        "feature_set": "self-contained quick-start OHLCV features",
        "target": TARGET_COLUMN,
        "forward_horizon": FORWARD_HORIZON,
        "train_start": pd.Timestamp(train_df["date"].min()).date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "embargo_days": EMBARGO_DAYS,
        "val_start": val_start.date().isoformat(),
        "val_end": pd.Timestamp(val_df["date"].max()).date().isoformat(),
        "train_cutoff": train_cutoff.date().isoformat(),
        "val_rank_ic": float(val_ic),
        "top_k": int(top_k),
        "n_stocks": int(len(weights)),
        "max_weight": float(weights.max()),
        "feature_gain": {k: float(feature_gain.get(k, 0.0)) for k in FEATURE_COLUMNS},
    }
    return weights, scores, metadata


def _write_submission(weights: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "stock_code": weights.index.astype(str).str.zfill(6),
        "weight": weights.astype(float).values,
    }).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--out", default=str(SUB_DIR / "phase2_baseline_xgb.csv"))
    parser.add_argument("--metadata-out", default="")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    weights, _scores, metadata = fit_predict_baseline_xgb(
        prices,
        as_of=args.as_of,
        top_k=args.top_k,
    )

    out_path = Path(args.out)
    _write_submission(weights, out_path)
    if args.metadata_out:
        import json

        meta_path = Path(args.metadata_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f">> Wrote {len(weights)} names to {out_path}")
    print(
        f"   as_of={metadata['as_of']} train={metadata['train_start']}->{metadata['train_end']} "
        f"val={metadata['val_start']}->{metadata['val_end']} val_ic={metadata['val_rank_ic']:+.4f}"
    )
    print(f"   max={weights.max():.4f} sum={weights.sum():.12f}")


if __name__ == "__main__":
    main()
