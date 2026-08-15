"""
Clean regime-style model for the CSI500 competition.

This is the reproducible replacement for the exploratory pool-style overlays:
it keeps the useful "recent market style" idea, but removes hand-picked stock
pools, manual blacklists, and frozen intermediate portfolios.

Pipeline
--------
1. Train the already-tested h3 LGB support model as of the prediction date.
2. Train a small ranker support model as an orthogonal ranking signal.
3. Compute recent, no-lookahead cross-sectional ICs for public OHLCV,
   market-relative, and industry-rank style factors.
4. Build a linear style score from the strongest positive recent-IC factors.
5. Blend model rank score + style score, then select top-k names.

Default final-style configuration intentionally mirrors the promising
regime_final_style_feat_w10_k50_equal direction:

    80% h3 LGB + 20% ranker support, then 10% clean style overlay,
    final portfolio top50 equal-weight.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).parent
FACTOR_DIR = ROOT / "factor_experiments_slim_asof20260430"
sys.path.insert(0, str(FACTOR_DIR))

import model_experiments as me  # noqa: E402
import self_test_multihorizon_lgb as slim  # noqa: E402
from features import (  # noqa: E402
    FEATURE_COLUMNS,
    INDUSTRY_RANK_FEATURE_COLUMNS,
    MARKET_RELATIVE_FEATURE_COLUMNS,
    _add_industry_rank_features,
    _add_market_relative_features,
    build_features,
    prediction_frame,
    target_column_for_horizon,
    training_frame,
)


DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
SUB_DIR = ROOT / "submissions"

DEFAULT_H3_WEIGHT = 0.80
DEFAULT_RANKER_WEIGHT = 0.20
DEFAULT_STYLE_WEIGHT = 0.10
DEFAULT_TOP_K = 50
DEFAULT_PORTFOLIO_MODE = "equal"

MODEL_HORIZON_WEIGHTS = {3: 0.50, 5: 0.30, 10: 0.20}
STYLE_HORIZON_WEIGHTS = {3: 0.60, 5: 0.30, 10: 0.10}
STYLE_WINDOW_TRADING_DAYS = 30
STYLE_HALFLIFE_DAYS = 12
STYLE_MAX_FEATURE_IC = 0.07
STYLE_MIN_ABS_IC = 0.004
STYLE_MIN_FEATURES = 8
STYLE_MAX_FEATURES = 18


@dataclass
class CleanRegimeStyleResult:
    weights: pd.Series
    final_scores: pd.Series
    base_scores: pd.Series
    style_scores: pd.Series
    selected_features: pd.DataFrame
    ic_table: pd.DataFrame
    metadata: dict


def _parse_weights(raw: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":")
        out[int(key)] = float(value)
    total = sum(out.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return {k: v / total for k, v in out.items()}


def _normalise_01(s: pd.Series) -> pd.Series:
    s = s.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    if mx <= mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def _rank_score(s: pd.Series) -> pd.Series:
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    return s.rank(method="average", pct=True, na_option="keep").fillna(0.5)


def _clean_feature_frame(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feat_cols:
        values = out[col].replace([np.inf, -np.inf], np.nan)
        fill = values.median() if values.notna().any() else 0.0
        out[col] = values.fillna(fill)
    return out


def _load_prices(path: str | Path) -> pd.DataFrame:
    prices = pd.read_parquet(path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    return prices


def _load_index(path: str | Path) -> pd.DataFrame:
    index_df = pd.read_parquet(path)
    index_df["date"] = pd.to_datetime(index_df["date"])
    return index_df


def _load_industry(path: str | Path) -> pd.DataFrame:
    industry = pd.read_csv(path, dtype={"stock_code": str})
    industry["stock_code"] = industry["stock_code"].astype(str).str.zfill(6)
    if "industry" not in industry.columns:
        raise ValueError(f"{path} must contain an industry column")
    return industry[["stock_code", "industry"]].drop_duplicates("stock_code")


def build_style_panel(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    industry: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a factor panel for the data-driven style overlay."""
    panel = build_features(prices)
    panel = _add_market_relative_features(panel, index_df)
    panel = _add_industry_rank_features(panel, industry)

    candidates = (
        list(FEATURE_COLUMNS)
        + list(MARKET_RELATIVE_FEATURE_COLUMNS)
        + list(INDUSTRY_RANK_FEATURE_COLUMNS)
    )
    active: list[str] = []
    seen: set[str] = set()
    for col in candidates:
        if col in panel.columns and col not in seen:
            active.append(col)
            seen.add(col)
    return panel, active


def compute_recent_feature_ic(
    panel: pd.DataFrame,
    feat_cols: list[str],
    as_of: pd.Timestamp,
    horizon: int,
    window_trading_days: int = STYLE_WINDOW_TRADING_DAYS,
    halflife_days: int = STYLE_HALFLIFE_DAYS,
) -> pd.Series:
    """No-lookahead recent IC using only labels available by as_of."""
    target = target_column_for_horizon(horizon)
    all_dates = np.array(pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"].unique()))).values)
    as_of_idx = int(np.searchsorted(all_dates, np.datetime64(as_of)))
    if as_of_idx >= len(all_dates) or pd.Timestamp(all_dates[as_of_idx]) != as_of:
        raise ValueError(f"as_of {as_of.date()} is not a trading date in the feature panel")
    cutoff_idx = max(0, as_of_idx - int(horizon))
    cutoff_date = pd.Timestamp(all_dates[cutoff_idx])

    base = panel[(panel["date"] <= cutoff_date) & (panel["volume"] > 0) & (panel["high"] > panel["low"])]
    base = base.dropna(subset=[target])
    dates = np.array(pd.DatetimeIndex(sorted(pd.to_datetime(base["date"].unique()))).values)
    if len(dates) == 0:
        raise RuntimeError(f"No training dates available for h{horizon} as of {as_of.date()}")
    recent_dates = dates[-min(window_trading_days, len(dates)):]
    max_pos = len(recent_dates) - 1

    weighted_ic = {feat: 0.0 for feat in feat_cols}
    total_weight = {feat: 0.0 for feat in feat_cols}
    for i, date_value in enumerate(recent_dates):
        date = pd.Timestamp(date_value)
        grp = base[base["date"] == date]
        tgt = grp[target]
        w = float(np.power(0.5, (max_pos - i) / max(halflife_days, 1)))
        for feat in feat_cols:
            vals = grp[feat]
            mask = vals.notna() & tgt.notna()
            if int(mask.sum()) < 20:
                continue
            rho, _ = spearmanr(vals[mask], tgt[mask])
            if not np.isnan(rho):
                weighted_ic[feat] += float(rho) * w
                total_weight[feat] += w

    return pd.Series({
        feat: weighted_ic[feat] / total_weight[feat]
        for feat in feat_cols
        if total_weight[feat] > 0
    })


def compute_blended_style_ic(
    panel: pd.DataFrame,
    feat_cols: list[str],
    as_of: pd.Timestamp,
    horizon_weights: dict[int, float] | None = None,
    window_trading_days: int = STYLE_WINDOW_TRADING_DAYS,
    halflife_days: int = STYLE_HALFLIFE_DAYS,
) -> pd.DataFrame:
    horizon_weights = horizon_weights or STYLE_HORIZON_WEIGHTS
    rows = []
    blended = pd.Series(0.0, index=feat_cols, dtype=float)
    for horizon, weight in horizon_weights.items():
        ic = compute_recent_feature_ic(
            panel,
            feat_cols,
            as_of,
            horizon=int(horizon),
            window_trading_days=window_trading_days,
            halflife_days=halflife_days,
        )
        blended = blended.add(ic.reindex(feat_cols).fillna(0.0) * float(weight), fill_value=0.0)
        for feat, value in ic.items():
            rows.append({
                "feature": feat,
                "horizon": int(horizon),
                "horizon_weight": float(weight),
                "ic": float(value),
            })

    ic_table = pd.DataFrame(rows)
    blended_df = blended.sort_values(ascending=False).reset_index()
    blended_df.columns = ["feature", "blended_ic"]
    return blended_df.merge(ic_table, on="feature", how="left")


def select_style_features(
    ic_table: pd.DataFrame,
    min_abs_ic: float = STYLE_MIN_ABS_IC,
    min_features: int = STYLE_MIN_FEATURES,
    max_features: int = STYLE_MAX_FEATURES,
) -> pd.DataFrame:
    blended = (
        ic_table[["feature", "blended_ic"]]
        .drop_duplicates("feature")
        .sort_values("blended_ic", ascending=False)
        .reset_index(drop=True)
    )
    selected = blended[
        (blended["blended_ic"] > 0) & (blended["blended_ic"].abs() >= min_abs_ic)
    ].copy()
    if len(selected) < min_features:
        selected = blended[blended["blended_ic"] > 0].head(min_features).copy()
    selected = selected.head(max_features).copy()
    clipped = selected["blended_ic"].clip(lower=-STYLE_MAX_FEATURE_IC, upper=STYLE_MAX_FEATURE_IC)
    selected["clipped_ic"] = clipped
    selected["style_weight"] = clipped.abs() / clipped.abs().sum()
    return selected


def style_score_from_features(
    pred_df: pd.DataFrame,
    selected_features: pd.DataFrame,
) -> pd.Series:
    if selected_features.empty:
        raise RuntimeError("No style features selected")

    pred = _clean_feature_frame(pred_df, selected_features["feature"].tolist())
    scores = pd.Series(0.0, index=pred["stock_code"].astype(str).str.zfill(6).values)
    for row in selected_features.itertuples(index=False):
        ranks = pred[row.feature].rank(pct=True, na_option="keep").fillna(0.5)
        component = ranks if row.clipped_ic >= 0 else 1.0 - ranks
        scores = scores.add(
            pd.Series(component.values, index=pred["stock_code"].astype(str).str.zfill(6).values)
            * float(row.style_weight),
            fill_value=0.0,
        )
    return _normalise_01(scores)


def h3_lgb_scores(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    all_dates: np.ndarray,
    as_of: pd.Timestamp,
) -> pd.Series:
    old_horizons = slim.TARGET_HORIZONS
    old_weights = slim.TARGET_HORIZON_WEIGHTS
    try:
        slim.TARGET_HORIZONS = [3]
        slim.TARGET_HORIZON_WEIGHTS = {3: 1.0}
        _weights, scores, _folds, _fold_weights = slim._fit_predict_for_asof(
            panel,
            prices,
            all_dates,
            as_of,
        )
        return scores
    finally:
        slim.TARGET_HORIZONS = old_horizons
        slim.TARGET_HORIZON_WEIGHTS = old_weights


def ranker_scores(
    panel: pd.DataFrame,
    industry: pd.DataFrame,
    all_dates: np.ndarray,
    as_of: pd.Timestamp,
    horizon_weights: dict[int, float] | None = None,
) -> pd.Series:
    horizon_weights = horizon_weights or MODEL_HORIZON_WEIGHTS
    as_of_idx = int(np.searchsorted(all_dates, np.datetime64(as_of)))
    pred_df = prediction_frame(panel, as_of=as_of)
    if pred_df.empty:
        raise RuntimeError(f"No ranker prediction rows for {as_of.date()}")

    horizon_preds = []
    weights = []
    for horizon, weight in horizon_weights.items():
        target_col = target_column_for_horizon(horizon)
        cutoff_idx = max(0, as_of_idx - int(horizon))
        train_cutoff = pd.Timestamp(all_dates[cutoff_idx])
        train_pool = training_frame(panel, max_date=train_cutoff, target_column=target_col)
        bundle = me.train_experiment(
            train_pool=train_pool,
            experiment="ranker_lgb",
            base_target_column=target_col,
            target_horizon=int(horizon),
            recency_halflife_days=60.0,
            industry=industry,
        )
        ensemble, ensemble_weights, _selected = me.select_ensemble(
            bundle.models,
            bundle.cv,
            method="best2",
            n_ensemble=me.N_ENSEMBLE,
        )
        pred_stack = np.stack([model.predict(pred_df[FEATURE_COLUMNS]) for model in ensemble])
        horizon_preds.append(np.average(pred_stack, axis=0, weights=ensemble_weights))
        weights.append(float(weight))

    weights_arr = np.array(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    combined = np.average(np.stack(horizon_preds), axis=0, weights=weights_arr)
    return pd.Series(
        combined,
        index=pred_df["stock_code"].astype(str).str.zfill(6),
        name="ranker_score",
    )


def build_components(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    industry: pd.DataFrame,
    as_of: pd.Timestamp,
    log_dir: Path,
    quiet_train: bool = False,
    style_horizon_weights: dict[int, float] | None = None,
    style_window_days: int = STYLE_WINDOW_TRADING_DAYS,
    style_halflife_days: int = STYLE_HALFLIFE_DAYS,
) -> tuple[dict[str, pd.Series], pd.Series, pd.DataFrame, pd.DataFrame]:
    """Train support models and compute the clean style score."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Keep temporary CV/model artifacts separate from production files.
    slim.te.LOG_DIR = log_dir / "lgb_cv_tmp"
    slim.te.MODEL_DIR = log_dir / "lgb_models_tmp"
    slim.te.LOG_DIR.mkdir(parents=True, exist_ok=True)
    slim.te.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    support_panel = build_features(prices)
    all_dates = np.array(pd.DatetimeIndex(sorted(pd.to_datetime(support_panel["date"].unique()))).values)
    train_log = log_dir / f"clean_regime_style_train_{as_of.strftime('%Y%m%d')}.log"
    log_fh = train_log.open("w", encoding="utf-8") if quiet_train else None
    try:
        ctx = contextlib.redirect_stdout(log_fh) if log_fh is not None else contextlib.nullcontext()
        with ctx:
            h3 = h3_lgb_scores(support_panel, prices, all_dates, as_of)
            rnk = ranker_scores(support_panel, industry, all_dates, as_of)
    finally:
        if log_fh is not None:
            log_fh.close()

    style_panel, style_features = build_style_panel(prices, index_df, industry)
    style_ic_table = compute_blended_style_ic(
        style_panel,
        style_features,
        as_of,
        horizon_weights=style_horizon_weights,
        window_trading_days=style_window_days,
        halflife_days=style_halflife_days,
    )
    selected = select_style_features(style_ic_table)
    pred_style = style_panel[style_panel["date"] == as_of].copy()
    style = style_score_from_features(pred_style, selected)

    pred_support = prediction_frame(support_panel, as_of=as_of)
    pred_idx = pred_support["stock_code"].astype(str).str.zfill(6)
    tradable = pred_support.set_index(pred_idx)["is_tradable"]
    recent_ok = slim.te.recent_tradability(prices, as_of, lookback_days=1)
    if recent_ok is not None:
        tradable = tradable.reindex(h3.index).fillna(False) & recent_ok.reindex(h3.index).fillna(False)

    return {"h3_lgb": h3, "ranker": rnk, "style": style}, tradable, selected, style_ic_table


def make_clean_regime_style_portfolio(
    components: dict[str, pd.Series],
    tradable: pd.Series,
    selected_features: pd.DataFrame,
    ic_table: pd.DataFrame,
    as_of: pd.Timestamp,
    h3_weight: float = DEFAULT_H3_WEIGHT,
    ranker_weight: float = DEFAULT_RANKER_WEIGHT,
    style_weight: float = DEFAULT_STYLE_WEIGHT,
    top_k: int = DEFAULT_TOP_K,
    portfolio_mode: str = DEFAULT_PORTFOLIO_MODE,
) -> CleanRegimeStyleResult:
    h3_weight = float(h3_weight)
    ranker_weight = float(ranker_weight)
    style_weight = float(style_weight)
    support_total = h3_weight + ranker_weight
    if support_total <= 0:
        raise ValueError("h3_weight + ranker_weight must be positive")
    h3_frac = h3_weight / support_total
    ranker_frac = ranker_weight / support_total

    all_codes = sorted(set().union(*(set(s.index) for s in components.values())))
    h3_rank = _rank_score(components["h3_lgb"]).reindex(all_codes).fillna(0.0)
    ranker_rank = _rank_score(components["ranker"]).reindex(all_codes).fillna(0.0)
    style_norm = _normalise_01(components["style"]).reindex(all_codes).fillna(0.0)

    base_score = h3_frac * h3_rank + ranker_frac * ranker_rank
    base_score = _normalise_01(base_score)
    final_scores = (1.0 - style_weight) * base_score + style_weight * style_norm

    is_tradable = tradable.reindex(final_scores.index).fillna(False)
    weights = slim.te.build_portfolio(
        final_scores,
        top_k=top_k,
        weighting=portfolio_mode,
        is_tradable=is_tradable,
    )
    metadata = {
        "as_of": as_of.date().isoformat(),
        "model": "clean regime-style",
        "support": {
            "h3_lgb_weight": h3_frac,
            "ranker_weight": ranker_frac,
            "ranker_horizon_weights": MODEL_HORIZON_WEIGHTS,
        },
        "style": {
            "style_weight": style_weight,
            "horizon_weights": STYLE_HORIZON_WEIGHTS,
            "window_trading_days": STYLE_WINDOW_TRADING_DAYS,
            "halflife_days": STYLE_HALFLIFE_DAYS,
            "min_abs_ic": STYLE_MIN_ABS_IC,
            "max_feature_ic": STYLE_MAX_FEATURE_IC,
            "selected_features": selected_features["feature"].tolist(),
        },
        "portfolio": {
            "top_k": top_k,
            "mode": portfolio_mode,
            "n_stocks": int((weights > 0).sum()),
            "max_weight": float(weights.max()),
            "sum_weight": float(weights.sum()),
        },
        "no_hand_pool": True,
        "hard_blacklist": False,
    }
    return CleanRegimeStyleResult(
        weights=weights,
        final_scores=final_scores,
        base_scores=base_score,
        style_scores=style_norm,
        selected_features=selected_features,
        ic_table=ic_table,
        metadata=metadata,
    )


def fit_clean_regime_style(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    industry: pd.DataFrame,
    as_of: pd.Timestamp,
    log_dir: Path,
    quiet_train: bool = False,
    h3_weight: float = DEFAULT_H3_WEIGHT,
    ranker_weight: float = DEFAULT_RANKER_WEIGHT,
    style_weight: float = DEFAULT_STYLE_WEIGHT,
    top_k: int = DEFAULT_TOP_K,
    portfolio_mode: str = DEFAULT_PORTFOLIO_MODE,
) -> CleanRegimeStyleResult:
    components, tradable, selected, ic_table = build_components(
        prices,
        index_df,
        industry,
        as_of,
        log_dir=log_dir,
        quiet_train=quiet_train,
    )
    return make_clean_regime_style_portfolio(
        components,
        tradable,
        selected,
        ic_table,
        as_of=as_of,
        h3_weight=h3_weight,
        ranker_weight=ranker_weight,
        style_weight=style_weight,
        top_k=top_k,
        portfolio_mode=portfolio_mode,
    )


def write_result(result: CleanRegimeStyleResult, out_path: Path, log_prefix: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "stock_code": result.weights.index.astype(str).str.zfill(6),
        "weight": result.weights.astype(float).values,
    }).to_csv(out_path, index=False)

    selected_path = log_prefix.with_name(log_prefix.name + "_selected_features.csv")
    ic_path = log_prefix.with_name(log_prefix.name + "_ic_table.csv")
    meta_path = log_prefix.with_name(log_prefix.name + "_metadata.json")
    scores_path = log_prefix.with_name(log_prefix.name + "_scores.csv")

    result.selected_features.to_csv(selected_path, index=False)
    result.ic_table.to_csv(ic_path, index=False)
    meta_path.write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
    pd.DataFrame({
        "stock_code": result.final_scores.index.astype(str).str.zfill(6),
        "final_score": result.final_scores.values,
        "base_score": result.base_scores.reindex(result.final_scores.index).values,
        "style_score": result.style_scores.reindex(result.final_scores.index).values,
    }).sort_values("final_score", ascending=False).to_csv(scores_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--industry-file", default=str(DATA_DIR / "industry.csv"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--out", default=str(SUB_DIR / "clean_regime_style_w10_k50_equal.csv"))
    parser.add_argument("--log-prefix", default=str(LOG_DIR / "clean_regime_style"))
    parser.add_argument("--h3-weight", type=float, default=DEFAULT_H3_WEIGHT)
    parser.add_argument("--ranker-weight", type=float, default=DEFAULT_RANKER_WEIGHT)
    parser.add_argument("--style-weight", type=float, default=DEFAULT_STYLE_WEIGHT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--portfolio-mode", choices=["equal", "sqrt_rank", "rank"], default=DEFAULT_PORTFOLIO_MODE)
    parser.add_argument("--quiet-train", action="store_true")
    args = parser.parse_args()

    prices = _load_prices(args.prices)
    index_df = _load_index(args.index)
    industry = _load_industry(args.industry_file)
    as_of = pd.Timestamp(str(args.as_of)) if args.as_of else prices["date"].max()

    print("\n" + "=" * 72)
    print("train_clean_regime_style.py")
    print(f"  as_of       : {as_of.date()}")
    print(f"  support     : {args.h3_weight:.0%} h3_lgb + {args.ranker_weight:.0%} ranker")
    print(f"  style       : {args.style_weight:.0%} recent-IC style overlay")
    print(f"  portfolio   : top{args.top_k} {args.portfolio_mode}")
    print("  hand pool   : none")
    print("  blacklist   : none")
    print("=" * 72)

    result = fit_clean_regime_style(
        prices,
        index_df,
        industry,
        as_of=as_of,
        log_dir=Path(args.log_prefix).parent / "clean_regime_style_tmp",
        quiet_train=args.quiet_train,
        h3_weight=args.h3_weight,
        ranker_weight=args.ranker_weight,
        style_weight=args.style_weight,
        top_k=args.top_k,
        portfolio_mode=args.portfolio_mode,
    )
    write_result(result, Path(args.out), Path(args.log_prefix))

    print("\n>> Selected clean style features")
    cols = ["feature", "blended_ic", "clipped_ic", "style_weight"]
    print(result.selected_features[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\n>> Submission: {args.out}")
    print(f"   stocks={len(result.weights)} max_w={result.weights.max():.4f} sum={result.weights.sum():.6f}")
    print(f"   top10={', '.join(result.weights.index.astype(str).str.zfill(6)[:10])}")
    print(f">> Logs: {args.log_prefix}_*.csv/json")


if __name__ == "__main__":
    main()
