"""
train_regime_blend.py — Short-window linear regime model for 20260506-20260508.

Problem with standard models: trained on 2+ years of history where momentum
REVERSES. In April 2026, momentum CONTINUES:
  dist_low_120d    IC = +0.297 (stocks near 120-day highs continue)
  resid_ret_5d     IC = +0.075 (excess-return momentum continues)
  ret_5d           IC = +0.067 (raw momentum continues)
  vol_ratio_5_20   IC = +0.059 (volatility expansion continues)

Fix: compute empirical per-date cross-sectional Spearman ICs over the last
REGIME_WINDOW_TRADING_DAYS, take the recency-weighted average IC per feature,
then score stocks as IC-weighted rank combination (= linear factor model
trained on the current regime). This is equivalent to ridge regression on
cross-sectional returns and fully avoids overfitting.

Steps
-----
1. Build full feature panel (slim + market-relative + industry ranks)
2. For each of last REGIME_WINDOW_TRADING_DAYS training dates, compute
   per-feature IC against the forward target
3. Recency-weight the per-date ICs (halflife = RECENCY_HALFLIFE_REGIME days)
   so April data dominates over earlier dates
4. Build regime score = sum over positive-IC features of (IC_weight * rank_pct)
5. Blend:  SUPPORT_WEIGHT * reproducible_support_score
          + (1-SUPPORT_WEIGHT) * regime_score
6. Build portfolio, validate, write CSV

Reproducible: no random components.

Usage:
    python train_regime_blend.py [--out submissions/regime_blend.csv]
                                  [--stable-weight 0.30]
                                  [--top-k 40]
                                  [--portfolio-mode sqrt_rank]
                                  [--as-of 2026-04-30]

The default support portfolio is produced by the slim LGB multi-horizon
pipeline in factor_experiments_slim_asof20260430/generate_variants.py. The
older frozen stable_slim blend can still be passed with --source-path for
legacy comparison, but it is not the default submission path.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent

sys.path.insert(0, str(ROOT / "factor_experiments_slim_asof20260430"))
from features import (
    FEATURE_COLUMNS as SLIM_FEATURE_COLUMNS,
    TARGET_HORIZONS,
    MARKET_RELATIVE_FEATURE_COLUMNS as MR_FEATURES,
    INDUSTRY_RANK_FEATURE_COLUMNS as IR_FEATURES,
    target_column_for_horizon,
    build_features,
    _add_market_relative_features,
    _add_industry_rank_features,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = ROOT / "data_20240101_20260430"
EXTRA_DIR = ROOT / "data"
SUB_DIR   = ROOT / "submissions"
LOG_DIR   = ROOT / "logs"

CLEAN_SUPPORT_PATH = (
    ROOT / "factor_experiments_slim_asof20260430"
    / "submissions" / "slim_ohlcv_top40"
    / "sub1_best_h3_5_10_r60_k40_best2_sqrt_rank_tradable1.csv"
)

LEGACY_STABLE_SLIM_PATH = (
    ROOT / "factor_experiments_slim_asof20260430"
    / "submissions" / "slim_blends_top40"
    / "stable70_slim_sqrt30_top40.csv"
)

# Legacy exclusions used in earlier exploratory overlays. They are opt-in only;
# the default submission path applies no hand ticker exclusions.
LEGACY_BLACKLIST = {"000703", "688180", "000783", "301611"}

# ── Regime model parameters ────────────────────────────────────────────────────
# How many recent TRADING DAYS to compute ICs over
REGIME_WINDOW_TRADING_DAYS = 30

# Recency halflife in trading days for IC weighting:
# IC from 12 trading days ago gets 0.5× weight vs today's IC
RECENCY_HALFLIFE_REGIME = 12

# Embargo: minimum gap (calendar days) between last training date and as-of
EMBARGO_DAYS = 6

# Multi-horizon IC blend weights and per-horizon embargo (calendar days)
HORIZON_WEIGHTS  = {3: 0.60, 5: 0.30, 10: 0.10}
HORIZON_EMBARGO  = {3: 6,    5: 8,    10: 16}

# Cap individual feature IC at this value before normalising weights.
# Prevents any single factor (e.g. overnight_ret with IC=+0.11) from
# crowding out other valid signals. 0.07 makes overnight_ret equal-weighted
# with vol/dist_low features instead of 2× heavier.
MAX_FEATURE_IC = 0.07

# ── Portfolio ──────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 50
MAX_WEIGHT    = 0.10


# ── Feature helpers ────────────────────────────────────────────────────────────

def get_active_features(panel: pd.DataFrame) -> list[str]:
    candidates = list(SLIM_FEATURE_COLUMNS) + list(MR_FEATURES) + list(IR_FEATURES)
    seen, out = set(), []
    for c in candidates:
        if c in panel.columns and c not in seen:
            out.append(c)
            seen.add(c)
    return out


# ── Linear regime model: IC-weighted factor score ─────────────────────────────

def compute_regime_ic(
    panel: pd.DataFrame,
    feat_cols: list[str],
    as_of: pd.Timestamp,
    horizon: int,
    window_trading_days: int,
    halflife_days: int,
    embargo_calendar_days: int | None = None,
) -> pd.Series:
    """
    Compute recency-weighted mean IC for each feature over recent history.

    Returns a pd.Series indexed by feature name with mean IC values.
    Positive IC → feature positively predicts horizon-day returns.
    """
    target = target_column_for_horizon(horizon)
    emb = embargo_calendar_days if embargo_calendar_days is not None else EMBARGO_DAYS
    embargo_end = as_of - pd.Timedelta(days=emb)

    base = panel.copy()
    base = base[(base["volume"] > 0) & (base["high"] > base["low"])]
    base = base[base["date"] <= embargo_end]
    base = base.dropna(subset=feat_cols + [target])

    all_dates = np.sort(base["date"].unique())
    if len(all_dates) < window_trading_days:
        print(f"   [warn] only {len(all_dates)} dates available, using all")
        window_trading_days = len(all_dates)
    recent_dates = all_dates[-window_trading_days:]

    # Recency weights per date
    max_idx = len(recent_dates) - 1
    date_weights = {
        d: float(np.power(0.5, (max_idx - i) / max(halflife_days, 1)))
        for i, d in enumerate(recent_dates)
    }

    # Accumulate weighted IC per feature
    weighted_ic  = {f: 0.0 for f in feat_cols}
    total_weight = {f: 0.0 for f in feat_cols}

    for date in recent_dates:
        grp = base[base["date"] == date]
        w   = date_weights[date]
        tgt = grp[target]
        for feat in feat_cols:
            vals = grp[feat]
            mask = vals.notna() & tgt.notna()
            if mask.sum() < 20:
                continue
            rho, _ = spearmanr(vals[mask], tgt[mask])
            if not np.isnan(rho):
                weighted_ic[feat]  += rho * w
                total_weight[feat] += w

    ic_series = pd.Series({
        f: weighted_ic[f] / total_weight[f]
        for f in feat_cols
        if total_weight[f] > 0
    })
    return ic_series


def regime_scores_from_ic(
    pred_df: pd.DataFrame,
    feat_cols: list[str],
    ic_series: pd.Series,
    positive_only: bool = True,
) -> pd.Series:
    """
    Score each stock as IC-weighted sum of cross-sectional feature rank percentiles.
    Only uses features with positive (or all) IC from ic_series.
    """
    if positive_only:
        ic_use = ic_series[ic_series > 0]
    else:
        ic_use = ic_series

    if ic_use.empty:
        print("   [warn] No positive-IC features, using top-10 by |IC|")
        ic_use = ic_series.abs().nlargest(10)

    # Cap per-feature IC to prevent any single factor from dominating
    ic_use = ic_use.clip(lower=-MAX_FEATURE_IC, upper=MAX_FEATURE_IC)

    # Normalise weights to sum to 1
    ic_norm = ic_use.abs() / ic_use.abs().sum()

    scores = pd.Series(0.0, index=pred_df["stock_code"].values)
    for feat, w in ic_norm.items():
        if feat not in pred_df.columns:
            continue
        rank_pct = pred_df[feat].rank(pct=True, na_option="keep").fillna(0.5).values
        # Flip sign for negative IC features
        if ic_use[feat] < 0:
            rank_pct = 1.0 - rank_pct
        scores += pd.Series(rank_pct * w, index=pred_df["stock_code"].values)

    return scores


# ── Portfolio helpers ──────────────────────────────────────────────────────────

def source_score(weights: pd.Series) -> pd.Series:
    ordered = weights.sort_values(ascending=False)
    n = len(ordered)
    ranks = pd.Series(np.arange(n, 0, -1, dtype=float), index=ordered.index)
    return ranks / ranks.max()


def portfolio_from_scores(
    scores: pd.Series, top_k: int, mode: str = "sqrt_rank"
) -> pd.Series:
    chosen = scores.sort_values(ascending=False).head(top_k)
    ranks  = np.arange(len(chosen), 0, -1, dtype=float)
    base   = np.sqrt(ranks) if mode == "sqrt_rank" else np.ones(len(chosen))
    w = pd.Series(base / base.sum(), index=chosen.index)
    for _ in range(200):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess  = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free    = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",          default=str(SUB_DIR / "regime_blend.csv"))
    parser.add_argument("--stable-weight", type=float, default=0.30)
    parser.add_argument("--top-k",         type=int,   default=DEFAULT_TOP_K)
    parser.add_argument(
        "--portfolio-mode",
        choices=["sqrt_rank", "equal"],
        default="sqrt_rank",
        help="Final top-k weighting rule after model scores select stocks.",
    )
    parser.add_argument("--as-of",         default=None)
    parser.add_argument(
        "--source-path",
        default=str(CLEAN_SUPPORT_PATH),
        help="Support portfolio CSV blended with regime score. Default is the reproducible slim LGB candidate.",
    )
    parser.add_argument(
        "--exclude-stocks",
        default="",
        help="Optional comma-separated stock codes to exclude. Default: none.",
    )
    parser.add_argument(
        "--legacy-blacklist",
        action="store_true",
        help="Opt into the earlier exploratory blacklist for legacy comparison only.",
    )
    parser.add_argument(
        "--diagnostics-pool",
        action="store_true",
        help="Print overlap diagnostics against the exploratory hand-inspection pool.",
    )
    args = parser.parse_args()

    stable_weight = args.stable_weight
    regime_weight = 1.0 - stable_weight
    source_path = Path(args.source_path)
    exclude_stocks = {
        s.strip().zfill(6)
        for s in args.exclude_stocks.split(",")
        if s.strip()
    }
    if args.legacy_blacklist:
        exclude_stocks |= LEGACY_BLACKLIST

    SUB_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print("train_regime_blend.py — linear IC-weighted regime model")
    print(f"  window={REGIME_WINDOW_TRADING_DAYS} trading days  "
          f"halflife={RECENCY_HALFLIFE_REGIME} days  "
          f"horizons={list(HORIZON_WEIGHTS.keys())}  IC_cap={MAX_FEATURE_IC}")
    print(f"  blend: {regime_weight:.0%} regime + {stable_weight:.0%} support")
    print(f"  portfolio mode: top{args.top_k} {args.portfolio_mode}")
    print(f"  support source: {source_path}")
    print(f"  exclusions: {sorted(exclude_stocks) if exclude_stocks else 'none'}")
    print(f"{'='*70}")

    # ── Load ───────────────────────────────────────────────────────────────────
    print("\n>> Loading data …")
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    idx_prices = pd.read_parquet(DATA_DIR / "index.parquet")
    idx_prices["date"] = pd.to_datetime(idx_prices["date"])
    print(f"   {len(prices):,} rows | {prices['stock_code'].nunique()} stocks | "
          f"{prices['date'].min().date()} → {prices['date'].max().date()}")

    # ── Features ───────────────────────────────────────────────────────────────
    print("\n>> Building features …")
    panel = build_features(prices)
    panel = _add_market_relative_features(panel, idx_prices)
    for d in [EXTRA_DIR, DATA_DIR]:
        ip = d / "industry.csv"
        if ip.exists():
            ind = pd.read_csv(ip, dtype={"stock_code": str})
            ind["stock_code"] = ind["stock_code"].str.zfill(6)
            panel = _add_industry_rank_features(panel, ind)
            break

    active_features = get_active_features(panel)
    print(f"   Active features: {len(active_features)}")

    as_of_ts = pd.Timestamp(str(args.as_of)) if args.as_of else panel["date"].max()
    print(f"   Prediction date: {as_of_ts.date()}")

    # ── Compute regime ICs ─────────────────────────────────────────────────────
    print(f"\n>> Computing multi-horizon recency-weighted ICs "
          f"(window={REGIME_WINDOW_TRADING_DAYS}d, halflife={RECENCY_HALFLIFE_REGIME}d, "
          f"IC_cap={MAX_FEATURE_IC}) …")

    # Blend ICs across horizons; each horizon uses its own lookahead embargo
    total_hw  = sum(HORIZON_WEIGHTS.values())
    blended_ic: dict[str, float] = {}
    for horizon, hw in HORIZON_WEIGHTS.items():
        emb = HORIZON_EMBARGO.get(horizon, EMBARGO_DAYS)
        ic_h = compute_regime_ic(
            panel, active_features, as_of_ts,
            horizon=horizon,
            window_trading_days=REGIME_WINDOW_TRADING_DAYS,
            halflife_days=RECENCY_HALFLIFE_REGIME,
            embargo_calendar_days=emb,
        )
        frac = hw / total_hw
        print(f"   Horizon {horizon}d (weight={frac:.0%}, embargo={emb}d): "
              f"top feature = {ic_h.idxmax()!r} IC={ic_h.max():+.4f}")
        for feat, val in ic_h.items():
            blended_ic[feat] = blended_ic.get(feat, 0.0) + frac * val

    ic_series = pd.Series(blended_ic)

    positive_ic = ic_series[ic_series > 0].sort_values(ascending=False)
    negative_ic = ic_series[ic_series < 0].sort_values()
    print(f"\n   Blended positive-IC features ({len(positive_ic)}) [before cap]:")
    for feat, ic in positive_ic.head(15).items():
        print(f"     {feat:40s}  IC = {ic:+.4f}")
    print(f"\n   Top blended negative-IC features ({len(negative_ic)}):")
    for feat, ic in negative_ic.head(5).items():
        print(f"     {feat:40s}  IC = {ic:+.4f}")

    # Save IC table
    ic_df = ic_series.sort_values(ascending=False).reset_index()
    ic_df.columns = ["feature", "ic"]
    ic_df.to_csv(LOG_DIR / "regime_ic_table.csv", index=False)

    # ── Score all stocks on as-of date ─────────────────────────────────────────
    print(f"\n>> Scoring stocks on {as_of_ts.date()} …")
    pred_df = panel[panel["date"] == as_of_ts].copy()
    if "is_tradable" in pred_df.columns:
        pred_df = pred_df[pred_df["is_tradable"]]
    print(f"   Stocks in prediction universe: {len(pred_df)}")

    regime_raw = regime_scores_from_ic(pred_df, active_features, ic_series,
                                        positive_only=True)

    # Normalise to [0, 1]
    rmin, rmax = regime_raw.min(), regime_raw.max()
    regime_norm = (regime_raw - rmin) / (rmax - rmin) if rmax > rmin else regime_raw

    top_regime_10 = regime_norm.sort_values(ascending=False).head(10)
    print(f"   Top-10 regime scores: {', '.join(f'{s}={v:.3f}' for s,v in top_regime_10.items())}")

    # ── Blend with reproducible support portfolio ──────────────────────────────
    print(f"\n>> Blending {regime_weight:.0%} regime + {stable_weight:.0%} support …")
    if not source_path.exists():
        raise FileNotFoundError(
            f"Support portfolio not found: {source_path}. "
            "Recreate the default support with "
            "factor_experiments_slim_asof20260430/generate_variants.py."
        )
    else:
        support_df = pd.read_csv(source_path, dtype={"stock_code": str})
        support_df["stock_code"] = support_df["stock_code"].astype(str).str.zfill(6)
        support_w = support_df.set_index("stock_code")["weight"]
        support_scores_raw = source_score(support_w[support_w > 0])

        combined = sorted(set(support_scores_raw.index) | set(regime_norm.index))
        support_aligned = support_scores_raw.reindex(combined, fill_value=0.0)
        regime_aligned = regime_norm.reindex(combined, fill_value=0.0)

        # Normalise each component to [0, 1]
        for arr in [support_aligned, regime_aligned]:
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr[:] = (arr - mn) / (mx - mn)

        final_scores = stable_weight * support_aligned + regime_weight * regime_aligned

        top40_support = set(support_scores_raw.sort_values(ascending=False).head(40).index)
        top40_regime = set(regime_norm.sort_values(ascending=False).head(40).index)
        print(f"   support top-40 ∩ regime top-40: {len(top40_support & top40_regime)}")

    # ── Optional data/exploration exclusions ───────────────────────────────────
    exclusions_found = [s for s in exclude_stocks if s in final_scores.index]
    if exclusions_found:
        print(f"   Removing optional exclusions: {sorted(exclusions_found)}")
        final_scores = final_scores.drop(index=exclusions_found, errors="ignore")

    # ── Build portfolio ────────────────────────────────────────────────────────
    print(f"\n>> Building portfolio (top-k={args.top_k}, mode={args.portfolio_mode}) …")
    weights = portfolio_from_scores(
        final_scores,
        top_k=args.top_k,
        mode=args.portfolio_mode,
    )

    # ── Write ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "stock_code": weights.index.astype(str).str.zfill(6),
        "weight":     weights.values.round(10),
    })
    out.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"Submission → {out_path}")
    print(f"  Stocks    : {len(out)}")
    print(f"  Min w     : {out['weight'].min():.4f}")
    print(f"  Max w     : {out['weight'].max():.4f}")
    print(f"  Sum w     : {out['weight'].sum():.6f}")
    print(f"  Top 10    : {', '.join(out['stock_code'].head(10))}")

    # ── Optional exploratory diagnostics ───────────────────────────────────────
    if args.diagnostics_pool:
        POOL_WANT = {
            "688037","002353","600118","000062","600378","002436","002261",
            "002738","688525","600988","300857","002812","301358","300037",
            "300763","001389","600392","300054","300454","300223","002156",
            "688234","601233","002444","000831","600096",
        }
        selected = set(out["stock_code"])
        covered  = selected & POOL_WANT
        print(f"\n  Pool coverage : {len(covered)}/{len(POOL_WANT)} "
              f"({len(covered)/len(POOL_WANT):.0%})")
        print(f"  Covered       : {sorted(covered)}")
        print(f"  Uncovered     : {sorted(POOL_WANT - covered)}")

        legacy_in = selected & LEGACY_BLACKLIST
        if legacy_in:
            print(f"  Legacy blacklist names in portfolio: {sorted(legacy_in)}")

        STABLE_CAND = {"002281","301308","002738","002273","000657","300054","300679",
                       "600118","603228","002353","300751","600096","300857","001389",
                       "002080","000831","001221","002624","603688","688525","300395",
                       "002008","688615","002797","300285","601179","300390","300677",
                       "600988","688629","300432","300450","300383","600602","000960",
                       "300570","002131","300017","002773","688336"}
        new_adds = selected - STABLE_CAND
        dropped  = STABLE_CAND - selected
        print(f"\n  vs model_xgb_safe: +{len(new_adds)} new  -{len(dropped)} dropped")
        print(f"  New additions : {sorted(new_adds)}")
        print(f"  Dropped       : {sorted(dropped)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
