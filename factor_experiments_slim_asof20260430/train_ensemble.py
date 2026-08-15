"""
train_ensemble.py  —  LightGBM with walk-forward CV (matching sub1_best).

Usage:
    python train_ensemble.py --out submissions/sub1_best.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

from features import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, prediction_frame,
    target_column_for_horizon,
)

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
DEFAULT_PRICES_PATH = (
    DATA_DIR / "prices.parquet"
    if (DATA_DIR / "prices.parquet").exists()
    else ROOT.parent / "data" / "prices.parquet"
)
DEFAULT_INDEX_PATH = (
    DATA_DIR / "index.parquet"
    if (DATA_DIR / "index.parquet").exists()
    else ROOT.parent / "data" / "index.parquet"
)
DEFAULT_INDUSTRY_PATH = (
    DATA_DIR / "industry.csv"
    if (DATA_DIR / "industry.csv").exists()
    else ROOT.parent / "data" / "industry.csv"
)
MODEL_DIR = ROOT / "models"
LOG_DIR   = ROOT / "logs"
SUB_DIR   = ROOT / "submissions"
for d in [MODEL_DIR, LOG_DIR, SUB_DIR]:
    d.mkdir(exist_ok=True)

# Walk-forward parameters
TRAIN_DAYS   = 180
VAL_DAYS     = 20
EMBARGO_DAYS = 5
STEP_DAYS    = 20
N_FOLDS      = 8
N_ENSEMBLE   = 3

# Portfolio rules
MIN_STOCKS    = 30
MAX_WEIGHT    = 0.10
DEFAULT_TOP_K = 50
DEFAULT_PORTFOLIO_WEIGHTING = "rank"
DEFAULT_FOLD_WEIGHTING = "equal"
DEFAULT_RECENT_TRADABLE_DAYS = 1
DEFAULT_RECENCY_HALFLIFE_DAYS = 0.0

# LightGBM hyperparameters
LGB_PARAMS = dict(
    objective         = "regression",
    metric            = "rmse",
    boosting_type     = "gbdt",
    num_leaves        = 63,
    max_depth         = -1,
    learning_rate     = 0.05,
    n_estimators      = 800,
    subsample         = 0.8,
    subsample_freq    = 1,
    colsample_bytree  = 0.7,
    min_child_samples = 20,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)


def rank_ic(y_true, y_pred, dates):
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 10:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    if not ics:
        return float("nan"), float("nan")
    arr = np.array(ics)
    return float(arr.mean()), float(arr.mean() / (arr.std() + 1e-9))


def recency_sample_weight(df, halflife_days=DEFAULT_RECENCY_HALFLIFE_DAYS):
    """Return training sample weights that emphasize recent trading dates.

    halflife_days is measured in trading-day positions, not calendar days.
    A value <= 0 preserves the original unweighted training behavior.
    """
    if halflife_days is None or halflife_days <= 0:
        return None
    dates = np.sort(df["date"].unique())
    if len(dates) == 0:
        return None
    date_to_pos = {d: i for i, d in enumerate(dates)}
    pos = df["date"].map(date_to_pos).to_numpy(dtype=float)
    age = float(len(dates) - 1) - pos
    weights = np.power(0.5, age / float(halflife_days))
    return weights / weights.mean()


def train_lgb(tr, vl, target_column=TARGET_COLUMN,
              recency_halflife_days=DEFAULT_RECENCY_HALFLIFE_DAYS):
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    sample_weight = recency_sample_weight(tr, recency_halflife_days)
    model.fit(
        tr[FEATURE_COLUMNS], tr[target_column],
        sample_weight=sample_weight,
        eval_set=[(vl[FEATURE_COLUMNS], vl[target_column])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return model


def walk_forward_cv(train_pool, target_column=TARGET_COLUMN,
                    target_horizon=FORWARD_HORIZON,
                    recency_halflife_days=DEFAULT_RECENCY_HALFLIFE_DAYS):
    all_dates = np.sort(train_pool["date"].unique())
    n_dates   = len(all_dates)

    required = TRAIN_DAYS + EMBARGO_DAYS + VAL_DAYS
    if n_dates < required + 10:
        raise RuntimeError(
            f"Need {required + 10} trading days; got {n_dates}.\n"
            "Run: python download_data.py --start 20250101 --end 20260421"
        )

    folds = []
    val_end_idx = n_dates - 1
    for _ in range(N_FOLDS):
        ve = val_end_idx
        vs = ve - VAL_DAYS + 1
        te = vs - EMBARGO_DAYS - 1
        ts = max(0, te - TRAIN_DAYS + 1)
        if ts < 0 or te <= ts + 10:
            break
        folds.append((ts, te, vs, ve))
        val_end_idx -= STEP_DAYS
    folds = list(reversed(folds))

    models  = []
    cv_rows = []

    print(f"\n{'='*65}")
    print(f"Walk-forward CV: {len(folds)} folds | LGB only | target={target_column}")
    print(f"  Train: {TRAIN_DAYS} days | Val: {VAL_DAYS} | Embargo: {EMBARGO_DAYS}")
    if recency_halflife_days and recency_halflife_days > 0:
        print(f"  Recency halflife: {recency_halflife_days:g} trading days")
    print(f"{'='*65}")

    for i, (ts, te, vs, ve) in enumerate(folds, 1):
        t_start = pd.Timestamp(all_dates[ts])
        t_end   = pd.Timestamp(all_dates[te])
        v_start = pd.Timestamp(all_dates[vs])
        v_end   = pd.Timestamp(all_dates[ve])

        tr = train_pool[(train_pool["date"] >= t_start) & (train_pool["date"] <= t_end)]
        vl = train_pool[(train_pool["date"] >= v_start) & (train_pool["date"] <= v_end)]

        print(f"\nFold {i}/{len(folds)}")
        print(f"  Train: {t_start.date()} -> {t_end.date()}  ({len(tr):,} rows)")
        print(f"  Val  : {v_start.date()} -> {v_end.date()}  ({len(vl):,} rows)")

        model = train_lgb(
            tr, vl,
            target_column=target_column,
            recency_halflife_days=recency_halflife_days,
        )
        preds = model.predict(vl[FEATURE_COLUMNS])
        ic, icir = rank_ic(vl[target_column].values, preds, vl["date"].values)
        models.append(model)
        print(f"  LGB: IC = {ic:+.4f}  |  ICIR = {icir:+.4f}")

        model_name = (
            f"lgbm_fold{i:02d}.txt"
            if target_horizon == FORWARD_HORIZON and not recency_halflife_days
            else f"lgbm_h{int(target_horizon):02d}_fold{i:02d}.txt"
        )
        model.booster_.save_model(str(MODEL_DIR / model_name))

        cv_rows.append({
            "fold": i,
            "target_horizon": int(target_horizon),
            "target_column": target_column,
            "recency_halflife_days": recency_halflife_days,
            "train_start": str(t_start.date()),
            "train_end":   str(t_end.date()),
            "val_start":   str(v_start.date()),
            "val_end":     str(v_end.date()),
            "ic_lgb":      round(ic, 4),
            "icir_lgb":    round(icir, 4),
        })

    cv_df = pd.DataFrame(cv_rows)
    cv_path = (
        LOG_DIR / "cv_results.csv"
        if target_horizon == FORWARD_HORIZON and not recency_halflife_days
        else LOG_DIR / f"cv_results_h{int(target_horizon):02d}.csv"
    )
    cv_df.to_csv(cv_path, index=False)

    ics = cv_df["ic_lgb"].dropna()
    print(f"\n{'='*65}")
    print(f"CV Summary")
    print(f"  Mean IC : {ics.mean():+.4f}")
    print(f"  Std  IC : {ics.std():.4f}")
    print(f"  ICIR    : {ics.mean() / (ics.std() + 1e-9):+.4f}")
    print(f"  Saved   : {cv_path}")
    print(f"{'='*65}\n")

    return models, cv_df


def select_ensemble(models, cv_df, method=DEFAULT_FOLD_WEIGHTING,
                    n_ensemble=N_ENSEMBLE):
    """Select recent fold models and assign prediction weights.

    The default is the original sub1_best behavior: equal-weight the last
    N_ENSEMBLE folds. Other methods are small portfolio experiments, not a
    change to the protected baseline.
    """
    recent_models = list(models[-n_ensemble:])
    recent_cv = cv_df.tail(n_ensemble).reset_index(drop=True)

    if method == "last":
        selected_models = [recent_models[-1]]
        weights = np.array([1.0])
        selected_cv = recent_cv.tail(1)
    elif method == "best2":
        n_pick = min(2, len(recent_models))
        order = recent_cv["ic_lgb"].to_numpy().argsort()[::-1][:n_pick]
        selected_models = [recent_models[i] for i in order]
        selected_cv = recent_cv.iloc[order]
        weights = np.ones(len(selected_models), dtype=float)
    else:
        selected_models = recent_models
        selected_cv = recent_cv
        if method == "positive_ic":
            weights = selected_cv["ic_lgb"].clip(lower=0).to_numpy(dtype=float)
            if weights.sum() <= 0:
                weights = np.ones(len(selected_models), dtype=float)
        else:
            weights = np.ones(len(selected_models), dtype=float)

    weights = weights / weights.sum()
    return selected_models, weights, selected_cv


def _apply_group_cap(weights, groups, max_group_weight):
    if max_group_weight is None or max_group_weight >= 1.0:
        return weights

    groups = groups.reindex(weights.index).fillna("unknown")
    w = weights.copy()
    for _ in range(100):
        group_sums = w.groupby(groups).sum()
        over_groups = group_sums[group_sums > max_group_weight + 1e-10]
        if over_groups.empty:
            break

        before = w.sum()
        for group, group_sum in over_groups.items():
            idx = groups[groups == group].index.intersection(w.index)
            w.loc[idx] *= max_group_weight / group_sum

        leftover = before - w.sum()
        if leftover <= 1e-12:
            break

        group_sums = w.groupby(groups).sum()
        free_idx = groups[groups.map(group_sums) < max_group_weight - 1e-10].index
        free_idx = free_idx.intersection(w.index)
        if len(free_idx) == 0 or w.loc[free_idx].sum() <= 0:
            break
        w.loc[free_idx] += leftover * w.loc[free_idx] / w.loc[free_idx].sum()

    w /= w.sum()
    return w


def build_portfolio(scores, top_k=DEFAULT_TOP_K,
                    weighting=DEFAULT_PORTFOLIO_WEIGHTING,
                    is_tradable=None,
                    industry=None,
                    max_industry_weight=None):
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k ({top_k}) must be >= {MIN_STOCKS}")
    scores = scores.copy()
    if is_tradable is not None:
        not_tradable = is_tradable[~is_tradable].index
        scores[scores.index.isin(not_tradable)] -= 1e6
    chosen = scores.sort_values(ascending=False).head(top_k)
    n = len(chosen)
    ranks = np.arange(n, 0, -1, dtype=float)
    if weighting == "equal":
        base = np.ones(n, dtype=float)
    elif weighting == "sqrt_rank":
        base = np.sqrt(ranks)
    else:
        base = ranks
    w = pd.Series(base / base.sum(), index=chosen.index)
    for _ in range(100):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    w /= w.sum()
    if industry is not None and max_industry_weight is not None:
        w = _apply_group_cap(w, industry, max_industry_weight)
        # Re-apply single-name cap after group redistribution.
        for _ in range(100):
            over = w > MAX_WEIGHT
            if not over.any():
                break
            excess = (w[over] - MAX_WEIGHT).sum()
            w[over] = MAX_WEIGHT
            free = ~over
            if not free.any():
                break
            w[free] += excess * w[free] / w[free].sum()
        w /= w.sum()
    assert abs(w.sum() - 1.0) < 1e-4
    assert (w <= MAX_WEIGHT + 1e-6).all()
    assert (w > 0).sum() >= MIN_STOCKS
    if industry is not None and max_industry_weight is not None:
        group_sums = w.groupby(industry.reindex(w.index).fillna("unknown")).sum()
        assert (group_sums <= max_industry_weight + 1e-6).all()
    return w


def recent_tradability(prices, as_of, lookback_days=DEFAULT_RECENT_TRADABLE_DAYS):
    """Require stocks to be tradable over the most recent lookback window."""
    if lookback_days <= 0 or not {"volume", "high", "low"}.issubset(prices.columns):
        return None

    as_of = pd.Timestamp(as_of)
    dates = np.sort(prices.loc[prices["date"] <= as_of, "date"].unique())
    if len(dates) == 0:
        return None

    recent_dates = dates[-lookback_days:]
    recent = prices[prices["date"].isin(recent_dates)].copy()
    recent["stock_code"] = recent["stock_code"].astype(str).str.zfill(6)
    recent["tradable"] = (recent["volume"] > 0) & (recent["high"] > recent["low"])
    return recent.groupby("stock_code")["tradable"].all()


def load_industry(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Provide --industry-file or create data/industry.csv."
        )
    industry = pd.read_csv(path, dtype={"stock_code": str})
    if "industry" not in industry.columns:
        raise ValueError(f"{path} must contain columns: stock_code, industry")
    industry["stock_code"] = industry["stock_code"].astype(str).str.zfill(6)
    return industry[["stock_code", "industry"]].drop_duplicates("stock_code")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DEFAULT_PRICES_PATH))
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--industry-file", default=str(DEFAULT_INDUSTRY_PATH))
    parser.add_argument("--as-of",  default=None)
    parser.add_argument("--top-k",  type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--out",    default=str(SUB_DIR / "submission.csv"))
    parser.add_argument(
        "--market-relative",
        action="store_true",
        help="Add CSI500-relative return/beta/residual features.",
    )
    parser.add_argument(
        "--industry-ranks",
        action="store_true",
        help="Add within-industry rank features from --industry-file.",
    )
    parser.add_argument(
        "--fold-weighting",
        choices=["equal", "positive_ic", "best2", "last"],
        default=DEFAULT_FOLD_WEIGHTING,
        help="How to combine recent CV fold models. 'equal' reproduces sub1_best.",
    )
    parser.add_argument(
        "--portfolio-weighting",
        choices=["rank", "sqrt_rank", "equal"],
        default=DEFAULT_PORTFOLIO_WEIGHTING,
        help="How to weight selected names after ranking by model score.",
    )
    parser.add_argument(
        "--recent-tradable-days",
        type=int,
        default=DEFAULT_RECENT_TRADABLE_DAYS,
        help="Exclude stocks not tradable on all of the most recent N trading days.",
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=FORWARD_HORIZON,
        help="Forward-return target horizon in trading days, e.g. 3 or 5.",
    )
    parser.add_argument(
        "--recency-halflife-days",
        type=float,
        default=DEFAULT_RECENCY_HALFLIFE_DAYS,
        help="Trading-day halflife for training sample weights; 0 disables.",
    )
    parser.add_argument(
        "--max-industry-weight",
        type=float,
        default=None,
        help="Optional cap on total portfolio weight per industry, e.g. 0.35.",
    )
    args = parser.parse_args()

    print(f"\n>> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    print(f"   {len(prices):,} rows | {prices['stock_code'].nunique()} stocks | "
          f"{prices['date'].min().date()} -> {prices['date'].max().date()}")
    index_prices = None
    if args.market_relative:
        print(f">> Loading {args.index}")
        index_prices = pd.read_parquet(args.index)
        index_prices["date"] = pd.to_datetime(index_prices["date"])
    industry = None
    if args.industry_ranks or args.max_industry_weight is not None:
        print(f">> Loading {args.industry_file}")
        industry = load_industry(args.industry_file)

    print("\n>> Building features ...")
    panel = build_features(
        prices,
        index_prices=index_prices,
        use_market_relative=args.market_relative,
        industry=industry,
        use_industry_ranks=args.industry_ranks,
    )
    print(f"   Panel: {len(panel):,} rows | {len(FEATURE_COLUMNS)} features")

    as_of_ts     = pd.Timestamp(str(args.as_of)) if args.as_of else panel["date"].max()
    all_dates    = np.sort(panel["date"].unique())
    as_of_idx    = int(np.searchsorted(all_dates, np.datetime64(as_of_ts)))
    target_column = target_column_for_horizon(args.target_horizon)
    cutoff_idx   = max(0, as_of_idx - args.target_horizon)
    train_cutoff = pd.Timestamp(all_dates[cutoff_idx])

    train_pool = training_frame(
        panel, max_date=train_cutoff, target_column=target_column
    )
    print(
        f"   Training pool: {len(train_pool):,} rows up to {train_cutoff.date()} "
        f"| target={target_column}"
    )

    models, cv_df = walk_forward_cv(
        train_pool,
        target_column=target_column,
        target_horizon=args.target_horizon,
        recency_halflife_days=args.recency_halflife_days,
    )
    ensemble, ensemble_weights, selected_cv = select_ensemble(
        models, cv_df, method=args.fold_weighting
    )

    print(">> Predicting ...")
    pred_df = prediction_frame(panel, as_of=args.as_of)
    if pred_df.empty:
        raise RuntimeError("No prediction rows.")

    pred_date = pred_df["date"].iloc[0]
    print(f"   Date: {pred_date.date()} | Stocks: {len(pred_df)}")

    all_preds = np.stack([m.predict(pred_df[FEATURE_COLUMNS]) for m in ensemble])
    pred_df = pred_df.assign(score=np.average(all_preds, axis=0, weights=ensemble_weights))
    scores = pred_df.set_index("stock_code")["score"]
    scores.index = scores.index.astype(str).str.zfill(6)

    is_tradable = (
        pred_df.set_index("stock_code")["is_tradable"]
        if "is_tradable" in pred_df.columns else None
    )
    if is_tradable is not None:
        is_tradable.index = is_tradable.index.astype(str).str.zfill(6)
    recent_ok = recent_tradability(
        prices, pred_date, lookback_days=args.recent_tradable_days
    )
    if recent_ok is not None:
        recent_ok = recent_ok.reindex(scores.index).fillna(False)
        is_tradable = recent_ok if is_tradable is None else (is_tradable & recent_ok)
    industry_series = None
    if industry is not None:
        industry_series = industry.set_index("stock_code")["industry"].reindex(scores.index)

    weights = build_portfolio(
        scores,
        top_k=args.top_k,
        weighting=args.portfolio_weighting,
        is_tradable=is_tradable,
        industry=industry_series,
        max_industry_weight=args.max_industry_weight,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
    out.to_csv(out_path, index=False)

    print(f"\n>> Submission -> {out_path}")
    print(f"   Ensemble : {len(ensemble)} LGB models")
    print(f"   Target   : {target_column}")
    if args.recency_halflife_days > 0:
        print(f"   Recency halflife: {args.recency_halflife_days:g} trading days")
    print(f"   Fold mode: {args.fold_weighting}")
    print(f"   Fold wts : {dict(zip(selected_cv['fold'].astype(int), ensemble_weights.round(4)))}")
    print(f"   Port mode: {args.portfolio_weighting}")
    print(f"   Recent tradable days: {args.recent_tradable_days}")
    if args.max_industry_weight is not None:
        print(f"   Max industry weight: {args.max_industry_weight:.0%}")
    print(f"   Stocks   : {len(out)}")
    print(f"   Min w    : {out['weight'].min():.4f}")
    print(f"   Max w    : {out['weight'].max():.4f}")
    print(f"   Sum w    : {out['weight'].sum():.6f}")
    print(f"\nNext step: python validate_submission.py {out_path}")


if __name__ == "__main__":
    main()
