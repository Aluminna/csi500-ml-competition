"""
Generate model-family experiments for the CSI500 portfolio.

This is intentionally separate from train_ensemble.py/generate_variants.py so
the current candidate workflow stays intact.  The experiments here reuse the
same features, fold schedule, horizon ensemble, and portfolio construction,
but swap the learning target/model:

    rank_lgb              LGBMRegressor on daily cross-sectional target rank
    zscore_lgb            LGBMRegressor on daily cross-sectional target z-score
    ranker_lgb            LGBMRanker with daily stock universe as each query
    rank_lgb_xgb_blend    average of LGBMRegressor and XGBRegressor rank models
    rank_xgb_conservative shallow, strongly regularized XGBoost rank model
    industry_resid_lgb    LGBMRegressor on target minus same-industry mean
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception:  # noqa: BLE001 - handled when the blend experiment is requested.
    xgb = None

from features import (
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    build_features,
    prediction_frame,
    target_column_for_horizon,
    training_frame,
)
from train_ensemble import (
    DEFAULT_INDEX_PATH,
    DEFAULT_INDUSTRY_PATH,
    DEFAULT_PRICES_PATH,
    DEFAULT_RECENCY_HALFLIFE_DAYS,
    EMBARGO_DAYS,
    LGB_PARAMS,
    LOG_DIR,
    N_ENSEMBLE,
    N_FOLDS,
    STEP_DAYS,
    SUB_DIR,
    TRAIN_DAYS,
    VAL_DAYS,
    build_portfolio,
    load_industry,
    rank_ic,
    recent_tradability,
    recency_sample_weight,
    select_ensemble,
)


EXPERIMENTS = [
    "rank_lgb",
    "zscore_lgb",
    "ranker_lgb",
    "rank_lgb_xgb_blend",
    "rank_xgb_conservative",
    "industry_resid_lgb",
]


def _parse_csv_list(raw: str, cast=str):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def _horizon_label(horizons: list[int]) -> str:
    return "h" + "_".join(str(int(h)) for h in horizons)


def _format_halflife(value: float) -> str:
    if value <= 0:
        return ""
    if float(value).is_integer():
        return f"r{int(value)}"
    return f"r{str(value).replace('.', 'p')}"


def _variant_name(experiment: str, horizons: list[int], recency_halflife_days: float,
                  top_k: int, fold_mode: str, port_mode: str, recent_days: int) -> str:
    recency = _format_halflife(recency_halflife_days)
    parts = ["sub1_exp", experiment, _horizon_label(horizons)]
    if recency:
        parts.append(recency)
    parts.extend([f"k{top_k}", fold_mode, port_mode, f"tradable{recent_days}"])
    return "_".join(parts) + ".csv"


def _make_folds(train_pool: pd.DataFrame):
    all_dates = np.sort(train_pool["date"].unique())
    n_dates = len(all_dates)
    required = TRAIN_DAYS + EMBARGO_DAYS + VAL_DAYS
    if n_dates < required + 10:
        raise RuntimeError(f"Need {required + 10} trading days; got {n_dates}.")

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
    return list(reversed(folds)), all_dates


def _daily_rank_target(df: pd.DataFrame, base_col: str) -> pd.Series:
    return df.groupby("date")[base_col].rank(method="average", pct=True) - 0.5


def _daily_zscore_target(df: pd.DataFrame, base_col: str) -> pd.Series:
    grouped = df.groupby("date")[base_col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return ((df[base_col] - mean) / std).clip(-5, 5).fillna(0.0)


def _daily_ranker_label(df: pd.DataFrame, base_col: str, bins: int = 5) -> pd.Series:
    pct = df.groupby("date")[base_col].rank(method="first", pct=True)
    label = np.floor(pct * bins).astype(int)
    return label.clip(0, bins - 1)


def _industry_residual_target(
    df: pd.DataFrame,
    base_col: str,
    industry: pd.DataFrame,
) -> pd.Series:
    ind = industry.set_index("stock_code")["industry"]
    groups = df["stock_code"].astype(str).str.zfill(6).map(ind).fillna("unknown")
    means = df.groupby([df["date"], groups])[base_col].transform("mean")
    return df[base_col] - means


def _prepare_targets(
    train_pool: pd.DataFrame,
    base_col: str,
    experiment: str,
    industry: pd.DataFrame | None,
) -> pd.DataFrame:
    df = train_pool.copy()
    if experiment in {"rank_lgb", "rank_lgb_xgb_blend", "rank_xgb_conservative"}:
        df["exp_target"] = _daily_rank_target(df, base_col)
    elif experiment == "zscore_lgb":
        df["exp_target"] = _daily_zscore_target(df, base_col)
    elif experiment == "ranker_lgb":
        df["exp_target"] = _daily_ranker_label(df, base_col)
    elif experiment == "industry_resid_lgb":
        if industry is None:
            raise ValueError("industry_resid_lgb requires data/industry.csv")
        df["exp_target"] = _industry_residual_target(df, base_col, industry)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    return df.dropna(subset=["exp_target"]).reset_index(drop=True)


class BlendModel:
    def __init__(self, lgb_model, xgb_model, lgb_weight: float = 0.5):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.lgb_weight = lgb_weight

    def predict(self, x):
        return (
            self.lgb_weight * self.lgb_model.predict(x)
            + (1.0 - self.lgb_weight) * self.xgb_model.predict(x)
        )


def _fit_lgb_regressor(tr: pd.DataFrame, vl: pd.DataFrame,
                       recency_halflife_days: float):
    params = LGB_PARAMS.copy()
    params["objective"] = "regression"
    params["metric"] = "rmse"
    model = lgb.LGBMRegressor(**params)
    sample_weight = recency_sample_weight(tr, recency_halflife_days)
    model.fit(
        tr[FEATURE_COLUMNS],
        tr["exp_target"],
        sample_weight=sample_weight,
        eval_set=[(vl[FEATURE_COLUMNS], vl["exp_target"])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return model


def _fit_xgb_regressor(tr: pd.DataFrame, vl: pd.DataFrame):
    if xgb is None:
        raise RuntimeError("xgboost import failed; cannot run rank_lgb_xgb_blend.")
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=2026,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(
        tr[FEATURE_COLUMNS],
        tr["exp_target"],
        eval_set=[(vl[FEATURE_COLUMNS], vl["exp_target"])],
        verbose=False,
    )
    return model


def _fit_conservative_xgb_regressor(
    tr: pd.DataFrame,
    vl: pd.DataFrame,
    recency_halflife_days: float,
):
    """Fit a low-capacity XGB model intended as an ensemble diversifier."""
    if xgb is None:
        raise RuntimeError("xgboost import failed; cannot run rank_xgb_conservative.")
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=260,
        max_depth=2,
        learning_rate=0.025,
        subsample=0.70,
        colsample_bytree=0.65,
        min_child_weight=30,
        gamma=0.05,
        reg_alpha=2.0,
        reg_lambda=8.0,
        random_state=20260503,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    sample_weight = recency_sample_weight(tr, recency_halflife_days)
    model.fit(
        tr[FEATURE_COLUMNS],
        tr["exp_target"],
        sample_weight=sample_weight,
        eval_set=[(vl[FEATURE_COLUMNS], vl["exp_target"])],
        verbose=False,
    )
    return model


def _group_sizes_by_date(df: pd.DataFrame) -> list[int]:
    return df.groupby("date", sort=True).size().astype(int).tolist()


def _sort_by_group(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["date", "stock_code"]).reset_index(drop=True)


def _fit_lgb_ranker(tr: pd.DataFrame, vl: pd.DataFrame):
    tr = _sort_by_group(tr)
    vl = _sort_by_group(vl)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.04,
        n_estimators=500,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        tr[FEATURE_COLUMNS],
        tr["exp_target"].astype(int),
        group=_group_sizes_by_date(tr),
        eval_set=[(vl[FEATURE_COLUMNS], vl["exp_target"].astype(int))],
        eval_group=[_group_sizes_by_date(vl)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return model


@dataclass
class HorizonBundle:
    models: list
    cv: pd.DataFrame


def train_experiment(
    train_pool: pd.DataFrame,
    experiment: str,
    base_target_column: str,
    target_horizon: int,
    recency_halflife_days: float,
    industry: pd.DataFrame | None,
) -> HorizonBundle:
    work = _prepare_targets(train_pool, base_target_column, experiment, industry)
    folds, all_dates = _make_folds(work)
    models = []
    rows = []

    print(f"\n{'=' * 72}")
    print(f"Experiment={experiment} | target={base_target_column} | h={target_horizon}")
    print(f"Train target: exp_target | folds={len(folds)}")
    if recency_halflife_days > 0 and experiment != "ranker_lgb":
        print(f"Recency halflife: {recency_halflife_days:g} trading days")
    print(f"{'=' * 72}")

    for i, (ts, te, vs, ve) in enumerate(folds, 1):
        t_start = pd.Timestamp(all_dates[ts])
        t_end = pd.Timestamp(all_dates[te])
        v_start = pd.Timestamp(all_dates[vs])
        v_end = pd.Timestamp(all_dates[ve])
        tr = work[(work["date"] >= t_start) & (work["date"] <= t_end)]
        vl = work[(work["date"] >= v_start) & (work["date"] <= v_end)]

        print(f"Fold {i}/{len(folds)} | train {t_start.date()}->{t_end.date()} | "
              f"val {v_start.date()}->{v_end.date()}")

        if experiment == "ranker_lgb":
            model = _fit_lgb_ranker(tr, vl)
        elif experiment == "rank_lgb_xgb_blend":
            lgb_model = _fit_lgb_regressor(tr, vl, recency_halflife_days)
            xgb_model = _fit_xgb_regressor(tr, vl)
            model = BlendModel(lgb_model, xgb_model, lgb_weight=0.5)
        elif experiment == "rank_xgb_conservative":
            model = _fit_conservative_xgb_regressor(
                tr, vl, recency_halflife_days
            )
        else:
            model = _fit_lgb_regressor(tr, vl, recency_halflife_days)

        preds = model.predict(vl[FEATURE_COLUMNS])
        ic, icir = rank_ic(vl[base_target_column].values, preds, vl["date"].values)
        models.append(model)
        rows.append({
            "fold": i,
            "experiment": experiment,
            "target_horizon": int(target_horizon),
            "target_column": base_target_column,
            "train_start": str(t_start.date()),
            "train_end": str(t_end.date()),
            "val_start": str(v_start.date()),
            "val_end": str(v_end.date()),
            "ic_lgb": round(ic, 4),
            "icir_lgb": round(icir, 4),
        })
        print(f"  raw target IC = {ic:+.4f} | ICIR = {icir:+.4f}")

    cv = pd.DataFrame(rows)
    mean_ic = cv["ic_lgb"].mean()
    print(f">> {experiment} h{target_horizon}: mean raw-target IC {mean_ic:+.4f}")
    return HorizonBundle(models=models, cv=cv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DEFAULT_PRICES_PATH))
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--industry-file", default=str(DEFAULT_INDUSTRY_PATH))
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--target-horizons", default="3,5,10")
    parser.add_argument("--horizon-weights", default="")
    parser.add_argument("--recency-halflife-days", type=float, default=DEFAULT_RECENCY_HALFLIFE_DAYS)
    parser.add_argument("--top-ks", default="40,50")
    parser.add_argument("--portfolio-modes", default="equal,sqrt_rank")
    parser.add_argument("--fold-modes", default="best2,positive_ic")
    parser.add_argument("--recent-days", default="1")
    parser.add_argument("--market-relative", action="store_true")
    parser.add_argument("--industry-ranks", action="store_true")
    args = parser.parse_args()

    experiments = _parse_csv_list(args.experiments)
    target_horizons = _parse_csv_list(args.target_horizons, int)
    if args.horizon_weights.strip():
        horizon_weights = np.array(_parse_csv_list(args.horizon_weights, float), dtype=float)
        if len(horizon_weights) != len(target_horizons):
            raise ValueError("--horizon-weights length must match --target-horizons")
    else:
        horizon_weights = np.ones(len(target_horizons), dtype=float)
    horizon_weights = horizon_weights / horizon_weights.sum()
    top_ks = _parse_csv_list(args.top_ks, int)
    port_modes = _parse_csv_list(args.portfolio_modes)
    fold_modes = _parse_csv_list(args.fold_modes)
    recent_days_list = _parse_csv_list(args.recent_days, int)

    industry_needed = (
        "industry_resid_lgb" in experiments
        or args.industry_ranks
    )
    industry = load_industry(args.industry_file) if industry_needed else None

    print(f"\n>> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_prices = None
    if args.market_relative:
        print(f">> Loading {args.index}")
        index_prices = pd.read_parquet(args.index)
        index_prices["date"] = pd.to_datetime(index_prices["date"])

    print("\n>> Building features")
    panel = build_features(
        prices,
        index_prices=index_prices,
        use_market_relative=args.market_relative,
        industry=industry,
        use_industry_ranks=args.industry_ranks,
    )
    print(f"   panel={len(panel):,} rows | features={len(FEATURE_COLUMNS)}")

    as_of_ts = pd.Timestamp(str(args.as_of))
    all_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(all_dates, np.datetime64(as_of_ts)))
    pred_df = prediction_frame(panel, as_of=args.as_of)
    if pred_df.empty:
        raise RuntimeError("No prediction rows.")
    pred_date = pred_df["date"].iloc[0]
    print(f">> Predict date: {pred_date.date()} | stocks={len(pred_df)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    rows = []
    cv_rows = []
    for experiment in experiments:
        bundles: dict[int, HorizonBundle] = {}
        for horizon in target_horizons:
            target_column = target_column_for_horizon(horizon)
            cutoff_idx = max(0, as_of_idx - horizon)
            train_cutoff = pd.Timestamp(all_dates[cutoff_idx])
            train_pool = training_frame(panel, max_date=train_cutoff, target_column=target_column)
            print(
                f"\n>> {experiment} h{horizon}: training rows={len(train_pool):,} "
                f"up to {train_cutoff.date()}"
            )
            bundle = train_experiment(
                train_pool=train_pool,
                experiment=experiment,
                base_target_column=target_column,
                target_horizon=horizon,
                recency_halflife_days=args.recency_halflife_days,
                industry=industry,
            )
            bundles[horizon] = bundle
            cv_rows.append(bundle.cv)

        for fold_mode in fold_modes:
            horizon_preds = []
            selected_parts = []
            weight_parts = []
            for horizon in target_horizons:
                bundle = bundles[horizon]
                ensemble, ensemble_weights, selected_cv = select_ensemble(
                    bundle.models,
                    bundle.cv,
                    method=fold_mode,
                    n_ensemble=N_ENSEMBLE,
                )
                preds = np.stack([m.predict(pred_df[FEATURE_COLUMNS]) for m in ensemble])
                horizon_preds.append(np.average(preds, axis=0, weights=ensemble_weights))
                selected_parts.append(
                    f"h{horizon}:" + ",".join(selected_cv["fold"].astype(int).astype(str))
                )
                weight_parts.append(
                    f"h{horizon}:" + ",".join(f"{w:.4f}" for w in ensemble_weights)
                )

            combined = np.average(np.stack(horizon_preds), axis=0, weights=horizon_weights)
            scores = pd.Series(
                combined,
                index=pred_df["stock_code"].astype(str).str.zfill(6),
                name="score",
            )
            base_is_tradable = (
                pred_df.set_index("stock_code")["is_tradable"]
                if "is_tradable" in pred_df.columns else None
            )
            if base_is_tradable is not None:
                base_is_tradable.index = base_is_tradable.index.astype(str).str.zfill(6)

            for recent_days in recent_days_list:
                recent_ok = recent_tradability(prices, pred_date, lookback_days=recent_days)
                if recent_ok is not None:
                    recent_ok = recent_ok.reindex(scores.index).fillna(False)
                    is_tradable = (
                        recent_ok if base_is_tradable is None
                        else base_is_tradable.reindex(scores.index).fillna(False) & recent_ok
                    )
                else:
                    is_tradable = base_is_tradable

                for top_k in top_ks:
                    for port_mode in port_modes:
                        weights = build_portfolio(
                            scores,
                            top_k=top_k,
                            weighting=port_mode,
                            is_tradable=is_tradable,
                        )
                        name = _variant_name(
                            experiment=experiment,
                            horizons=target_horizons,
                            recency_halflife_days=args.recency_halflife_days,
                            top_k=top_k,
                            fold_mode=fold_mode,
                            port_mode=port_mode,
                            recent_days=recent_days,
                        )
                        out_path = out_dir / name
                        out = pd.DataFrame({
                            "stock_code": weights.index.astype(str).str.zfill(6),
                            "weight": weights.values,
                        })
                        out.to_csv(out_path, index=False)
                        rows.append({
                            "file": str(out_path),
                            "experiment": experiment,
                            "fold_mode": fold_mode,
                            "portfolio_mode": port_mode,
                            "top_k": top_k,
                            "recent_tradable_days": recent_days,
                            "target_horizons": ",".join(map(str, target_horizons)),
                            "horizon_weights": ",".join(f"{w:.4f}" for w in horizon_weights),
                            "recency_halflife_days": args.recency_halflife_days,
                            "n_names": int((out["weight"] > 0).sum()),
                            "max_weight": float(out["weight"].max()),
                            "sum_weight": float(out["weight"].sum()),
                            "folds": ";".join(selected_parts),
                            "fold_weights": ";".join(weight_parts),
                        })

    summary = pd.DataFrame(rows)
    summary_path = LOG_DIR / f"{out_dir.name}_summary.csv"
    summary.to_csv(summary_path, index=False)
    if cv_rows:
        cv_path = LOG_DIR / f"{out_dir.name}_cv.csv"
        pd.concat(cv_rows, ignore_index=True).to_csv(cv_path, index=False)
        print(f">> CV: {cv_path}")
    print(f">> Wrote {len(summary)} variants to {out_dir}")
    print(f">> Summary: {summary_path}")


if __name__ == "__main__":
    main()
