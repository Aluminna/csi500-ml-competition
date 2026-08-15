"""Portfolio-level style and tradability controls.

These helpers deliberately sit outside the feature/model code.  They use only
information available up to the prediction date, then constrain the final
static submission so model experiments can be compared with and without the
same risk controls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MIN_STOCKS = 30
MAX_WEIGHT = 0.10


def build_style_context(
    prices: pd.DataFrame,
    as_of,
    lookback_days: int = 20,
    size_buckets: int = 5,
) -> pd.DataFrame:
    """Build per-stock liquidity, size-proxy, and limit-risk context.

    ``turnover`` in the provided data is a turnover fraction.  When present, a
    rough float-market-cap proxy is ``close * volume / turnover``.  It is noisy
    on one day, so we use a recent median for bucket assignment.
    """
    required = {"date", "stock_code", "close", "volume", "amount", "turnover"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing columns for style controls: {missing}")

    as_of = pd.Timestamp(as_of)
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df = df[df["date"] <= as_of].sort_values(["stock_code", "date"])
    if df.empty:
        raise ValueError(f"No prices available on or before {as_of.date()}")

    recent = df.groupby("stock_code", group_keys=False).tail(int(lookback_days)).copy()
    turnover = recent["turnover"].astype(float)
    recent["float_cap_proxy_row"] = np.where(
        turnover > 0,
        recent["close"].astype(float) * recent["volume"].astype(float) / turnover,
        np.nan,
    )

    agg = recent.groupby("stock_code").agg(
        recent_count=("date", "size"),
        last_date=("date", "max"),
        amount_mean_20d=("amount", "mean"),
        turnover_mean_20d=("turnover", "mean"),
        volume_mean_20d=("volume", "mean"),
        float_cap_proxy=("float_cap_proxy_row", "median"),
    )

    last = df.groupby("stock_code", group_keys=False).tail(1).set_index("stock_code")
    for col in ["close", "volume", "turnover", "high", "low", "pct_change"]:
        if col in last.columns:
            agg[f"last_{col}"] = last[col]
    if "pct_change" not in last.columns:
        agg["last_pct_change"] = np.nan

    agg["amount_rank"] = agg["amount_mean_20d"].rank(pct=True)
    agg["turnover_rank"] = agg["turnover_mean_20d"].rank(pct=True)
    agg["size_rank"] = agg["float_cap_proxy"].rank(pct=True)
    agg["log_float_cap_proxy"] = np.log(agg["float_cap_proxy"].clip(lower=0) + 1.0)

    size_source = agg["log_float_cap_proxy"].replace([np.inf, -np.inf], np.nan)
    valid = size_source.dropna()
    agg["size_bucket"] = "size_unknown"
    if len(valid) >= size_buckets:
        labels = [f"size_q{i}" for i in range(1, int(size_buckets) + 1)]
        try:
            agg.loc[valid.index, "size_bucket"] = pd.qcut(
                valid.rank(method="first"),
                q=int(size_buckets),
                labels=labels,
            ).astype(str)
        except ValueError:
            agg.loc[valid.index, "size_bucket"] = "size_all"

    agg["last_tradable"] = True
    if {"last_volume", "last_high", "last_low"}.issubset(agg.columns):
        agg["last_tradable"] = (
            (agg["last_volume"] > 0)
            & (agg["last_high"] > agg["last_low"])
        )
    return agg


def control_mask(
    scores_index: pd.Index,
    context: pd.DataFrame,
    min_amount_quantile: float | None = None,
    min_turnover_quantile: float | None = None,
    max_last_abs_pct_change: float | None = None,
    min_recent_count: int = 5,
) -> pd.Series:
    """Return eligible names after liquidity and recent limit-move filters."""
    idx = pd.Index(scores_index.astype(str).str.zfill(6), name="stock_code")
    ctx = context.reindex(idx)
    ok = ctx["last_tradable"].fillna(False) & (ctx["recent_count"].fillna(0) >= min_recent_count)

    if min_amount_quantile is not None and min_amount_quantile > 0:
        threshold = context["amount_mean_20d"].quantile(float(min_amount_quantile))
        ok &= ctx["amount_mean_20d"] >= threshold

    if min_turnover_quantile is not None and min_turnover_quantile > 0:
        threshold = context["turnover_mean_20d"].quantile(float(min_turnover_quantile))
        ok &= ctx["turnover_mean_20d"] >= threshold

    if max_last_abs_pct_change is not None and max_last_abs_pct_change > 0:
        ok &= ctx["last_pct_change"].abs().fillna(0) < float(max_last_abs_pct_change)

    return pd.Series(ok.to_numpy(dtype=bool), index=idx)


def _base_weights(n: int, weighting: str) -> np.ndarray:
    ranks = np.arange(n, 0, -1, dtype=float)
    if weighting == "equal":
        base = np.ones(n, dtype=float)
    elif weighting == "sqrt_rank":
        base = np.sqrt(ranks)
    else:
        base = ranks
    return base / base.sum()


def _apply_single_name_cap(weights: pd.Series, max_weight: float = MAX_WEIGHT) -> pd.Series:
    w = weights.copy()
    for _ in range(100):
        over = w > max_weight
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        free = ~over
        if not free.any() or w[free].sum() <= 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def _apply_group_cap(
    weights: pd.Series,
    groups: pd.Series,
    max_group_weight: float,
) -> pd.Series:
    if max_group_weight is None or max_group_weight >= 1.0:
        return weights / weights.sum()

    group_values = groups.reindex(weights.index).fillna("unknown").astype(str)
    w = weights.copy()
    for _ in range(100):
        group_sums = w.groupby(group_values).sum()
        over_groups = group_sums[group_sums > max_group_weight + 1e-10]
        if over_groups.empty:
            break

        before = w.sum()
        for group, group_sum in over_groups.items():
            idx = group_values[group_values == group].index.intersection(w.index)
            if len(idx) > 0 and group_sum > 0:
                w.loc[idx] *= max_group_weight / group_sum

        leftover = before - w.sum()
        if leftover <= 1e-12:
            break

        group_sums = w.groupby(group_values).sum()
        free_idx = group_values[group_values.map(group_sums) < max_group_weight - 1e-10].index
        free_idx = free_idx.intersection(w.index)
        if len(free_idx) == 0 or w.loc[free_idx].sum() <= 0:
            break
        w.loc[free_idx] += leftover * w.loc[free_idx] / w.loc[free_idx].sum()

    return w / w.sum()


def build_controlled_portfolio(
    scores: pd.Series,
    top_k: int,
    weighting: str,
    is_tradable: pd.Series | None = None,
    group_caps: list[tuple[pd.Series, float]] | None = None,
    min_stocks: int = MIN_STOCKS,
    max_weight: float = MAX_WEIGHT,
) -> pd.Series:
    """Build a long-only portfolio with optional multiple group caps."""
    if top_k < min_stocks:
        raise ValueError(f"top_k ({top_k}) must be >= {min_stocks}")

    scores = scores.copy()
    scores.index = scores.index.astype(str).str.zfill(6)
    if is_tradable is not None:
        ok = is_tradable.reindex(scores.index).fillna(False)
        scores.loc[~ok] -= 1e6

    chosen = scores.sort_values(ascending=False).head(top_k)
    w = pd.Series(_base_weights(len(chosen), weighting), index=chosen.index)
    caps = group_caps or []
    for _ in range(100):
        before = w.copy()
        w = _apply_single_name_cap(w, max_weight=max_weight)
        for groups, cap in caps:
            w = _apply_group_cap(w, groups, cap)
        w = _apply_single_name_cap(w, max_weight=max_weight)
        if float((w - before).abs().max()) < 1e-12:
            break

    w /= w.sum()
    assert abs(w.sum() - 1.0) < 1e-4
    assert (w <= max_weight + 1e-6).all()
    assert (w > 0).sum() >= min_stocks
    for groups, cap in caps:
        if cap is None or cap >= 1.0:
            continue
        group_sums = w.groupby(groups.reindex(w.index).fillna("unknown")).sum()
        assert (group_sums <= cap + 1e-6).all()
    return w
