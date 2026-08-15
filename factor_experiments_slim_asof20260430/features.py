"""
features.py  —  Feature engineering for the CSI500 competition.

Slim alpha-factor experiment for the CSI500 competition.

Starts from the stable 40-factor base and adds a small set of public,
short-horizon OHLCV factors: attention/lottery, price anchors, overnight
returns, turnover shock, and short reversal decomposition.
No industry neutralization, no rank transform (applied optionally in training).

Public interface:
    FEATURE_COLUMNS   — list of 40 feature column names
    TARGET_COLUMN     — "target_5d" (default / backward compatible)
    FORWARD_HORIZON   — 5 (default / backward compatible)
    target_column_for_horizon(horizon)
    build_features(prices)
    training_frame(panel, min_date, max_date)
    prediction_frame(panel, as_of)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FORWARD_HORIZON = 5
TARGET_COLUMN   = "target_5d"
TARGET_HORIZONS = (3, 5, 10)


def target_column_for_horizon(horizon: int) -> str:
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    return f"target_{horizon}d"

# Frozen tuple — never mutated at runtime
FEATURE_COLUMNS = [
    # Momentum (5)
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    # Slim short-horizon decomposition / reversal (3)
    "ret_3d", "ret_5d_ex_1d", "reversal_3d",
    # Momentum spread / trend (3)
    "ret_5d_vs_20d", "ret_10d_vs_60d", "reversal_1d",
    # Volatility (5)
    "vol_5d", "vol_20d", "vol_ratio_5_20",
    "ret_skew_20d", "ret_kurt_20d",
    # Volume / liquidity (6)
    "volume_z_20d", "turnover_ma_5d", "turnover_ma_20d",
    "turnover_ratio_5_20", "amihud_20d", "vwap_bias",
    # Price level (4)
    "close_over_ma5", "close_over_ma20", "close_over_ma60", "ma5_over_ma20",
    # Technical (5)
    "rsi_14", "macd_hist", "bb_pct", "price_to_high_20", "price_to_low_20",
    # Intraday patterns (3)
    "intraday_range", "upper_shadow", "lower_shadow",
    # Volume-price divergence (2)
    "ret_vol_corr_10d", "price_volume_div",
    # Slim public-factor additions (9)
    "overnight_ret_3d", "overnight_ret_5d",
    "dist_high_120d", "dist_low_120d",
    "max_ret_20d", "min_ret_20d", "up_8pct_count_20d",
    "turnover_vol_20d", "amount_ratio_5_20",
    # Cross-sectional ranks (7)
    "ret_1d_rank", "ret_5d_rank", "ret_20d_rank",
    "vol_20d_rank", "turnover_rank", "amihud_rank", "intraday_range_rank",
]

MARKET_RELATIVE_FEATURE_COLUMNS = [
    "rel_ret_1d", "rel_ret_5d", "rel_ret_10d", "rel_ret_20d",
    "rel_vol_20d", "beta_20d", "resid_ret_5d", "resid_ret_20d",
]

INDUSTRY_RANK_SOURCE_COLUMNS = [
    "ret_5d", "ret_20d", "vol_20d", "turnover_ma_20d", "amihud_20d",
    "intraday_range",
]
INDUSTRY_RANK_FEATURE_COLUMNS = [
    f"{col}_industry_rank" for col in INDUSTRY_RANK_SOURCE_COLUMNS
]


def _per_stock_features(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute time-series features for one stock."""
    df = grp.sort_values("date").copy()
    close  = df["close"].astype(float)
    open_  = df["open"].astype(float)
    volume = df["volume"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    amount = df["amount"].astype(float) if "amount" in df.columns else close * volume
    dr = close.pct_change(1)

    # Momentum
    df["ret_1d"]  = dr
    df["ret_3d"]  = close.pct_change(3)
    df["ret_5d"]  = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["ret_60d"] = close.pct_change(60)
    df["reversal_1d"]    = -dr
    df["reversal_3d"]    = -df["ret_3d"]
    df["ret_5d_ex_1d"]   = df["ret_5d"] - df["ret_1d"]
    df["ret_5d_vs_20d"]  = df["ret_5d"]  - df["ret_20d"]
    df["ret_10d_vs_60d"] = df["ret_10d"] - df["ret_60d"]

    # Volatility
    df["vol_5d"]         = dr.rolling(5).std()
    df["vol_20d"]        = dr.rolling(20).std()
    df["vol_ratio_5_20"] = df["vol_5d"] / (df["vol_20d"].replace(0, np.nan))
    df["ret_skew_20d"]   = dr.rolling(20).skew()
    df["ret_kurt_20d"]   = dr.rolling(20).kurt()

    # Volume / liquidity
    vm20 = volume.rolling(20).mean()
    vs20 = volume.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (volume - vm20) / vs20
    to = df["turnover"].astype(float) if "turnover" in df.columns else pd.Series(np.nan, index=df.index)
    df["turnover_ma_5d"]      = to.rolling(5).mean()
    df["turnover_ma_20d"]     = to.rolling(20).mean()
    df["turnover_ratio_5_20"] = df["turnover_ma_5d"] / (df["turnover_ma_20d"].replace(0, np.nan))
    df["amihud_20d"]          = (dr.abs() / (amount + 1e-9)).rolling(20).mean()
    vwap = amount.rolling(5).sum() / (volume.rolling(5).sum() + 1e-9)
    df["vwap_bias"] = close / (vwap + 1e-9) - 1.0
    df["turnover_vol_20d"] = to.rolling(20, min_periods=10).std()
    amount_ma20 = amount.rolling(20, min_periods=10).mean()
    df["amount_ratio_5_20"] = (
        amount.rolling(5, min_periods=3).mean()
        / (amount_ma20 + 1e-9)
        - 1.0
    )

    # Price level
    df["close_over_ma5"]  = close / close.rolling(5).mean()  - 1.0
    df["close_over_ma20"] = close / close.rolling(20).mean() - 1.0
    df["close_over_ma60"] = close / close.rolling(60).mean() - 1.0
    df["ma5_over_ma20"]   = close.rolling(5).mean() / close.rolling(20).mean() - 1.0

    # Technical
    delta = close.diff()
    up   = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + up / down)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_pct"] = (close - (ma20 - 2*std20)) / (4*std20 + 1e-9)
    df["price_to_high_20"] = close / (high.rolling(20).max() + 1e-9) - 1
    df["price_to_low_20"]  = close / (low.rolling(20).min()  + 1e-9) - 1

    # Intraday patterns
    daily_range = high - low
    df["intraday_range"] = daily_range / (close + 1e-9)
    df["upper_shadow"]   = (high - np.maximum(close, open_)) / (daily_range + 1e-9)
    df["lower_shadow"]   = (np.minimum(close, open_) - low)  / (daily_range + 1e-9)

    # Volume-price divergence
    df["ret_vol_corr_10d"] = dr.rolling(10).corr(volume)
    df["price_volume_div"] = df["ret_5d"] - volume.pct_change(5)

    # Slim public-factor additions.
    prev_close = close.shift(1)
    overnight_ret_1d = open_ / (prev_close + 1e-9) - 1.0
    df["overnight_ret_3d"] = overnight_ret_1d.rolling(3, min_periods=2).sum()
    df["overnight_ret_5d"] = overnight_ret_1d.rolling(5, min_periods=3).sum()

    high120 = high.rolling(120, min_periods=60).max()
    low120 = low.rolling(120, min_periods=60).min()
    df["dist_high_120d"] = close / (high120 + 1e-9) - 1.0
    df["dist_low_120d"] = close / (low120 + 1e-9) - 1.0

    df["max_ret_20d"] = dr.rolling(20, min_periods=10).max()
    df["min_ret_20d"] = dr.rolling(20, min_periods=10).min()
    df["up_8pct_count_20d"] = dr.gt(0.08).rolling(20, min_periods=10).sum()

    # Targets. Keep target_5d as the backward-compatible default while also
    # exposing shorter/longer horizons for live-window experiments.
    for horizon in TARGET_HORIZONS:
        df[target_column_for_horizon(horizon)] = close.shift(-horizon) / close - 1.0
    return df


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    for out_col, src in [
        ("ret_1d_rank",         "ret_1d"),
        ("ret_5d_rank",         "ret_5d"),
        ("ret_20d_rank",        "ret_20d"),
        ("vol_20d_rank",        "vol_20d"),
        ("turnover_rank",       "turnover_ma_20d"),
        ("amihud_rank",         "amihud_20d"),
        ("intraday_range_rank", "intraday_range"),
    ]:
        panel[out_col] = panel.groupby("date")[src].rank(
            method="average", pct=True, na_option="keep"
        )
    return panel


def _add_market_relative_features(panel: pd.DataFrame,
                                  index_prices: pd.DataFrame) -> pd.DataFrame:
    """Add features measured relative to the CSI500 index.

    These use only index information available up to each feature date. They
    align the prediction target with the competition's excess-return objective
    without changing the portfolio evaluator.
    """
    idx = index_prices.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")
    idx_close = idx["close"].astype(float)
    idx_ret = idx_close.pct_change(1)

    idx_features = pd.DataFrame({
        "date": idx["date"],
        "idx_ret_1d": idx_ret,
        "idx_ret_5d": idx_close.pct_change(5),
        "idx_ret_10d": idx_close.pct_change(10),
        "idx_ret_20d": idx_close.pct_change(20),
        "idx_vol_20d": idx_ret.rolling(20).std(),
    })

    out = panel.merge(idx_features, on="date", how="left")
    out["rel_ret_1d"] = out["ret_1d"] - out["idx_ret_1d"]
    out["rel_ret_5d"] = out["ret_5d"] - out["idx_ret_5d"]
    out["rel_ret_10d"] = out["ret_10d"] - out["idx_ret_10d"]
    out["rel_ret_20d"] = out["ret_20d"] - out["idx_ret_20d"]
    out["rel_vol_20d"] = out["vol_20d"] - out["idx_vol_20d"]

    out = out.sort_values(["stock_code", "date"]).reset_index(drop=True)

    def _beta(grp: pd.DataFrame) -> pd.Series:
        cov = grp["ret_1d"].rolling(20).cov(grp["idx_ret_1d"])
        var = grp["idx_ret_1d"].rolling(20).var()
        return cov / (var + 1e-9)

    out["beta_20d"] = (
        out.groupby("stock_code", group_keys=False)[["ret_1d", "idx_ret_1d"]]
        .apply(_beta)
    )
    out["resid_ret_5d"] = out["ret_5d"] - out["beta_20d"] * out["idx_ret_5d"]
    out["resid_ret_20d"] = out["ret_20d"] - out["beta_20d"] * out["idx_ret_20d"]
    out = out.drop(
        columns=[
            "idx_ret_1d", "idx_ret_5d", "idx_ret_10d", "idx_ret_20d",
            "idx_vol_20d",
        ],
        errors="ignore",
    )
    return out.sort_values(["date", "stock_code"]).reset_index(drop=True)


def _add_industry_rank_features(panel: pd.DataFrame,
                                industry: pd.DataFrame) -> pd.DataFrame:
    """Add within-industry cross-sectional ranks for selected features."""
    industry = industry.copy()
    industry["stock_code"] = industry["stock_code"].astype(str).str.zfill(6)
    if "industry" not in industry.columns:
        raise ValueError("industry data must contain an 'industry' column")

    cols = ["stock_code", "industry"]
    out = panel.merge(industry[cols].drop_duplicates("stock_code"),
                      on="stock_code", how="left")
    out["industry"] = out["industry"].fillna("unknown")
    for src, dst in zip(INDUSTRY_RANK_SOURCE_COLUMNS, INDUSTRY_RANK_FEATURE_COLUMNS):
        out[dst] = out.groupby(["date", "industry"])[src].rank(
            method="average", pct=True, na_option="keep"
        )
    return out.sort_values(["date", "stock_code"]).reset_index(drop=True)


def build_features(prices: pd.DataFrame,
                   index_prices: pd.DataFrame | None = None,
                   use_market_relative: bool = False,
                   industry: pd.DataFrame | None = None,
                   use_industry_ranks: bool = False) -> pd.DataFrame:
    """Build (date, stock_code) feature panel."""
    required = {"date", "stock_code", "close", "volume"}
    missing  = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    chunks = []
    for code, grp in prices.groupby("stock_code", sort=False):
        feat = _per_stock_features(grp)
        feat["stock_code"] = code
        chunks.append(feat)
    panel = pd.concat(chunks, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = _cross_sectional_ranks(panel)
    if use_market_relative:
        if index_prices is None:
            raise ValueError("index_prices is required when use_market_relative=True")
        panel = _add_market_relative_features(panel, index_prices)
        for col in MARKET_RELATIVE_FEATURE_COLUMNS:
            if col not in FEATURE_COLUMNS:
                FEATURE_COLUMNS.append(col)
    if use_industry_ranks:
        if industry is None:
            raise ValueError("industry data is required when use_industry_ranks=True")
        panel = _add_industry_rank_features(panel, industry)
        for col in INDUSTRY_RANK_FEATURE_COLUMNS:
            if col not in FEATURE_COLUMNS:
                FEATURE_COLUMNS.append(col)
    return panel.sort_values(["date", "stock_code"]).reset_index(drop=True)


def training_frame(panel: pd.DataFrame,
                   min_date=None, max_date=None,
                   target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Return rows with all features + target present. No transforms."""
    df = panel.dropna(subset=FEATURE_COLUMNS + [target_column]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df.reset_index(drop=True)


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Return latest-date rows with all features. Adds is_tradable flag."""
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(str(as_of))
    df = panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()
    if "volume" in df.columns and "high" in df.columns and "low" in df.columns:
        df["is_tradable"] = (df["volume"] > 0) & (df["high"] > df["low"])
    else:
        df["is_tradable"] = True
    return df.reset_index(drop=True)
