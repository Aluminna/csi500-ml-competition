from __future__ import annotations

"""Generate the orig66/xgb34 top34 Phase 2 submission from two model portfolios.

Final recipe:
  66% updated original submission-1 LGB-style portfolio
  34% official-style XGB baseline portfolio
  deterministic top-34 equal-weight construction
"""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SUB_DIR = ROOT / "submissions"

FINAL_AS_OF = "20260508"
ORIGINAL_WEIGHT = 0.66
BASELINE_WEIGHT = 0.34
TOP_K = 34
REWEIGHT = "equal"

CURRENT_ORIGINAL = SUB_DIR / "original_submission1_model_asof20260508.csv"
CURRENT_BASELINE = SUB_DIR / "phase2_baseline_xgb_asof20260508.csv"
DEFAULT_OUT = SUB_DIR / "Yuxin_He_week2_orig66_xgb34_k34.csv"


def load_weights(path: Path) -> pd.Series:
    df = pd.read_csv(path, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    weights = df.set_index("stock_code")["weight"].astype(float)
    weights = weights[weights > 0]
    if weights.empty:
        raise ValueError(f"{path} has no positive weights.")
    if weights.index.duplicated().any():
        raise ValueError(f"{path} contains duplicate stock_code values.")
    return weights / weights.sum()


def blend_portfolios(original: pd.Series, baseline: pd.Series) -> pd.Series:
    blended = original.mul(ORIGINAL_WEIGHT).add(
        baseline.mul(BASELINE_WEIGHT),
        fill_value=0.0,
    )
    blended = blended[blended > 0].sort_values(ascending=False)
    if blended.empty:
        raise ValueError("The blended portfolio is empty.")
    return blended / blended.sum()


def topk_equal(scores: pd.Series, top_k: int = TOP_K) -> pd.Series:
    chosen = scores.sort_values(ascending=False).head(top_k)
    if len(chosen) < top_k:
        raise ValueError(f"Only {len(chosen)} candidates available for top{top_k}.")
    return pd.Series(1.0 / len(chosen), index=chosen.index)


def write_submission(weights: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "stock_code": weights.index.astype(str).str.zfill(6),
            "weight": weights.astype(float).to_numpy(),
        }
    ).to_csv(out_path, index=False)


def generate() -> pd.Series:
    original = load_weights(CURRENT_ORIGINAL)
    baseline = load_weights(CURRENT_BASELINE)
    blended = blend_portfolios(original, baseline)
    return topk_equal(blended, TOP_K)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    weights = generate()
    write_submission(weights, out_path)
    print("Generated Phase 2 orig66/xgb34 top34 submission")
    print(f"  as_of   : {FINAL_AS_OF}")
    print(f"  recipe  : original={ORIGINAL_WEIGHT:.3f}, baseline={BASELINE_WEIGHT:.3f}")
    print(f"  sources : {CURRENT_ORIGINAL.relative_to(ROOT)}, {CURRENT_BASELINE.relative_to(ROOT)}")
    print(f"  rule    : top{TOP_K} {REWEIGHT}")
    print(f"  output  : {out_path}")
    print(f"  n       : {len(weights)}")
    print(f"  sum     : {weights.sum():.12f}")
    print(f"  max     : {weights.max():.6f}")


if __name__ == "__main__":
    main()
