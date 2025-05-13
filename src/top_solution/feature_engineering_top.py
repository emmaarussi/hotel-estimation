"""Feature engineering pipeline inspired by top Kaggle solutions.
Creates list-wise rank features, competitor/time features via CombinedFeatureEngineer
and stores output as feather for fast loading.

Usage
-----
python feature_engineering_top.py --mode train --input data/raw/train.csv \
                                  --output data/processed/top_solution/train.feather

python feature_engineering_top.py --mode test --input data/raw/test.csv  \
                                  --output data/processed/top_solution/test.feather
"""

from __future__ import annotations
import os
import argparse
import sys
from pathlib import Path

# Ensure project src directory is on PYTHONPATH so that tess_try2 can be imported
project_src = Path(__file__).resolve().parents[2]
if str(project_src) not in sys.path:
    sys.path.insert(0, str(project_src))

# also include project_root/src in path
src_dir = project_src / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np

# Re-use comprehensive engineer from tess_try2
from tess_try2.combined_feature_engineering import CombinedFeatureEngineer

LISTWISE_NUMERIC_COLS = [
    "price_usd",
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "prop_log_historical_price",
]

def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add percentile rank of key numeric columns within each search (srch_id)."""
    if "srch_id" not in df.columns:
        print("srch_id missing – skip rank features")
        return df
    for col in LISTWISE_NUMERIC_COLS:
        if col in df.columns:
            df[f"rank_pct_{col}"] = (
                df.groupby("srch_id")[col].rank(method="average", pct=True)
            )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    print(f"Reading data from {args.input} …")
    df = pd.read_csv(args.input)

    # Add listwise rank features first (fast)
    df = add_rank_features(df)

    print("Running CombinedFeatureEngineer …")
    cfe = CombinedFeatureEngineer()
    df_fe = cfe.create_enhanced_features(
        df,
        is_training=args.mode == "train",
        target_col="relevance_score" if args.mode == "train" else None,
    )

    print(f"Saving feather to {args.output}")
    df_fe.reset_index(drop=True).to_feather(args.output)


if __name__ == "__main__":
    main()
