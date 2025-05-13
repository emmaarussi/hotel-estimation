"""Combine baseline tess-try LambdaMART predictions with new Optuna-tuned model.
Produces ensemble NDCG scores and saves predictions.
"""

from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import ndcg_score

DATA_DIR = Path("data/processed/top_solution")
MODEL_DIR = Path("data/models/top_solution")
BASELINE_PRED_PATH = Path("data/models/tess-try/baseline_ranker_pred.csv")


def load_preds(path):
    df = pd.read_csv(path)
    return df.set_index(["srch_id", "prop_id"])["pred"].rename(Path(path).stem)


def evaluate_ndcg(merged_df, k=5):
    scores = []
    for srch_id, grp in merged_df.groupby(level=0):
        if len(grp) < k:
            continue
        y_true = grp["relevance_score"].values
        y_pred = grp.values
        scores.append(ndcg_score([y_true], [y_pred], k=k))
    return np.mean(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=str(DATA_DIR / "test.feather"))
    args = parser.parse_args()

    print("Loading processed test features …")
    df_test = pd.read_feather(args.test)
    id_cols = ["srch_id", "prop_id"]

    # Load new model
    model = lgb.Booster(model_file=str(MODEL_DIR / "lgb_ranker_optuna.txt"))
    # Align feature set between training and test to avoid shape mismatch
    model_feats = model.feature_name()
    # Ensure all expected columns exist in test; if absent, add zeros
    for col in model_feats:
        if col not in df_test.columns:
            df_test[col] = 0.0
    X_test = df_test[model_feats]  # keep order identical to training
    df_test["pred_optuna"] = model.predict(X_test)

    baseline = None
    if BASELINE_PRED_PATH.exists():
        print(f"Loading baseline predictions from {BASELINE_PRED_PATH} …")
        baseline = load_preds(BASELINE_PRED_PATH)

    # Merge predictions
    df_test.set_index(id_cols, inplace=True)
    if baseline is not None:
        df_test = df_test.join(baseline, how="left")
        df_test["pred_combined"] = (
            df_test[["pred_optuna", baseline.name]]
            .rank(method="average", pct=True, axis=0)
            .mean(axis=1)
        )
    else:
        print("Baseline predictions not found – using only Optuna model scores for submission.")
        df_test["pred_combined"] = df_test["pred_optuna"]

    # Evaluate if relevance_score available (only in training/validation scenarios)
    if "relevance_score" in df_test.columns:
        cols_to_eval = ["pred_optuna", "pred_combined"]
        if baseline is not None:
            cols_to_eval.insert(1, baseline.name)
        for name in cols_to_eval:
            df_test[name] = df_test[name].astype(float)
            score = evaluate_ndcg(
                df_test[["relevance_score", name]].rename(columns={name: "pred"}), k=5
            )
            print(f"NDCG@5 {name}: {score:.4f}")

    # Save combined predictions
    out_path = MODEL_DIR / "ensemble_predictions.csv"
    df_test.reset_index()[id_cols + ["pred_combined"]].to_csv(out_path, index=False)
    print(f"Ensemble predictions saved to {out_path}")


if __name__ == "__main__":
    main()
