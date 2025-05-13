"""Train LightGBM Ranker with Bayesian optimisation (Optuna) using leak-free split.
Outputs best model and study summary.
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
import pickle
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score

DATA_DIR = Path("data/processed/top_solution")
MODEL_DIR = Path("data/models/top_solution")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def ndcg5(y_true, y_pred, group):
    """Compute NDCG@5 for LightGBM callback."""
    # Convert to numpy groups
    offsets = group.cumsum()
    start = 0
    scores = []
    for end in offsets:
        yt = y_true[start:end]
        yp = y_pred[start:end]
        if len(yt) >= 5:
            scores.append(ndcg_score([yt], [yp], k=5))
        start = end
    return sum(scores) / (len(scores) + 1e-12)


def objective(trial: optuna.Trial, X, y, groups):
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "force_row_wise": True,
    }

    # GroupKFold 5
    gkf = GroupKFold(n_splits=5)
    scores = []
    for train_idx, valid_idx in gkf.split(X, y, groups):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        group_train = groups.iloc[train_idx].value_counts().sort_index().values
        group_valid = groups.iloc[valid_idx].value_counts().sort_index().values

        lgb_train = lgb.Dataset(X_train, y_train, group=group_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, group=group_valid, reference=lgb_train)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, first_metric_only=True, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        score = ndcg5(y_valid.values, y_pred, groups.iloc[valid_idx])
        scores.append(score)

    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(DATA_DIR / "train.feather"))
    args = parser.parse_args()

    print("Loading processed features …")
    df = pd.read_feather(args.train)

    y = df["relevance_score"]
    groups = df["srch_id"]
    feature_cols = [c for c in df.columns if c not in ["relevance_score", "srch_id", "booking_bool", "click_bool"]]
    X = df[feature_cols]

    print(f"Training with {len(X)} rows and {len(feature_cols)} features …")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y, groups), n_trials=10, timeout=60*60)

    print("Best NDCG@5:", study.best_value)
    print("Best params:", study.best_params)

    # Train final model on full data with best params
    best_params = study.best_params
    best_params.update({
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "boosting_type": "gbdt",
        "verbosity": -1,
        "force_row_wise": True,
    })

    group_sizes = groups.value_counts().sort_index().values
    lgb_dataset = lgb.Dataset(X, y, group=group_sizes)
    model = lgb.train(
        best_params,
        lgb_dataset,
        num_boost_round=500,
    )

    model_path = MODEL_DIR / "lgb_ranker_optuna.txt"
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    with open(MODEL_DIR / "study.pkl", "wb") as f:
        pickle.dump(study, f)


if __name__ == "__main__":
    main()
