import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import ndcg_score
from feature_engineering import EnhancedFeatureEngineer

def mean_group_ndcg(y_true, y_pred, groups, k=5):
    scores = []
    for group in np.unique(groups):
        idx = groups == group
        if idx.sum() < 2:
            continue
        score = ndcg_score([y_true[idx]], [y_pred[idx]], k=k)
        scores.append(score)
    return np.mean(scores) if scores else np.nan

# --- Load models ---
lgb_model = lgb.Booster(model_file='data/models/lgb_best_model.txt')
xgb_model = xgb.Booster()
xgb_model.load_model('data/models/xgb_best_model.json')

# --- Load data ---
train_full = pd.read_csv('data/raw/training_set_VU_DM.csv')
np.random.seed(123)
unique_ids = train_full['srch_id'].unique()
np.random.shuffle(unique_ids)
n = len(unique_ids)
val_ids = unique_ids[int(0.5 * n):int(0.7 * n)]
test_ids = unique_ids[int(0.9 * n):]

val = train_full[train_full['srch_id'].isin(val_ids)].copy()
test = train_full[train_full['srch_id'].isin(test_ids)].copy()

feature_engineer = EnhancedFeatureEngineer()
val_clean = feature_engineer.create_enhanced_features(val, is_training=True)
test_clean = feature_engineer.create_enhanced_features(test, is_training=True)

X_val = val_clean.drop(['relevance_score', 'srch_id', 'prop_id'], axis=1)
y_val = val_clean['relevance_score'].values
groups_val = val_clean['srch_id'].values

X_test = test_clean.drop(['relevance_score', 'srch_id', 'prop_id'], axis=1)
y_test = test_clean['relevance_score'].values
groups_test = test_clean['srch_id'].values

# --- Get base model predictions for stacking ---
lgb_val_pred = lgb_model.predict(X_val)
xgb_val_pred = xgb_model.predict(xgb.DMatrix(X_val))
lgb_test_pred = lgb_model.predict(X_test)
xgb_test_pred = xgb_model.predict(xgb.DMatrix(X_test))

# --- Train meta-model (stacker) on validation set ---
meta_X_val = np.vstack([lgb_val_pred, xgb_val_pred]).T
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X_val, y_val)

# --- Predict with meta-model on test set ---
meta_X_test = np.vstack([lgb_test_pred, xgb_test_pred]).T
ensemble_test_pred = meta_model.predict(meta_X_test)

# --- Evaluate ---
lgb_score = mean_group_ndcg(y_test, lgb_test_pred, groups_test, k=5)
xgb_score = mean_group_ndcg(y_test, xgb_test_pred, groups_test, k=5)
blend_score = mean_group_ndcg(y_test, ensemble_test_pred, groups_test, k=5)

print(f"NDCG@5 on held-out test set (LightGBM): {lgb_score:.4f}")
print(f"NDCG@5 on held-out test set (XGBoost): {xgb_score:.4f}")
print(f"NDCG@5 on held-out test set (Stacked ensemble): {blend_score:.4f}")

# --- Save test set predictions for inspection ---
test_submission = pd.DataFrame({
    'srch_id': test_clean['srch_id'],
    'prop_id': test_clean['prop_id'],
    'eval_score': ensemble_test_pred
})
test_submission = test_submission.sort_values(['srch_id', 'eval_score'], ascending=[True, False])
test_submission[['srch_id', 'prop_id']].to_csv('data/submission/stacked_ensemble_testset.csv', index=False)