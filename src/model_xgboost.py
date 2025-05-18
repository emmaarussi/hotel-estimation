import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from pathlib import Path
from datetime import datetime
from feature_engineering import EnhancedFeatureEngineer

# ----------------------------------------------------------------------------
# 1. Load and split data by 'srch_id' into train/val/test subsets
# ----------------------------------------------------------------------------
path_train = Path('data/raw/training_set_VU_DM.csv')
path_test = Path('data/raw/test_set_VU_DM.csv')

# Compatibility for notebooks
if path_test.exists():
    df_full = pd.read_csv(path_train)
    df_kaggle = pd.read_csv(path_test)
else:
    df_full = pd.read_csv('..' / path_train)
    df_kaggle = pd.read_csv('..' / path_test)

# Shuffle and split srch_id for reproducible subsets
need = df_full['srch_id'].unique()
np.random.seed(123)
np.random.shuffle(need)
n = len(need)
ids_train = need[: int(0.1 * n)]
ids_val   = need[int(0.5 * n) : int(0.7 * n)]
ids_test  = need[int(0.8 * n) : ]

train = df_full[df_full['srch_id'].isin(ids_train)].copy()
val   = df_full[df_full['srch_id'].isin(ids_val)].copy()
test  = df_full[df_full['srch_id'].isin(ids_test)].copy()

# ----------------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------------
fe = EnhancedFeatureEngineer()
train_fe = fe.create_enhanced_features(train, is_training=True)
val_fe   = fe.create_enhanced_features(val,   is_training=True)
test_fe  = fe.create_enhanced_features(test,  is_training=True)

# Helper to extract X, y, and group
def prepare(df):
    X = df.drop(['relevance_score','srch_id','prop_id'], axis=1)
    y = df['relevance_score'].values
    group = df.groupby('srch_id').size().to_list()
    return X, y, group

X_train, y_train, group_train = prepare(train_fe)
X_val,   y_val,   group_val   = prepare(val_fe)
X_test,  y_test,  group_test  = prepare(test_fe)

# Create DMatrix for ranking with group info
dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group(group_train)
dval = xgb.DMatrix(X_val, label=y_val)
dval.set_group(group_val)
dtest = xgb.DMatrix(X_test, label=y_test)
dtest.set_group(group_test)

# ----------------------------------------------------------------------------
# 3. Hyperparameter tuning with Optuna
# ----------------------------------------------------------------------------
def objective(trial):
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@5',
        'tree_method': 'hist',
        'seed': 123,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3,log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'lambdarank_pair_method': 'topk',
        'lambdarank_num_pair_per_sample': trial.suggest_int('lambdarank_num_pair_per_sample', 1, 20),
    }
    evals_result = {}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=5,
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=False
    )
    # Return best NDCG@5 on validation
    return evals_result['validation']['ndcg@5'][bst.best_iteration]

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
print("Best hyperparameters:\n", best_params)

# ----------------------------------------------------------------------------
# 4. Final training on train + validation
# ----------------------------------------------------------------------------
# Combine train & val
df_tv = pd.concat([train, val], ignore_index=True)
df_tv_fe = fe.create_enhanced_features(df_tv, is_training=True)
X_tv, y_tv, group_tv = prepare(df_tv_fe)

dtrain_full = xgb.DMatrix(X_tv, label=y_tv)
dtrain_full.set_group(group_tv)

# Merge best_params with fixed settings
final_params = {
    'objective': 'rank:ndcg',
    'eval_metric': 'ndcg@5',
    'tree_method': 'hist',
    'seed': 123,
    **best_params,
}

# Train with best number of rounds from tuning
best_nrounds = study.best_trial.user_attrs.get('best_nrounds', None)
# fallback: use early stopping on test
if best_nrounds is None:
    best_nrounds = study.best_trial.params.get('n_estimators', 1000)

bst_final = xgb.train(
    final_params,
    dtrain_full,
    num_boost_round=best_nrounds,
    evals=[(dtest, 'test')],
    early_stopping_rounds=50,
    verbose_eval=True
)

# Save the model
bst_final.save_model('data/models/xgb_best_model.json')

# ----------------------------------------------------------------------------
# 5. Feature importance & evaluation
# ----------------------------------------------------------------------------
importance = bst_final.get_score(importance_type='weight')
feat_imp = pd.Series(importance).sort_values(ascending=False)
print("Top 20 features by weight importance:")
print(feat_imp.head(20))

# NDCG@5 on held-out test
ev = bst_final.eval(dtest)
print(f"Test eval: {ev}")

# ----------------------------------------------------------------------------
# 6. Prepare Kaggle submission
# ----------------------------------------------------------------------------
df_k_fe = fe.create_enhanced_features(df_kaggle, is_training=False)
X_kaggle = df_k_fe.drop(['srch_id', 'prop_id'], axis=1)
dm_kaggle = xgb.DMatrix(X_kaggle)
preds = bst_final.predict(dm_kaggle)

submission = pd.DataFrame({
    'srch_id': df_k_fe['srch_id'],
    'prop_id': df_k_fe['prop_id'],
    'eval_score': preds
})
submission = submission.sort_values(['srch_id','eval_score'], ascending=[True, False])

fname = f"data/submission/xgb_rank_{datetime.now().strftime('%m%d_%H%M%S')}.csv"
submission[['srch_id','prop_id']].to_csv(fname, index=False)
print(f"Saved submission to {fname}")