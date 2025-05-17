import pandas as pd
import lightgbm as lgb_std
import optuna.integration.lightgbm as lgb #https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
from lightgbm import early_stopping, log_evaluation
from pathlib import Path
import numpy as np
from datetime import datetime
import sys
import optuna

from feature_engineering import EnhancedFeatureEngineer

path_train = Path('data/raw/training_set_VU_DM.csv')
path_test = Path('data/raw/test_set_VU_DM.csv')

# Checking path for compatibility with notebooks
if path_test.exists():
    train_full = pd.read_csv(path_train)
    kaggle_test = pd.read_csv(path_test)
else:
    train_full = pd.read_csv('..' / path_train)
    kaggle_test = pd.read_csv('..' / path_test)
    
    
# Create random subset of training data based on srch id
# Saves training time and allows for offline testing of the submissions!!
np.random.seed(123)
unique_ids = train_full['srch_id'].unique()
np.random.shuffle(unique_ids)
n = len(unique_ids)
train_ids = unique_ids[:int(0.3 * n)]
val_ids = unique_ids[int(0.5 * n):int(0.7 * n)]
test_ids = unique_ids[int(0.8 * n):]

train = train_full[train_full['srch_id'].isin(train_ids)].copy()
val = train_full[train_full['srch_id'].isin(val_ids)].copy()
test = train_full[train_full['srch_id'].isin(test_ids)].copy()

feature_engineer = EnhancedFeatureEngineer()

# Feature engineering
train_clean = feature_engineer.create_enhanced_features(train, is_training=True)
val_clean = feature_engineer.create_enhanced_features(val, is_training=True)
test_clean = feature_engineer.create_enhanced_features(test, is_training=True)

X_train = train_clean.drop(['relevance_score', 'srch_id', 'prop_id'], axis=1)
y_train = train_clean['relevance_score']
group_train = train_clean.groupby('srch_id').size().to_list()

X_val = val_clean.drop(['relevance_score', 'srch_id', 'prop_id'], axis=1)
y_val = val_clean['relevance_score']
group_val = val_clean.groupby('srch_id').size().to_list()

X_test = test_clean.drop(['relevance_score', 'srch_id', 'prop_id'], axis=1)
y_test = test_clean['relevance_score']
group_test = test_clean.groupby('srch_id').size().to_list()

lgb_train = lgb.Dataset(X_train, y_train, group=group_train)
lgb_val = lgb.Dataset(X_val, y_val, group=group_val, reference=lgb_train)
lgb_test = lgb.Dataset(X_test, y_test, group=group_test, reference=lgb_train)


params = {
     'objective': 'lambdarank',
     'metric': 'ndcg',
     'eval_at': 5,
     "boosting_type": "gbdt",
     "verbosity": -1
}

best_params, tuning_history = dict(), list()

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_val, lgb_test],
    valid_names=['train', 'val', 'test'],
    callbacks=[early_stopping(100), log_evaluation(100)]
)

# Evaluate NDCG5 on the test set
best_iter = model.best_iteration
test_scores = model.eval_valid(feval=None)
best_params = model.params

# Print top 20 most important features
importances = model.feature_importance()
feature_names = X_train.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("Top 20 features by importance:")
print(feat_imp.head(20))

# Printing best scores
# Gives evaluation metric on held out training data (extra test set)
# Comparable score as kaggle submission
print(f"test score: {model.best_score['test']['ndcg@5']}")

print('Best parameters')
for k,v in best_params.items():
    print(k,v)
    
## Retrain using the best model:
train_val_ids = np.concatenate([train_ids, val_ids])
train_val = train_full[train_full['srch_id'].isin(train_val_ids)].copy()
train_val_clean = feature_engineer.create_enhanced_features(train_val, is_training=True)
X_train_val = train_val_clean.drop(['relevance_score', 'srch_id', 'prop_id'], axis=1)
y_train_val = train_val_clean['relevance_score']
group_train_val = train_val_clean.groupby('srch_id').size().to_list()
lgb_train_val = lgb.Dataset(X_train_val, y_train_val, group=group_train_val)

retrained_model = lgb_std.train(
    best_params,
    lgb_train_val,
    num_boost_round=500,
    valid_sets = [lgb_train_val, lgb_test],
    valid_names = ['train_val', 'test'],
    callbacks=[early_stopping(100), log_evaluation(100)]
)

print(f"retrained model score: {retrained_model.best_score['test']['ndcg@5']}")

## Preparing submission
kaggle_test_clean = feature_engineer.create_enhanced_features(kaggle_test, is_training=False)
X_kaggle_test = kaggle_test_clean.drop(['srch_id', 'prop_id'], axis=1)

# Predict on the real test set
kaggle_test_pred = retrained_model.predict(X_kaggle_test)

submission = pd.DataFrame({
    'srch_id': kaggle_test_clean['srch_id'],
    'prop_id': kaggle_test_clean['prop_id'],
    'eval_score': kaggle_test_pred
})

#TODO: save best model, feature importance etc, tuning......

submission = submission.sort_values(['srch_id', 'eval_score'], ascending=[True, False])
submission = submission[['srch_id', 'prop_id']]
submission.to_csv(f'data/submission/test_{datetime.now().strftime("%m_%d_%H_%M_%S")}.csv', index=False)
