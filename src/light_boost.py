import pandas as pd
import lightgbm as lgb
from pathlib import Path
import numpy as np
from datetime import datetime

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
train_ids = unique_ids[:int(0.4 * n)]
val_ids = unique_ids[int(0.4 * n):int(0.5 * n)]
test_ids = unique_ids[int(0.5 * n):]

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
    'learning_rate': 0.1,
    'num_leaves': 31
}

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_train, lgb_val, lgb_test],
    valid_names=['train', 'val', 'test'],
    early_stopping_rounds=10
)

# Evaluate on test set 
test_pred = model.predict(X_test)
test_results = pd.concat([test_clean['srch_id'], test_clean['prop_id'], pd.Series(test_pred)], axis=1)
test_results.columns = ['srch_id', 'prop_id', 'eval_score']
test_results = test_results.sort_values(['srch_id', 'eval_score'], ascending=[True, False])


# Evaluate NDCG5 on the test set
best_iter = model.best_iteration
test_scores = model.eval_valid(feval=None)

# Printing best scores
# Gives evaluation metric on held out training data (extra test set)
# Comparable score as kaggle submission
print(f"test score: {model.best_score['test']['ndcg@5']}")

## Preparing submission
kaggle_test_clean = feature_engineer.create_enhanced_features(kaggle_test, is_training=False)
X_kaggle_test = kaggle_test_clean.drop(['srch_id', 'prop_id'], axis=1)

# Predict on the real test set
kaggle_test_pred = model.predict(X_kaggle_test)

submission = pd.DataFrame({
    'srch_id': kaggle_test_clean['srch_id'],
    'prop_id': kaggle_test_clean['prop_id'],
    'eval_score': kaggle_test_pred
})

#TODO: save best model, feature importance etc, tuning......

submission = submission.sort_values(['srch_id', 'eval_score'], ascending=[True, False])
submission = submission[['srch_id', 'prop_id']]
submission.to_csv(f'data/submission/test_{datetime.now().strftime("%m_%d_%H_%M_%S")}.csv', index=False)
