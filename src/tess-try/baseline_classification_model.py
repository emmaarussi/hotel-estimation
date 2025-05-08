import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix, average_precision_score
from sklearn.preprocessing import StandardScaler
import os
import time
import joblib
from functools import partial

def train_baseline_classification_models(data_path, output_dir='data/models/baseline', sample_size=None):
    """Train baseline classification models for hotel booking prediction"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the enhanced dataset (optionally use a sample for faster development)
    print(f"Loading data from {data_path}...")
    if sample_size:
        df = pd.read_csv(data_path, nrows=sample_size)
        print(f"Using sample of {sample_size} rows")
    else:
        df = pd.read_csv(data_path)
        print(f"Using full dataset with {len(df)} rows")
    
    # Based on feature importance analysis and top Expedia solutions, select the most important features
    important_features = [
        # Core predictive features (from our feature importance analysis)
        'prop_historical_br',     # Historical booking rate
        'prop_historical_br_log', # Log-transformed historical booking rate (from our enhanced feature engineering)
        'prop_historical_ctr',    # Historical click rate
        'prop_historical_ctr_log', # Log-transformed historical click rate (from our enhanced feature engineering)
        'position',               # Position in search results
        'prop_avg_position',      # Average historical position (from our enhanced feature engineering)
        
        # Price competitiveness features (from 2nd place solution)
        'price_rank',             # Rank of price within search results
        'price_percentile',       # Price percentile within search
        'price_diff_from_mean',   # Price difference from mean
        'price_diff_from_hist',   # Price difference from historical average (from our enhanced feature engineering)
        'price_ratio_to_mean',    # Price ratio to search mean (from our enhanced feature engineering)
        'price_ratio_to_hist',    # Price ratio to historical average (from our enhanced feature engineering)
        
        # Normalized price features (from 3rd place solution)
        'price_norm_by_srch_destination_id', # Price normalized by destination (from our enhanced feature engineering)
        'price_norm_by_prop_country_id',     # Price normalized by country (from our enhanced feature engineering)
        
        # Quality metrics
        'prop_starrating',        # Star rating
        'prop_starrating_monotonic', # Monotonic star rating (from our enhanced feature engineering)
        'prop_review_score',      # Review score
        'prop_review_monotonic',  # Monotonic review score (from our enhanced feature engineering)
        
        # Value metrics (from 4th place solution)
        'review_score_per_dollar', # Value for money based on reviews
        'star_per_dollar',        # Stars per dollar
        
        # Composite features (from 4th place solution)
        'star_review_composite',  # Star-review composite (from our enhanced feature engineering)
        'location_review_composite', # Location-review composite (from our enhanced feature engineering)
        
        # Location features
        'prop_location_score1',   # Location score 1
        'prop_location_score2',   # Location score 2
        'prop_location_monotonic', # Monotonic location score (from our enhanced feature engineering)
        'location_rank',          # Location rank
        
        # Search context
        'destination_search_volume', # Search volume for destination
        'star_rank',              # Star rating rank
        'is_domestic',            # Domestic flag
        'log_orig_distance',      # Log-transformed distance (from our enhanced feature engineering)
        
        # User-property match features (from 3rd place solution)
        'user_star_diff',         # Difference between user's historical star preference and hotel stars
        'user_price_diff',        # Difference between user's historical price and hotel price
        'user_price_diff_log'     # Log-transformed price difference
    ]
    
    # Ensure all features exist in the dataset
    features_to_use = [f for f in important_features if f in df.columns]
    print(f"Using {len(features_to_use)} features: {features_to_use}")
    
    # Prepare features and target
    print("Preparing features and target...")
    X = df[features_to_use]
    y_click = df['click_bool']
    y_booking = df['booking_bool']
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_click_train, y_click_temp, y_booking_train, y_booking_temp = train_test_split(
        X, y_click, y_booking, test_size=0.3, random_state=42)
    
    X_val, X_test, y_click_val, y_click_test, y_booking_val, y_booking_test = train_test_split(
        X_temp, y_click_temp, y_booking_temp, test_size=0.5, random_state=42)
    
    print(f"Train set: {X_train.shape[0]} rows")
    print(f"Validation set: {X_val.shape[0]} rows")
    print(f"Test set: {X_test.shape[0]} rows")
    
    # Create a balanced training set (4th place solution approach)
    print("Creating balanced training set...")
    pos_samples = X_train[y_booking_train == 1]
    pos_y_click = y_click_train[y_booking_train == 1]
    pos_y_booking = y_booking_train[y_booking_train == 1]
    
    # Sample negative examples (3:1 ratio of negative to positive)
    neg_indices = np.where(y_booking_train == 0)[0]
    neg_sample_size = min(len(pos_samples) * 3, len(neg_indices))
    neg_sample_indices = np.random.choice(neg_indices, size=neg_sample_size, replace=False)
    
    neg_samples = X_train.iloc[neg_sample_indices]
    neg_y_click = y_click_train.iloc[neg_sample_indices]
    neg_y_booking = y_booking_train.iloc[neg_sample_indices]
    
    # Combine positive and negative samples
    X_train_balanced = pd.concat([pos_samples, neg_samples])
    y_click_train_balanced = pd.concat([pos_y_click, neg_y_click])
    y_booking_train_balanced = pd.concat([pos_y_booking, neg_y_booking])
    
    print(f"Balanced train set: {len(X_train_balanced)} rows (Positive: {len(pos_samples)}, Negative: {len(neg_samples)})")
    
    # Check if we should train country-specific models (3rd place solution approach)
    country_specific = False
    if 'prop_country_id' in df.columns and len(df['prop_country_id'].unique()) > 1:
        country_counts = df['prop_country_id'].value_counts()
        major_countries = country_counts[country_counts > len(df) * 0.05].index.tolist()
        if len(major_countries) > 1:
            country_specific = True
            print(f"Will train separate models for {len(major_countries)} major countries")
    
    # Scale numerical features for logistic regression
    scaler = StandardScaler()
    X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression model for booking prediction
    print("\nTraining Logistic Regression model for booking prediction...")
    start_time = time.time()
    
    # Train with cross-validation for better generalization (2nd place solution approach)
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    lr_models = []
    
    # Instead of using VotingClassifier which is causing issues,
    # we'll implement a simpler ensemble approach
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_balanced_scaled, y_booking_train_balanced)):
        print(f"Training fold {fold+1}/{n_folds}")
        X_fold_train = X_train_balanced_scaled[train_idx]
        y_fold_train = y_booking_train_balanced.iloc[train_idx]
        
        # Train model on this fold
        fold_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        fold_model.fit(X_fold_train, y_fold_train)
        lr_models.append(fold_model)
    
    # Use the first model as our base model for feature importance and saving
    # We'll still use all models for prediction by averaging their outputs
    lr_model = lr_models[0]
    
    lr_train_time = time.time() - start_time
    
    # Evaluate logistic regression ensemble
    # Average predictions from all models in the ensemble
    lr_val_preds_list = []
    for model in lr_models:
        lr_val_preds_list.append(model.predict_proba(X_val_scaled)[:, 1])
    
    # Average the predictions
    lr_val_preds = np.mean(lr_val_preds_list, axis=0)
    lr_val_auc = roc_auc_score(y_booking_val, lr_val_preds)
    
    # Calculate precision-recall AUC
    precision, recall, _ = precision_recall_curve(y_booking_val, lr_val_preds)
    lr_val_pr_auc = auc(recall, precision)
    
    # Get F1 score at optimal threshold
    lr_thresholds = np.linspace(0, 1, 100)
    lr_f1_scores = [f1_score(y_booking_val, lr_val_preds >= threshold) for threshold in lr_thresholds]
    lr_best_idx = np.argmax(lr_f1_scores)
    lr_best_threshold = lr_thresholds[lr_best_idx]
    lr_best_f1 = lr_f1_scores[lr_best_idx]
    
    print(f"Logistic Regression Results:")
    print(f"  Training time: {lr_train_time:.2f} seconds")
    print(f"  Validation AUC: {lr_val_auc:.4f}")
    print(f"  Validation PR-AUC: {lr_val_pr_auc:.4f}")
    
    # Get feature importance for logistic regression
    # Calculate average coefficients across all models in the ensemble
    all_coefs = []
    for model in lr_models:
        if hasattr(model, 'coef_'):
            all_coefs.append(np.abs(model.coef_[0]))
    
    # Average the coefficients
    if all_coefs:
        avg_coef = np.mean(all_coefs, axis=0)
        lr_importance = pd.DataFrame({
            'feature': features_to_use,
            'importance': avg_coef
        }).sort_values('importance', ascending=False).reset_index(drop=True)
    else:
        # Fallback if no coefficients found
        lr_importance = pd.DataFrame({
            'feature': features_to_use,
            'importance': np.ones(len(features_to_use))
        }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Train LightGBM model for booking prediction
    print("\nTraining LightGBM model for booking prediction...")
    start_time = time.time()
    
    # Define LightGBM parameters (enhanced based on top solutions)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,  # L1 regularization (from top solutions)
        'lambda_l2': 0.1,  # L2 regularization (from top solutions)
        'verbose': -1,
        'seed': 42
    }
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_balanced, y_booking_train_balanced)
    val_data = lgb.Dataset(X_val, y_booking_val, reference=train_data)
    
    # Simplify training to avoid parameter issues
    print("Training LightGBM model...")
    
    # Train with early stopping
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    lgb_train_time = time.time() - start_time
    
    # Evaluate LightGBM model
    lgb_val_preds = lgb_model.predict(X_val)
    lgb_val_auc = roc_auc_score(y_booking_val, lgb_val_preds)
    
    # Calculate precision-recall AUC
    precision, recall, _ = precision_recall_curve(y_booking_val, lgb_val_preds)
    lgb_val_pr_auc = auc(recall, precision)
    
    # Get F1 score at optimal threshold
    lgb_thresholds = np.linspace(0, 1, 100)
    lgb_f1_scores = [f1_score(y_booking_val, lgb_val_preds >= threshold) for threshold in lgb_thresholds]
    lgb_best_threshold = lgb_thresholds[np.argmax(lgb_f1_scores)]
    lgb_best_f1 = max(lgb_f1_scores)
    
    print(f"LightGBM Results:")
    print(f"  Training time: {lgb_train_time:.2f} seconds")
    print(f"  Validation AUC: {lgb_val_auc:.4f}")
    print(f"  Validation PR-AUC: {lgb_val_pr_auc:.4f}")
    print(f"  Best F1 Score: {lgb_best_f1:.4f} (threshold: {lgb_best_threshold:.2f})")
    
    # Get feature importance from LightGBM
    lgb_importance = pd.DataFrame({
        'feature': features_to_use,
        'importance': lgb_model.feature_importance(importance_type='gain')
    })
    lgb_importance = lgb_importance.sort_values('importance', ascending=False)
    
    # Evaluate on test set
    print("\nEvaluating models on test set...")
    
    # Logistic Regression Ensemble
    lr_test_preds_list = []
    for model in lr_models:
        lr_test_preds_list.append(model.predict_proba(X_test_scaled)[:, 1])
    
    # Average the predictions
    lr_test_preds = np.mean(lr_test_preds_list, axis=0)
    lr_test_auc = roc_auc_score(y_booking_test, lr_test_preds)
    precision, recall, _ = precision_recall_curve(y_booking_test, lr_test_preds)
    lr_test_pr_auc = auc(recall, precision)
    lr_test_f1 = f1_score(y_booking_test, lr_test_preds >= lr_best_threshold)
    
    # LightGBM
    lgb_test_preds = lgb_model.predict(X_test)
    lgb_test_auc = roc_auc_score(y_booking_test, lgb_test_preds)
    precision, recall, _ = precision_recall_curve(y_booking_test, lgb_test_preds)
    lgb_test_pr_auc = auc(recall, precision)
    lgb_test_f1 = f1_score(y_booking_test, lgb_test_preds >= lgb_best_threshold)
    
    print(f"Logistic Regression Test Results:")
    print(f"  Test AUC: {lr_test_auc:.4f}")
    print(f"  Test PR-AUC: {lr_test_pr_auc:.4f}")
    print(f"  Test F1 Score: {lr_test_f1:.4f}")
    
    print(f"LightGBM Test Results:")
    print(f"  Test AUC: {lgb_test_auc:.4f}")
    print(f"  Test PR-AUC: {lgb_test_pr_auc:.4f}")
    print(f"  Test F1 Score: {lgb_test_f1:.4f}")
    
    # Save models
    print("\nSaving models...")
    import joblib
    # Save all logistic regression models in the ensemble
    for i, model in enumerate(lr_models):
        joblib.dump(model, f"{output_dir}/logistic_regression_model_fold_{i}.pkl")
    lgb_model.save_model(f"{output_dir}/lightgbm_model.txt")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    
    # Save feature importances
    lr_importance.to_csv(f"{output_dir}/lr_feature_importance.csv", index=False)
    lgb_importance.to_csv(f"{output_dir}/lgb_feature_importance.csv", index=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    sns.barplot(x='importance', y='feature', data=lr_importance.head(10))
    plt.title('Logistic Regression Feature Importance')
    plt.tight_layout()
    
    plt.subplot(2, 1, 2)
    sns.barplot(x='importance', y='feature', data=lgb_importance.head(10))
    plt.title('LightGBM Feature Importance')
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/feature_importance_comparison.png", dpi=300, bbox_inches='tight')
    
    # Plot confusion matrices
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    lr_cm = confusion_matrix(y_booking_test, lr_test_preds >= lr_best_threshold, normalize='true')
    sns.heatmap(lr_cm, annot=True, fmt='.2%', cmap='Blues')
    plt.title('Logistic Regression Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.subplot(1, 2, 2)
    lgb_cm = confusion_matrix(y_booking_test, lgb_test_preds >= lgb_best_threshold, normalize='true')
    sns.heatmap(lgb_cm, annot=True, fmt='.2%', cmap='Blues')
    plt.title('LightGBM Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
    
    # Return results
    results = {
        'lr_model': lr_model,
        'lgb_model': lgb_model,
        'lr_importance': lr_importance,
        'lgb_importance': lgb_importance,
        'lr_metrics': {
            'val_auc': lr_val_auc,
            'val_pr_auc': lr_val_pr_auc,
            'val_f1': lr_best_f1,
            'test_auc': lr_test_auc,
            'test_pr_auc': lr_test_pr_auc,
            'test_f1': lr_test_f1,
            'threshold': lr_best_threshold
        },
        'lgb_metrics': {
            'val_auc': lgb_val_auc,
            'val_pr_auc': lgb_val_pr_auc,
            'val_f1': lgb_best_f1,
            'test_auc': lgb_test_auc,
            'test_pr_auc': lgb_test_pr_auc,
            'test_f1': lgb_test_f1,
            'threshold': lgb_best_threshold
        }
    }
    
    return results

if __name__ == "__main__":
    # Train baseline classification models
    data_path = "data/processed/featured_training_set.csv"
    
    # For testing, you can use a smaller sample
    # Uncomment the line below to use a sample
    # sample_size = 100000  # Use a smaller sample for faster development
    sample_size = None  # Use the full dataset
    
    print("Starting baseline classification model training...")
    results = train_baseline_classification_models(data_path, sample_size=sample_size)
    
    print("\nBaseline classification model training complete!")
    print("Results saved in data/models/baseline/")
