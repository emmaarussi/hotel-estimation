import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os

def analyze_feature_importance(data_path, output_dir='data/analysis/feature_importance', n_top_features=30):
    """Analyze feature importance using LightGBM model"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the enhanced dataset
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    print("Preparing features and target...")
    X = df.drop(['click_bool', 'booking_bool', 'gross_bookings_usd', 'relevance_score', 'srch_id', 'date_time'], 
               errors='ignore', axis=1)
    y = df['relevance_score']
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create LightGBM datasets
    print("Creating LightGBM datasets...")
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Set parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Train model
    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=10)
        ]
    )
    
    # Get feature importance
    print("Calculating feature importance...")
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance(importance_type='gain')
    })
    importance = importance.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Save feature importance to CSV
    importance.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    
    # Plot top N features
    print(f"Plotting top {n_top_features} features...")
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=importance.head(n_top_features))
    plt.title(f'Top {n_top_features} Most Important Features')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_{n_top_features}_features.png", dpi=300, bbox_inches='tight')
    
    # Create feature importance groups
    print("Creating feature importance groups...")
    # Calculate cumulative importance
    importance['cumulative_importance'] = importance['importance'].cumsum() / importance['importance'].sum()
    
    # Group features by importance
    critical_features = importance[importance['cumulative_importance'] <= 0.5]['feature'].tolist()
    important_features = importance[(importance['cumulative_importance'] > 0.5) & 
                                  (importance['cumulative_importance'] <= 0.9)]['feature'].tolist()
    useful_features = importance[(importance['cumulative_importance'] > 0.9) & 
                               (importance['cumulative_importance'] <= 0.99)]['feature'].tolist()
    marginal_features = importance[importance['cumulative_importance'] > 0.99]['feature'].tolist()
    
    feature_groups = pd.DataFrame({
        'group': ['Critical', 'Important', 'Useful', 'Marginal'],
        'count': [len(critical_features), len(important_features), len(useful_features), len(marginal_features)],
        'cumulative_importance': [0.5, 0.4, 0.09, 0.01]
    })
    
    # Save feature groups to CSV
    feature_groups.to_csv(f"{output_dir}/feature_groups.csv", index=False)
    
    # Plot feature importance distribution
    plt.figure(figsize=(10, 6))
    plt.bar(feature_groups['group'], feature_groups['count'], alpha=0.7)
    plt.title('Feature Count by Importance Group')
    plt.ylabel('Number of Features')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_groups.png", dpi=300, bbox_inches='tight')
    
    # Print results
    print("\nFeature Importance Analysis Results:")
    print(f"Total number of features: {len(feature_names)}")
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))
    
    print("\nFeature Groups:")
    print(feature_groups)
    
    print(f"\nCritical Features (Top {len(critical_features)} features that account for 50% of importance):")
    for i, feature in enumerate(critical_features[:10], 1):
        print(f"{i}. {feature}")
    if len(critical_features) > 10:
        print(f"...and {len(critical_features) - 10} more")
    
    return importance, feature_groups, model

if __name__ == "__main__":
    # Run feature importance analysis
    data_path = "data/processed/featured_training_set.csv"
    importance, groups, model = analyze_feature_importance(data_path)
    
    print("\nAnalysis complete! Results saved in data/analysis/feature_importance/")
