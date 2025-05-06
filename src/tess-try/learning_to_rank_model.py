import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import os
import time

def train_learning_to_rank_model(data_path, output_dir='data/models/ranking', sample_size=None):
    """Train a learning-to-rank model using LightGBM with lambdarank objective"""
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
    
    # Ensure we have the relevance score (5*booking + 1*click)
    if 'relevance_score' not in df.columns:
        print("Creating relevance score...")
        df['relevance_score'] = 5 * df['booking_bool'] + 1 * df['click_bool']
    
    # Drop unnecessary columns
    drop_cols = ['click_bool', 'booking_bool', 'gross_bookings_usd', 'date_time']
    drop_cols = [col for col in drop_cols if col in df.columns]
    
    # Get all feature columns
    feature_cols = [col for col in df.columns if col not in drop_cols + ['relevance_score', 'srch_id']]
    print(f"Using {len(feature_cols)} features")
    
    # Split data by search_id to maintain the ranking context
    print("Splitting data by search_id...")
    unique_searches = df['srch_id'].unique()
    train_searches, temp_searches = train_test_split(unique_searches, test_size=0.3, random_state=42)
    val_searches, test_searches = train_test_split(temp_searches, test_size=0.5, random_state=42)
    
    # Create train, validation, and test sets
    train_df = df[df['srch_id'].isin(train_searches)]
    val_df = df[df['srch_id'].isin(val_searches)]
    test_df = df[df['srch_id'].isin(test_searches)]
    
    print(f"Train set: {len(train_df)} rows, {len(train_searches)} searches")
    print(f"Validation set: {len(val_df)} rows, {len(val_searches)} searches")
    print(f"Test set: {len(test_df)} rows, {len(test_searches)} searches")
    
    # Prepare features and target
    X_train = train_df[feature_cols]
    y_train = train_df['relevance_score']
    q_train = train_df['srch_id']
    
    X_val = val_df[feature_cols]
    y_val = val_df['relevance_score']
    q_val = val_df['srch_id']
    
    X_test = test_df[feature_cols]
    y_test = test_df['relevance_score']
    q_test = test_df['srch_id']
    
    # Create LightGBM datasets with query information
    print("Creating LightGBM datasets...")
    
    # Convert query IDs to group sizes
    train_groups = train_df.groupby('srch_id').size().values
    val_groups = val_df.groupby('srch_id').size().values
    test_groups = test_df.groupby('srch_id').size().values
    
    print(f"Number of training groups: {len(train_groups)}")
    print(f"Number of validation groups: {len(val_groups)}")
    print(f"Number of test groups: {len(test_groups)}")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, y_train, group=train_groups)
    
    # Only create validation dataset if there are validation groups
    if len(val_groups) > 0:
        val_data = lgb.Dataset(X_val, y_val, group=val_groups, reference=train_data)
    else:
        val_data = None
    
    # Define LightGBM parameters for lambdarank
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',  # Use a single metric to avoid issues
        'eval_at': [5, 10],  # This evaluates NDCG@5 and NDCG@10
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'min_sum_hessian_in_leaf': 1e-3,
        'verbose': -1,
        'seed': 42
    }
    
    # Train model
    print("\nTraining LightGBM ranking model...")
    start_time = time.time()
    
    # Check if validation data exists
    if val_data is None:
        print("Warning: No validation data available. Using training data for validation.")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data],  # Use training data for validation if no validation data
            callbacks=[
                lgb.log_evaluation(period=100)
            ]
        )
    else:
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    })
    importance = importance.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    
    # Make predictions
    test_preds = model.predict(X_test)
    
    # Calculate NDCG@k for each query group
    ndcg_scores = []
    ndcg5_scores = []
    ndcg10_scores = []
    
    # Group test data by search_id
    test_groups_df = test_df.groupby('srch_id')
    
    # Calculate NDCG for each search
    for srch_id, group in test_groups_df:
        # Get true relevance scores and predictions for this search
        true_relevance = group['relevance_score'].values
        preds = test_preds[test_df['srch_id'] == srch_id]
        
        # Skip if all relevance scores are 0
        if np.sum(true_relevance) == 0:
            continue
        
        # Calculate NDCG@5 and NDCG@10
        try:
            # Reshape for sklearn's ndcg_score function
            true_relevance = true_relevance.reshape(1, -1)
            preds = preds.reshape(1, -1)
            
            # Calculate scores
            if len(true_relevance[0]) >= 5:
                ndcg5 = ndcg_score(true_relevance, preds, k=5)
                ndcg5_scores.append(ndcg5)
            
            if len(true_relevance[0]) >= 10:
                ndcg10 = ndcg_score(true_relevance, preds, k=10)
                ndcg10_scores.append(ndcg10)
        except Exception as e:
            print(f"Error calculating NDCG for search {srch_id}: {e}")
    
    # Calculate average NDCG scores
    avg_ndcg5 = np.mean(ndcg5_scores)
    avg_ndcg10 = np.mean(ndcg10_scores)
    
    print(f"Average NDCG@5: {avg_ndcg5:.4f}")
    print(f"Average NDCG@10: {avg_ndcg10:.4f}")
    
    # Save model and results
    print("\nSaving model and results...")
    model.save_model(f"{output_dir}/lightgbm_ranking_model.txt")
    importance.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    
    # Save evaluation metrics
    metrics = pd.DataFrame({
        'Metric': ['NDCG@5', 'NDCG@10', 'Training Time (s)'],
        'Value': [avg_ndcg5, avg_ndcg10, train_time]
    })
    metrics.to_csv(f"{output_dir}/evaluation_metrics.csv", index=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance.head(20))
    plt.title('Top 20 Features for Learning-to-Rank Model')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
    
    # Create a function to visualize rankings
    def visualize_rankings(model, df, feature_cols, search_ids, output_dir):
        """Visualize rankings for specific search IDs"""
        for srch_id in search_ids:
            # Get data for this search
            search_df = df[df['srch_id'] == srch_id].copy()
            
            if len(search_df) == 0:
                continue
            
            # Make predictions
            search_df['predicted_score'] = model.predict(search_df[feature_cols])
            
            # Sort by predicted score and original position
            search_df['original_rank'] = search_df['position'].rank(method='dense')
            search_df['predicted_rank'] = search_df['predicted_score'].rank(method='dense', ascending=False)
            
            # Select columns for visualization
            vis_df = search_df[['prop_id', 'position', 'original_rank', 'predicted_rank', 
                               'predicted_score', 'relevance_score', 'prop_starrating', 
                               'price_usd', 'prop_review_score']].sort_values('predicted_score', ascending=False)
            
            # Save to CSV
            vis_df.to_csv(f"{output_dir}/search_{srch_id}_ranking.csv", index=False)
            
            # Create visualization
            plt.figure(figsize=(14, 8))
            
            # Plot original vs predicted ranks
            plt.subplot(1, 2, 1)
            plt.scatter(vis_df['original_rank'], vis_df['predicted_rank'], 
                      c=vis_df['relevance_score'], cmap='viridis', 
                      s=100, alpha=0.7, edgecolors='k')
            plt.colorbar(label='Relevance Score')
            plt.xlabel('Original Rank')
            plt.ylabel('Predicted Rank')
            plt.title(f'Original vs Predicted Rankings\nSearch ID: {srch_id}')
            plt.grid(True, alpha=0.3)
            
            # Plot price vs star rating with relevance
            plt.subplot(1, 2, 2)
            scatter = plt.scatter(vis_df['price_usd'], vis_df['prop_starrating'],
                               c=vis_df['relevance_score'], cmap='viridis',
                               s=100, alpha=0.7, edgecolors='k')
            plt.colorbar(scatter, label='Relevance Score')
            
            # Add rank labels to points
            for i, row in vis_df.head(10).iterrows():
                plt.annotate(f"{int(row['predicted_rank'])}", 
                           (row['price_usd'], row['prop_starrating']),
                           xytext=(5, 5), textcoords='offset points')
            
            plt.xlabel('Price (USD)')
            plt.ylabel('Star Rating')
            plt.title('Price vs Star Rating\nColor = Relevance, Labels = Predicted Rank')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/search_{srch_id}_visualization.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Visualize rankings for a few example searches
    print("\nVisualizing example rankings...")
    example_searches = test_searches[:5]  # Take first 5 test searches
    visualize_rankings(model, test_df, feature_cols, example_searches, output_dir)
    
    # Return results
    results = {
        'model': model,
        'importance': importance,
        'metrics': {
            'ndcg@5': avg_ndcg5,
            'ndcg@10': avg_ndcg10,
            'training_time': train_time
        }
    }
    
    return results

if __name__ == "__main__":
    # Train learning-to-rank model
    data_path = "data/processed/featured_training_set.csv"
    
    # For testing, you can use a smaller sample
    # Uncomment the line below to use a sample
    # sample_size = 100000  # Use a smaller sample for faster development
    sample_size = None  # Use the full dataset
    
    print("Starting learning-to-rank model training...")
    results = train_learning_to_rank_model(data_path, sample_size=sample_size)
    
    print("\nLearning-to-rank model training complete!")
    print("Results saved in data/models/ranking/")
