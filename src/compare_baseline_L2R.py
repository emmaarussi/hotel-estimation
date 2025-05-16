import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import argparse

# Import our model training functions - using relative imports within tess-try directory
from baseline_classification_model import train_baseline_classification_models
from learning_to_rank_model import train_learning_to_rank_model

def run_models(data_path, output_dir='data/models/tess-try', sample_size=None):
    """Run both modeling approaches and compare results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Train baseline classification models
    print("\n" + "="*80)
    print("TRAINING BASELINE CLASSIFICATION MODELS")
    print("="*80)
    baseline_results = train_baseline_classification_models(
        data_path, 
        output_dir=f"{output_dir}/baseline", 
        sample_size=sample_size
    )
    
    # Train learning-to-rank model
    print("\n" + "="*80)
    print("TRAINING LEARNING-TO-RANK MODEL")
    print("="*80)
    ranking_results = train_learning_to_rank_model(
        data_path, 
        output_dir=f"{output_dir}/ranking", 
        sample_size=sample_size
    )
    
    # Compare results
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Create comparison table with enhanced metrics
    comparison = pd.DataFrame([
        {
            'Model': 'Logistic Regression (Classification)',
            'Task': 'Binary Classification (booking)',
            'AUC': baseline_results['lr_metrics']['test_auc'],
            'PR-AUC': baseline_results['lr_metrics']['test_pr_auc'],
            'F1 Score': baseline_results['lr_metrics']['test_f1'],
            'Top Feature': baseline_results['lr_importance']['feature'].iloc[0],
            'Training Time (s)': baseline_results['lr_metrics'].get('training_time', 0)
        },
        {
            'Model': 'LightGBM (Classification)',
            'Task': 'Binary Classification (booking)',
            'AUC': baseline_results['lgb_metrics']['test_auc'],
            'PR-AUC': baseline_results['lgb_metrics']['test_pr_auc'],
            'F1 Score': baseline_results['lgb_metrics']['test_f1'],
            'Top Feature': baseline_results['lgb_importance']['feature'].iloc[0],
            'Training Time (s)': baseline_results['lgb_metrics'].get('training_time', 0)
        },
        {
            'Model': 'LightGBM (LambdaRank)',
            'Task': 'Learning-to-Rank (relevance score)',
            'AUC': None,  # Not applicable for ranking model
            'PR-AUC': None,  # Not applicable for ranking model
            'F1 Score': None,  # Not applicable for ranking model
            'NDCG@5': ranking_results['metrics']['ndcg@5'],
            'NDCG@10': ranking_results['metrics']['ndcg@10'],
            'MRR': ranking_results['metrics'].get('mrr', None),  # Mean Reciprocal Rank (from top solutions)
            'Top Feature': ranking_results['importance']['feature'].iloc[0],
            'Training Time (s)': ranking_results['metrics']['training_time']
        }
    ])
    
    # Save comparison table
    comparison.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    # Print comparison table
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))
    
    # Create feature importance comparison visualization
    print("\nCreating feature importance comparison visualization...")
    
    # Get top 10 features from each model
    lr_top10 = set(baseline_results['lr_importance']['feature'].head(10))
    lgb_top10 = set(baseline_results['lgb_importance']['feature'].head(10))
    rank_top10 = set(ranking_results['importance']['feature'].head(10))
    
    # Find common top features
    common_features = lr_top10.intersection(lgb_top10).intersection(rank_top10)
    
    print(f"\nCommon top features across all models: {common_features}")
    
    # Check if we have country-specific models
    if 'country_models' in ranking_results:
        print("\nCountry-specific model analysis:")
        country_models = ranking_results['country_models']
        country_importances = ranking_results['country_importances']
        
        # Compare top features across countries
        print("Top 3 features by country:")
        for country_id, importance_df in country_importances.items():
            top3 = importance_df.head(3)['feature'].tolist()
            print(f"  Country {country_id}: {', '.join(top3)}")
            
        # Create country-specific comparison visualization
        plt.figure(figsize=(12, 8))
        country_ids = list(country_importances.keys())
        
        for i, country_id in enumerate(country_ids[:min(4, len(country_ids))]):
            plt.subplot(2, 2, i+1)
            top5 = country_importances[country_id].head(5)
            sns.barplot(x='importance', y='feature', data=top5)
            plt.title(f'Country {country_id} Top Features')
            plt.tight_layout()
            
        plt.savefig(f"{output_dir}/country_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create Venn diagram of top features
    try:
        from matplotlib_venn import venn3
        plt.figure(figsize=(10, 8))
        venn3([lr_top10, lgb_top10, rank_top10], 
              ('Logistic Regression', 'LightGBM Classification', 'LightGBM Ranking'))
        plt.title('Top 10 Features Overlap Between Models')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_features_venn_diagram.png", dpi=300, bbox_inches='tight')
    except ImportError:
        print("matplotlib_venn not installed. Skipping Venn diagram.")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Create a summary of enhancements based on top Expedia solutions
    print("\nEnhancements implemented from top Expedia competition solutions:")
    print("1. Feature Engineering:")
    print("   - Monotonic features (3rd place solution)")
    print("   - Group-based normalization (3rd place solution)")
    print("   - Composite features using F1×max(F2)+F2 formula (4th place solution)")
    print("   - User-property match features (3rd place solution)")
    
    print("2. Model Training:")
    print("   - Balanced sampling (4th place solution)")
    print("   - Cross-validation (2nd place solution)")
    print("   - Country-specific models (3rd place solution)")
    print("   - Ensemble techniques (1st place solution)")
    
    print("3. Evaluation Metrics:")
    print("   - Mean Reciprocal Rank (MRR)")
    print("   - NDCG@5 and NDCG@10")
    
    print("\nAll results saved in:")
    print(f"- Classification models: {output_dir}/baseline/")
    print(f"- Ranking model: {output_dir}/ranking/")
    print(f"- Comparison: {output_dir}/model_comparison.csv")
    
    return {
        'baseline': baseline_results,
        'ranking': ranking_results,
        'comparison': comparison
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hotel ranking models')
    parser.add_argument('--sample', type=int, default=None, 
                      help='Sample size to use (default: use full dataset)')
    parser.add_argument('--output', type=str, default='data/models/tess-try',
                      help='Output directory for models and results')
    args = parser.parse_args()
    
    data_path = "data/processed/featured_training_set.csv"
    
    print(f"Starting hotel ranking model training...")
    if args.sample:
        print(f"Using sample of {args.sample} rows")
    else:
        print("Using full dataset")
    
    results = run_models(data_path, output_dir=args.output, sample_size=args.sample)
    
    print("\nHotel ranking model training and comparison complete!")