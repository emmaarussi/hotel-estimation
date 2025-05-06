import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import argparse

# Import our model training functions
from baseline_classification_model import train_baseline_classification_models
from learning_to_rank_model import train_learning_to_rank_model

def run_models(data_path, output_dir='data/models', sample_size=None):
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
    
    # Create comparison table
    comparison = pd.DataFrame([
        {
            'Model': 'Logistic Regression (Classification)',
            'Task': 'Binary Classification (booking)',
            'AUC': baseline_results['lr_metrics']['test_auc'],
            'PR-AUC': baseline_results['lr_metrics']['test_pr_auc'],
            'F1 Score': baseline_results['lr_metrics']['test_f1'],
            'Top Feature': baseline_results['lr_importance']['feature'].iloc[0]
        },
        {
            'Model': 'LightGBM (Classification)',
            'Task': 'Binary Classification (booking)',
            'AUC': baseline_results['lgb_metrics']['test_auc'],
            'PR-AUC': baseline_results['lgb_metrics']['test_pr_auc'],
            'F1 Score': baseline_results['lgb_metrics']['test_f1'],
            'Top Feature': baseline_results['lgb_importance']['feature'].iloc[0]
        },
        {
            'Model': 'LightGBM (LambdaRank)',
            'Task': 'Learning-to-Rank (relevance score)',
            'NDCG@5': ranking_results['metrics']['ndcg@5'],
            'NDCG@10': ranking_results['metrics']['ndcg@10'],
            'Top Feature': ranking_results['importance']['feature'].iloc[0]
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
    parser.add_argument('--output', type=str, default='data/models',
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
