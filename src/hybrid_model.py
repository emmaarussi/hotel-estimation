import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from feature_engineering import EnhancedFeatureEngineer

class TwoStageHybridModel:
    """
    A two-stage hybrid model that combines classification and ranking:
    1. Stage 1: Fast classification model to pre-filter promising hotels
    2. Stage 2: Refined ranking model for final ordering of pre-filtered hotels
    """
    
    def __init__(
        self,
        classification_params: Optional[Dict] = None,
        ranking_params: Optional[Dict] = None,
        filter_percentile: float = 0.3  # Keep top 30% from classifier by default
    ):
        # Default parameters for the classification model (optimized for speed)
        self.classification_params = classification_params or {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'goss',  # Gradient-based One-Side Sampling for faster training
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'verbose': -1,
            'random_state': 42
        }
        
        # Default parameters for the ranking model (optimized for precision)
        self.ranking_params = ranking_params or {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10],
            'boosting_type': 'gbdt',
            'n_estimators': 200,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'verbose': -1,
            'random_state': 42
        }
        
        self.filter_percentile = filter_percentile
        self.classification_model = None
        self.ranking_model = None
        self.feature_scaler = StandardScaler()
        
    def _prepare_ranking_data(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        groups: np.ndarray
    ) -> Tuple[lgb.Dataset, List[int]]:
        """Prepare data for LightGBM ranking model."""
        group_sizes = y.groupby(groups).size().values
        return lgb.Dataset(X, y, group=group_sizes), group_sizes
    
    def _get_top_hotels(
        self,
        X: pd.DataFrame,
        classifier_scores: np.ndarray,
        groups: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Get top hotels based on classifier scores within each group."""
        df_scores = pd.DataFrame({
            'group': groups,
            'score': classifier_scores,
            'index': range(len(X))
        })
        
        # Calculate threshold for each group
        thresholds = df_scores.groupby('group')['score'].quantile(1 - self.filter_percentile)
        
        # Get indices of hotels to keep
        keep_indices = []
        for group in df_scores['group'].unique():
            group_scores = df_scores[df_scores['group'] == group]
            threshold = thresholds[group]
            keep_idx = group_scores[group_scores['score'] >= threshold]['index'].values
            keep_indices.extend(keep_idx)
        
        keep_indices = sorted(keep_indices)
        return X.iloc[keep_indices], np.array(keep_indices)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        groups: np.ndarray,
        eval_fraction: float = 0.2
    ) -> Dict:
        """
        Train both models using cross-validation.
        
        Args:
            X: Feature matrix
            y: DataFrame with 'click_bool' and 'booking_bool' columns
            groups: Array of group IDs (e.g., search_ids)
            eval_fraction: Fraction of data to use for evaluation
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("Scaling features...")
        X_scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Split data for evaluation
        gkf = GroupKFold(n_splits=int(1/eval_fraction))
        train_idx, eval_idx = next(gkf.split(X, y, groups))
        
        X_train = X_scaled.iloc[train_idx]
        y_train = y.iloc[train_idx]
        groups_train = groups[train_idx]
        
        X_eval = X_scaled.iloc[eval_idx]
        y_eval = y.iloc[eval_idx]
        groups_eval = groups[eval_idx]
        
        print("\nTraining classification model...")
        self.classification_model = lgb.train(
            self.classification_params,
            lgb.Dataset(X_train, y_train['booking_bool']),
            valid_sets=[lgb.Dataset(X_eval, y_eval['booking_bool'])],
            callbacks=[lgb.early_stopping(50)]
        )
        
        # Get classifier predictions for training ranking model
        print("\nFiltering hotels using classifier...")
        classifier_scores_train = self.classification_model.predict(X_train)
        X_train_filtered, train_indices = self._get_top_hotels(
            X_train, classifier_scores_train, groups_train
        )
        y_train_filtered = y_train.iloc[train_indices]
        groups_train_filtered = groups_train[train_indices]
        
        print("\nTraining ranking model on filtered data...")
        train_data, group_sizes = self._prepare_ranking_data(
            X_train_filtered,
            y_train_filtered['booking_bool'],
            groups_train_filtered
        )
        
        self.ranking_model = lgb.train(
            self.ranking_params,
            train_data,
            callbacks=[lgb.early_stopping(50)]
        )
        
        # Evaluate the full pipeline
        print("\nEvaluating hybrid model...")
        metrics = self.evaluate(X_eval, y_eval, groups_eval)
        
        return metrics
    
    def predict(
        self,
        X: pd.DataFrame,
        groups: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using the two-stage model.
        
        Returns:
            Tuple of (classifier_scores, ranking_scores)
        """
        X_scaled = pd.DataFrame(
            self.feature_scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Stage 1: Classification
        classifier_scores = self.classification_model.predict(X_scaled)
        
        # Filter hotels
        X_filtered, indices = self._get_top_hotels(X_scaled, classifier_scores, groups)
        
        # Stage 2: Ranking
        ranking_scores = np.zeros(len(X))
        ranking_scores[indices] = self.ranking_model.predict(X_filtered)
        
        return classifier_scores, ranking_scores
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        groups: np.ndarray
    ) -> Dict:
        """Evaluate the model using various metrics."""
        classifier_scores, ranking_scores = self.predict(X, groups)
        
        # Calculate metrics
        metrics = {}
        
        # Classification metrics
        metrics['classifier_auc'] = roc_auc_score(
            y['booking_bool'],
            classifier_scores
        )
        metrics['classifier_ap'] = average_precision_score(
            y['booking_bool'],
            classifier_scores
        )
        
        # Ranking metrics
        df_results = pd.DataFrame({
            'group': groups,
            'true_label': y['booking_bool'],
            'classifier_score': classifier_scores,
            'ranking_score': ranking_scores
        })
        
        ndcg_values = []
        for _, group_data in df_results.groupby('group'):
            if len(group_data) > 1:  # Need at least 2 items for ranking
                ndcg = ndcg_score(
                    [group_data['true_label'].values],
                    [group_data['ranking_score'].values],
                    k=min(10, len(group_data))
                )
                ndcg_values.append(ndcg)
        
        metrics['ranking_ndcg10'] = np.mean(ndcg_values)
        
        # Calculate improvement over baseline
        baseline_ndcg = []
        for _, group_data in df_results.groupby('group'):
            if len(group_data) > 1:
                ndcg = ndcg_score(
                    [group_data['true_label'].values],
                    [group_data['classifier_score'].values],
                    k=min(10, len(group_data))
                )
                baseline_ndcg.append(ndcg)
        
        metrics['baseline_ndcg10'] = np.mean(baseline_ndcg)
        metrics['ndcg_improvement'] = (
            metrics['ranking_ndcg10'] - metrics['baseline_ndcg10']
        ) / metrics['baseline_ndcg10'] * 100
        
        return metrics


def main():
    """Example usage of the TwoStageHybridModel."""
    print("Loading data...")
    df = pd.read_csv("data/processed/featured_training_set.csv")
    
    # Prepare features and targets
    feature_cols = [col for col in df.columns 
                   if col not in ['click_bool', 'booking_bool', 'gross_bookings_usd']]
    X = df[feature_cols]
    y = df[['click_bool', 'booking_bool']]
    groups = df['srch_id'].values
    
    # Initialize and train the model
    model = TwoStageHybridModel(filter_percentile=0.3)
    metrics = model.fit(X, y, groups)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Classification AUC: {metrics['classifier_auc']:.4f}")
    print(f"Classification AP: {metrics['classifier_ap']:.4f}")
    print(f"Ranking NDCG@10: {metrics['ranking_ndcg10']:.4f}")
    print(f"Baseline NDCG@10: {metrics['baseline_ndcg10']:.4f}")
    print(f"NDCG Improvement: {metrics['ndcg_improvement']:.1f}%")


if __name__ == "__main__":
    main()
