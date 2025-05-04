import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import optuna
from preprocess import load_and_preprocess_data, prepare_features_and_targets

def create_lgb_dataset(X, y, group_ids=None):
    """Create LightGBM dataset with group information for ranking"""
    if group_ids is not None:
        groups = y.groupby(group_ids).size().values
        return lgb.Dataset(X, y, group=groups)
    return lgb.Dataset(X, y)

def objective(trial, train_data, valid_data):
    """Optuna objective for hyperparameter optimization"""
    param = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.001, 0.1),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "random_state": 42
    }
    
    model = lgb.train(param, train_data, valid_sets=[valid_data], 
                      num_boost_round=param["n_estimators"],
                      early_stopping_rounds=50, verbose_eval=False)
    
    return model.best_score["valid_0"]["ndcg@10"]

def train_ranking_model(X_train, y_train, X_val, y_val, feature_names):
    """Train LightGBM ranking model with hyperparameter optimization"""
    print("Creating training datasets...")
    train_data = create_lgb_dataset(X_train, y_train['booking_bool'], 
                                  group_ids=y_train.index)
    valid_data = create_lgb_dataset(X_val, y_val['booking_bool'], 
                                  group_ids=y_val.index)
    
    print("Optimizing hyperparameters...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_data, valid_data), 
                  n_trials=20, show_progress_bar=True)
    
    best_params = study.best_params
    best_params.update({
        "objective": "lambdarank",
        "metric": "ndcg",
        "boosting_type": "gbdt",
        "random_state": 42
    })
    
    print("Training final model with best parameters...")
    final_model = lgb.train(best_params, train_data, valid_sets=[valid_data],
                           num_boost_round=best_params["n_estimators"],
                           early_stopping_rounds=50)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': final_model.feature_importance(importance_type='gain')
    })
    importance = importance.sort_values('importance', ascending=False)
    
    return final_model, importance

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    
    # Calculate AUC-ROC and Average Precision for clicks and bookings
    click_auc = roc_auc_score(y_test['click_bool'], predictions)
    booking_auc = roc_auc_score(y_test['booking_bool'], predictions)
    click_ap = average_precision_score(y_test['click_bool'], predictions)
    booking_ap = average_precision_score(y_test['booking_bool'], predictions)
    
    print("\nModel Evaluation Metrics:")
    print(f"Click AUC-ROC: {click_auc:.4f}")
    print(f"Booking AUC-ROC: {booking_auc:.4f}")
    print(f"Click Average Precision: {click_ap:.4f}")
    print(f"Booking Average Precision: {booking_ap:.4f}")
    
    return {
        'click_auc': click_auc,
        'booking_auc': booking_auc,
        'click_ap': click_ap,
        'booking_ap': booking_ap
    }

if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data("data/raw/training_set_VU_DM.csv")
    
    # Prepare features and targets
    print("Preparing features and targets...")
    results = prepare_features_and_targets(df)
    X_train, X_test, y_train, y_test, feature_names = results[:5]
    
    # Train model
    print("Training model...")
    model, feature_importance = train_ranking_model(X_train, y_train, X_test, y_test, 
                                                  feature_names)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print top features
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
