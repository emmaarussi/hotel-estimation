# Hotel Ranking Model - Tess's Implementation

## Overview
This directory contains the implementation of enhanced feature engineering and two modeling approaches for the hotel ranking task:

1. **Baseline Classification Models** - Predicting booking directly using logistic regression and LightGBM
2. **Learning-to-Rank Model** - Using LightGBM with lambdarank objective to optimize ranking directly

## Files

### Feature Engineering
- `feature_engineering_tess.py` - Enhanced feature engineering script that adds cross-features and interaction terms

### Feature Analysis
- `analyze_feature_importance.py` - Script to analyze feature importance and identify the most predictive features

### Models
- `baseline_classification_model.py` - Implementation of classification models (Logistic Regression and LightGBM)
- `learning_to_rank_model.py` - Implementation of learning-to-rank model using LightGBM with lambdarank objective
- `run_models.py` - Script to run both modeling approaches and compare results

## Workflow

1. **Data Preprocessing**
   - Used the original preprocessing.py script to clean the data
   - Created clean_training_set.csv in data/processed/

2. **Enhanced Feature Engineering**
   - Added ranking features (review_rank, star_rank, location_rank)
   - Added geo features (is_domestic, log_orig_distance)
   - Added interaction terms (review_score_per_dollar, star_x_score)
   - Created relevance score target (5×booking + 1×click)
   - Output: featured_training_set.csv

3. **Feature Importance Analysis**
   - Identified most important features using LightGBM
   - Found that historical booking rate and click-through rate account for over 80% of predictive power
   - Position in search results is the third most important feature

4. **Model Training and Evaluation**
   - Implemented and compared two modeling approaches
   - Evaluated classification models using AUC, PR-AUC, and F1 score
   - Evaluated ranking model using NDCG@5 and NDCG@10
   - Identified common important features across models

## Running the Code

```bash
# Run feature engineering
source venv/bin/activate && python3 feature_engineering_tess.py

# Analyze feature importance
source venv/bin/activate && python3 analyze_feature_importance.py

# Run both models and compare
source venv/bin/activate && python3 run_models.py

# Run with a sample (for faster testing)
source venv/bin/activate && python3 run_models.py --sample 100000
```

## Results

Results from model training are saved in the following directories:
- Classification models: data/models/baseline/
- Ranking model: data/models/ranking/
- Comparison: data/models/model_comparison.csv
