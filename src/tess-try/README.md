# Hotel Ranking Model - Tess's Enhanced Implementation

## Overview
This directory contains my enhanced implementation for the hotel ranking task, inspired by top solutions from the Expedia Personalized Hotel Recommendations Challenge. My implementation builds upon the original project but adds significant improvements in feature engineering and modeling approaches.

I've implemented two advanced modeling approaches:

1. **Enhanced Baseline Classification Models** - Predicting booking directly using ensemble logistic regression and LightGBM with balanced sampling
2. **Advanced Learning-to-Rank Model** - Using LightGBM with lambdarank objective and country-specific models to optimize ranking directly

## Project Structure

### Code (src/tess-try/)
- `feature_engineering_tess.py` - Enhanced feature engineering script with monotonic features and group-based normalization
- `analyze_feature_importance.py` - Script to analyze feature importance and identify the most predictive features
- `baseline_classification_model.py` - Implementation of classification models with ensemble techniques and balanced sampling
- `learning_to_rank_model.py` - Implementation of learning-to-rank model with country-specific modeling
- `run_models.py` - Script to run both modeling approaches and compare results
- `README.md` - This documentation file

### Data and Results
- `data/processed/tess-try/` - Processed datasets with enhanced features
- `data/models/tess-try/baseline/` - Classification model outputs and visualizations
- `data/models/tess-try/ranking/` - Learning-to-rank model outputs and visualizations
- `data/models/tess-try/model_comparison.csv` - Comparison of all model performances
- `data/analysis/tess-try/feature_importance/` - Feature importance analysis results

## Workflow

1. **Data Preprocessing**
   - Used the original preprocessing.py script to clean the data
   - Created clean_training_set.csv in data/processed/

2. **Enhanced Feature Engineering**
   - Added ranking features (review_rank, star_rank, location_rank)
   - Added geo features (is_domestic, log_orig_distance)
   - Added interaction terms (review_score_per_dollar, star_x_score)
   - Created monotonic features (prop_starrating_monotonic, prop_review_monotonic, prop_location_monotonic)
   - Implemented group-based normalization (price_norm_by_srch_destination_id, price_norm_by_prop_country_id)
   - Added composite features using F1×max(F2)+F2 formula (star_review_composite, location_review_composite)
   - Added user-property match features (user_star_diff, user_price_diff)
   - Created relevance score target (5×booking + 1×click)
   - Output: featured_training_set.csv

3. **Feature Importance Analysis**
   - Identified most important features using LightGBM
   - Found that historical booking rate and click-through rate account for over 80% of predictive power
   - Position in search results is the third most important feature

4. **Model Training and Evaluation**
   - Implemented and compared two modeling approaches based on top Expedia competition solutions
   - Used balanced sampling (3:1 negative to positive ratio) for classification models
   - Implemented cross-validation with 5 folds for logistic regression
   - Created ensemble approach for logistic regression by averaging predictions
   - Implemented country-specific models for major countries
   - Enhanced regularization with L1 and L2 penalties
   - Evaluated classification models using AUC, PR-AUC, and F1 score
   - Evaluated ranking model using NDCG@5, NDCG@10, and Mean Reciprocal Rank (MRR)
   - Identified common important features across models and countries

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
- Classification models: data/models/tess-try/baseline/
- Ranking model: data/models/tess-try/ranking/
- Comparison: data/models/tess-try/model_comparison.csv

### Performance Metrics

#### Classification Models:
- **Logistic Regression Ensemble**:
  - AUC: 0.980
  - PR-AUC: 0.645
  - F1 Score: 0.554

- **LightGBM Classification**:
  - AUC: 0.981
  - PR-AUC: 0.602
  - F1 Score: 0.563

#### Learning-to-Rank Model:
- **LightGBM with LambdaRank**:
  - NDCG@5: 0.814
  - NDCG@10: 0.804
  - MRR: 0.796

### Key Findings

1. The most important features across all models are:
   - prop_historical_ctr (historical click-through rate)
   - prop_historical_br (historical booking rate)
   - position (position in search results)
   - price-related features (price_diff_from_mean, price_percentile)

2. Country-specific models show different feature importance patterns:
   - Country 219: prop_historical_ctr, prop_historical_br, position
   - Country 100: prop_historical_ctr, position, price_diff_from_mean
