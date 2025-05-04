# Hotel Click and Booking Prediction

This project implements a machine learning system to predict which hotel properties users are most likely to click on and book based on search results data. The model uses LightGBM for learning-to-rank optimization to improve search result rankings.

## Project Structure

```
hotel-estimation/
├── data/
│   └── raw/         # Raw training data (training_set_VU_DM.csv)
├── src/
│   ├── preprocess.py  # Data preprocessing and feature engineering
│   └── train_model.py  # Model training and evaluation
└── requirements.txt    # Project dependencies
```

## Features

### Data Preprocessing
- Temporal feature extraction (hour, day, month)
- Price-related feature engineering
- Competitor analysis features
- Location score processing
- Historical user preference features
- Search context features

### Model Implementation
- LightGBM ranking model
- Hyperparameter optimization using Optuna
- Learning-to-rank objective (LambdaRank)
- Evaluation metrics:
  - AUC-ROC for clicks and bookings
  - Average Precision
  - NDCG@10 for ranking quality

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
- Place the training dataset (`training_set_VU_DM.csv`) in the `data/raw/` directory

3. Run the model:
```bash
python src/train_model.py
```

## Model Details

The system uses a LightGBM model optimized for ranking tasks with the following key components:

1. Feature Engineering:
   - Temporal patterns
   - Price comparisons
   - User history
   - Property characteristics
   - Search context

2. Training Pipeline:
   - Data preprocessing
   - Feature scaling and encoding
   - Hyperparameter optimization
   - Model training with early stopping
   - Performance evaluation

3. Evaluation Metrics:
   - Click-through rate prediction
   - Booking probability prediction
   - Ranking quality assessment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
