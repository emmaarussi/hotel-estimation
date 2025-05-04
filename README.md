# Hotel Click and Booking Prediction

This project implements a machine learning system to predict which hotel properties users are most likely to click on and book based on search results data. The model uses advanced data analysis and visualization techniques to understand patterns in hotel bookings and clicks.

## Project Structure

```
hotel-estimation/
├── data/
│   ├── raw/           # Raw training data
│   ├── processed/     # Preprocessed and cleaned data
│   └── analysis/      # Analysis outputs
│       ├── plots/     # Visualization plots
│       └── tables/    # LaTeX tables with statistics
├── src/
│   ├── data_analysis.py     # Data analysis and visualization
│   ├── preprocessing.py     # Data cleaning and preprocessing
│   ├── feature_engineering.py # Feature creation and transformation
│   └── train_model.py       # Model training and evaluation
└── requirements.txt         # Project dependencies
```

## Features

### Data Analysis
- Dataset composition: 54 variables (53 numerical, 1 categorical)
- Enhanced distribution plots with outlier handling and statistical panels
- Comprehensive variable type analysis and memory usage optimization
- Missing value patterns and visualization
- Search behavior analysis:
  - Average stay: 2.37 days (75% ≤ 3 days)
  - Booking window: 38.5 days average (75% within 51 days)
  - Room occupancy: 1.97 adults, 0.34 children per booking
- Competitive rate analysis with price differentials

### Data Preprocessing
- Memory-efficient data processing using chunks
- Data type optimization
- Missing value handling
- Outlier detection and treatment

### Feature Engineering
- Price-related features
- Temporal features (booking window, length of stay)
- Competitor analysis features
- Location score processing
- Property characteristics
- Search context features

### Visualizations
- Distribution plots with outlier analysis
- Missing value heatmaps
- Price impact visualizations
- Competitor price difference analysis
- Search pattern distributions

### Analysis Outputs
- LaTeX report with comprehensive findings in `data/analysis/report/`
- Statistical tables in `data/analysis/tables/`:
  - Data types and basic information
  - Numerical variable statistics
  - Missing value patterns
- Enhanced distribution plots with statistical panels
- Search pattern visualizations
- Competitive pricing analysis

## Getting Started

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place the training data in `data/raw/`
5. Run the analysis:
   ```bash
   python src/data_analysis.py
   ```

## Data Files

Note: Large data files are not included in the repository. You'll need to obtain:
- `training_set_VU_DM.csv` (place in `data/raw/`)

## Output

The analysis generates:
- Visualization plots in `data/analysis/plots/`
- Statistical tables in `data/analysis/tables/`
- Processed data in `data/processed/`
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
