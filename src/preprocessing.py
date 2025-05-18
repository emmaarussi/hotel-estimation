import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class HotelDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_stats = {}
        
    def _handle_price_outliers(self, df):
        """
        Handle price outliers based on star rating
        Found extreme outliers in price_usd (up to 19M)
        """
        # Use predefined price caps per star rating (based on analysis)
        price_caps = {
            0: 100,    # Unknown/unrated hotels
            1: 120,     # 1-star hotels
            2: 150,    # 2-star hotels
            3: 200,    # 3-star hotels
            4: 250,    # 4-star hotels
            5: 370    # 5-star hotels
        }
        
        # Cap prices based on star rating
        for star, cap in price_caps.items():
            mask = df['prop_starrating'] == star
            df.loc[mask & (df['price_usd'] > cap), 'price_usd'] = cap
        
        return df
    
    def _handle_missing_competitor_data(self, df):
        """
        Handle missing competitor data
        - Most competitor data is missing (52-98% missing rates)
        - Simplified to only keep aggregate metrics
        """
        # Calculate mean price difference where available
        price_diffs = []
        comp_available_count = 0
        
        for i in range(1, 9):
            mask = ~df[f'comp{i}_rate'].isna()
            comp_available_count += mask.astype(int)
            if mask.any():
                diff = df.loc[mask, 'price_usd'] - df.loc[mask, f'comp{i}_rate']
                price_diffs.append(diff)
        
        # Only keep aggregate metrics
        df['comp_rates_available'] = comp_available_count
        df['mean_comp_price_diff'] = pd.concat(price_diffs, axis=1).mean(axis=1).fillna(0)
        
        # Drop individual competitor columns
        comp_cols = [f'comp{i}_{suffix}' for i in range(1, 9) 
                    for suffix in ['rate', 'inv', 'rate_percent_diff']]
        df.drop(columns=comp_cols, errors='ignore', inplace=True)
        
        return df
    
    def _handle_missing_location_scores(self, df):
        """
        Handle missing location scores
        - prop_location_score2 has 22% missing values
        - Missing values might indicate less popular/new locations
        """
        # Simple imputation based on location_score1
        df['prop_location_score2'].fillna(df['prop_location_score1'] * 0.7, inplace=True)
        return df
    
    def _handle_missing_distances(self, df):
        """
        Handle missing distances
        - orig_destination_distance has 32% missing values
        - Missing values might indicate domestic/nearby searches
        """
        # For same country searches, use a small default distance
        same_country = df['visitor_location_country_id'] == df['prop_country_id']
        df.loc[same_country & df['orig_destination_distance'].isna(), 'orig_destination_distance'] = 50
        
        # For different countries, use larger default distance
        diff_country = df['visitor_location_country_id'] != df['prop_country_id']
        df.loc[diff_country & df['orig_destination_distance'].isna(), 'orig_destination_distance'] = 500
        
        return df
    
    def _handle_missing_review_scores(self, df):
        """
        Handle missing review scores
        - Only 0.15% missing values
        - Missing might indicate new properties
        """
        # Fill missing reviews with mean per star rating
        for star in df['prop_starrating'].unique():
            mask = (df['prop_starrating'] == star) & (df['prop_review_score'].isna())
            mean_score = df[df['prop_starrating'] == star]['prop_review_score'].mean()
            df.loc[mask, 'prop_review_score'] = mean_score
            
        return df
    
    def _handle_missing_historical_data(self, df):
        """
        Handle missing historical data
        - Simplified to only keep binary flag for booking history
        - Remove high-missing-value historical features
        """
        # Create binary flag for users with history
        df['has_booking_history'] = (~df['visitor_hist_starrating'].isna()).astype(int)
        
        # Drop high-missing-value columns
        df.drop(columns=['visitor_hist_starrating', 'visitor_hist_adr_usd'], 
                errors='ignore', inplace=True)
        
        return df
    
    def _normalize_numerical_features(self, df):
        """
        Normalize numerical features to prevent scale issues
        Removed redundant and derived features
        """
        # Save original price before normalization
        df['price_usd_original'] = df['price_usd'].copy()
        
        # Core numerical features only
        numerical_features = [
            'price_usd',
            'orig_destination_distance',
            'prop_location_score1',
            'prop_review_score',
            'prop_starrating'
        ]
        
        # Store means and stds for future use
        self.feature_stats = {
            'means': df[numerical_features].mean().to_dict(),
            'stds': df[numerical_features].std().to_dict()
        }
        
        # Normalize features
        df[numerical_features] = (df[numerical_features] - df[numerical_features].mean()) / df[numerical_features].std()
        
        # Drop redundant/derived features
        columns_to_drop = [
            'random_bool',  # Not useful for prediction
            'prop_starrating_monotonic',  # Redundant with prop_starrating
            'prop_review_monotonic',      # Redundant with prop_review_score
            'prop_location_monotonic',    # Redundant with prop_location_score
            'prop_historical_ctr_log',    # Keep only non-log version
            'prop_historical_br_log',     # Keep only non-log version
            'prop_price_std',             # High missing rate
            'prop_position_std',          # High missing rate
            'dest_month'                  # High memory usage, can be recreated
        ]
        df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
        
        return df
    
    def preprocess(self, df, is_training=True):
        """
        Main preprocessing function
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        is_training : bool
            Whether this is training data
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed dataframe
        """
        print("Handling price outliers...")
        df = self._handle_price_outliers(df)
        
        print("Handling missing competitor data...")
        df = self._handle_missing_competitor_data(df)
        
        print("Handling missing location scores...")
        df = self._handle_missing_location_scores(df)
        
        print("Handling missing distances...")
        df = self._handle_missing_distances(df)
        
        print("Handling missing review scores...")
        df = self._handle_missing_review_scores(df)
        
        print("Handling missing historical data...")
        df = self._handle_missing_historical_data(df)
        
        if is_training:
            print("Normalizing numerical features...")
            df = self._normalize_numerical_features(df)
        
        return df

def save_preprocessed_data(input_file, output_file, chunk_size=100000):
    """
    Process the data in chunks and save to CSV
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str
        Path to output CSV file
    chunk_size : int
        Number of rows to process at once
    """
    # Get total number of rows first
    print("Reading file information...")
    total_rows = sum(1 for _ in open(input_file)) - 1  # subtract header
    chunks_total = (total_rows + chunk_size - 1) // chunk_size
    
    print(f"Total rows: {total_rows:,}")
    print(f"Will process in {chunks_total:,} chunks")
    
    preprocessor = HotelDataPreprocessor()
    
    # Process first chunk
    print(f"\nProcessing chunk 1 of {chunks_total}...")
    first_chunk = pd.read_csv(input_file, nrows=chunk_size)
    processed_chunk = preprocessor.preprocess(first_chunk)
    
    # Save first chunk with headers
    processed_chunk.to_csv(output_file, index=False)
    print(f"Created output file: {output_file}")
    
    # Process remaining chunks
    reader = pd.read_csv(input_file, skiprows=range(1, chunk_size+1), chunksize=chunk_size)
    
    for chunk_number, chunk in enumerate(reader, start=1):
        print(f"Processing chunk {chunk_number + 1} of {chunks_total}...")
        processed_chunk = preprocessor.preprocess(chunk, is_training=False)
        processed_chunk.to_csv(output_file, mode='a', header=False, index=False)
    
    print("\nPreprocessing complete!")
    return output_file

if __name__ == "__main__":
    input_file = "data/raw/training_set_VU_DM.csv"
    output_file = "data/processed/clean_training_set.csv"
    
    print("Starting preprocessing pipeline...")
    output_path = save_preprocessed_data(input_file, output_file)
    
    # Verify the output
    print("\nVerifying output file...")
    df_sample = pd.read_csv(output_path, nrows=5)
    print("\nFirst 5 rows of processed data:")
    print(df_sample.head())
    
    # Get file size
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nProcessed file size: {size_mb:.2f} MB")
