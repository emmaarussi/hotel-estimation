import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class EnhancedFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.global_stats = {}
        
    def _create_temporal_features(self, df):
        """Create time-based features"""
        df['date_time'] = pd.to_datetime(df['date_time'])
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek
        df['month'] = df['date_time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Time-based aggregations per destination
        time_aggs = df.groupby('srch_destination_id').agg({
            'price_usd': ['mean', 'std']
        }).reset_index()
        time_aggs.columns = ['srch_destination_id', 'dest_avg_price', 'dest_price_std']
        df = df.merge(time_aggs, on='srch_destination_id', how='left')

        # Advanced temporal features (from top solutions)(nieuw van tess!)
        # Month-destination interaction (captures seasonality effects)
        df['dest_month'] = df['srch_destination_id'].astype(str) + '_' + df['month'].astype(str)
        
        # Booking window features
        df['advance_booking_days'] = (df['srch_co_time'] - df['srch_ci_time']).dt.days if 'srch_co_time' in df.columns and 'srch_ci_time' in df.columns else df['srch_booking_window']
        df['booking_window_bucket'] = pd.qcut(df['srch_booking_window'], q=5, labels=False, duplicates='drop')
        
        return df
    
    def _create_price_features(self, df):
        """Create price-related features"""
        # Basic price features
        df['price_per_night'] = df['price_usd'] / df['srch_length_of_stay'].clip(lower=1)
        df['total_price'] = df['price_usd'] * df['srch_length_of_stay']
        
        # Price position within search results
        df['price_rank'] = df.groupby('srch_id')['price_usd'].rank(method='dense')
        df['price_percentile'] = df.groupby('srch_id')['price_usd'].rank(pct=True)
        
        # Price competitiveness features
        df['price_diff_from_mean'] = df['price_usd'] - df.groupby('srch_id')['price_usd'].transform('mean')
        df['price_ratio_to_mean'] = df['price_usd'] / df.groupby('srch_id')['price_usd'].transform('mean')
        
        # Normalize price by star rating
        df['price_per_star'] = df['price_usd'] / df['prop_starrating'].clip(lower=1)

        # Advanced price features (from top solutions)(nieuw van tess!)
        # Group-based price normalization (3rd place solution approach)
        for group_col in ['srch_destination_id', 'prop_country_id']:
            if group_col in df.columns:
                group_mean = df.groupby(group_col)['price_usd'].transform('mean')
                group_std = df.groupby(group_col)['price_usd'].transform('std')
                df[f'price_norm_by_{group_col}'] = (df['price_usd'] - group_mean) / group_std.clip(lower=1)
        
        # Historical price comparison (2nd place solution approach)
        if 'prop_avg_price' in df.columns:
            df['price_diff_from_hist'] = df['price_usd'] - df['prop_avg_price']
            df['price_ratio_to_hist'] = df['price_usd'] / df['prop_avg_price'].clip(lower=1)
        
        return df
    
    def _create_competitive_features(self, df):
        """Create features based on competitor data"""
        # Count available competitor rates
        comp_rate_cols = [f'comp{i}_rate' for i in range(1, 9)]
        df['comp_rates_available'] = df[comp_rate_cols].notnull().sum(axis=1)
        
        # Average price difference with competitors
        for i in range(1, 9):
            df[f'comp{i}_price_diff'] = df['price_usd'] - df[f'comp{i}_rate']
        
        price_diff_cols = [f'comp{i}_price_diff' for i in range(1, 9)]
        df['avg_comp_price_diff'] = df[price_diff_cols].mean(axis=1)
        df['cheaper_than_comp_count'] = (df[price_diff_cols] < 0).sum(axis=1)
        
        # Competitive position score
        df['comp_position_score'] = df['cheaper_than_comp_count'] / df['comp_rates_available'].clip(lower=1)
        
        return df
    
    def _create_property_features(self, df):
        """Create property-related features"""
        # Location score features
        df['location_score_diff'] = df['prop_location_score1'] - df['prop_location_score2']
        df['location_score_product'] = df['prop_location_score1'] * df['prop_location_score2']
        
        # Property history features
        prop_history = df.groupby('prop_id').agg({
            'click_bool': ['mean', 'count'],
            'booking_bool': ['mean', 'count'],
            'price_usd': ['mean', 'std']
        }).reset_index()
        
        prop_history.columns = ['prop_id', 'prop_historical_ctr', 'prop_click_count',
                              'prop_historical_br', 'prop_booking_count',
                              'prop_avg_price', 'prop_price_std']
        
        df = df.merge(prop_history, on='prop_id', how='left')


        # Add ranking features for important metrics
        df['review_rank'] = df.groupby('srch_id')['prop_review_score'].rank(method='dense', ascending=False)
        df['star_rank'] = df.groupby('srch_id')['prop_starrating'].rank(method='dense', ascending=False)
        df['location_rank'] = df.groupby('srch_id')['prop_location_score1'].rank(method='dense', ascending=False)
        
        # Monotonic features (3rd place solution approach) (nieuw van tess)
        # Calculate target means for monotonic transformations
        if 'booking_bool' in df.columns and df['booking_bool'].sum() > 0:
            booking_star_mean = df.loc[df['booking_bool'] == 1, 'prop_starrating'].mean()
            booking_review_mean = df.loc[df['booking_bool'] == 1, 'prop_review_score'].mean()
            booking_location_mean = df.loc[df['booking_bool'] == 1, 'prop_location_score1'].mean()
            
            # Create monotonic features (distance from optimal value)
            df['prop_starrating_monotonic'] = abs(df['prop_starrating'] - booking_star_mean)
            df['prop_review_monotonic'] = abs(df['prop_review_score'] - booking_review_mean)
            df['prop_location_monotonic'] = abs(df['prop_location_score1'] - booking_location_mean)
        
        # Log transform of historical metrics (helps with skewed distributions)
        for col in ['prop_historical_ctr', 'prop_historical_br']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col])
        
        # Historical property position (4th place solution approach)
        if 'position' in df.columns:
            prop_position = df.groupby('prop_id')['position'].agg(['mean', 'median', 'std']).reset_index()
            prop_position.columns = ['prop_id', 'prop_avg_position', 'prop_median_position', 'prop_position_std']
            df = df.merge(prop_position, on='prop_id', how='left')
        
        return df
    
    def _create_search_context_features(self, df):
        """Create features from search context"""
        # Basic search features
        df['total_guests'] = df['srch_adults_count'] + df['srch_children_count']
        df['rooms_per_person'] = df['srch_room_count'] / df['total_guests'].clip(lower=1)
        df['advance_booking_days'] = df['srch_booking_window']
        
        # Search volume features
        search_volume = df.groupby('srch_destination_id').size().reset_index()
        search_volume.columns = ['srch_destination_id', 'destination_search_volume']
        df = df.merge(search_volume, on='srch_destination_id', how='left')

        # Add domestic flag (nieuw van tess)
        df['is_domestic'] = (df['visitor_location_country_id'] == df['prop_country_id']).astype(int)
        
        # Add log-distance feature
        df['log_orig_distance'] = np.log1p(df['orig_destination_distance'])
        
        return df
    
    def _create_user_features(self, df):
        """Create user history and preference features"""
        # User history indicators
        df['user_has_history'] = (~df['visitor_hist_starrating'].isna()).astype(int)
        
        # Preference matching features
        df['star_rating_diff'] = df['prop_starrating'] - df['visitor_hist_starrating']
        df['price_diff_from_hist'] = df['price_usd'] - df['visitor_hist_adr_usd']
        
        # Fill missing values with 0 (no history)
        history_cols = ['visitor_hist_starrating', 'visitor_hist_adr_usd',
                       'star_rating_diff', 'price_diff_from_hist']
        df[history_cols] = df[history_cols].fillna(0)
        
        return df


    def _create_interaction_features(self, df):
        """Create interaction terms between features"""
        # Quality per price features
        df['review_score_per_dollar'] = df['prop_review_score'] / df['price_usd'].clip(lower=1)
        df['star_per_dollar'] = df['prop_starrating'] / df['price_usd'].clip(lower=1)
        df['location_score_per_dollar'] = df['prop_location_score1'] / df['price_usd'].clip(lower=1)
        
        # Cross-features
        df['star_x_score'] = df['prop_starrating'] * df['prop_review_score']
        
        # Geographical features
        if 'visitor_location_country_id' in df.columns and 'prop_country_id' in df.columns:
            df['is_domestic'] = (df['visitor_location_country_id'] == df['prop_country_id']).astype(int)
        
        # Log-transform of distance (handle skewed distribution)
        if 'orig_destination_distance' in df.columns:
            df['log_orig_distance'] = np.log1p(df['orig_destination_distance'])
        
        # Advanced interaction features (from top solutions)
        # Composite features using the F1×max(F2)+F2 formula (4th place solution)
        if 'prop_starrating' in df.columns and 'prop_review_score' in df.columns:
            max_review = df['prop_review_score'].max()
            df['star_review_composite'] = df['prop_starrating'] * max_review + df['prop_review_score']
        
        if 'prop_location_score1' in df.columns and 'prop_review_score' in df.columns:
            max_location = df['prop_location_score1'].max()
            df['location_review_composite'] = df['prop_location_score1'] * max_review + df['prop_review_score']
        
        # User-property match features (3rd place solution)
        if 'visitor_hist_starrating' in df.columns and 'prop_starrating' in df.columns:
            df['user_star_diff'] = abs(df['visitor_hist_starrating'] - df['prop_starrating'])
        
        if 'visitor_hist_adr_usd' in df.columns and 'price_usd' in df.columns:
            df['user_price_diff'] = df['price_usd'] - df['visitor_hist_adr_usd']
            df['user_price_ratio'] = df['price_usd'] / df['visitor_hist_adr_usd'].clip(lower=1)
            # Log transform to handle skewness
            df['user_price_diff_log'] = np.log1p(abs(df['user_price_diff'])) * np.sign(df['user_price_diff'])
        
        return df
    
    def _create_relevance_score(self, df):
        """Create a relevance score target (5*booking + 1*click)"""
        # This is a common approach in learning-to-rank problems
        df['relevance_score'] = 5 * df['booking_bool'] + 1 * df['click_bool']
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Fill missing review scores with mean
        df['prop_review_score'].fillna(df['prop_review_score'].mean(), inplace=True)
        
        # Fill missing location scores with 0
        df['prop_location_score2'].fillna(0, inplace=True)
        
        # Fill missing distances with median
        df['orig_destination_distance'].fillna(df['orig_destination_distance'].median(), inplace=True)
        
        return df
    
    def create_enhanced_features(self, df, is_training=True):
        """Create all features including cross-features for the learning-to-rank problem"""
        print("Creating temporal features...")
        df = self._create_temporal_features(df)
        
        print("Creating price features...")
        df = self._create_price_features(df)
        
        print("Creating competitive features...")
        df = self._create_competitive_features(df)
        
        print("Creating property features...")
        df = self._create_property_features(df)
        
        print("Creating search context features...")
        df = self._create_search_context_features(df)
        
        print("Creating user features...")
        df = self._create_user_features(df)
        
        print("Creating interaction features...")
        df = self._create_interaction_features(df)
        
        print("Creating relevance score...")
        df = self._create_relevance_score(df)
        
        print("Handling missing values...")
        df = self._handle_missing_values(df)
        
        # Drop original datetime column and other unnecessary columns
        cols_to_drop = ['date_time'] + [f'comp{i}_{x}' for i in range(1, 9) 
                                      for x in ['rate', 'inv', 'rate_percent_diff']]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        return df


def process_and_save_features(input_file, output_file, chunk_size=100000):
    """Process the data in chunks and save to CSV with enhanced features"""
    # Get total number of rows first
    print("Reading file information...")
    total_rows = sum(1 for _ in open(input_file)) - 1  # subtract header
    chunks_total = (total_rows + chunk_size - 1) // chunk_size
    
    print(f"Total rows: {total_rows:,}")
    print(f"Will process in {chunks_total:,} chunks")
    
    feature_engineer = EnhancedFeatureEngineer()
    
    # Process first chunk
    print(f"\nProcessing chunk 1 of {chunks_total}...")
    first_chunk = pd.read_csv(input_file, nrows=chunk_size)
    processed_chunk = feature_engineer.create_enhanced_features(first_chunk)
    
    # Save first chunk with headers
    processed_chunk.to_csv(output_file, index=False)
    print(f"Created output file: {output_file}")
    
    # Process remaining chunks
    reader = pd.read_csv(input_file, skiprows=range(1, chunk_size+1), chunksize=chunk_size)
    
    for chunk_number, chunk in enumerate(reader, start=1):
        print(f"Processing chunk {chunk_number + 1} of {chunks_total}...")
        processed_chunk = feature_engineer.create_enhanced_features(chunk, is_training=False)
        processed_chunk.to_csv(output_file, mode='a', header=False, index=False)
    
    print("\nFeature engineering complete!")
    return output_file

if __name__ == "__main__":
    # Example usage
    input_file = "data/processed/clean_training_set.csv"  # Use the cleaned data from preprocessing
    output_file = "data/processed/featured_training_set.csv"
    
    print("Starting enhanced feature engineering pipeline...")
    output_path = process_and_save_features(input_file, output_file)
    
    # Verify the output
    print("\nVerifying output file...")
    df_sample = pd.read_csv(output_path, nrows=5)
    print("\nFirst 5 rows of processed data:")
    print(df_sample.head())
    
    # Get file size
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nProcessed file size: {size_mb:.2f} MB")
    
    # Print feature list
    print("\nFeatures created:")
    feature_cols = [col for col in df_sample.columns 
                   if col not in ['click_bool', 'booking_bool', 'gross_bookings_usd']]
    for col in sorted(feature_cols):
        print(f"- {col}")



  