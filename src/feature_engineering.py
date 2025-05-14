# Standard data manipulation and analysis libraries
import pandas as pd
import numpy as np

# For feature scaling and normalization
from sklearn.preprocessing import StandardScaler

# For datetime manipulation
from datetime import datetime

# For efficient dictionary-based statistics tracking
from collections import defaultdict

class EnhancedFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.global_stats = {}
        # Aggregation dictionaries for incremental features
        self.hotel_stats = defaultdict(lambda: {
            'cnt': 0,
            'click': 0,
            'book': 0,
            'price_sum': 0.0,
            'price_sq_sum': 0.0
        })
        self.dest_stats = defaultdict(lambda: {
            'cnt': 0,
            'click': 0,
            'book': 0
        })
        self.country_stats = defaultdict(lambda: {
            'cnt': 0,
            'book': 0
        })

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

        # Advanced price features 
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
        """Create features based on competitor data, robust to missing comp*_rate columns."""
        # Find which comp*_rate columns exist
        comp_rate_cols = [col for col in [f'comp{i}_rate' for i in range(1, 9)] if col in df.columns]
        if comp_rate_cols:
            df['comp_rates_available'] = df[comp_rate_cols].notnull().sum(axis=1)
            # Average price difference with competitors
            price_diff_cols = []
            for col in comp_rate_cols:
                diff_col = col.replace('_rate', '_price_diff')
                df[diff_col] = df['price_usd'] - df[col]
                price_diff_cols.append(diff_col)
            df['avg_comp_price_diff'] = df[price_diff_cols].mean(axis=1)
            df['cheaper_than_comp_count'] = (df[price_diff_cols] < 0).sum(axis=1)
            df['comp_position_score'] = df['cheaper_than_comp_count'] / df['comp_rates_available'].clip(lower=1)
        else:
            # If no comp*_rate columns exist, fill with zeros or NaNs
            df['comp_rates_available'] = 0
            df['avg_comp_price_diff'] = np.nan
            df['cheaper_than_comp_count'] = 0
            df['comp_position_score'] = 0
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
        
        # Log transform of historical metrics (handle skewed distributions)
        df['prop_historical_ctr_log'] = np.log1p(df['prop_historical_ctr'])
        df['prop_historical_br_log'] = np.log1p(df['prop_historical_br'])
        df['prop_log_historical_price'] = np.log1p(df['prop_avg_price'])
        
        # Position-based features (if position is available)
        if 'position' in df.columns:
            position_stats = df.groupby('prop_id').agg({
                'position': ['mean', 'median', 'std']
            }).reset_index()
            
            position_stats.columns = ['prop_id', 'prop_avg_position', 'prop_median_position',
                                     'prop_position_std']
            
            df = df.merge(position_stats, on='prop_id', how='left')
            
            # Position statistics per hotel and month (from 2nd place solution)
            if 'month' in df.columns:
                hotel_month_pos = df.groupby(['prop_id', 'month'])['position'].agg(['mean', 'median', 'std']).reset_index()
                hotel_month_pos.columns = ['prop_id', 'month', 'hotel_month_pos_mean', 'hotel_month_pos_median', 'hotel_month_pos_std']
                df = df.merge(hotel_month_pos, on=['prop_id', 'month'], how='left')
        
        # Destination-based features
        dest_stats = df.groupby('srch_destination_id').agg({
            'click_bool': 'mean',
            'booking_bool': 'mean'
        }).reset_index()
        
        dest_stats.columns = ['srch_destination_id', 'dest_ctr', 'dest_br']
        df = df.merge(dest_stats, on='srch_destination_id', how='left')
        
        # Target encoding for categorical variables (from 2nd place solution)
        if 'prop_country_id' in df.columns:
            country_stats = df.groupby('prop_country_id')['booking_bool'].mean().reset_index()
            country_stats.columns = ['prop_country_id', 'country_booking_rate_enc']
            df = df.merge(country_stats, on='prop_country_id', how='left')
        
        # Monotonic transformations (from 3rd place solution)
        df['prop_starrating_monotonic'] = df['prop_starrating'].rank(pct=True)
        df['prop_review_monotonic'] = df['prop_review_score'].rank(pct=True)
        df['prop_location_monotonic'] = df['prop_location_score1'].rank(pct=True)
        
        # Binning of numerical variables (from 4th place solution)
        df['price_bin'] = pd.qcut(df['price_usd'], 10, labels=False, duplicates='drop')
        if 'orig_destination_distance' in df.columns:
            df['distance_bin'] = pd.qcut(df['orig_destination_distance'].fillna(df['orig_destination_distance'].median()), 
                                        10, labels=False, duplicates='drop')
        
        # Property brand indicator
        if 'prop_brand_bool' not in df.columns:
            # If not already present, try to infer from property ID patterns
            # This is a placeholder - actual implementation would depend on domain knowledge
            df['prop_brand_bool'] = 0
        
        # Random boolean feature (useful for randomization in models)
        df['random_bool'] = np.random.randint(0, 2, size=len(df))
        
        # Incremental features (updated as we process data)
        # These capture global statistics that evolve as we see more data
        rows = df.to_dict('records')
        hotel_features = []
        
        for row in rows:
            pid = row['prop_id']
            did = row['srch_destination_id']
            cid = row['prop_country_id'] if 'prop_country_id' in row else 0
            
            # Update statistics
            self.hotel_stats[pid]['cnt'] += 1
            self.hotel_stats[pid]['click'] += row['click_bool']
            self.hotel_stats[pid]['book'] += row['booking_bool']
            self.hotel_stats[pid]['price_sum'] += row['price_usd']
            self.hotel_stats[pid]['price_sq_sum'] += row['price_usd'] ** 2

            self.dest_stats[did]['cnt'] += 1
            self.dest_stats[did]['click'] += row['click_bool']
            self.dest_stats[did]['book'] += row['booking_bool']

            self.country_stats[cid]['cnt'] += 1
            self.country_stats[cid]['book'] += row['booking_bool']

            # assign new columns
            hotel_features.append({
                'hotel_ctr': self.hotel_stats[pid]['click'] / max(1, self.hotel_stats[pid]['cnt']),
                'hotel_br': self.hotel_stats[pid]['book'] / max(1, self.hotel_stats[pid]['cnt']),
                'hotel_price_mean': self.hotel_stats[pid]['price_sum'] / max(1, self.hotel_stats[pid]['cnt']),
                'hotel_price_std': np.sqrt(
                    max(0, self.hotel_stats[pid]['price_sq_sum'] / max(1, self.hotel_stats[pid]['cnt']) - 
                        (self.hotel_stats[pid]['price_sum'] / max(1, self.hotel_stats[pid]['cnt'])) ** 2)
                )
            })

        # Add the incremental features back to the dataframe
        for col in ['hotel_ctr', 'hotel_br', 'hotel_price_mean', 'hotel_price_std']:
            df[col] = [f[col] for f in hotel_features]

        # --- visitor history deltas ---
        if 'visitor_hist_starrating' in df.columns:
            df['visitor_star_diff'] = (df['prop_starrating'] - df['visitor_hist_starrating']).abs()
        if 'visitor_hist_adr_usd' in df.columns:
            df['visitor_price_log_diff'] = np.log1p(df['price_usd']) - np.log1p(df['visitor_hist_adr_usd'])

        # --- monotonic transform example ---
        df['prop_starrating_monotonic'] = (df['prop_starrating'] - 4).abs()

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
        
        # ------------------------
        # Phase-1 extra features (inspired by top Kaggle solutions)
        # ------------------------
        # 1. Search-level property count
        df['num_props_in_search'] = df.groupby('srch_id')['prop_id'].transform('count')

        # 2. Within-search price statistics
        df['price_min_search'] = df.groupby('srch_id')['price_usd'].transform('min')
        df['price_max_search'] = df.groupby('srch_id')['price_usd'].transform('max')
        df['price_std_search'] = df.groupby('srch_id')['price_usd'].transform('std').fillna(0)

        # 3. Rank features inside each search result page
        df['review_score_rank'] = df.groupby('srch_id')['prop_review_score'].rank(method='dense')
        df['starrating_rank'] = df.groupby('srch_id')['prop_starrating'].rank(method='dense')

        # 4. Reciprocal display rank (position)
        if 'position' in df.columns:
            df['reciprocal_rank'] = 1 / df['position'].clip(lower=1)
        
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
        
        # Cross-features - Composite scores (from top solutions)
        df['star_x_score'] = df['prop_starrating'] * df['prop_review_score']
        
        # Use monotonic features if available (from 3rd place solution)
        if 'prop_starrating_monotonic' in df.columns and 'prop_review_monotonic' in df.columns:
            df['star_review_composite'] = df['prop_starrating_monotonic'] * df['prop_review_monotonic']
        
        if 'prop_location_monotonic' in df.columns and 'prop_review_monotonic' in df.columns:
            df['location_review_composite'] = df['prop_location_monotonic'] * df['prop_review_monotonic']
        
        # Geographical features
        if 'visitor_location_country_id' in df.columns and 'prop_country_id' in df.columns:
            df['is_domestic'] = (df['visitor_location_country_id'] == df['prop_country_id']).astype(int)
        
        # Log-transform of distance (handle skewed distribution)
        if 'orig_destination_distance' in df.columns:
            df['log_orig_distance'] = np.log1p(df['orig_destination_distance'])
        
        # User preference matching (2nd place solution approach)
        if 'visitor_hist_starrating' in df.columns and 'prop_starrating' in df.columns:
            df['user_star_diff'] = abs(df['visitor_hist_starrating'] - df['prop_starrating'])
            # Add normalized version
            df['visitor_star_diff'] = (df['visitor_hist_starrating'] - df['prop_starrating']) / df['prop_starrating'].clip(lower=1)
        
        if 'visitor_hist_adr_usd' in df.columns and 'price_usd' in df.columns:
            df['user_price_diff'] = df['price_usd'] - df['visitor_hist_adr_usd']
            df['user_price_ratio'] = df['price_usd'] / df['visitor_hist_adr_usd'].clip(lower=1)
            # Log transform to handle skewness (from 3rd place solution)
            df['user_price_diff_log'] = np.log1p(abs(df['user_price_diff'])) * np.sign(df['user_price_diff'])
            df['visitor_price_log_diff'] = np.log1p(df['price_usd']) - np.log1p(df['visitor_hist_adr_usd'].clip(lower=1))
        
        # Listwise rank features (from 4th place solution)
        df['price_rank_pct'] = df.groupby('srch_id')['price_usd'].rank(pct=True)
        df['star_rank_pct'] = df.groupby('srch_id')['prop_starrating'].rank(pct=True)
        df['review_rank_pct'] = df.groupby('srch_id')['prop_review_score'].rank(pct=True)
        
        # Add ranking features for important metrics if not already present
        df['review_rank'] = df.groupby('srch_id')['prop_review_score'].rank(method='dense', ascending=False)
        df['star_rank'] = df.groupby('srch_id')['prop_starrating'].rank(method='dense', ascending=False)
        df['location_rank'] = df.groupby('srch_id')['prop_location_score1'].rank(method='dense', ascending=False)
        
        # Has booking history indicator
        df['has_booking_history'] = (~df['visitor_hist_adr_usd'].isna()).astype(int)
        
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
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Create enhanced features for Expedia hotel data")
    parser.add_argument(
        "--input",
        "-i",
        dest="input_file",
        default="data/processed/clean_training_set.csv",
        help="Path to the cleaned CSV file produced by preprocessing.",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_file",
        default="data/processed/featured_training_set.csv",
        help="Path to save the feature-engineered CSV file.",
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        dest="chunk_size",
        type=int,
        default=100000,
        help="Number of rows to process per chunk.",
    )

    args = parser.parse_args()

    print("Starting enhanced feature-engineering pipeline…")
    output_path = process_and_save_features(args.input_file, args.output_file, args.chunk_size)

    # Verify the output
    print("\nVerifying output file…")
    df_sample = pd.read_csv(output_path, nrows=5)
    print("\nFirst 5 rows of processed data:")
    print(df_sample.head())

    # Get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nProcessed file size: {size_mb:.2f} MB")

    # Print feature list
    print("\nFeatures created:")
    feature_cols = [
        col for col in df_sample.columns if col not in [
            "click_bool",
            "booking_bool",
            "gross_bookings_usd",
        ]
    ]
    for col in sorted(feature_cols):
        print(f"- {col}")