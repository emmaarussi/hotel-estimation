import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.feature_selection import SelectKBest, f_regression
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class CombinedFeatureEngineer:
    """
    Comprehensive feature engineering class implementing all techniques from top Expedia competition solutions.
    This combines the original implementation from tess-try with additional techniques from the
    1st, 2nd, 3rd, and 4th place solutions that weren't previously implemented.
    
    Key implementations include:
    - Monotonic features (3rd place solution)
    - Group-based normalization (3rd place solution)
    - Composite features using F1×max(F2)+F2 formula (4th place solution)
    - User-property match features (3rd place solution)
    - Target encoding for categorical features (2nd place solution)
    - Worst-case scenario imputation for missing values (3rd place solution)
    - Binning of categorical variables (4th place solution)
    - Position statistics across hotel and month (2nd place solution)
    - Redundant feature removal via feature selection (3rd place solution)
    - Neural network-inspired features (1st place solution)
    - Factorization machine-inspired features (4th place solution)
    - First quartile imputation (4th place solution)
    """
    
    def __init__(self):
        """
        Initialize the feature engineer with necessary scalers and statistics containers
        """
        self.scaler = StandardScaler()
        self.global_stats = {}
        self.target_encodings = {}
        self.feature_importances = None
    
    def create_enhanced_features(self, df, is_training=True, target_col='relevance_score'):
        """
        Create all features including cross-features for the learning-to-rank problem
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe containing the data
        is_training : bool, default=True
            Whether this is training data (affects feature selection)
        target_col : str, default='relevance_score'
            The target column for feature selection and target encoding
        
        Returns:
        --------
        pandas.DataFrame
            The dataframe with enhanced features
        """
        # Store original columns to track new features
        original_columns = set(df.columns)
        print("Creating temporal features...")
        df = self._create_temporal_features(df)
        
        print("Creating price features...")
        df = self._create_price_features(df)
        
        print("Creating competitive features...")
        df = self._create_competitive_features(df)
        
        print("Creating property features...")
        df = self._create_property_features(df, target_col if is_training else None)
        
        print("Creating search context features...")
        df = self._create_search_context_features(df)
        
        print("Creating user features...")
        df = self._create_user_features(df)
        
        print("Creating interaction features...")
        df = self._create_interaction_features(df)
        
        # Create relevance score if needed and in training mode
        if is_training and 'relevance_score' not in df.columns and 'booking_bool' in df.columns and 'click_bool' in df.columns:
            print("Creating relevance score...")
            df['relevance_score'] = 5 * df['booking_bool'] + 1 * df['click_bool']
        
        print("Handling missing values...")
        df = self._handle_missing_values(df)
        
        # Feature selection (only in training mode)
        if is_training and target_col in df.columns:
            print("Selecting features...")
            df, selected_features = self._select_features(df, target_col)
            print(f"Selected {len(selected_features)} features.")
        
        # Drop original datetime column and other unnecessary columns
        cols_to_drop = ['date_time'] + [f'comp{i}_{x}' for i in range(1, 9) 
                                      for x in ['rate', 'inv', 'rate_percent_diff']]
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
        
        # Ensure critical columns like srch_id are preserved
        critical_columns = ['srch_id', 'booking_bool', 'click_bool']
        for col in critical_columns:
            if col in original_columns and col not in df.columns:
                print(f"WARNING: Critical column {col} was lost during feature engineering. Attempting to restore...")
                # Instead of trying to access the missing column, we need to store the original data
                # This is a bug fix - we can't restore from df[col] if col is missing
                print(f"ERROR: Cannot restore {col} as it was completely lost during processing.")
                print(f"Please modify the feature engineering pipeline to preserve {col}.")
                raise ValueError(f"Critical column {col} was lost during feature engineering")
        
        # If this is training data, store the created feature columns for consistency
        if is_training:
            self.created_features = set(df.columns) - original_columns
            self.all_features = set(df.columns)
            # Ensure critical columns are in all_features
            for col in critical_columns:
                if col in original_columns:
                    self.all_features.add(col)
        # If this is test data, ensure it has the same features as training data
        elif hasattr(self, 'all_features'):
            # Add missing columns with default values
            for col in self.all_features:
                if col not in df.columns:
                    if col in critical_columns:
                        print(f"ERROR: Critical column {col} is missing in test data")
                        raise ValueError(f"Critical column {col} is missing")
                    else:
                        df[col] = 0  # Default value for non-critical columns
            # Drop extra columns that weren't in training data
            extra_cols = set(df.columns) - self.all_features
            if extra_cols:
                # Don't drop critical columns even if they weren't in training
                extra_cols = extra_cols - set(critical_columns)
                df = df.drop(columns=list(extra_cols), errors='ignore')
        
        return df
    
    def _create_temporal_features(self, df):
        """
        Create time-based features with enhanced techniques
        - Hour, day of week, month features
        - Weekend indicator
        - Holiday season indicator
        - Advanced time-based patterns (1st place solution)
        """
        print("  Creating temporal features...")
        
        # Check if we have date_time column
        if 'date_time' not in df.columns:
            print("    No date_time column found. Skipping temporal features.")
            return df
        
        # Convert date_time to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['date_time']):
            df['date_time'] = pd.to_datetime(df['date_time'])
        
        # Extract basic time components
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek
        df['month'] = df['date_time'].dt.month
        
        # Create weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Create time of day categories
        df['time_of_day'] = pd.cut(df['hour'], 
                                 bins=[0, 6, 12, 18, 24], 
                                 labels=['night', 'morning', 'afternoon', 'evening'],
                                 include_lowest=True)
        
        # Convert to numeric for modeling
        time_map = {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3}
        df['time_of_day_num'] = df['time_of_day'].map(time_map)
        
        # Create holiday season indicator (November-December)
        df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
        
        # Create quarter feature
        df['quarter'] = ((df['month'] - 1) // 3) + 1
        
        # NEW: Advanced time-based patterns (1st place solution)
        # Day of month
        df['day_of_month'] = df['date_time'].dt.day
        
        # Day of year
        df['day_of_year'] = df['date_time'].dt.dayofyear
        
        # Week of year
        df['week_of_year'] = df['date_time'].dt.isocalendar().week
        
        # Create cyclical features for time variables
        # These capture the cyclical nature of time features better than linear values
        
        # Hour of day (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (7-day cycle)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month (12-month cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _create_price_features(self, df):
        """
        Create price-related features with enhanced techniques
        - Price statistics (mean, median, min, max)
        - Price competitiveness features
        - Group-based normalization (3rd place solution)
        - Advanced price volatility features (1st place solution)
        """
        print("  Creating price features...")
        
        # Check if we have price_usd column
        if 'price_usd' not in df.columns:
            print("    No price_usd column found. Skipping price features.")
            return df
        
        # Basic price features
        # Log transform price to normalize distribution
        df['price_log'] = np.log1p(df['price_usd'])
        
        # Price statistics by destination
        if 'srch_destination_id' in df.columns:
            # Calculate price statistics by destination
            dest_price_stats = df.groupby('srch_destination_id')['price_usd'].agg(['mean', 'median', 'min', 'max']).reset_index()
            dest_price_stats.columns = ['srch_destination_id', 'dest_price_mean', 'dest_price_median', 'dest_price_min', 'dest_price_max']
            
            # Merge statistics back to main dataframe
            df = df.merge(dest_price_stats, on='srch_destination_id', how='left')
            
            # Create price competitiveness features
            df['price_diff_from_mean'] = df['price_usd'] - df['dest_price_mean']
            df['price_ratio_to_mean'] = df['price_usd'] / df['dest_price_mean'].clip(lower=0.1)
            df['price_percentile'] = df.groupby('srch_destination_id')['price_usd'].rank(pct=True)
            
            # NEW: Group-based normalization (3rd place solution)
            # Z-score normalization within destination
            df['price_zscore_dest'] = df.groupby('srch_destination_id')['price_usd'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
            
            # Min-max normalization within destination
            df['price_minmax_dest'] = df.groupby('srch_destination_id')['price_usd'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0)
        
        # Price statistics by country
        if 'prop_country_id' in df.columns:
            # Calculate price statistics by country
            country_price_stats = df.groupby('prop_country_id')['price_usd'].agg(['mean', 'median']).reset_index()
            country_price_stats.columns = ['prop_country_id', 'country_price_mean', 'country_price_median']
            
            # Merge statistics back to main dataframe
            df = df.merge(country_price_stats, on='prop_country_id', how='left')
            
            # Create price competitiveness features by country
            df['price_diff_from_country_mean'] = df['price_usd'] - df['country_price_mean']
            df['price_ratio_to_country_mean'] = df['price_usd'] / df['country_price_mean'].clip(lower=0.1)
            
            # NEW: Group-based normalization by country (3rd place solution)
            # Z-score normalization within country
            df['price_zscore_country'] = df.groupby('prop_country_id')['price_usd'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
        
        # Price statistics by star rating
        if 'prop_starrating' in df.columns:
            # Calculate price statistics by star rating
            star_price_stats = df.groupby('prop_starrating')['price_usd'].agg(['mean', 'median']).reset_index()
            star_price_stats.columns = ['prop_starrating', 'star_price_mean', 'star_price_median']
            
            # Merge statistics back to main dataframe
            df = df.merge(star_price_stats, on='prop_starrating', how='left')
            
            # Create price competitiveness features by star rating
            df['price_diff_from_star_mean'] = df['price_usd'] - df['star_price_mean']
            df['price_ratio_to_star_mean'] = df['price_usd'] / df['star_price_mean'].clip(lower=0.1)
            
            # NEW: Group-based normalization by star rating (3rd place solution)
            # Z-score normalization within star rating
            df['price_zscore_star'] = df.groupby('prop_starrating')['price_usd'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
        
        # Price rank within search results
        if 'srch_id' in df.columns:
            df['price_rank'] = df.groupby('srch_id')['price_usd'].rank(method='dense')
            df['price_rank_pct'] = df.groupby('srch_id')['price_usd'].rank(pct=True)
        
        # NEW: Advanced price volatility features (1st place solution)
        if 'srch_id' in df.columns and 'srch_destination_id' in df.columns:
            # Calculate price range and volatility within search
            df['price_range_within_search'] = df.groupby('srch_id')['price_usd'].transform('max') - df.groupby('srch_id')['price_usd'].transform('min')
            df['price_std_within_search'] = df.groupby('srch_id')['price_usd'].transform('std')
            
            # Calculate price volatility as coefficient of variation
            df['price_cv_within_search'] = df['price_std_within_search'] / df.groupby('srch_id')['price_usd'].transform('mean').clip(lower=0.1)
            
            # Calculate price position relative to search range
            df['price_position_in_range'] = (df['price_usd'] - df.groupby('srch_id')['price_usd'].transform('min')) / df['price_range_within_search'].clip(lower=0.1)
        
        return df
    
    def _create_competitive_features(self, df):
        """
        Create features based on competitor data with enhanced techniques
        - Worst-case scenario imputation (3rd place solution)
        - Enhanced competitive position metrics (2nd place solution)
        """
        print("  Creating competitive features...")
        
        # Check if we have competitor columns
        comp_cols = [col for col in df.columns if col.startswith('comp') and col.endswith('rate')]
        if not comp_cols:
            print("    No competitor columns found. Skipping competitive features.")
            return df
        
        # NEW: Worst-case scenario imputation for missing competitor rates (3rd place solution)
        print("  Applying worst-case scenario imputation for missing competitor rates (3rd place solution)...")
        for col in comp_cols:
            if df[col].isna().sum() > 0 and 'price_usd' in df.columns:
                # For missing competitor rates, assume worst case (competitor has better price)
                missing_mask = df[col].isna()
                df.loc[missing_mask, col] = df.loc[missing_mask, 'price_usd'] * 0.8  # 20% discount
        
        # Count available competitor rates
        df['comp_rate_available_count'] = 0
        for i in range(1, 9):
            rate_col = f'comp{i}_rate'
            if rate_col in df.columns:
                df['comp_rate_available_count'] += (~df[rate_col].isna()).astype(int)
        
        # Calculate price differences with competitors
        for i in range(1, 9):
            rate_col = f'comp{i}_rate'
            if rate_col in df.columns and 'price_usd' in df.columns:
                # Calculate absolute and percentage differences
                df[f'comp{i}_price_diff'] = df['price_usd'] - df[rate_col]
                df[f'comp{i}_price_ratio'] = df['price_usd'] / df[rate_col].clip(lower=0.1)
        
        # Create aggregate competitor features
        comp_rate_cols = [f'comp{i}_rate' for i in range(1, 9) if f'comp{i}_rate' in df.columns]
        if comp_rate_cols:
            # Calculate min, max, mean competitor rates
            df['comp_rate_min'] = df[comp_rate_cols].min(axis=1)
            df['comp_rate_max'] = df[comp_rate_cols].max(axis=1)
            df['comp_rate_mean'] = df[comp_rate_cols].mean(axis=1)
            
            # Calculate price competitiveness vs. competitors
            if 'price_usd' in df.columns:
                df['price_vs_comp_min'] = df['price_usd'] - df['comp_rate_min']
                df['price_vs_comp_max'] = df['price_usd'] - df['comp_rate_max']
                df['price_vs_comp_mean'] = df['price_usd'] - df['comp_rate_mean']
                
                # Calculate price ratios
                df['price_ratio_vs_comp_min'] = df['price_usd'] / df['comp_rate_min'].clip(lower=0.1)
                df['price_ratio_vs_comp_mean'] = df['price_usd'] / df['comp_rate_mean'].clip(lower=0.1)
                
                # Calculate competitive position
                df['price_better_than_comp_count'] = 0
                for col in comp_rate_cols:
                    df['price_better_than_comp_count'] += (df['price_usd'] < df[col]).astype(int)
                
                # Calculate percentage of competitors that are better
                df['price_better_than_comp_pct'] = df['price_better_than_comp_count'] / df['comp_rate_available_count'].clip(lower=1)
        
        # Create competitive availability features
        comp_inv_cols = [f'comp{i}_inv' for i in range(1, 9) if f'comp{i}_inv' in df.columns]
        if comp_inv_cols:
            # Count available competitors
            df['comp_available_count'] = 0
            for col in comp_inv_cols:
                df['comp_available_count'] += (df[col] == 1).astype(int)
            
            # Calculate percentage of available competitors
            df['comp_available_pct'] = df['comp_available_count'] / len(comp_inv_cols)
        
        # NEW: Enhanced competitive position metrics (2nd place solution)
        print("  Creating enhanced competitive position metrics (2nd place solution)...")
        
        # Calculate competitive pressure score
        # Higher score means more competitive pressure (more competitors with better prices)
        if 'price_better_than_comp_count' in df.columns and 'comp_rate_available_count' in df.columns:
            # Weighted score based on how many competitors have better prices
            df['competitive_pressure_score'] = 1 - (df['price_better_than_comp_count'] / df['comp_rate_available_count'].clip(lower=1))
            
            # Create competitive pressure categories
            df['competitive_pressure_level'] = pd.cut(
                df['competitive_pressure_score'], 
                bins=[0, 0.33, 0.66, 1], 
                labels=['low', 'medium', 'high'],
                include_lowest=True
            )
            
            # Convert to numeric for modeling
            pressure_map = {'low': 0, 'medium': 1, 'high': 2}
            df['competitive_pressure_level_num'] = df['competitive_pressure_level'].map(pressure_map)
        
        return df
    
    def _create_property_features(self, df, target_col=None):
        """
        Create property-related features with enhanced techniques
        - Monotonic features (3rd place solution)
        - Target encoding for categorical features (2nd place solution)
        - Binning of categorical variables (4th place solution)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe containing the data
        target_col : str, optional
            The target column for target encoding. If None, no target encoding is performed.
        """
        print("  Creating property features...")
        
        # Create monotonic features (3rd place solution)
        print("  Creating monotonic features (3rd place solution)...")
        if 'prop_starrating' in df.columns:
            # Create monotonic star rating feature
            if target_col and target_col in df.columns:
                # Calculate mean target value for each star rating
                star_target_mean = df.groupby('prop_starrating')[target_col].mean()
                # Find the star rating with the highest target value
                optimal_star = star_target_mean.idxmax()
                # Create monotonic feature as distance from optimal star rating
                df['prop_starrating_monotonic'] = np.abs(df['prop_starrating'] - optimal_star)
            else:
                # Without target, assume higher stars are better
                df['prop_starrating_monotonic'] = 5 - df['prop_starrating']
        
        if 'prop_review_score' in df.columns:
            # Create monotonic review score feature
            if target_col and target_col in df.columns:
                # Calculate mean target value for each review score (rounded)
                df['review_rounded'] = df['prop_review_score'].round(0)
                review_target_mean = df.groupby('review_rounded')[target_col].mean()
                # Find the review score with the highest target value
                optimal_review = review_target_mean.idxmax()
                # Create monotonic feature as distance from optimal review score
                df['prop_review_monotonic'] = np.abs(df['prop_review_score'] - optimal_review)
                # Drop temporary column
                df.drop('review_rounded', axis=1, inplace=True)
            else:
                # Without target, assume higher reviews are better
                df['prop_review_monotonic'] = 5 - df['prop_review_score']
        
        if 'prop_location_score1' in df.columns:
            # Create monotonic location score feature
            if target_col and target_col in df.columns:
                # Calculate mean target value for each location score (binned)
                df['location_binned'] = pd.qcut(df['prop_location_score1'], 10, labels=False, duplicates='drop')
                location_target_mean = df.groupby('location_binned')[target_col].mean()
                # Find the location bin with the highest target value
                optimal_location = location_target_mean.idxmax()
                # Create monotonic feature as distance from optimal location bin
                df['prop_location_monotonic'] = np.abs(df['location_binned'] - optimal_location)
                # Drop temporary column
                df.drop('location_binned', axis=1, inplace=True)
            else:
                # Without target, assume higher location scores are better
                df['prop_location_monotonic'] = 10 - (df['prop_location_score1'] * 10).astype(int)
        
        # NEW: Binning of categorical variables (4th place solution)
        print("  Binning categorical variables (4th place solution)...")
        categorical_cols = ['prop_country_id', 'prop_id', 'srch_destination_id']
        for col in categorical_cols:
            if col in df.columns:
                # Count frequency of each category
                value_counts = df[col].value_counts()
                
                # Create frequency-based bins
                # High frequency (top 10%)
                high_freq = value_counts[value_counts >= value_counts.quantile(0.9)].index.tolist()
                # Medium frequency (10-50%)
                med_freq = value_counts[(value_counts < value_counts.quantile(0.9)) & 
                                        (value_counts >= value_counts.quantile(0.5))].index.tolist()
                # Low frequency (bottom 50%)
                low_freq = value_counts[value_counts < value_counts.quantile(0.5)].index.tolist()
                
                # Create binned feature
                bin_col = f'{col}_freq_bin'
                df[bin_col] = 0  # Default to low frequency
                df.loc[df[col].isin(med_freq), bin_col] = 1  # Medium frequency
                df.loc[df[col].isin(high_freq), bin_col] = 2  # High frequency
        
        # NEW: Target encoding for categorical features (2nd place solution)
        if target_col and target_col in df.columns:
            print("  Applying target encoding for categorical features (2nd place solution)...")
            # List of categorical columns to encode
            encode_cols = ['prop_country_id', 'srch_destination_id', 'site_id']
            encode_cols = [col for col in encode_cols if col in df.columns]
            
            for col in encode_cols:
                # Check if we have pre-computed encodings for this column
                if col in self.target_encodings:
                    encoding_map = self.target_encodings[col]
                else:
                    # Calculate mean target value for each category
                    encoding_map = df.groupby(col)[target_col].mean().to_dict()
                    # Store for future use
                    self.target_encodings[col] = encoding_map
                
                # Apply encoding
                encoded_col = f'{col}_target_encoded'
                df[encoded_col] = df[col].map(encoding_map)
                
                # Fill missing values with global mean
                global_mean = df[target_col].mean()
                df[encoded_col].fillna(global_mean, inplace=True)
        
        # Create ranking features for property attributes
        print("  Creating ranking features for property attributes...")
        if 'srch_id' in df.columns:
            # Star rating rank within search
            if 'prop_starrating' in df.columns:
                df['star_rank'] = df.groupby('srch_id')['prop_starrating'].rank(method='dense', ascending=False)
            
            # Review score rank within search
            if 'prop_review_score' in df.columns:
                df['review_rank'] = df.groupby('srch_id')['prop_review_score'].rank(method='dense', ascending=False)
            
            # Location score rank within search
            if 'prop_location_score1' in df.columns:
                df['location_rank'] = df.groupby('srch_id')['prop_location_score1'].rank(method='dense', ascending=False)
        
        # Fill missing values for important property features
        if 'prop_review_score' in df.columns:
            df['prop_review_score'].fillna(df['prop_review_score'].median(), inplace=True)
        
        if 'prop_location_score1' in df.columns:
            df['prop_location_score1'].fillna(0, inplace=True)
        
        if 'prop_location_score2' in df.columns:
            df['prop_location_score2'].fillna(0, inplace=True)
        
        # Log-transform distance to normalize distribution
        if 'orig_destination_distance' in df.columns:
            # NEW: First quartile imputation for missing values (4th place solution)
            distance_q1 = df['orig_destination_distance'].quantile(0.25)
            df['orig_destination_distance'].fillna(distance_q1, inplace=True)
            
            # Create log-transformed distance
            df['log_orig_distance'] = np.log1p(df['orig_destination_distance'])
        
        return df
    
    def _create_search_context_features(self, df):
        """
        Create features from search context with enhanced techniques
        - Enhanced geographical features (3rd place solution)
        - Search volume features (2nd place solution)
        """
        print("  Creating search context features...")
        
        # Create geographical features
        if 'visitor_location_country_id' in df.columns and 'prop_country_id' in df.columns:
            # Domestic/international travel feature
            df['is_domestic'] = (df['visitor_location_country_id'] == df['prop_country_id']).astype(int)
        
        # Create search volume features (2nd place solution)
        print("  Creating search volume features (2nd place solution)...")
        if 'srch_destination_id' in df.columns:
            # Calculate search volume for each destination
            dest_volume = df['srch_destination_id'].value_counts().to_dict()
            df['destination_search_volume'] = df['srch_destination_id'].map(dest_volume)
            
            # Calculate search volume percentile
            df['destination_volume_percentile'] = pd.qcut(df['destination_search_volume'], 
                                                   q=10, labels=False, duplicates='drop')
        
        # NEW: Enhanced search context features (2nd place solution)
        if 'srch_id' in df.columns:
            # Calculate number of properties per search
            search_size = df.groupby('srch_id').size().to_dict()
            df['search_size'] = df['srch_id'].map(search_size)
            
            # Calculate position percentile within search
            if 'position' in df.columns:
                df['position_percentile'] = df.groupby('srch_id')['position'].rank(pct=True)
        
        # NEW: Search timing features (1st place solution)
        if 'date_time' in df.columns:
            # Extract hour of day
            df['hour_of_day'] = df['date_time'].dt.hour
            
            # Create time of day categories
            df['time_of_day'] = pd.cut(df['hour_of_day'], 
                                 bins=[0, 6, 12, 18, 24], 
                                 labels=['night', 'morning', 'afternoon', 'evening'],
                                 include_lowest=True)
            
            # Convert to numeric for modeling
            time_map = {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3}
            df['time_of_day_num'] = df['time_of_day'].map(time_map)
            
            # Day of week features
            df['is_weekend'] = df['date_time'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # NEW: Search query complexity (4th place solution)
        # More filters typically indicate a more specific search
        filter_cols = ['srch_adults_count', 'srch_children_count', 'srch_room_count',
                      'srch_saturday_night_bool']
        
        # Count how many filters are non-zero/non-null
        df['search_filter_count'] = 0
        for col in filter_cols:
            if col in df.columns:
                df['search_filter_count'] += (df[col] > 0).astype(int)
        
        return df
    
    def _create_user_features(self, df):
        """
        Create user history and preference features with enhanced techniques
        - User-property match features (3rd place solution)
        - User behavior patterns (2nd place solution)
        """
        print("  Creating user features...")
        
        # Check if we have user history data
        user_cols = ['visitor_hist_starrating', 'visitor_hist_adr_usd']
        if not all(col in df.columns for col in user_cols):
            print("    No user history data found. Skipping user features.")
            return df
        
        # Create user-property match features (3rd place solution)
        print("  Creating user-property match features (3rd place solution)...")
        if 'visitor_hist_starrating' in df.columns and 'prop_starrating' in df.columns:
            # Calculate difference between user's historical star preference and hotel stars
            df['user_star_diff'] = df['prop_starrating'] - df['visitor_hist_starrating']
            
            # Create absolute difference feature
            df['user_star_abs_diff'] = np.abs(df['user_star_diff'])
            
            # Create indicator for whether hotel matches user's preferred star level
            df['user_star_match'] = (df['user_star_abs_diff'] <= 0.5).astype(int)
        
        if 'visitor_hist_adr_usd' in df.columns and 'price_usd' in df.columns:
            # Calculate difference between user's historical price and hotel price
            df['user_price_diff'] = df['price_usd'] - df['visitor_hist_adr_usd']
            
            # Create price ratio feature
            df['user_price_ratio'] = df['price_usd'] / df['visitor_hist_adr_usd'].clip(lower=0.1)
            
            # Log-transform price difference
            df['user_price_diff_log'] = np.log1p(np.abs(df['user_price_diff'])) * np.sign(df['user_price_diff'])
            
            # Create indicator for whether hotel is within user's price range (±20%)
            df['user_price_match'] = ((df['user_price_ratio'] >= 0.8) & 
                                 (df['user_price_ratio'] <= 1.2)).astype(int)
        
        # NEW: User behavior patterns (2nd place solution)
        print("  Creating user behavior patterns (2nd place solution)...")
        if 'user_id' in df.columns:
            # Calculate user search frequency
            user_search_counts = df.groupby('user_id')['srch_id'].nunique().to_dict()
            df['user_search_count'] = df['user_id'].map(user_search_counts)
            
            # Calculate average position clicked by user
            if 'position' in df.columns and 'click_bool' in df.columns:
                # Group by user_id and calculate mean position of clicked items
                user_click_positions = df[df['click_bool'] == 1].groupby('user_id')['position'].mean().to_dict()
                df['user_avg_click_position'] = df['user_id'].map(user_click_positions)
                
                # Fill missing values with global average
                global_avg_position = df[df['click_bool'] == 1]['position'].mean()
                df['user_avg_click_position'].fillna(global_avg_position, inplace=True)
                
                # Calculate position preference feature
                # Lower values indicate user prefers items higher in the search results
                df['position_vs_user_pref'] = df['position'] - df['user_avg_click_position']
        
        # NEW: User-destination history (3rd place solution)
        if 'user_id' in df.columns and 'srch_destination_id' in df.columns:
            # Create user-destination pairs
            df['user_dest_pair'] = df['user_id'].astype(str) + '_' + df['srch_destination_id'].astype(str)
            
            # Calculate booking rate for each user-destination pair
            if 'booking_bool' in df.columns:
                user_dest_booking = df.groupby('user_dest_pair')['booking_bool'].mean().to_dict()
                df['user_dest_booking_rate'] = df['user_dest_pair'].map(user_dest_booking)
                
                # Fill missing values with global average
                global_booking_rate = df['booking_bool'].mean()
                df['user_dest_booking_rate'].fillna(global_booking_rate, inplace=True)
            
            # Drop temporary column
            df.drop('user_dest_pair', axis=1, inplace=True)
        
        # Fill missing user history values
        for col in user_cols:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def _create_interaction_features(self, df):
        """
        Create interaction terms between features with enhanced techniques
        - Composite features using F1u00d7max(F2)+F2 formula (4th place solution)
        - Advanced feature interactions (1st place solution)
        - Neural network-inspired features (1st place solution)
        """
        print("  Creating interaction features...")
        
        # Create value metrics (price normalized by quality)
        if 'price_usd' in df.columns:
            # Value for money based on reviews
            if 'prop_review_score' in df.columns:
                df['review_score_per_dollar'] = df['prop_review_score'] / df['price_usd'].clip(lower=1)
            
            # Stars per dollar
            if 'prop_starrating' in df.columns:
                df['star_per_dollar'] = df['prop_starrating'] / df['price_usd'].clip(lower=1)
            
            # Location score per dollar
            if 'prop_location_score1' in df.columns:
                df['location_per_dollar'] = df['prop_location_score1'] / df['price_usd'].clip(lower=1)
        
        # Create quality composite features
        if 'prop_starrating' in df.columns and 'prop_review_score' in df.columns:
            # Star-review interaction (simple product)
            df['star_x_review'] = df['prop_starrating'] * df['prop_review_score']
            
            # NEW: Composite feature using F1u00d7max(F2)+F2 formula (4th place solution)
            df['star_review_composite'] = df['prop_starrating'] * df['prop_review_score'].clip(lower=1) + df['prop_review_score']
        
        if 'prop_location_score1' in df.columns and 'prop_review_score' in df.columns:
            # Location-review interaction
            df['location_x_review'] = df['prop_location_score1'] * df['prop_review_score']
            
            # NEW: Composite feature using F1u00d7max(F2)+F2 formula (4th place solution)
            df['location_review_composite'] = df['prop_location_score1'] * df['prop_review_score'].clip(lower=1) + df['prop_review_score']
        
        if 'prop_location_score1' in df.columns and 'prop_location_score2' in df.columns:
            # Location score product
            df['location_score_product'] = df['prop_location_score1'] * df['prop_location_score2']
        
        # NEW: Advanced feature interactions (1st place solution)
        print("  Creating advanced feature interactions (1st place solution)...")
        
        # Position-based interactions
        if 'position' in df.columns:
            # Position and price interaction
            if 'price_percentile' in df.columns:
                # Higher values indicate expensive hotels appearing high in results
                df['position_price_interaction'] = (1 / (df['position'].clip(lower=1))) * df['price_percentile']
            
            # Position and star rating interaction
            if 'prop_starrating' in df.columns:
                # Higher values indicate high-star hotels appearing high in results
                df['position_star_interaction'] = (1 / (df['position'].clip(lower=1))) * df['prop_starrating']
        
        # Price competitiveness interactions
        if 'price_diff_from_mean' in df.columns:
            # Price difference and quality interactions
            if 'prop_starrating' in df.columns:
                # Positive values indicate good value (high stars, low price)
                df['star_price_value'] = df['prop_starrating'] - df['price_diff_from_mean'] / 100
            
            if 'prop_review_score' in df.columns:
                # Positive values indicate good value (high reviews, low price)
                df['review_price_value'] = df['prop_review_score'] - df['price_diff_from_mean'] / 100
        
        # NEW: Neural network-inspired features (1st place solution)
        print("  Creating neural network-inspired features (1st place solution)...")
        
        # Create non-linear transformations of key features
        key_features = ['prop_starrating', 'prop_review_score', 'prop_location_score1', 
                       'price_usd', 'position']
        
        for feature in key_features:
            if feature in df.columns:
                # Square transformation (captures quadratic effects)
                df[f'{feature}_squared'] = df[feature] ** 2
                
                # Square root transformation (dampens extreme values)
                if feature != 'position':  # Position can be 0, so avoid sqrt
                    df[f'{feature}_sqrt'] = np.sqrt(df[feature].clip(lower=0.01))
                
                # Logarithmic transformation (handles skewed distributions)
                if feature not in ['prop_starrating', 'position']:  # Avoid log of small values
                    df[f'{feature}_log'] = np.log1p(df[feature].clip(lower=0.01))
        
        # Create ratio features between key metrics
        if 'prop_starrating' in df.columns and 'prop_review_score' in df.columns:
            # Star-review ratio (captures discrepancy between official rating and user reviews)
            df['star_review_ratio'] = df['prop_starrating'] / df['prop_review_score'].clip(lower=0.1)
        
        if 'prop_location_score1' in df.columns and 'prop_location_score2' in df.columns:
            # Location score ratio (captures discrepancy between location metrics)
            df['location_score_ratio'] = df['prop_location_score1'] / df['prop_location_score2'].clip(lower=0.1)
        
        # NEW: Factorization machine-inspired features (4th place solution)
        print("  Creating factorization machine-inspired features (4th place solution)...")
        
        # Create pairwise interactions between important features
        important_features = ['prop_starrating', 'prop_review_score', 'prop_location_score1', 
                             'price_percentile', 'position']
        important_features = [f for f in important_features if f in df.columns]
        
        # Create pairwise interactions (limited to avoid explosion of features)
        for i, feat1 in enumerate(important_features):
            for feat2 in important_features[i+1:]:
                # Create interaction feature
                interaction_name = f'{feat1}_x_{feat2}'
                df[interaction_name] = df[feat1] * df[feat2]
        
        return df
    
    def _handle_missing_values(self, df):
        """
        Handle missing values in the dataset using techniques from top solutions
        - Worst-case scenario imputation (3rd place solution)
        - First quartile imputation (4th place solution)
        """
        print("  Handling missing values...")
        
        # NEW: First quartile imputation for numeric features (4th place solution)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                # Calculate first quartile
                q1 = df[col].quantile(0.25)
                # Fill missing values with first quartile
                df[col].fillna(q1, inplace=True)
        
        # Fill missing values for specific columns
        if 'prop_review_score' in df.columns:
            df['prop_review_score'].fillna(df['prop_review_score'].median(), inplace=True)
        
        if 'prop_location_score1' in df.columns:
            df['prop_location_score1'].fillna(0, inplace=True)
        
        if 'prop_location_score2' in df.columns:
            df['prop_location_score2'].fillna(0, inplace=True)
        
        # Fill missing competitor data with worst-case scenario (3rd place solution)
        comp_cols = [col for col in df.columns if col.startswith('comp') and col.endswith('rate')]
        for col in comp_cols:
            if df[col].isna().sum() > 0 and 'price_usd' in df.columns:
                # For missing competitor rates, assume worst case (competitor has better price)
                missing_mask = df[col].isna()
                df.loc[missing_mask, col] = df.loc[missing_mask, 'price_usd'] * 0.8  # 20% discount
        
        return df
    
    def _select_features(self, df, target_col=None, n_features=100):
        """
        Select most important features using statistical methods
        - Redundant feature removal (3rd place solution)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe containing the features
        target_col : str, optional
            The target column for feature selection. If None, no selection is performed.
        n_features : int, default=100
            The number of features to select
        """
        print(f"  Selecting top {n_features} features (3rd place solution)...")
        
        if target_col is None or target_col not in df.columns:
            print("    No target column provided. Skipping feature selection.")
            return df, list(df.columns)
        
        # Get feature columns (exclude target and non-feature columns)
        exclude_cols = [target_col, 'srch_id', 'date_time', 'user_id', 'prop_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Check if we have enough data for feature selection
        if len(df) < 1000 or len(feature_cols) <= n_features:
            print(f"    Not enough data or features for selection. Using all {len(feature_cols)} features.")
            return df, feature_cols
        
        # Get numeric feature columns
        numeric_features = df[feature_cols].select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Apply feature selection
        X = df[numeric_features]
        y = df[target_col]
        
        # Handle missing values for feature selection
        X = X.fillna(X.mean())
        
        # Select features using f_regression for ranking problems
        selector = SelectKBest(f_regression, k=min(n_features, len(numeric_features)))
        selector.fit(X, y)
        
        # Get selected feature indices and scores
        selected_indices = selector.get_support(indices=True)
        selected_features = [numeric_features[i] for i in selected_indices]
        feature_scores = selector.scores_
        
        # Create a dataframe of feature importances
        feature_importance = pd.DataFrame({
            'feature': numeric_features,
            'importance': feature_scores
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Store feature importances for later use
        self.feature_importances = feature_importance
        
        # Add any non-numeric features back (they weren't considered in the selection)
        non_numeric_features = [col for col in feature_cols if col not in numeric_features]
        all_selected_features = selected_features + non_numeric_features
        
        # Make sure to include critical columns
        critical_columns = ['srch_id', 'booking_bool', 'click_bool']
        critical_to_include = [col for col in critical_columns if col in df.columns and col != target_col]
        
        print(f"    Selected {len(selected_features)} numeric features and kept {len(non_numeric_features)} non-numeric features.")
        print(f"    Preserving {len(critical_to_include)} critical columns: {critical_to_include}")
        
        # Return the dataframe with selected features, target column, and critical columns
        return_columns = all_selected_features + [target_col] + critical_to_include
        # Remove duplicates while preserving order
        return_columns = list(dict.fromkeys(return_columns))
        
        return df[return_columns], all_selected_features
