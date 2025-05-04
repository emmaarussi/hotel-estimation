import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class RankingFeatureEngineer:
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
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Fill missing review scores with mean
        df['prop_review_score'].fillna(df['prop_review_score'].mean(), inplace=True)
        
        # Fill missing location scores with 0
        df['prop_location_score2'].fillna(0, inplace=True)
        
        # Fill missing distances with median
        df['orig_destination_distance'].fillna(df['orig_destination_distance'].median(), inplace=True)
        
        return df
    
    def create_ranking_features(self, df, is_training=True):
        """
        Create all features for the learning-to-rank problem
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        is_training : bool
            Whether this is training data (for fitting scalers)
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with engineered features
        """
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
        
        print("Handling missing values...")
        df = self._handle_missing_values(df)
        
        # Drop original datetime column and other unnecessary columns
        cols_to_drop = ['date_time'] + [f'comp{i}_{x}' for i in range(1, 9) 
                                      for x in ['rate', 'inv', 'rate_percent_diff']]
        df = df.drop(columns=cols_to_drop)
        
        return df

if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    df = pd.read_csv("data/raw/training_set_VU_DM.csv")
    
    print("Creating features...")
    feature_engineer = RankingFeatureEngineer()
    df_featured = feature_engineer.create_ranking_features(df)
    
    print("\nFinal feature set shape:", df_featured.shape)
    print("\nFeatures created:", sorted([col for col in df_featured.columns 
                                       if col not in ['click_bool', 'booking_bool', 'gross_bookings_usd']]))
