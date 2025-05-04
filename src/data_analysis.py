import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set style for plots
sns.set_theme(style='whitegrid')
sns.set_palette("husl")

def create_latex_table(df, caption, label, filename):
    """Create a LaTeX table from a DataFrame"""
    latex_table = df.to_latex(
        index=True,
        float_format=lambda x: '{:,.2f}'.format(x) if isinstance(x, (int, float)) else str(x),
        caption=caption,
        label=label,
        escape=True
    )
    
    with open(f'data/analysis/tables/{filename}.tex', 'w') as f:
        f.write(latex_table)

def analyze_basic_statistics(df):
    """Analyze basic statistics of all variables"""
    # Get data types and basic info
    data_types = pd.DataFrame({
        'Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null %': df.isnull().sum() / len(df) * 100,
        'Unique Values': df.nunique(),
        'Memory Usage': df.memory_usage(deep=True)
    })
    
    # Calculate statistics for numerical columns
    numeric_stats = df.describe(include=[np.number])
    
    # Calculate statistics for categorical columns
    categorical_stats = df.describe(include=['object'])
    
    # Create LaTeX tables
    create_latex_table(
        data_types,
        'Variable Types and Basic Information',
        'tab:data_types',
        'data_types'
    )
    
    create_latex_table(
        numeric_stats,
        'Numerical Variables Statistics',
        'tab:numeric_stats',
        'numeric_stats'
    )
    
    create_latex_table(
        categorical_stats,
        'Categorical Variables Statistics',
        'tab:categorical_stats',
        'categorical_stats'
    )
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Total number of variables: {len(df.columns)}")
    print(f"Number of numerical variables: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"Number of categorical variables: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"Number of boolean variables: {len(df.select_dtypes(include=['bool']).columns)}")
    print(f"Total number of rows: {len(df)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

def analyze_missing_values(df):
    """Analyze missing values and create LaTeX table"""
    # Calculate missing value statistics
    missing = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing Percentage': (df.isnull().sum() / len(df) * 100),
        'Total Values': len(df)
    })
    missing = missing[missing['Missing Count'] > 0].sort_values('Missing Percentage', ascending=False)
    
    # Create LaTeX table
    create_latex_table(
        missing,
        'Missing Values Analysis',
        'tab:missing_values',
        'missing_values'
    )
    
    # Create missing values heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.savefig('data/analysis/plots/missing_values_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_enhanced_distribution_plot(df, column, output_dir):
    """Create enhanced distribution plots with outlier handling and proper scaling.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to plot
        output_dir (str): Directory to save plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with two subplots and extra space for stats
    fig = plt.figure(figsize=(15, 10))
    
    # Create GridSpec with explicit spacing
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[4, 1],
        height_ratios=[1, 1],
        bottom=0.1,
        left=0.1,
        right=0.9,
        hspace=0.3,
        wspace=0.3
    )
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    stats_ax = fig.add_subplot(gs[:, 1])
    
    # Clear and configure stats panel
    stats_ax.clear()
    stats_ax.set_xticks([])
    stats_ax.set_yticks([])
    stats_ax.spines['top'].set_visible(False)
    stats_ax.spines['right'].set_visible(False)
    stats_ax.spines['bottom'].set_visible(False)
    stats_ax.spines['left'].set_visible(False)
    
    # Format and add title to stats panel
    title = column.replace('_', ' ').title()
    stats_ax.text(0.5, 1.05, f'Distribution Analysis:\n{title}',
                 ha='center', va='top',
                 fontsize=12, fontweight='bold',
                 transform=stats_ax.transAxes)
    
    # Calculate statistics for outlier detection
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Plot 1: Full distribution with outliers
    sns.histplot(data=df, x=column, bins=50, ax=ax1)
    ax1.set_title('Full Distribution (with outliers)')
    ax1.set_xlabel(column)
    ax1.set_ylabel('Count')
    
    # Add vertical lines for outlier bounds
    ax1.axvline(lower_bound, color='r', linestyle='--', alpha=0.5, label='Outlier bounds')
    ax1.axvline(upper_bound, color='r', linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Plot 2: Distribution without outliers
    filtered_data = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    sns.histplot(data=filtered_data, x=column, bins=50, ax=ax2)
    sns.kdeplot(data=filtered_data, x=column, ax=ax2, color='red', linewidth=2)
    ax2.set_title('Distribution (without outliers)')
    ax2.set_xlabel(column)
    ax2.set_ylabel('Count')
    
    # Add statistics to the right panel
    stats_text = [
        f'Statistics:',
        f'',
        f'Mean: {df[column].mean():.2f}',
        f'Median: {df[column].median():.2f}',
        f'Std: {df[column].std():.2f}',
        f'',
        f'Outlier Analysis:',
        f'',
        f'Q1: {Q1:.2f}',
        f'Q3: {Q3:.2f}',
        f'IQR: {IQR:.2f}',
        f'',
        f'Lower bound: {lower_bound:.2f}',
        f'Upper bound: {upper_bound:.2f}',
        f'',
        f'Outlier %: {100 * (1 - len(filtered_data) / len(df)):.1f}%'
    ]
    
    # Hide stats axis frame and ticks
    stats_ax.axis('off')
    
    # Add stats text
    stats_ax.text(0.1, 0.9, '\n'.join(stats_text),
                 transform=stats_ax.transAxes,
                 va='top', family='monospace')

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Reserve space for right-side stats
    output_path = f'{output_dir}/{column}_enhanced_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_numerical_distributions(df):
    """Analyze numerical distributions"""
    # Get numerical columns
    numerical_cols = ['price_usd', 'prop_starrating', 'prop_review_score',
                      'srch_booking_window', 'srch_length_of_stay',
                      'prop_location_score1', 'prop_location_score2']
    
    # Filter only existing columns
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    if len(numerical_cols) > 0:
        # Create statistics table if not already done
        if not os.path.exists('data/analysis/tables/numerical_statistics.tex'):
            stats = df[numerical_cols].describe()
            create_latex_table(
                stats,
                'Numerical Feature Statistics',
                'tab:numerical_stats',
                'numerical_statistics'
            )
        
        # Always generate plots if they don't exist
        for col in numerical_cols:
            standard_path = f'data/analysis/plots/{col}_distribution.png'
            enhanced_path = f'data/analysis/plots/{col}_enhanced_distribution.png'
            
            # Standard distribution plot
            if not os.path.exists(standard_path):
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=col, bins=50)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(standard_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Enhanced distribution plot
            if not os.path.exists(enhanced_path):
                create_enhanced_distribution_plot(df, col, 'data/analysis/plots')


def analyze_target_variables(df):
    """Analyze click and booking rates"""
    # Calculate conversion rates by star rating
    star_conversions = {
        'Total Searches': df.groupby('prop_starrating').size(),
        'Click Rate': df.groupby('prop_starrating')['click_bool'].mean() * 100,
        'Booking Rate': df.groupby('prop_starrating')['booking_bool'].mean() * 100
    }
    
    conversion = pd.DataFrame(star_conversions)
    
    # Only create plots and tables for the first chunk
    if not os.path.exists('data/analysis/tables/conversion_rates.tex'):
        create_latex_table(
            conversion,
            'Conversion Rates by Star Rating',
            'tab:conversion_rates',
            'conversion_rates'
        )
        
        # Create conversion rates plot
        plt.figure(figsize=(10, 6))
        conversion[['Click Rate', 'Booking Rate']].plot(kind='bar')
        plt.title('Click and Booking Rates by Star Rating')
        plt.xlabel('Star Rating')
        plt.ylabel('Rate (%)')
        plt.tight_layout()
        plt.savefig('data/analysis/plots/conversion_rates.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Analyze price quartiles impact
    df['price_quartile'] = pd.qcut(df['price_usd'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    price_impact = pd.DataFrame({
        'Click Rate': df.groupby('price_quartile')['click_bool'].mean() * 100,
        'Booking Rate': df.groupby('price_quartile')['booking_bool'].mean() * 100
    })
    
    # Only create plots and tables for the first chunk
    if not os.path.exists('data/analysis/tables/price_quartile_impact.tex'):
        create_latex_table(
            price_impact,
            'Conversion Rates by Price Quartile',
            'tab:price_impact',
            'price_quartile_impact'
        )
        
        # Create price impact plot
        plt.figure(figsize=(10, 6))
        price_impact.plot(kind='bar')
        plt.title('Click and Booking Rates by Price Quartile')
        plt.xlabel('Price Quartile')
        plt.ylabel('Rate (%)')
        plt.tight_layout()
        plt.savefig('data/analysis/plots/price_impact.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def analyze_search_patterns(df):
    """Analyze search patterns and create visualizations"""
    # Only create plots and tables for the first chunk
    if not os.path.exists('data/analysis/tables/booking_window_stats.tex'):
        # Analyze booking window statistics
        booking_window_stats = df['srch_booking_window'].describe()
        create_latex_table(
            pd.DataFrame(booking_window_stats),
            'Booking Window Statistics (Days)',
            'tab:booking_window',
            'booking_window_stats'
        )
        
        # Create booking window distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='srch_booking_window', bins=50)
        plt.title('Distribution of Booking Window')
        plt.xlabel('Days before Check-in')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('data/analysis/plots/booking_window_dist.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Analyze length of stay patterns
    stay_patterns = pd.DataFrame({
        'Searches': df.groupby('srch_length_of_stay').size(),
        'Click Rate': df.groupby('srch_length_of_stay')['click_bool'].mean() * 100,
        'Booking Rate': df.groupby('srch_length_of_stay')['booking_bool'].mean() * 100
    }).head(10)  # Show first 10 lengths
    
    if not os.path.exists('data/analysis/tables/length_of_stay_patterns.tex'):
        create_latex_table(
            stay_patterns,
            'Search Patterns by Length of Stay',
            'tab:stay_patterns',
            'length_of_stay_patterns'
        )
        
        # Create length of stay plot
        plt.figure(figsize=(12, 6))
        stay_patterns[['Click Rate', 'Booking Rate']].plot(kind='bar')
        plt.title('Conversion Rates by Length of Stay')
        plt.xlabel('Length of Stay (Days)')
        plt.ylabel('Rate (%)')
        plt.tight_layout()
        plt.savefig('data/analysis/plots/stay_patterns.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def analyze_categorical_distributions(df):
    """Analyze categorical distributions"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        dist = df[col].value_counts(normalize=True) * 100
        create_latex_table(
            dist.to_frame('Percentage'),
            f'Distribution of {col}',
            f'tab:{col}_dist',
            f'{col}_distribution'
        )
        
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'data/analysis/plots/{col}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

def analyze_search_patterns(df):
    """Analyze search patterns and context"""
    print("\nSearch Patterns Analysis:")
    
    # Length of stay distribution
    los_stats = df['srch_length_of_stay'].describe()
    print("\nLength of Stay Statistics:")
    print(los_stats)
    
    # Booking window distribution
    bw_stats = df['srch_booking_window'].describe()
    print("\nBooking Window Statistics:")
    print(bw_stats)
    
    # Room and guest statistics
    room_stats = {
        'avg_adults': df['srch_adults_count'].mean(),
        'avg_children': df['srch_children_count'].mean(),
        'avg_rooms': df['srch_room_count'].mean(),
        'avg_guests_per_room': (df['srch_adults_count'] + df['srch_children_count']) / df['srch_room_count'].clip(lower=1)
    }
    print("\nRoom and Guest Statistics:")
    print(pd.Series(room_stats))
    
    return los_stats, bw_stats, room_stats

def analyze_competitive_rates(df):
    """Analyze competitive rates and create visualizations"""
    # Only create plots and tables for the first chunk
    if not os.path.exists('data/analysis/tables/competitor_rate_analysis.tex'):
        # Calculate competitor rate availability
        comp_cols = [f'comp{i}_rate' for i in range(1, 9)]
        availability = pd.DataFrame({
            'Available (%)': df[comp_cols].notna().mean() * 100,
            'Mean Price Diff (%)': df[[f'comp{i}_rate_percent_diff' 
                                     for i in range(1, 9)]].mean()
        })
        
        create_latex_table(
            availability,
            'Competitor Rate Availability and Price Differences',
            'tab:competitor_rates',
            'competitor_rate_analysis'
        )
        
        # Create competitor price difference distribution plot
        plt.figure(figsize=(12, 6))
        for i in range(1, 9):
            data = df[df[f'comp{i}_rate_percent_diff'].notna()]
            if len(data) > 0:
                sns.kdeplot(data=data,
                           x=f'comp{i}_rate_percent_diff',
                           label=f'Competitor {i}')
        plt.title('Distribution of Price Differences with Competitors')
        plt.xlabel('Price Difference (%)')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig('data/analysis/plots/competitor_price_diffs.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create competitor availability plot
        plt.figure(figsize=(10, 6))
        availability['Available (%)'].plot(kind='bar')
        plt.title('Competitor Rate Availability')
        plt.xlabel('Competitor')
        plt.ylabel('Availability (%)')
        plt.tight_layout()
        plt.savefig('data/analysis/plots/competitor_availability.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def optimize_dtypes(df):
    """Optimize data types to reduce memory usage"""
    # Convert integer columns to smaller types where possible
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Convert float columns to float32
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

def main():
    """Main analysis function with memory optimization"""
    print("Loading data...")
    
    # Read data types for optimization
    dtypes = {
        'srch_id': 'int32',
        'site_id': 'int16',
        'visitor_location_country_id': 'int16',
        'prop_country_id': 'int16',
        'prop_id': 'int32',
        'prop_starrating': 'int8',
        'prop_review_score': 'float32',
        'prop_brand_bool': 'int8',
        'prop_location_score1': 'float32',
        'prop_location_score2': 'float32',
        'prop_log_historical_price': 'float32',
        'position': 'int16',
        'price_usd': 'float32',
        'promotion_flag': 'int8',
        'srch_length_of_stay': 'int8',
        'srch_booking_window': 'int16',
        'srch_adults_count': 'int8',
        'srch_children_count': 'int8',
        'srch_room_count': 'int8',
        'srch_saturday_night_bool': 'int8',
        'random_bool': 'int8',
        'click_bool': 'int8',
        'booking_bool': 'int8'
    }
    
    # Read data in chunks
    chunk_size = 100000
    chunks = pd.read_csv("data/raw/training_set_VU_DM.csv", 
                        dtype=dtypes,
                        chunksize=chunk_size)
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}...")
        chunk = optimize_dtypes(chunk)
        
        if i == 0:  # First chunk
            print("\nAnalyzing basic statistics...")
            analyze_basic_statistics(chunk)
            
            print("\nAnalyzing missing values...")
            analyze_missing_values(chunk)
            
            print("\nAnalyzing numerical distributions...")
            analyze_numerical_distributions(chunk)
        
        # These analyses can be done on the full dataset by accumulating results
        print("Analyzing target variables...")
        analyze_target_variables(chunk)
        
        print("Analyzing search patterns...")
        analyze_search_patterns(chunk)
        
        print("Analyzing competitive rates...")
        analyze_competitive_rates(chunk)
        
        # Free memory
        del chunk
    
    print("\nAnalysis complete! Results saved in:")
    print("- LaTeX tables: data/analysis/tables/")
    print("- Plots: data/analysis/plots/")

if __name__ == "__main__":
    main()




