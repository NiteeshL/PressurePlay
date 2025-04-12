import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    print("Loading data...")
    # Load the matches data
    matches_df = pd.read_csv('matches.csv')
    
    # Initial data exploration
    print(f"Dataset shape: {matches_df.shape}")
    print("\nSample data:")
    print(matches_df.head())
    
    # Check for missing values
    print("\nMissing values in each column:")
    print(matches_df.isnull().sum())
    
    # Convert date to datetime with flexible parsing
    # Use pandas to infer the format instead of specifying it
    matches_df['date'] = pd.to_datetime(matches_df['date'], dayfirst=False, errors='coerce')
    
    # Add these columns only if the date parsing was successful
    matches_df['year'] = matches_df['date'].dt.year
    matches_df['month'] = matches_df['date'].dt.month
    
    # Fill missing values in year and month with the mode (most common value)
    if matches_df['year'].isnull().any():
        year_mode = matches_df['year'].mode()[0]
        matches_df['year'].fillna(year_mode, inplace=True)
    
    if matches_df['month'].isnull().any():
        month_mode = matches_df['month'].mode()[0]
        matches_df['month'].fillna(month_mode, inplace=True)
    
    return matches_df

def engineer_pressure_features(df):
    print("\nEngineering pressure-related features...")
    
    # Create a feature for close matches (matches decided by small margins)
    # For matches won by runs, consider close if margin is less than 20 runs
    # For matches won by wickets, consider close if margin is 3 or fewer wickets
    df['close_match'] = ((df['win_by_runs'] > 0) & (df['win_by_runs'] < 20)) | \
                        ((df['win_by_wickets'] > 0) & (df['win_by_wickets'] <= 3))
    
    # Create a feature for matches where team is chasing (batting second)
    df['is_chasing'] = df['toss_decision'] == 'field'
    
    # Create a feature for high-stakes games (playoffs, finals, etc.)
    # We could identify these based on match_id ranges or specific dates
    # For now, let's use a simple heuristic based on the last few matches of each season
    df = df.sort_values(['season', 'date'])
    df['match_num_in_season'] = df.groupby('season').cumcount() + 1
    matches_per_season = df.groupby('season').size()
    df['is_high_stakes'] = False
    
    for season in df['season'].unique():
        # Consider last 5 matches of each season as high stakes
        season_matches = matches_per_season[season]
        high_stakes_threshold = max(0, season_matches - 5)
        df.loc[(df['season'] == season) & (df['match_num_in_season'] > high_stakes_threshold), 'is_high_stakes'] = True
    
    # Create a pressure index based on several factors
    df['pressure_index'] = 0
    
    # Add pressure for close matches
    df.loc[df['close_match'], 'pressure_index'] += 2
    
    # Add pressure for chasing
    df.loc[df['is_chasing'], 'pressure_index'] += 1
    
    # Add pressure for high stakes matches
    df.loc[df['is_high_stakes'], 'pressure_index'] += 3
    
    # Calculate strike rate proxy for the match
    # We don't have ball-by-ball data, but we can use win_by_runs as a proxy for the batting team's performance
    # Higher win_by_runs might indicate better batting performance
    # Need to handle cases where win_by_wickets > 0 differently
    
    df['batting_performance'] = np.nan
    
    # For teams batting first and winning
    mask_batting_first_win = (df['toss_decision'] == 'bat') & (df['win_by_runs'] > 0)
    df.loc[mask_batting_first_win, 'batting_performance'] = df.loc[mask_batting_first_win, 'win_by_runs']
    
    # For teams batting first and losing
    mask_batting_first_lose = (df['toss_decision'] == 'bat') & (df['win_by_wickets'] > 0)
    # We don't have direct metrics here, so we'll use a placeholder negative score
    df.loc[mask_batting_first_lose, 'batting_performance'] = -df.loc[mask_batting_first_lose, 'win_by_wickets'] * 10
    
    # For teams batting second and winning
    mask_batting_second_win = (df['toss_decision'] == 'field') & (df['win_by_wickets'] > 0)
    # Higher wickets remaining generally means better batting performance
    df.loc[mask_batting_second_win, 'batting_performance'] = df.loc[mask_batting_second_win, 'win_by_wickets'] * 15
    
    # For teams batting second and losing
    mask_batting_second_lose = (df['toss_decision'] == 'field') & (df['win_by_runs'] > 0) 
    df.loc[mask_batting_second_lose, 'batting_performance'] = -df.loc[mask_batting_second_lose, 'win_by_runs']
    
    # Create a win indicator (1 for win, 0 for loss)
    df['toss_winner_won'] = (df['toss_winner'] == df['winner']).astype(int)
    
    return df

def visualize_data(df):
    print("\nCreating visualizations...")
    
    # Create directory for plots if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Plot 1: Distribution of pressure index
    plt.figure(figsize=(10, 6))
    sns.histplot(df['pressure_index'], kde=True)
    plt.title('Distribution of Pressure Index')
    plt.xlabel('Pressure Index')
    plt.ylabel('Frequency')
    plt.savefig('plots/pressure_index_distribution.png')
    plt.close()
    
    # Plot 2: Batting performance vs pressure index
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='pressure_index', y='batting_performance', data=df)
    plt.title('Batting Performance vs. Pressure Index')
    plt.xlabel('Pressure Index')
    plt.ylabel('Batting Performance Proxy')
    plt.savefig('plots/batting_vs_pressure.png')
    plt.close()
    
    # Plot 3: Win rate by pressure index
    plt.figure(figsize=(10, 6))
    pressure_win_rate = df.groupby('pressure_index')['toss_winner_won'].mean()
    pressure_win_rate.plot(kind='bar')
    plt.title('Win Rate by Pressure Index')
    plt.xlabel('Pressure Index')
    plt.ylabel('Win Rate')
    plt.savefig('plots/win_rate_by_pressure.png')
    plt.close()
    
    # Plot 4: Batting performance by year
    plt.figure(figsize=(12, 6))
    yearly_performance = df.groupby('year')['batting_performance'].mean()
    yearly_performance.plot(kind='line', marker='o')
    plt.title('Average Batting Performance by Year')
    plt.xlabel('Year')
    plt.ylabel('Avg. Batting Performance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plots/batting_performance_by_year.png')
    plt.close()

def build_model(df):
    print("\nBuilding machine learning model...")
    
    # Drop rows with missing batting_performance
    df_model = df.dropna(subset=['batting_performance'])
    
    # Select relevant features
    features = ['season', 'pressure_index', 'is_chasing', 'close_match', 'is_high_stakes', 'year', 'month']
    X = df_model[features]
    y = df_model['batting_performance']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define preprocessing steps
    numeric_features = ['pressure_index', 'year', 'month']
    categorical_features = ['season', 'is_chasing', 'close_match', 'is_high_stakes']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create pipeline with preprocessor and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    # Feature importance
    if hasattr(pipeline['regressor'], 'feature_importances_'):
        # Get feature names after one-hot encoding
        ohe = pipeline.named_steps['preprocessor'].transformers_[1][1]
        cat_feature_names = []
        for i, col in enumerate(categorical_features):
            cat_values = ohe.categories_[i]
            cat_feature_names.extend([f"{col}_{val}" for val in cat_values])
        
        all_feature_names = numeric_features + cat_feature_names
        
        # Get feature importances
        importances = pipeline['regressor'].feature_importances_
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sorted_indices = np.argsort(importances)[::-1]
        plt.title('Feature Importance for Batting Performance Prediction')
        plt.bar(range(len(importances)), importances[sorted_indices], align='center')
        plt.xticks(range(len(importances)), [all_feature_names[i] for i in sorted_indices], rotation=90)
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png')
        plt.close()
    
    return pipeline

def analyze_pressure_impact(model, df):
    print("\nAnalyzing impact of pressure on batting performance...")
    
    # Create a test dataset with varying pressure levels
    test_data = pd.DataFrame({
        'season': [2017] * 6,
        'year': [2017] * 6,
        'month': [4] * 6,
        'is_chasing': [True] * 6,
        'close_match': [False, False, True, True, True, True],
        'is_high_stakes': [False, True, False, True, True, True],
        'pressure_index': [1, 4, 3, 6, 8, 10]
    })
    
    # Make predictions
    predictions = model.predict(test_data)
    
    # Add predictions to test data
    test_data['predicted_performance'] = predictions
    
    # Plot the relationship between pressure and predicted performance
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='pressure_index', y='predicted_performance', data=test_data, marker='o')
    plt.title('Predicted Batting Performance vs. Pressure Index')
    plt.xlabel('Pressure Index')
    plt.ylabel('Predicted Batting Performance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plots/predicted_performance_by_pressure.png')
    plt.close()
    
    print("\nResults:")
    print(test_data[['pressure_index', 'is_chasing', 'close_match', 'is_high_stakes', 'predicted_performance']])
    
    # Calculate correlation
    correlation = df['pressure_index'].corr(df['batting_performance'], method='pearson')
    print(f"\nCorrelation between Pressure Index and Batting Performance: {correlation:.4f}")
    
    # Interpret correlation
    if correlation < -0.5:
        print("Strong negative correlation: Higher pressure significantly reduces batting performance")
    elif correlation < -0.3:
        print("Moderate negative correlation: Higher pressure moderately reduces batting performance")
    elif correlation < -0.1:
        print("Weak negative correlation: Higher pressure slightly reduces batting performance")
    elif correlation < 0.1:
        print("No significant correlation: Pressure has minimal impact on batting performance")
    elif correlation < 0.3:
        print("Weak positive correlation: Higher pressure slightly improves batting performance")
    elif correlation < 0.5:
        print("Moderate positive correlation: Higher pressure moderately improves batting performance")
    else:
        print("Strong positive correlation: Higher pressure significantly improves batting performance")

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Engineer features
    df_with_features = engineer_pressure_features(df)
    
    # Visualize data
    visualize_data(df_with_features)
    
    # Build model
    model = build_model(df_with_features)
    
    # Analyze pressure impact
    analyze_pressure_impact(model, df_with_features)
    
    print("\nAnalysis complete! Check the 'plots' directory for visualizations.")

if __name__ == "__main__":
    main()
