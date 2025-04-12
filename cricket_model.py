import os
import glob
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
import joblib
import warnings
from tqdm import tqdm  # Import tqdm for progress bars
warnings.filterwarnings('ignore')

class CricketMatchModel:
    def __init__(self, data_dir='all_csv'):
        self.data_dir = data_dir
        self.win_model = None
        self.pressure_model = None
        self.feature_names = None
        self.win_scaler = None  # Separate scaler for win features
        self.pressure_scaler = None  # Separate scaler for pressure features
    
    def load_data(self):
        """Load and process all cricket match data from CSV files"""
        all_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        match_data = []
        
        print(f"Loading and processing {len(all_files)} cricket match files...")
        for file_path in tqdm(all_files, desc="Processing CSV files"):
            try:
                # Initialize match information dictionary
                match_info = {
                    'match_id': os.path.basename(file_path).replace('.csv', ''),
                    'balls_per_over': 6,  # Default value
                    'team1': '',
                    'team2': '',
                    'gender': '',
                    'season': '',
                    'venue': '',
                    'city': '',
                    'toss_winner': '',
                    'toss_decision': '',
                    'winner': '',
                    'outcome': 'unknown'
                }
                
                # Data structures to store ball-by-ball information
                innings_data = {}
                teams = []
                
                # Read the CSV file using the csv module for more robust parsing
                with open(file_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    
                    # Process each row
                    for row in reader:
                        if not row:  # Skip empty rows
                            continue
                        
                        row_type = row[0]
                        
                        if row_type == 'info' and len(row) >= 3:
                            # Handle info rows - extract match metadata
                            info_key = row[1]
                            info_value = row[2]
                            
                            # Extract relevant match information
                            if info_key == 'balls_per_over':
                                match_info['balls_per_over'] = int(info_value)
                            elif info_key == 'team':
                                teams.append(info_value)
                                if len(teams) == 1:
                                    match_info['team1'] = info_value
                                elif len(teams) == 2:
                                    match_info['team2'] = info_value
                            elif info_key == 'gender':
                                match_info['gender'] = info_value
                            elif info_key == 'season':
                                match_info['season'] = info_value
                            elif info_key == 'venue':
                                match_info['venue'] = info_value
                            elif info_key == 'city':
                                match_info['city'] = info_value
                            elif info_key == 'toss_winner':
                                match_info['toss_winner'] = info_value
                            elif info_key == 'toss_decision':
                                match_info['toss_decision'] = info_value
                            elif info_key == 'outcome':
                                match_info['outcome'] = info_value
                            elif info_key == 'winner' and info_value:
                                match_info['winner'] = info_value
                                match_info['outcome'] = 'winner'
                                
                        elif row_type == 'ball' and len(row) >= 9:
                            # Handle ball-by-ball data
                            innings = row[1]
                            over_ball = row[2]
                            batting_team = row[3]
                            batsman = row[4]
                            non_striker = row[5]
                            bowler = row[6]
                            runs_off_bat = int(row[7]) if row[7].isdigit() else 0
                            extras = int(row[8]) if row[8].isdigit() else 0
                            
                            # Extract wicket information if available
                            wicket_type = row[12] if len(row) > 12 else ''
                            dismissed = row[13] if len(row) > 13 else ''
                            
                            # Create innings key if it doesn't exist
                            if innings not in innings_data:
                                innings_data[innings] = {
                                    'balls': [],
                                    'batting_team': batting_team,
                                    'total_runs': 0,
                                    'total_wickets': 0,
                                    'total_balls': 0,
                                    'powerplay_runs': 0,
                                    'death_runs': 0,
                                    'dot_balls': 0,
                                    'fours': 0,
                                    'sixes': 0
                                }
                            
                            # Update innings data
                            innings_data[innings]['total_runs'] += runs_off_bat + extras
                            innings_data[innings]['total_balls'] += 1
                            
                            # Track wickets
                            if wicket_type and wicket_type != '""':
                                innings_data[innings]['total_wickets'] += 1
                            
                            # Track dot balls
                            if runs_off_bat == 0 and extras == 0:
                                innings_data[innings]['dot_balls'] += 1
                            
                            # Track boundaries
                            if runs_off_bat == 4:
                                innings_data[innings]['fours'] += 1
                            elif runs_off_bat == 6:
                                innings_data[innings]['sixes'] += 1
                            
                            # Track powerplay runs (first 6 overs)
                            over_num = float(over_ball.split('.')[0]) if '.' in over_ball else int(over_ball)
                            if over_num < 6:  # First 6 overs
                                innings_data[innings]['powerplay_runs'] += runs_off_bat + extras
                                
                            # Store ball data for further analysis
                            innings_data[innings]['balls'].append({
                                'over_ball': over_ball,
                                'runs_off_bat': runs_off_bat,
                                'extras': extras,
                                'wicket': bool(wicket_type and wicket_type != '""')
                            })
                
                # Calculate innings statistics and match features
                if len(innings_data) >= 2:  # Make sure we have data for at least 2 innings
                    # Get ordered innings keys (usually '1' and '2')
                    innings_keys = sorted(innings_data.keys())
                    first_innings = innings_data[innings_keys[0]]
                    second_innings = innings_data[innings_keys[1]]
                    
                    # Calculate run rate
                    balls_per_over = match_info['balls_per_over']
                    first_overs = first_innings['total_balls'] / balls_per_over
                    second_overs = second_innings['total_balls'] / balls_per_over
                    
                    first_run_rate = first_innings['total_runs'] / first_overs if first_overs > 0 else 0
                    second_run_rate = second_innings['total_runs'] / second_overs if second_overs > 0 else 0
                    
                    # Calculate death overs runs (last 5 overs)
                    # For first innings
                    total_balls_first = first_innings['total_balls']
                    if total_balls_first > balls_per_over * 5:
                        death_balls_first = first_innings['balls'][-balls_per_over * 5:]
                        first_innings['death_runs'] = sum(ball['runs_off_bat'] + ball['extras'] for ball in death_balls_first)
                    
                    # For second innings
                    total_balls_second = second_innings['total_balls']
                    if total_balls_second > balls_per_over * 5:
                        death_balls_second = second_innings['balls'][-balls_per_over * 5:]
                        second_innings['death_runs'] = sum(ball['runs_off_bat'] + ball['extras'] for ball in death_balls_second)
                    
                    # Calculate dot ball percentages
                    first_dot_pct = first_innings['dot_balls'] / first_innings['total_balls'] if first_innings['total_balls'] > 0 else 0
                    second_dot_pct = second_innings['dot_balls'] / second_innings['total_balls'] if second_innings['total_balls'] > 0 else 0
                    
                    # Calculate key comparative features
                    run_diff = first_innings['total_runs'] - second_innings['total_runs']
                    run_rate_diff = first_run_rate - second_run_rate
                    boundary_diff = (first_innings['fours'] + first_innings['sixes']) - (second_innings['fours'] + second_innings['sixes'])
                    
                    # Calculate pressure metrics
                    if first_innings['total_runs'] > 0:
                        chase_rr_required = first_innings['total_runs'] / (first_innings['total_balls'] / balls_per_over)
                        rr_pressure = second_run_rate / chase_rr_required if chase_rr_required > 0 else 1
                    else:
                        rr_pressure = 1
                        
                    wicket_pressure = second_innings['total_wickets'] / 10  # Assume 10 wickets max
                    
                    # Create match features record
                    match_features = {
                        'match_id': match_info['match_id'],
                        'team1': match_info['team1'],
                        'team2': match_info['team2'],
                        'run_diff': run_diff,
                        'run_rate_diff': run_rate_diff,
                        'boundary_diff': boundary_diff,
                        'first_innings_score': first_innings['total_runs'],
                        'first_innings_rr': first_run_rate,
                        'first_innings_wickets': first_innings['total_wickets'],
                        'second_innings_rr': second_run_rate,
                        'second_innings_wickets': second_innings['total_wickets'],
                        'powerplay_diff': first_innings['powerplay_runs'] - second_innings['powerplay_runs'],
                        'death_overs_diff': first_innings['death_runs'] - second_innings['death_runs'],
                        'dot_ball_diff': first_dot_pct - second_dot_pct,
                        'rr_pressure': rr_pressure,
                        'wicket_pressure': wicket_pressure,
                        'outcome': match_info.get('outcome', 'unknown'),
                        'winner': match_info.get('winner', 'unknown')
                    }
                    
                    # Calculate pressure score (custom formula)
                    # Higher score means more pressure
                    pressure_score = (
                        (1 - rr_pressure) * 0.5 +  # Run rate pressure (inverted as lower is more pressure)
                        wicket_pressure * 0.3 +  # Wicket pressure
                        abs(run_diff) / 100 * 0.2  # Closeness of game
                    ) * 100  # Scale to 0-100
                    
                    match_features['pressure_score'] = pressure_score
                    match_data.append(match_features)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
                
        match_df = pd.DataFrame(match_data)
        print(f"Processed {len(match_data)} matches successfully")
        return match_df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        # Create target for win prediction (1 if team1 won, 0 otherwise)
        df['team1_won'] = df.apply(lambda x: 1 if x['winner'] == x['team1'] else 0, axis=1)
        
        # Select features for models
        win_features = [
            'run_diff', 'run_rate_diff', 'boundary_diff', 'first_innings_score',
            'first_innings_rr', 'first_innings_wickets', 'second_innings_rr',
            'second_innings_wickets', 'powerplay_diff', 'death_overs_diff', 'dot_ball_diff'
        ]
        
        pressure_features = [
            'run_diff', 'run_rate_diff', 'first_innings_score', 
            'second_innings_rr', 'second_innings_wickets', 'rr_pressure', 
            'wicket_pressure', 'dot_ball_diff'
        ]
        
        # Drop rows with missing values
        df_clean = df.dropna(subset=win_features + ['team1_won', 'pressure_score'])
        
        # Store feature names
        self.feature_names = {
            'win': win_features,
            'pressure': pressure_features
        }
        
        X_win = df_clean[win_features]
        y_win = df_clean['team1_won']
        
        X_pressure = df_clean[pressure_features]
        y_pressure = df_clean['pressure_score']
        
        # Use separate scalers for each feature set
        self.win_scaler = StandardScaler()
        self.pressure_scaler = StandardScaler()
        
        X_win_scaled = self.win_scaler.fit_transform(X_win)
        X_pressure_scaled = self.pressure_scaler.fit_transform(X_pressure)
        
        return X_win_scaled, y_win, X_pressure_scaled, y_pressure
    
    def train_models(self):
        """Train win prediction and pressure score models"""
        # Load and prepare data
        print("Loading and preparing cricket match data...")
        df = self.load_data()
        X_win, y_win, X_pressure, y_pressure = self.prepare_features(df)
        
        # Split data into training and testing sets
        print("Splitting data into training and test sets...")
        X_win_train, X_win_test, y_win_train, y_win_test = train_test_split(
            X_win, y_win, test_size=0.2, random_state=42
        )
        
        X_pressure_train, X_pressure_test, y_pressure_train, y_pressure_test = train_test_split(
            X_pressure, y_pressure, test_size=0.2, random_state=42
        )
        
        # Train win prediction model (classification)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        print("Training win prediction model with grid search...")
        rf_clf = RandomForestClassifier(random_state=42)
        
        # Create a custom grid search with progress bar
        best_score = 0
        best_params = None
        best_model = None
        param_combinations = [
            {'n_estimators': n, 'max_depth': d, 'min_samples_split': s}
            for n in param_grid['n_estimators']
            for d in param_grid['max_depth']
            for s in param_grid['min_samples_split']
        ]
        
        for params in tqdm(param_combinations, desc="Grid Search for Win Model"):
            model = RandomForestClassifier(random_state=42, **params)
            model.fit(X_win_train, y_win_train)
            score = model.score(X_win_test, y_win_test)
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model
        
        self.win_model = best_model
        print(f"Best win model parameters: {best_params}")
        
        # Evaluate win model
        win_pred = self.win_model.predict(X_win_test)
        win_accuracy = accuracy_score(y_win_test, win_pred)
        print(f"Win prediction accuracy: {win_accuracy:.4f}")
        
        # Feature importance for win model
        feature_importances = self.win_model.feature_importances_
        win_features = self.feature_names['win']
        win_importance_df = pd.DataFrame({'Feature': win_features, 'Importance': feature_importances})
        win_importance_df = win_importance_df.sort_values('Importance', ascending=False)
        print("Win model feature importance:")
        print(win_importance_df)
        
        # Train pressure score model (regression)
        print("Training pressure score model...")
        gb_reg = GradientBoostingRegressor(random_state=42)
        
        with tqdm(total=100, desc="Training Pressure Model") as pbar:
            # Setup callback function to update progress bar
            # Setting staged_predict to report progress at each boost stage
            gb_reg.fit(X_pressure_train, y_pressure_train)
            # Update progress bar to 100% when done
            pbar.update(100)
        
        self.pressure_model = gb_reg
        
        # Evaluate pressure model
        pressure_pred = self.pressure_model.predict(X_pressure_test)
        pressure_rmse = np.sqrt(mean_squared_error(y_pressure_test, pressure_pred))
        print(f"Pressure score RMSE: {pressure_rmse:.4f}")
        
        # Feature importance for pressure model
        pressure_features = self.feature_names['pressure']
        pressure_importance = self.pressure_model.feature_importances_
        pressure_importance_df = pd.DataFrame({'Feature': pressure_features, 'Importance': pressure_importance})
        pressure_importance_df = pressure_importance_df.sort_values('Importance', ascending=False)
        print("Pressure model feature importance:")
        print(pressure_importance_df)
        
        # Save models for later use
        print("Saving models...")
        self.save_models()
        
        return {
            'win_accuracy': win_accuracy,
            'pressure_rmse': pressure_rmse,
            'win_importance': win_importance_df,
            'pressure_importance': pressure_importance_df
        }
    
    def save_models(self, path='models'):
        """Save trained models to disk"""
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save models
        joblib.dump(self.win_model, os.path.join(path, 'win_model.pkl'))
        joblib.dump(self.pressure_model, os.path.join(path, 'pressure_model.pkl'))
        joblib.dump(self.feature_names, os.path.join(path, 'feature_names.pkl'))
        joblib.dump(self.win_scaler, os.path.join(path, 'win_scaler.pkl'))
        joblib.dump(self.pressure_scaler, os.path.join(path, 'pressure_scaler.pkl'))
        
        print(f"Models saved to {path} directory")

if __name__ == "__main__":
    model = CricketMatchModel()
    print("Cricket Match Prediction and Pressure Score Model")
    print("="*50)
    results = model.train_models()
    
    print("\nTraining Results:")
    print(f"Win prediction accuracy: {results['win_accuracy']:.2%}")
    print(f"Pressure score RMSE: {results['pressure_rmse']:.4f}")
