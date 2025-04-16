import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class CricketModelPipeline:
    def __init__(self, model_dir='models'):
        """Initialize the pipeline by loading the models"""
        self.model_dir = model_dir
        self.win_model = None
        self.pressure_model = None
        self.feature_names = None
        self.win_scaler = None
        self.pressure_scaler = None
        self.load_models()
    
    def load_models(self):
        """Load the trained models from disk"""
        try:
            self.win_model = joblib.load(os.path.join(self.model_dir, 'win_model.pkl'))
            self.pressure_model = joblib.load(os.path.join(self.model_dir, 'pressure_model.pkl'))
            self.feature_names = joblib.load(os.path.join(self.model_dir, 'feature_names.pkl'))
            self.win_scaler = joblib.load(os.path.join(self.model_dir, 'win_scaler.pkl'))
            self.pressure_scaler = joblib.load(os.path.join(self.model_dir, 'pressure_scaler.pkl'))
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Make sure you've trained the models first by running cricket_model.py")
    
    def process_match_data(self, match_data):
        """
        Process match data into features needed for prediction
        
        Parameters:
        match_data (dict): Dictionary with match information containing:
            - first_innings_score
            - first_innings_rr
            - first_innings_wickets
            - second_innings_score (current)
            - second_innings_rr (current)
            - second_innings_wickets (current)
            - balls_remaining
            - powerplay_diff
            - death_overs_diff
            - dot_ball_diff
        
        Returns:
        dict: Dictionary with processed features
        """
        # Calculate derived features
        run_diff = match_data['first_innings_score'] - match_data['second_innings_score']
        run_rate_diff = match_data['first_innings_rr'] - match_data['second_innings_rr']
        
        # Calculate required run rate
        if match_data['balls_remaining'] > 0:
            runs_needed = match_data['first_innings_score'] - match_data['second_innings_score'] + 1
            overs_remaining = match_data['balls_remaining'] / 6
            required_rr = runs_needed / overs_remaining if overs_remaining > 0 else float('inf')
        else:
            required_rr = 0
        
        # Calculate pressure metrics
        if required_rr > 0:
            rr_pressure = match_data['second_innings_rr'] / required_rr
        else:
            rr_pressure = 1  # No pressure if already winning
        
        wicket_pressure = match_data['second_innings_wickets'] / 10
        
        boundary_diff = match_data.get('boundary_diff', 0)
        
        # Create feature dictionaries for both models
        win_features = {
            'run_diff': run_diff,
            'run_rate_diff': run_rate_diff,
            'boundary_diff': boundary_diff,
            'first_innings_score': match_data['first_innings_score'],
            'first_innings_rr': match_data['first_innings_rr'],
            'first_innings_wickets': match_data['first_innings_wickets'],
            'second_innings_rr': match_data['second_innings_rr'],
            'second_innings_wickets': match_data['second_innings_wickets'],
            'powerplay_diff': match_data.get('powerplay_diff', 0),
            'death_overs_diff': match_data.get('death_overs_diff', 0),
            'dot_ball_diff': match_data.get('dot_ball_diff', 0)
        }
        
        pressure_features = {
            'run_diff': run_diff,
            'run_rate_diff': run_rate_diff,
            'first_innings_score': match_data['first_innings_score'],
            'second_innings_rr': match_data['second_innings_rr'],
            'second_innings_wickets': match_data['second_innings_wickets'],
            'rr_pressure': rr_pressure,
            'wicket_pressure': wicket_pressure,
            'dot_ball_diff': match_data.get('dot_ball_diff', 0)
        }
        
        return {
            'win_features': win_features,
            'pressure_features': pressure_features,
            'required_rr': required_rr,
            'runs_needed': runs_needed if match_data['balls_remaining'] > 0 else 0,
            'balls_remaining': match_data['balls_remaining']
        }
    
    def predict(self, match_data):
        """
        Predict win probability and pressure score
        
        Parameters:
        match_data (dict): Dictionary with match information
        
        Returns:
        dict: Dictionary with win probability and pressure score
        """
        if not all([self.win_model, self.pressure_model, self.feature_names, self.win_scaler, self.pressure_scaler]):
            return {"error": "Models not loaded properly", 
                    "team1_win_probability": 0.5,  # Default value to avoid KeyError
                    "team2_win_probability": 0.5,
                    "pressure_score": 50,
                    "required_run_rate": 0,
                    "runs_needed": 0,
                    "balls_remaining": 0}
        
        # Process match data
        processed_data = self.process_match_data(match_data)
        
        # Prepare features for win model
        win_features_df = pd.DataFrame([processed_data['win_features']])
        # Scale directly using the DataFrame to preserve feature names
        win_features_scaled = self.win_scaler.transform(win_features_df[self.feature_names['win']])
        
        # Predict win probability
        win_prob = self.win_model.predict_proba(win_features_scaled)[0][1]
        
        # Prepare features for pressure model
        pressure_features_df = pd.DataFrame([processed_data['pressure_features']])
        # Scale directly using the DataFrame to preserve feature names
        pressure_features_scaled = self.pressure_scaler.transform(pressure_features_df[self.feature_names['pressure']])
        
        # Predict pressure score
        pressure_score = self.pressure_model.predict(pressure_features_scaled)[0]
        
        return {
            'team1_win_probability': win_prob,
            'team2_win_probability': 1 - win_prob,
            'pressure_score': pressure_score,
            'required_run_rate': processed_data['required_rr'],
            'runs_needed': processed_data['runs_needed'],
            'balls_remaining': processed_data['balls_remaining']
        }
    
    def simulate_match_progression(self, initial_match_data, total_balls=120):
        """
        Simulate pressure and win probability throughout a match
        
        Parameters:
        initial_match_data (dict): Initial match state
        total_balls (int): Total balls in the innings
        
        Returns:
        dict: Simulated data for visualization
        """
        simulation_data = []
        
        # Make a copy of the initial data to modify
        match_data = initial_match_data.copy()
        
        # Define run and wicket progression rates for simulation
        avg_runs_per_ball = match_data['first_innings_score'] / total_balls
        wicket_intervals = np.linspace(0, total_balls, 11)  # 10 wickets over the innings
        
        for ball in range(total_balls, -1, -5):  # Step by 5 balls for efficiency
            # Update match data
            match_data['balls_remaining'] = ball
            match_data['second_innings_score'] = initial_match_data['first_innings_score'] * (1 - ball/total_balls) * 0.9
            
            # Estimate wickets based on ball progression
            estimated_wickets = np.sum(wicket_intervals <= (total_balls - ball))
            match_data['second_innings_wickets'] = min(estimated_wickets, 10)
            
            # Calculate run rate
            balls_played = total_balls - ball
            overs_played = balls_played / 6
            if overs_played > 0:
                match_data['second_innings_rr'] = match_data['second_innings_score'] / overs_played
            else:
                match_data['second_innings_rr'] = 0
            
            # Get predictions
            prediction = self.predict(match_data)
            
            simulation_data.append({
                'balls_played': balls_played,
                'overs_played': overs_played,
                'win_probability': prediction['team2_win_probability'],
                'pressure_score': prediction['pressure_score'],
                'required_run_rate': prediction['required_run_rate'],
                'runs_needed': prediction['runs_needed']
            })
        
        return simulation_data
    
    def visualize_match_pressure(self, simulation_data):
        """
        Visualize pressure and win probability throughout a match
        
        Parameters:
        simulation_data (list): List of dictionaries with simulation data
        
        Returns:
        None (displays plot)
        """
        df = pd.DataFrame(simulation_data)
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        
        # Plot win probability
        ax1.plot(df['overs_played'], df['win_probability'], 'b-', label='Win Probability')
        ax1.set_xlabel('Overs')
        ax1.set_ylabel('Win Probability', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_ylim(0, 1)
        
        # Plot pressure score
        ax2.plot(df['overs_played'], df['pressure_score'], 'r-', label='Pressure Score')
        ax2.set_ylabel('Pressure Score', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add required run rate as dotted line
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))
        ax3.plot(df['overs_played'], df['required_run_rate'], 'g--', label='Required RR')
        ax3.set_ylabel('Required Run Rate', color='g')
        ax3.tick_params(axis='y', labelcolor='g')
        
        # Add lines for key moments (pressure peaks)
        pressure_peaks = df[df['pressure_score'] > df['pressure_score'].quantile(0.75)]
        for idx, row in pressure_peaks.iterrows():
            plt.axvline(x=row['overs_played'], color='gray', alpha=0.3)
        
        # Add title and legend
        plt.title('Match Progression: Win Probability and Pressure Score')
        fig.tight_layout()
        
        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper center', 
                   bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        plt.show()
        
    def analyze_match_pressure_points(self, simulation_data, top_n=3):
        """
        Identify and analyze key pressure points in the match
        
        Parameters:
        simulation_data (list): List of dictionaries with simulation data
        top_n (int): Number of top pressure moments to identify
        
        Returns:
        list: Key pressure moments
        """
        df = pd.DataFrame(simulation_data)
        
        # Calculate pressure changes
        df['pressure_change'] = df['pressure_score'].diff().abs()
        
        # Find key pressure moments
        key_moments = df.sort_values('pressure_score', ascending=False).head(top_n)
        
        print("\nKey Pressure Moments in the Match:")
        for i, (idx, row) in enumerate(key_moments.iterrows(), 1):
            print(f"Moment {i}:")
            print(f"  Overs: {row['overs_played']:.1f}")
            print(f"  Pressure Score: {row['pressure_score']:.2f}")
            print(f"  Win Probability: {row['win_probability']:.2%}")
            print(f"  Required Run Rate: {row['required_run_rate']:.2f}")
            print(f"  Runs Needed: {row['runs_needed']}")
            print()
        
        return key_moments.to_dict('records')

# Example usage
if __name__ == "__main__":
    pipeline = CricketModelPipeline()
    
    # Example match data
    match_data = {
        'first_innings_score': 180,
        'first_innings_rr': 9.0,
        'first_innings_wickets': 8,
        'second_innings_score': 100,
        'second_innings_rr': 8.33,
        'second_innings_wickets': 4,
        'balls_remaining': 48,  # 8 overs remaining
        'powerplay_diff': 10,
        'death_overs_diff': 0,
        'dot_ball_diff': 0.05,
        'boundary_diff': 4
    }
    
    # Make prediction
    prediction = pipeline.predict(match_data)
    print("\nMatch Prediction:")
    print(f"Team 1 Win Probability: {prediction['team1_win_probability']:.2%}")
    print(f"Team 2 Win Probability: {prediction['team2_win_probability']:.2%}")
    print(f"Pressure Score (0-100): {prediction['pressure_score']:.2f}")
    print(f"Required Run Rate: {prediction['required_run_rate']:.2f}")
    print(f"Runs Needed: {prediction['runs_needed']}")
    
    # Simulate match progression
    simulation_data = pipeline.simulate_match_progression(match_data)
    
    # Visualize match progression
    pipeline.visualize_match_pressure(simulation_data)
    
    # Analyze key pressure points
    pipeline.analyze_match_pressure_points(simulation_data)
