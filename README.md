# PressurePlay: Cricket Match Analysis Tool

PressurePlay is a machine learning application that analyzes cricket matches to predict win probabilities and quantify pressure situations. The tool processes historical cricket match data to build predictive models and provides visualizations of pressure moments during matches.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Cricket match data in CSV format

### Installation Steps

1. **Clone or download the repository**

2. **Create a virtual environment**
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install required packages**
   ```
   pip install -r requirements.txt
   ```

## Data Preparation

1. **Prepare your cricket match data**
   - Create a directory named `all_csv` in the project root
   - Place your cricket match CSV files in this directory
   - Each CSV should contain ball-by-ball data for a complete match

2. **CSV format requirements**
   - Files should follow standard cricket data format with 'info' and 'ball' rows
   - Required columns are documented in the code comments of `cricket_model.py`

## Running the Application

### Step 1: Train the Models

Train the machine learning models using your cricket match data:

```
python cricket_model.py
```

This script will:
- Process all cricket matches in the `all_csv` directory
- Extract features for win prediction and pressure scoring
- Train classification and regression models
- Save the trained models to the `models` directory

The training process may take several minutes to hours depending on the amount of data.

### Step 2: Make Predictions

Once models are trained, you can use the pipeline to make predictions:

```
python model_pipeline.py
```

This will run an example prediction with sample match data. The script will:
- Load the trained models
- Generate predictions for the sample match scenario
- Visualize the pressure and win probability throughout the match
- Identify key pressure moments

### Step 3: Customizing Predictions

To analyze different match scenarios, modify the `match_data` dictionary in `model_pipeline.py`:

```python
match_data = {
    'first_innings_score': 180,         # Total runs scored in first innings
    'first_innings_rr': 9.0,            # Run rate in first innings
    'first_innings_wickets': 8,         # Wickets lost in first innings
    'second_innings_score': 100,        # Current score in second innings
    'second_innings_rr': 8.33,          # Current run rate in second innings
    'second_innings_wickets': 4,        # Current wickets lost in second innings
    'balls_remaining': 48,              # Balls remaining in second innings
    'powerplay_diff': 10,               # Difference in powerplay scores
    'death_overs_diff': 0,              # Difference in death overs scoring
    'dot_ball_diff': 0.05,              # Difference in dot ball percentage
    'boundary_diff': 4                  # Difference in boundaries hit
}
```

## Troubleshooting

### Common Issues:

1. **Missing model files error**
   - Ensure you've run `cricket_model.py` first to generate the model files
   - Check that the `models` directory contains all required model files

2. **Data format issues**
   - Make sure your CSV files follow the expected format
   - Check for missing or corrupted data in your match files

3. **Package dependencies**
   - If you encounter module errors, verify all dependencies are installed:
   ```
   pip install -r requirements.txt
   ```

4. **Model quality issues**
   - If predictions seem inaccurate, you may need more training data
   - Consider adjusting the feature set in `cricket_model.py`

## Advanced Usage

### Simulation and Visualization

The application can simulate how pressure and win probability evolve throughout a match:

```python
# Simulate match progression
simulation_data = pipeline.simulate_match_progression(match_data)

# Visualize match progression
pipeline.visualize_match_pressure(simulation_data)

# Analyze key pressure points
pipeline.analyze_match_pressure_points(simulation_data)
```

These functions provide valuable insights into how pressure builds and shifts during a cricket match.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
