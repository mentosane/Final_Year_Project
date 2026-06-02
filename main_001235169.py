# ==============================================================================================================================
# File:        [main_001235169.py]
# Author:      [Mihai-Serban Morar]
# Student ID:  [001235169]
# Date:        [15/04/2025]
# Description: The main_001235169.py script processes noise and speed data from the London Underground,
# # trains machine learning models to predict noise levels based on speed and curvature,
# # and suggests speed reductions to mitigate noise. It uses Random Forest and XGBoost models,
# # evaluates their performance, and provides visualizations of the noise reduction process.
# # The script also saves the model metrics and suggestions to CSV files for further analysis. 

# =============================================================================================================================

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import numpy as np

# 1. Load Datasets

df = pd.read_csv("All_Tube_Lines_Noise_Data_with_Speed_and_Curve_Info.csv")

df['Line'] = df['Line'].str.strip()

# 2. Group by Segment and Average Noise Levels

df_aggregated = df.groupby(['Line', 'From', 'To', 'Avg Curve Radius (m)']).agg(
    {
        'Average Speed (km/h)': 'mean',
        'dB LAeq': 'mean'
    }
).reset_index()

grouped = df_aggregated.groupby(['Line'])

# 3. Model Training and Evaluation Function
def train_and_evaluate_model(data, model_type="rf"):
    X = data[['Average Speed (km/h)', 'Avg Curve Radius (m)']]
    y = data['dB LAeq']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=250,min_samples_leaf=2,max_depth=10, random_state=26)
    elif model_type == "xgb":
        model = XGBRegressor(n_estimators=250, learning_rate=0.05,max_depth=10,subsample=0.8,colsample_bytree=0.8, random_state=26)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return model, {'RMSE': rmse, 'MAE': mae}

# 4. Improved Physics-Aware Speed Reduction Function

def suggest_speed_reduction(segment_data, model, current_speed, threshold=75, min_speed=20, speed_step=-0.5, visualize=False):
    curve_radius = segment_data['Avg Curve Radius (m)'].values[0]
    current_noise = model.predict(segment_data[['Average Speed (km/h)', 'Avg Curve Radius (m)']])[0]
    speed_reduction_log = []
    if current_noise < threshold:
        return "✅ No action needed."
    
    # Use a more realistic physics scaling for noise
    baseline_noise = (current_speed ** 2) / curve_radius
    for speed in np.arange(current_speed, min_speed - 0.5, speed_step):
        test_data = segment_data.copy()
        test_data['Average Speed (km/h)'] = speed
        model_noise = model.predict(test_data[['Average Speed (km/h)', 'Avg Curve Radius (m)']])[0]
        estimated_noise_reduction = (speed ** 2) / curve_radius
        adjusted_noise = model_noise - (baseline_noise - estimated_noise_reduction)
        speed_reduction_log.append((speed, adjusted_noise))
        if adjusted_noise < threshold:
            if visualize:
                plt.figure(figsize=(10, 6))
                speeds, noises = zip(*speed_reduction_log)
                plt.plot(speeds, noises, marker='o', label='Adjusted Noise Levels')
                plt.axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold} dB)')
                plt.xlabel('Speed (km/h)')
                plt.ylabel('Predicted Noise (dB)')
                plt.title(f"Noise Reduction Simulation for Segment: {segment_data['From'].values[0]} → {segment_data['To'].values[0]}")
                plt.legend()
                plt.gca().invert_xaxis()
                plt.show()
            return f"⚠️ High noise detected ({current_noise:.2f} dB). Suggest reducing speed to {speed:.1f} km/h to keep noise under {threshold} dB."
    return f"⚠️ Noise level ({current_noise:.2f} dB) exceeds threshold and no feasible slower speed found."

# 5. Ask if User Wants to Visualize Plots
show_plots = input("Would you like to visualize the speed reduction plots? (yes/no): ").strip().lower() == "yes"

# 6. Train and Store Models with Metrics
rf_models = {}
xgb_models = {}
metrics_rf = {}
metrics_xgb = {}
all_suggestions = []

metrics_comparison = []

for line, data in grouped:
    rf_model, rf_metrics = train_and_evaluate_model(data, model_type="rf")
    xgb_model, xgb_metrics = train_and_evaluate_model(data, model_type="xgb")
    rf_models[line] = rf_model
    xgb_models[line] = xgb_model
    metrics_rf[line] = rf_metrics
    metrics_xgb[line] = xgb_metrics
    metrics_comparison.append({
        'Line': line,
        'RF_RMSE': rf_metrics['RMSE'],
        'RF_MAE': rf_metrics['MAE'],
        'XGB_RMSE': xgb_metrics['RMSE'],
        'XGB_MAE': xgb_metrics['MAE']
    })
    for _, row in data.iterrows():
        segment = pd.DataFrame([row])
        current_speed = row['Average Speed (km/h)']
        rf_message = suggest_speed_reduction(segment, rf_model, current_speed, threshold=75, visualize=show_plots)
        xgb_message = suggest_speed_reduction(segment, xgb_model, current_speed, threshold=75, visualize=show_plots)
        all_suggestions.append({
            'Line': line,
            'Segment': f"{row['From']} → {row['To']}",
            'RF Suggestion': rf_message,
            'XGB Suggestion': xgb_message
        })

# Convert to DataFrame and Export
suggestions_df = pd.DataFrame(all_suggestions)
suggestions_df.to_csv("All_Line_Suggestions.csv", index=False)
print("Suggestions saved to 'All_Line_Suggestions.csv'.")

metrics_df = pd.DataFrame(metrics_comparison)
metrics_df.to_csv("Model_Metrics_Comparison.csv", index=False)
print("Model metrics saved to 'Model_Metrics_Comparison.csv'.")

# Optional: Display first 20 rows for quick review
print(metrics_df.head(20))