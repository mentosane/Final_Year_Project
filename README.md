# Noise Mitigation in the London Underground Using Machine Learning & Track Curvature

This public repository contains my final-year project exploring how **train speed** and **track curvature** relate to **noise levels (dB LAeq)** on the London Underground. 
The project trains machine learning models to **predict noise** and produces **speed-reduction suggestions** for segments predicted to exceed a chosen noise threshold.

In addition, a supporting script uses OpenStreetMap data to estimate **track curvature radii near stations** and generates an interactive HTML map.

---

## Key Features

- Predicts noise (**dB LAeq**) from:
	- **Average Speed (km/h)**
	- **Average Curve Radius (m)**
- Trains and compares two regression models per Tube line:
	- Random Forest Regressor (scikit-learn)
	- XGBoost Regressor (xgboost)
- Generates speed-reduction recommendations for high-noise segments (default threshold: **75 dB**)
- Exports results to CSV for reporting and further analysis
- (Optional) visualizes the noise reduction simulation as plots
- Computes curvature radii from OSM subway geometry and produces an interactive station map

---

## Repository Structure

- `main_001235169.py`  
  Main ML pipeline: loads data, aggregates segments, trains models, evaluates metrics, and generates speed suggestions.

- `Curvature Radius Calculator.py`  
  Downloads London subway geometry (OSMnx), estimates curvature radii using a 3-point method, associates curves with nearest stations, exports curvature summary and an interactive map.

- `All_Tube_Lines_Noise_Data_with_Speed_and_Curve_Info.csv`  
  Main dataset used by `main_001235169.py`.

- Outputs (generated when running scripts):
	- `All_Line_Suggestions.csv`
	- `Model_Metrics_Comparison.csv`
	- `Station_curve_radii.csv`
	- `station_curve_radii_map.html`

---

## Data & Attribution

This project uses:
- London Underground noise/speed dataset files included in this repo (see CSVs)
- Track and station geometry derived from **OpenStreetMap** via **OSMnx**

If you reuse this repository, please ensure you comply with the relevant dataset terms and OpenStreetMap/OSMnx attribution requirements.

---

## Models

### Inputs (features)
- `Average Speed (km/h)`
- `Avg Curve Radius (m)`

### Target
- `dB LAeq`

### Metrics
- RMSE
- MAE

Exported to:
- `Model_Metrics_Comparison.csv`

---

## Speed Reduction Logic (High Level)

If a segment’s predicted noise is above the threshold (default `75 dB`), the script simulates lower speeds in steps (default `-0.5 km/h`) down to a minimum speed (default `20 km/h`) and returns the first speed predicted to bring noise below the threshold.

A physics-aware adjustment term scales with:
- `(speed^2) / curve_radius`

Recommendations are exported to:
- `All_Line_Suggestions.csv`

---

## Setup

### Requirements
- Python 3.x

### Install dependencies (ML pipeline)

pip install pandas numpy matplotlib scikit-learn xgboost

### Install dependencies (curvature + mapping script)

pip install osmnx geopandas shapely folium branca

> Note: Geo/OSM packages can be easier to install via conda on some systems.

---

## How to Run

### 1) Run the ML pipeline
Make sure `All_Tube_Lines_Noise_Data_with_Speed_and_Curve_Info.csv` is in the same folder, then:

python main_001235169.py

You’ll be prompted: Would you like to visualize the speed reduction plots? (yes/no):

Outputs:
- `All_Line_Suggestions.csv`
- `Model_Metrics_Comparison.csv`

---

### 2) (Optional) Run curvature calculation + map generation
This script expects an input CSV named:
- `underground_noise_dataset.csv`

Run: python "Curvature Radius Calculator.py"

Outputs:
- `Station_curve_radii.csv`
- `station_curve_radii_map.html` (opens in your browser)

---

## Troubleshooting

- **Permission errors running scripts**: ensure you run inside a folder you own and have write access to.
- **XGBoost install issues**: try reinstalling with `pip install xgboost`, or use conda.
- **Geo stack issues (OSMnx/GeoPandas)**: consider:

conda install -c conda-forge osmnx geopandas shapely folium branca

---

## Author

Mihai-Serban Morar (Student ID: 001235169)

---

## License

**Academic Use Only (All Rights Reserved)**

This project is shared publicly for educational review and portfolio purposes.

You may:
- View and download the repository for **personal learning** and **academic evaluation**

You may not (without written permission from the author):
- Use this work (or substantial parts of it) in commercial products
- Redistribute it
- Submit it as your own work (in whole or in part)
- Create and distribute derivative works

If you would like to reuse any part of this project, please contact the author to request permission.
