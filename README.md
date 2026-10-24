# Mizoram Landslide Risk System
Setup: `pip install -r requirements.txt`  (run all commands inside this folder; internet needed)

Quick start: `python run_all.py`  (runs M1-M5, then starts the app at http://127.0.0.1:8000)

| Milestone | Command | Result |
|---|---|---|
| M1 Real dataset | `python m1_real_data.py` | data/landslide_dataset.csv |
| M2 ML model | `python m2_train_model.py` | models/landslide_model.joblib, output/metrics.txt |
| M3 GIS map | `python m3_gis_map.py` | output/risk_map.html (open in browser) |
| M4 Real-time weather | `python m4_realtime_weather.py` | output/live_risk.csv |
| M5 Alerts + field reports | `python m5_alerts_reports.py` | data/landslide.db |
| M6 Full app | `uvicorn m6_app:app --reload` | http://127.0.0.1:8000 (API docs: /docs) |

M1 sources: NASA Global Landslide Catalog (events), Open-Meteo/ERA5 (rainfall, soil moisture), Open-Meteo DEM (elevation, slope).
First run takes ~10-20 min; downloads are cached in data/raw/. NDVI is a constant placeholder.
The M3 map inputs and M4 town terrain values are still synthetic/placeholder.
Deploy: `docker build -t landslide . && docker run -p 8000:8000 landslide`
