"""MILESTONE 4 - Real-time weather -> AI prediction.  Run: python m4_realtime_weather.py
Weather: Open-Meteo (free, no key; past 3 days rain + soil moisture). Falls back to simulated values if offline.
STATIC values below are PLACEHOLDERS - replace with values extracted from a DEM/NDVI raster."""
import json, urllib.request, joblib
from common import *
STATIC = {  # slope(deg), elevation(m), ndvi, prev_landslide  -- approximate placeholders
    "Aizawl": (32, 1100, .55, 1), "Lunglei": (30, 1000, .58, 1), "Champhai": (26, 1500, .50, 0),
    "Serchhip": (28, 900, .60, 0), "Kolasib": (27, 700, .62, 1), "Saiha": (31, 900, .65, 0),
    "Lawngtlai": (29, 1000, .63, 0), "Mamit": (25, 500, .66, 0)}
def fetch_weather(lat, lon):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           "&hourly=precipitation,soil_moisture_0_to_1cm&past_days=3&forecast_days=1")
    try:
        h = json.load(urllib.request.urlopen(url, timeout=10))["hourly"]
        rain = float(sum(x or 0 for x in h["precipitation"][:72]))            # last 72 h
        sm = [x for x in h["soil_moisture_0_to_1cm"] if x is not None]
        return rain, 100 * float(np.mean(sm[-24:])) if sm else 40.0, "open-meteo"
    except Exception:
        r = np.random.default_rng(abs(hash((lat, lon))) % 2**32)
        return float(r.uniform(20, 250)), float(r.uniform(25, 80)), "simulated (offline)"
def predict_all():
    model = joblib.load(ROOT / "models/landslide_model.joblib"); rows = []
    for t, (la, lo) in TOWNS.items():
        rain, sm, src = fetch_weather(la, lo); sl, el, nd, pv = STATIC[t]
        p = float(model.predict_proba(pd.DataFrame([[rain, sm, sl, el, nd, pv]], columns=FEATURES))[0, 1])
        rows.append(dict(town=t, latitude=la, longitude=lo, rainfall=round(rain, 1), soil_moisture=round(sm, 1),
                         risk_probability=round(p, 3), risk_level=risk_level(p), source=src))
    df = pd.DataFrame(rows).sort_values("risk_probability", ascending=False)
    df.to_csv(ROOT / "output/live_risk.csv", index=False); return df
if __name__ == "__main__":
    print(predict_all().to_string(index=False))
