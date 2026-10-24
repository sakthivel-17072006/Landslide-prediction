"""Shared settings/helpers for all milestones."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).parent
for d in ("data", "models", "output"): (ROOT / d).mkdir(exist_ok=True)
FEATURES = ["rainfall", "soil_moisture", "slope", "elevation", "ndvi", "prev_landslide"]
LAT, LON = (21.95, 24.52), (92.26, 93.45)          # Mizoram bounding box
TOWNS = {"Aizawl": (23.7271, 92.7176), "Lunglei": (22.8671, 92.7656), "Champhai": (23.4569, 93.3254),
         "Serchhip": (23.3000, 92.8500), "Kolasib": (24.2260, 92.6760), "Saiha": (22.4850, 92.9780),
         "Lawngtlai": (22.5300, 92.8800), "Mamit": (23.9300, 92.4800)}
CLASSES = ["VERY LOW", "LOW", "MEDIUM", "HIGH", "VERY HIGH"]       # 5 landslide classes
def risk_level(p):
    return CLASSES[int(np.digitize(p, [.2, .4, .6, .8]))]
def nearest_town(la, lo):
    return min(TOWNS, key=lambda t: (TOWNS[t][0]-la)**2 + (TOWNS[t][1]-lo)**2)
def make_features(n, rng):
    d = pd.DataFrame({"latitude": rng.uniform(*LAT, n), "longitude": rng.uniform(*LON, n),
        "rainfall": rng.gamma(4, 40, n).clip(0, 450), "slope": rng.gamma(3, 8, n).clip(0, 70),
        "elevation": rng.uniform(50, 2100, n), "ndvi": rng.beta(5, 2.5, n),
        "prev_landslide": rng.binomial(1, 0.15, n)})
    d["soil_moisture"] = (25 + 0.14 * d.rainfall + rng.normal(0, 8, n)).clip(5, 100)
    return d
