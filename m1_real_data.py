"""MILESTONE 1 (REAL DATA) - builds data/landslide_dataset.csv from real public sources.   Run: python m1_real_data.py
Needs internet; takes ~10-20 min the first time (results are cached in data/raw/, so reruns/interruptions resume).

Sources
  Landslide events : NASA Global Landslide Catalog (data.nasa.gov, id dd9e-wu2v), India / NE-region events
  Rainfall + soil moisture : Open-Meteo Historical API (ERA5 / ERA5-Land reanalysis, ~25 km grid)
  Elevation + slope: Open-Meteo Elevation API (Copernicus 90 m DEM); slope computed from 4 neighbouring points
  prev_landslide   : earlier catalog events within ~11 km
  NDVI             : NOT available keyless -> constant 0.6 (carries no information; add Sentinel-2/MODIS NDVI later)
Landslide = 1 : catalog events.  Landslide = 0 : points 16-65 km from an event, on dates drawn from event dates (season matched)."""
import json, time, urllib.request, urllib.parse, urllib.error
from datetime import timedelta
from common import *

GLC_URLS = [  # tried in order (NASA moved the file; the old data.nasa.gov/api/views/... link now gives 404)
    "https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/Global_Landslide_Catalog_Export_rows.csv",
    "https://data-nasa-bucket-production.s3.us-east-1.amazonaws.com/legacy/Global_Landslide_Catalog_Export/Global_Landslide_Catalog_Export_rows.csv",
    "https://data.nasa.gov/api/views/9ns5-uuif/rows.csv?accessType=DOWNLOAD"]
ELEV_URL = "https://api.open-meteo.com/v1/elevation?"
HIST_URL = "https://archive-api.open-meteo.com/v1/archive?"
NER_LAT, NER_LON = (21.5, 29.5), (88.0, 97.5)
MAX_POS, NEG_RATIO, D = 600, 2, 0.005
RAW = ROOT / "data/raw"; RAW.mkdir(parents=True, exist_ok=True)

def http_get(url, tries=8):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "landslide-student-project"})
            return urllib.request.urlopen(req, timeout=90).read()
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode("utf8", "ignore").lower() if hasattr(e, "read") else ""
            if e.code == 429:                                   # Open-Meteo free-tier rate limit
                if "daily" in body: raise SystemExit("Open-Meteo DAILY limit reached. Progress is saved - rerun tomorrow.")
                wait = 600 if "hourly" in body else 65
                print(f"  rate-limited by Open-Meteo, waiting {wait}s (progress is saved)...", flush=True); time.sleep(wait); continue
            if i == tries - 1: raise
            time.sleep(4 * (i + 1))
        except Exception:
            if i == tries - 1: raise
            time.sleep(4 * (i + 1))
def http_json(url): return json.loads(http_get(url))

def download_glc(p):
    for u in GLC_URLS:
        try:
            b = http_get(u, tries=2)
            if b[:300].lstrip().lower().startswith((b"<!doctype", b"<html")): raise ValueError("got a web page, not CSV")
            p.write_bytes(b); print("Downloaded NASA catalog from", u); return
        except Exception as e: print("  link failed:", u[:70], "->", str(e)[:50])
    raise SystemExit("\nCould not download the NASA catalog automatically.\nManual fix: open https://data.nasa.gov/dataset/global-landslide-catalog-export , "
                     "click 'Go to resource' on the CSV, save it as  data/raw/glc.csv  (inside this project), then rerun.")

def load_events():
    p = RAW / "glc.csv"
    if not p.exists(): download_glc(p)
    df = pd.read_csv(p, low_memory=False); df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"lat": "latitude", "lng": "longitude", "lon": "longitude", "date": "event_date"})
    if not {"latitude", "longitude", "event_date"} <= set(df.columns):
        raise SystemExit(f"Unexpected GLC columns: {list(df.columns)}\nDelete data/raw/glc.csv and rerun, or send me this list.")
    try: df["date"] = pd.to_datetime(df.event_date, errors="coerce", format="mixed")
    except (TypeError, ValueError): df["date"] = pd.to_datetime(df.event_date, errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "date"])
    m = df.latitude.between(*NER_LAT) & df.longitude.between(*NER_LON)
    if "country_name" in df: m &= df.country_name.fillna("").str.lower().eq("india")
    df = df[m]
    if "location_accuracy" in df:
        good = df[df.location_accuracy.fillna("unknown").str.lower().isin(["exact", "1km", "5km", "10km"])]
        if len(good) >= 50: df = good
        else: print("WARNING: few precisely located events; keeping all accuracies")
    print(f"NER events from NASA GLC: {len(df)}")
    if len(df) < 30: raise SystemExit("Too few events found - check the catalog file / filters.")
    return df.sample(min(len(df), MAX_POS), random_state=0)[["latitude", "longitude", "date"]].reset_index(drop=True)

def elevations(points):
    """Open-Meteo counts every coordinate as one call (limit 600/min, 5000/hour) -> 100 per request, ~11 s apart, cached."""
    cp = RAW / "elev_cache.json"; cache = json.loads(cp.read_text()) if cp.exists() else {}
    key = lambda p: f"{p[0]:.5f},{p[1]:.5f}"
    todo = list(dict.fromkeys(key(p) for p in points if key(p) not in cache))
    for i in range(0, len(todo), 100):
        ch = [tuple(map(float, k.split(","))) for k in todo[i:i + 100]]
        q = urllib.parse.urlencode({"latitude": ",".join(f"{a:.5f}" for a, _ in ch), "longitude": ",".join(f"{b:.5f}" for _, b in ch)})
        for k, v in zip(todo[i:i + 100], http_json(ELEV_URL + q)["elevation"]): cache[k] = v
        cp.write_text(json.dumps(cache)); print(f"  elevation {min(i + 100, len(todo))}/{len(todo)}", flush=True); time.sleep(11)
    return [cache[key(p)] for p in points]

def terrain(df):
    pts = []
    for la, lo in zip(df.latitude, df.longitude): pts += [(la, lo), (la + D, lo), (la, lo + D)]     # centre, north, east
    e = np.array(elevations(pts), float).reshape(-1, 3); la = df.latitude.values
    dzdy = (e[:, 1] - e[:, 0]) / (D * 110574); dzdx = (e[:, 2] - e[:, 0]) / (D * 111320 * np.cos(np.radians(la)))
    return e[:, 0], np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

def weather(df):
    cp = RAW / "weather_cache.json"; cache = json.loads(cp.read_text()) if cp.exists() else {}
    rain, sm = [], []
    for n, r in enumerate(df.itertuples()):
        key = f"{r.latitude:.3f},{r.longitude:.3f},{r.date.date()}"
        if key not in cache:
            q = urllib.parse.urlencode(dict(latitude=round(r.latitude, 3), longitude=round(r.longitude, 3),
                start_date=(r.date - timedelta(days=2)).date().isoformat(), end_date=r.date.date().isoformat(),
                daily="precipitation_sum", hourly="soil_moisture_0_to_7cm", timezone="UTC"))
            try:
                j = http_json(HIST_URL + q); s = [x for x in j["hourly"]["soil_moisture_0_to_7cm"] if x is not None]
                cache[key] = [float(sum(x or 0 for x in j["daily"]["precipitation_sum"])), 100 * float(np.mean(s[-24:])) if s else None]
            except Exception as ex:
                print("weather failed:", key, str(ex)[:60]); rain.append(None); sm.append(None); continue
            time.sleep(.15)
            if n % 50 == 0: cp.write_text(json.dumps(cache)); print(f"  weather {n}/{len(df)}", flush=True)
        rain.append(cache[key][0]); sm.append(cache[key][1])
    cp.write_text(json.dumps(cache))
    return np.array(rain, float), np.array(sm, float)

def make_negatives(pos, rng):
    rows = []
    while len(rows) < NEG_RATIO * len(pos):
        b = pos.iloc[rng.integers(len(pos))]; dt = pos.date.iloc[rng.integers(len(pos))]
        la, lo = b.latitude + rng.uniform(-.6, .6), b.longitude + rng.uniform(-.6, .6)
        if np.hypot(la - b.latitude, lo - b.longitude) < .15 or not (NER_LAT[0] < la < NER_LAT[1] and NER_LON[0] < lo < NER_LON[1]): continue
        near = (abs(pos.latitude - la) < .15) & (abs(pos.longitude - lo) < .15) & ((pos.date - dt).abs() < pd.Timedelta(days=7))
        if not near.any(): rows.append((la, lo, dt))
    return pd.DataFrame(rows, columns=["latitude", "longitude", "date"])

def prev_flag(df, ev):
    return np.array([int(((abs(ev.latitude - la) < .1) & (abs(ev.longitude - lo) < .1) & (ev.date < d - pd.Timedelta(days=1))).any())
                     for la, lo, d in zip(df.latitude, df.longitude, df.date)])

if __name__ == "__main__":
    rng = np.random.default_rng(0); pos = load_events(); neg = make_negatives(pos, rng)
    pos["landslide"], neg["landslide"] = 1, 0; df = pd.concat([pos, neg], ignore_index=True)
    print("Fetching elevation/slope..."); df["elevation"], df["slope"] = terrain(df)
    print("Fetching rainfall/soil moisture (slow)..."); df["rainfall"], df["soil_moisture"] = weather(df)
    df["prev_landslide"] = prev_flag(df, pos); df["ndvi"] = 0.6
    df = df.dropna(subset=["rainfall", "elevation", "slope"]); df["soil_moisture"] = df.soil_moisture.fillna(df.soil_moisture.median())
    df[FEATURES + ["landslide", "latitude", "longitude", "date"]].to_csv(ROOT / "data/landslide_dataset.csv", index=False)
    print(f"\nSaved data/landslide_dataset.csv  rows={len(df)}  landslides={int(df.landslide.sum())}  non-landslides={int((1-df.landslide).sum())}")
    print("Next: python m2_train_model.py   (or just python run_all.py)")
