"""MILESTONE 6 - Complete app (API + dashboard).  Run: uvicorn m6_app:app --reload  -> http://127.0.0.1:8000
Docs: http://127.0.0.1:8000/docs"""
import joblib, shutil, uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from common import *
import m4_realtime_weather as weather, m5_alerts_reports as store
app = FastAPI(title="Mizoram Landslide Risk System")
model = joblib.load(ROOT / "models/landslide_model.joblib")
(UP := ROOT / "data/uploads").mkdir(exist_ok=True)
class Inputs(BaseModel):
    rainfall: float; soil_moisture: float; slope: float; elevation: float; ndvi: float; prev_landslide: int = 0
@app.get("/")
def dashboard(): return FileResponse(ROOT / "output/risk_map.html")
@app.post("/predict")
def predict(x: Inputs):
    p = float(model.predict_proba(pd.DataFrame([x.model_dump()])[FEATURES])[0, 1])
    return {"risk_probability": round(p, 3), "risk_level": risk_level(p)}
@app.get("/risk/live")
def live(): return weather.predict_all().to_dict("records")
@app.post("/alerts/run")
def alerts(threshold: float = 0.8): return {"sent": store.run_alerts(threshold)}
@app.get("/alerts")
def list_alerts(): return store.list_rows("alerts")
@app.post("/reports")
async def report(latitude: float = Form(...), longitude: float = Form(...), type: str = Form(...),
                 description: str = Form(""), photo: UploadFile | None = File(None)):
    path = None
    if photo and photo.filename:
        path = str(UP / f"{uuid.uuid4().hex}_{photo.filename}")
        with open(path, "wb") as f: shutil.copyfileobj(photo.file, f)
    store.add_report(latitude, longitude, type, description, path); return {"status": "saved"}
@app.get("/reports")
def list_reports(): return store.list_rows("field_reports")
