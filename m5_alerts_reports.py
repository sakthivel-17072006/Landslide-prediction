"""MILESTONE 5 - Alerts + field reporting.  Run: python m5_alerts_reports.py"""
import sqlite3, datetime
from common import *
import m4_realtime_weather as weather
DB = ROOT / "data/landslide.db"
def conn():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY, ts TEXT, town TEXT, risk REAL, level TEXT, message TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS field_reports(id INTEGER PRIMARY KEY, ts TEXT, latitude REAL, longitude REAL, type TEXT, description TEXT, photo TEXT)")
    return c
def send_sms(message):
    """STUB: prints only. Plug in Twilio / Fast2SMS / MSG91 here for real SMS."""
    print("[SMS]", message)
def run_alerts(threshold=0.8):
    df, sent = weather.predict_all(), []
    with conn() as c:
        for r in df[df.risk_probability >= threshold].itertuples():
            msg = (f"VERY HIGH landslide risk near {r.town} ({r.risk_probability:.0%}). Rainfall {r.rainfall} mm/3 days, "
                   "soil moisture high. Authorities: monitor vulnerable roads and slopes.")
            c.execute("INSERT INTO alerts(ts,town,risk,level,message) VALUES(?,?,?,?,?)",
                      (datetime.datetime.now().isoformat(timespec="seconds"), r.town, r.risk_probability, r.risk_level, msg))
            send_sms(msg); sent.append(msg)
    return sent
def add_report(lat, lon, rtype, description, photo=None):
    with conn() as c:
        c.execute("INSERT INTO field_reports(ts,latitude,longitude,type,description,photo) VALUES(?,?,?,?,?,?)",
                  (datetime.datetime.now().isoformat(timespec="seconds"), lat, lon, rtype, description, photo))
def list_rows(table):
    c = conn(); c.row_factory = sqlite3.Row
    return [dict(r) for r in c.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT 100")]
if __name__ == "__main__":
    print("Alerts sent:", len(run_alerts(threshold=0.6)), "(threshold lowered to 0.6 for demo)")
    add_report(23.73, 92.72, "Crack", "Crack on road edge near Aizawl")
    print("Reports:", list_rows("field_reports"))
