"""Runs every milestone in order (M1 downloads real data first), then starts the app.  Usage: python run_all.py"""
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
steps = ["m1_real_data.py", "m2_train_model.py", "m3_gis_map.py", "m4_realtime_weather.py", "m5_alerts_reports.py"]
for step in steps:
    print(f"\n===== Running {step} =====", flush=True)
    if subprocess.run([sys.executable, "-W", "ignore", step]).returncode != 0:
        sys.exit(f"Stopped: {step} failed. Fix the error above and rerun.")
print("\n===== Starting app at http://127.0.0.1:8000  (Ctrl+C to stop) =====", flush=True)
subprocess.run([sys.executable, "-m", "uvicorn", "m6_app:app", "--port", "8000"])
