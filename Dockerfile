FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python m1_real_data.py && python m2_train_model.py && python m3_gis_map.py
EXPOSE 8000
CMD ["uvicorn", "m6_app:app", "--host", "0.0.0.0", "--port", "8000"]
