from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import requests

# ML මොඩල් ලෝඩ් කරගැනීම
fuel_model = joblib.load("generator_fuel_model.pkl")
co2_model = joblib.load("generator_co2_model.pkl")

app = FastAPI()

FIREBASE_DB_URL = "https://generator-intelligence-default-rtdb.asia-southeast1.firebasedatabase.app"

class SensorData(BaseModel):
    active_power: float
    apparent_power: float
    power_factor: float
    load_factor: float

@app.get("/")
def home():
    return {"status": "API is running successfully!"}

@app.post("/predict")
def predict_performance(data: SensorData):
    try:
        # 1. ඩේටා අරගෙන ඇරේ එකක් හැදීම
        input_data = np.array([[data.active_power, data.apparent_power, data.power_factor, data.load_factor]])
        
        # 2. ප්‍රඩික්ෂන් ලබා ගැනීම
        pred_fuel = float(fuel_model.predict(input_data)[0])
        pred_co2 = float(co2_model.predict(input_data)[0])
        
        # 3. Firebase REST API එක හරහා කෙලින්ම ඩේටා යැවීම (කිසිම ෆයිල් එකක් ඕන නැත!)
        sensor_payload = {
            "Active_Power": data.active_power,
            "Apparent_Power": data.apparent_power,
            "Power_Factor": data.power_factor,
            "Load_Factor": data.load_factor
        }
        pred_payload = {
            "Fuel_Consumption_L_h": pred_fuel,
            "CO2_Emission_kg_h": pred_co2
        }
        
        requests.put(f"{FIREBASE_DB_URL}/Generator_System/Sensor_Data.json", json=sensor_payload)
        requests.put(f"{FIREBASE_DB_URL}/Generator_System/Predictions.json", json=pred_payload)
        
        # 4. රෙස්පොන්ස් එක යැවීම
        return {
            "status": "success",
            "Fuel_Consumption_L_h": pred_fuel,
            "CO2_Emission_kg_h": pred_co2
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


