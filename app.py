import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Generator Intelligence API")

# කොලැබ් එකෙන් ඩවුන්ලෝඩ් කරගත්ත මොඩල් ෆයිල් දෙක ലോඩ් කරගැනීම
try:
  fuel_model = joblib.load("generator_fuel_model.pkl")
  co2_model = joblib.load("generator_co2_model.pkl")
except Exception as e:
  print(f"Error loading models: {e}")


# ESP32 එකෙන් හෝ වෙනත් තැනකින් එන ඩේටා ෆෝමැට් එක (Data Schema)
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
    # ඩේටා ලිස්ට් එකකට හැඩගැසීම
    input_features = [[
        data.active_power,
        data.apparent_power,
        data.power_factor,
        data.load_factor,
    ]]

    # මොඩල් දෙකෙන්ම ප්‍රඩික්ෂන් ලබා ගැනීම
    pred_fuel = fuel_model.predict(input_features)[0]
    pred_co2 = co2_model.predict(input_features)[0]

    return {
        "Fuel_Consumption_L_h": round(float(pred_fuel), 3),
        "CO2_Emission_kg_h": round(float(pred_co2), 3),
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))