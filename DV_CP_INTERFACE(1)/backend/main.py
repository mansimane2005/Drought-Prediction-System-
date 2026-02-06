from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from drought_model import fetch_weather_data, train_and_predict
import requests
import traceback

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema for POST requests
class PredictionRequest(BaseModel):
    city_name: str
    start_date: str  # format: YYYY-MM-DD
    end_date: str    # format: YYYY-MM-DD

# Function to get lat/lon from city name
def get_coordinates(city_name):
    try:
        response = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1"
        )
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            lat = data["results"][0]["latitude"]
            lon = data["results"][0]["longitude"]
            return lat, lon
        else:
            return None, None
    except Exception:
        return None, None

# POST endpoint for prediction
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        lat, lon = get_coordinates(request.city_name)
        if lat is None or lon is None:
            return {"error": f"❌ Could not find coordinates for '{request.city_name}'"}

        # Format dates
        start = request.start_date.replace("-", "")
        end = request.end_date.replace("-", "")

        # Fetch weather data
        df, error = fetch_weather_data(lat, lon, start, end)
        if error:
            return {"error": error}

        # Train model and get predictions
        result_summary, prediction_df = train_and_predict(df)

        if prediction_df is None or prediction_df.empty:
            drought_status = "ℹ️ Prediction data is empty or invalid."
        elif 'Predicted_Drought' not in prediction_df.columns:
            drought_status = "ℹ️ 'Predicted_Drought' column missing."
        else:
            drought_present = prediction_df['Predicted_Drought'].any()
            drought_status = "🌵 Drought is predicted in this period." if drought_present else "✅ No drought predicted."

        # Convert DataFrame to list of dicts
        prediction = prediction_df.to_dict(orient="records")

        return {
            "lat": lat,
            "lon": lon,
            "result_summary": result_summary.replace("\n", "<br>"),
            "drought_status": drought_status,
            "prediction": prediction
        }

    except Exception as e:
        return {"error": f"❌ Internal error:\n{traceback.format_exc()}"}