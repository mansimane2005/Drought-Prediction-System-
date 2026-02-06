import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def fetch_weather_data(lat, lon, start, end):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "PRECTOTCORR,T2M",
        "start": start,
        "end": end,
        "latitude": lat,
        "longitude": lon,
        "community": "RE",
        "format": "JSON"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "properties" not in data or "parameter" not in data["properties"]:
        return None, "API error. Check coordinates or date."

    params_data = data["properties"]["parameter"]

    if 'PRECTOTCORR' not in params_data or 'T2M' not in params_data:
        return None, f"Missing expected parameters. Got: {list(params_data.keys())}"

    df = pd.DataFrame({
        "Date": params_data["PRECTOTCORR"].keys(),
        "Rainfall": params_data["PRECTOTCORR"].values(),
        "Temperature": params_data["T2M"].values()
    })

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df, None

def label_drought_condition(df, window=7, rain_sum_threshold=25):
    """
    Labels drought if total rainfall over a rolling window is below the threshold.
    """
    df['Rainfall'] = pd.to_numeric(df['Rainfall'], errors='coerce')
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
    df.dropna(inplace=True)

    df['SumRainfall'] = df['Rainfall'].rolling(window=window).sum()
    df['Drought'] = df['SumRainfall'].apply(lambda x: 1 if x is not None and x < rain_sum_threshold else 0)
    df.dropna(inplace=True)

    return df

def train_and_predict(df):
    df = label_drought_condition(df)

    X = df[['Rainfall', 'Temperature']]
    y = df['Drought']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)

    # Add prediction column
    df['Predicted_Drought'] = model.predict(scaler.transform(X))

    # Decision: if more than 30% days are drought, declare drought
    drought_days = df['Predicted_Drought'].sum()
    drought_ratio = drought_days / len(df)

    if drought_ratio > 0.3:
        drought_status = "🌵 Drought is predicted in this period."
    else:
        drought_status = "✅ No drought is predicted in this period."

    result_text = (
        f"✅ Model Accuracy: {acc:.2f}\n\n"
        f"📊 Classification Report:\n{report}\n"
        f"{drought_status}"
    )

    return result_text, df[['Date', 'Rainfall', 'Temperature', 'SumRainfall', 'Drought', 'Predicted_Drought']]
