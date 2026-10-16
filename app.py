
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained models
saved = joblib.load("weather_models.joblib")
models = saved["models"]
FEATURES = saved["features"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Weather Prediction API is running",
        "endpoint": "/predict",
        "method": "POST",
        "required_features": FEATURES
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please send a valid JSON object."
        }), 400

    missing = [name for name in FEATURES if name not in data]
    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing
        }), 400

    try:
        values = {name: float(data[name]) for name in FEATURES}
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("All feature values must be finite numbers.")

        # Basic input validation
        if not 1 <= values["month"] <= 12:
            raise ValueError("month must be between 1 and 12.")
        if not 0 <= values["wind_speed_kmh"] <= 300:
            raise ValueError("wind_speed_kmh must be between 0 and 300.")
        if not 0 <= values["previous_humidity_pct"] <= 100:
            raise ValueError("previous_humidity_pct must be between 0 and 100.")
        if not 0 <= values["previous_rainfall_mm"] <= 1000:
            raise ValueError("previous_rainfall_mm must be between 0 and 1000.")
        if not 800 <= values["pressure_hpa"] <= 1100:
            raise ValueError("pressure_hpa must be between 800 and 1100.")

        input_df = pd.DataFrame([values], columns=FEATURES)

        temperature = float(
            models["temperature_c"].predict(input_df)[0]
        )
        humidity = float(
            models["humidity_pct"].predict(input_df)[0]
        )
        rainfall = float(
            models["rainfall_mm"].predict(input_df)[0]
        )

        # Keep output values within basic physical bounds
        humidity = float(np.clip(humidity, 0, 100))
        rainfall = max(0.0, rainfall)

        return jsonify({
            "predicted_temperature_c": round(temperature, 2),
            "predicted_humidity_pct": round(humidity, 2),
            "predicted_rainfall_mm": round(rainfall, 2),
            "note": "Educational ML estimate, not an official weather forecast."
        })

    except (TypeError, ValueError, OverflowError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
