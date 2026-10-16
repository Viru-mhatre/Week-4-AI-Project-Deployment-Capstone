
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "weather.csv"
MODEL_PATH = "weather_models.joblib"

FEATURES = [
    "pressure_hpa",
    "wind_speed_kmh",
    "month",
    "previous_temp_c",
    "previous_humidity_pct",
    "previous_rainfall_mm"
]

TARGETS = [
    "temperature_c",
    "humidity_pct",
    "rainfall_mm"
]

# 1. Load and preprocess data
df = pd.read_csv(DATA_PATH)
df = df.replace([float("inf"), float("-inf")], float("nan"))
df = df.dropna(subset=FEATURES + TARGETS)

if len(df) < 20:
    raise ValueError("Please provide at least 20 complete rows.")

X = df[FEATURES]
y = df[TARGETS]

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {}
metrics = {}

# 3. Train and evaluate one model for each target
for target in TARGETS:
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )

    model.fit(X_train, y_train[target])
    predictions = model.predict(X_test)

    models[target] = model
    metrics[target] = {
        "MAE": round(float(mean_absolute_error(
            y_test[target], predictions
        )), 3),
        "R2": round(float(r2_score(
            y_test[target], predictions
        )), 3)
    }

# 4. Save models and their required input features
joblib.dump({
    "models": models,
    "features": FEATURES,
    "targets": TARGETS
}, MODEL_PATH)

with open("model_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nTraining completed!")
print("Saved model:", MODEL_PATH)
print("\nTest-set metrics:")
for target, values in metrics.items():
    print(target, values)

print("\nMetrics saved to model_metrics.json")
