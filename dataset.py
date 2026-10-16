
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1500

month = rng.integers(1, 13, n)
pressure = rng.normal(1010, 7, n).clip(980, 1035)
wind = rng.gamma(2.0, 5.0, n).clip(0, 45)

# Previous weather observations (simulated)
previous_temp = (
    27 + 5 * np.sin(2 * np.pi * (month - 3) / 12)
    + rng.normal(0, 2, n)
)
previous_humidity = (
    65 + 15 * np.sin(2 * np.pi * (month + 2) / 12)
    + rng.normal(0, 8, n)
).clip(20, 100)
previous_rainfall = rng.exponential(3, n)
previous_rainfall[rng.random(n) < 0.65] = 0

# Simulated target measurements, not real observations
temperature = (
    previous_temp
    + 0.12 * (pressure - 1010)
    - 0.06 * wind
    + rng.normal(0, 1.5, n)
)
humidity = (
    previous_humidity
    + 0.3 * previous_rainfall
    - 0.15 * (temperature - 27)
    + rng.normal(0, 5, n)
).clip(15, 100)

rain_probability = 1 / (
    1 + np.exp(-(-2.0 + 0.035 * (humidity - 65)
                 + 0.10 * previous_rainfall))
)
rainfall = np.where(
    rng.random(n) < rain_probability,
    rng.gamma(2.0, 4.0, n),
    0.0
)

df = pd.DataFrame({
    "pressure_hpa": pressure,
    "wind_speed_kmh": wind,
    "month": month,
    "previous_temp_c": previous_temp,
    "previous_humidity_pct": previous_humidity,
    "previous_rainfall_mm": previous_rainfall,
    "temperature_c": temperature,
    "humidity_pct": humidity,
    "rainfall_mm": rainfall
})

df.to_csv("weather.csv", index=False)
print("Dataset created: weather.csv")
print(df.head())
print("Rows and columns:", df.shape)
