
# Advanced Weather Prediction API

## Project Overview
This project demonstrates an end-to-end Machine Learning
application that predicts temperature, humidity and rainfall
using Python, Scikit-learn, Joblib and Flask.

The trained models are exposed through a REST API and can
be tested using Postman.

## Objectives
- Preprocess weather data.
- Train three regression models.
- Evaluate predictions using MAE and R-squared.
- Save models using Joblib.
- Develop and test a Flask REST API.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Flask
- Joblib
- Postman

## Project Files
- generate_dataset.py: creates a simulated starter dataset.
- train_model.py: trains and evaluates the models.
- app.py: runs the prediction API.
- weather.csv: generated dataset.
- weather_models.joblib: saved models.
- model_metrics.json: test-set evaluation results.

## Installation
Install the dependencies:

    python -m pip install -r requirements.txt

## Execution
Run these commands in order:

    python generate_dataset.py
    python train_model.py
    python app.py

## API Testing
Endpoint: POST http://127.0.0.1:5000/predict

Send a JSON body containing pressure, wind speed, month,
previous temperature, previous humidity and previous rainfall.

The API returns predicted temperature in Celsius, humidity
as a percentage, and rainfall in millimetres.

## Evaluation
MAE and R-squared scores are saved to model_metrics.json.
Report the actual results produced by your training script.

## Dataset Limitation
The included starter dataset is simulated for educational
purposes. It does not establish real-world forecast accuracy.
Use a suitable historical weather dataset for meaningful
weather prediction experiments.

## Conclusion
This project demonstrates data preparation, model training,
evaluation, serialization and deployment through a REST API.
