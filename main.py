from pathlib import Path

import joblib
from fastapi import FastAPI

app = FastAPI()


@app.post("/predict")
def prediction_service():

    model_file = Path(__file__).parent.resolve() / "models" / "regressor.joblib"
    regressor = joblib.load(model_file)

    input_data = ""

    prediction = regressor.predict(input_data)

    return prediction
