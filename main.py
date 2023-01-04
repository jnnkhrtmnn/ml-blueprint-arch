from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ObservationIn(BaseModel):
    x1: float
    x2: float


class PredictionOut(BaseModel):
    x1: float
    x2: float
    prediction: float


@app.post("/predict", response_model=PredictionOut)
def prediction_service(input_data: ObservationIn):

    model_file = Path(__file__).parent.resolve() / "models" / "regressor.joblib"
    regressor = joblib.load(model_file)

    X = np.array([[input_data.x1, input_data.x2]])

    assert X.shape == (1, 2)

    prediction = regressor.predict(X)

    return {"x1": input_data.x1, "x2": input_data.x2, "prediction": prediction[0]}
