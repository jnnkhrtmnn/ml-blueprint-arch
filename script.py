from pathlib import Path

import joblib

from src.data_generation import generate_data
from src.train import train_model

project_path = Path(__file__).parent.resolve()

data_file_path = project_path / "data" / "dummy_data.pkl"
dummy_data = generate_data()
dummy_data.to_pickle(data_file_path)

model_file_path = project_path / "models" / "regressor.joblib"
regressor = train_model(training_data=dummy_data)
joblib.dump(regressor, model_file_path)

# TODO: Include logging
# TODO: Add MLflow
