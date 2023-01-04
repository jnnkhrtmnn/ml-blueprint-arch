import argparse
import logging
from pathlib import Path

import joblib
import mlflow

from src.data_generation import generate_data
from src.train import train_model

logging.basicConfig(  # type: ignore
    format="%(asctime)s %(name)s %(levelname)-8s %(message)s",
    level=logging.INFO,  # type: ignore
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)  # type: ignore

project_path = Path(__file__).parent.resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--no_mlflow_logging", action=argparse.BooleanOptionalAction)
parser.set_defaults(no_mlflow_logging=False)

args = parser.parse_args()

if not args.no_mlflow_logging:
    experiment_name = "ml_blueprint_project"

    experiments = mlflow.list_experiments()
    if experiment_name not in [experiment.name for experiment in experiments]:
        mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name=experiment_name)
    mlflow.sklearn.autolog()

data_file_path = project_path / "data" / "dummy_data.pkl"
dummy_data = generate_data()
dummy_data.to_pickle(data_file_path)


model_file_path = project_path / "models" / "regressor.joblib"
regressor = train_model(training_data=dummy_data)
joblib.dump(regressor, model_file_path)
