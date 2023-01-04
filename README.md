# Deployable Machine Learning App Blueprint
Simple blueprint for machine learning projects using a prediction API run via  fastapi and Docker.
This project is not perfectly engineered, but should provide a very lightweight example.

## Set Up
I like to use pyenv and poetry to manamge my python versions and dependencies.
Simply run poetry install in the root directory of your project.

## Data Science example
As an example, some data is generated and a linear regression model is build to showcase the usage of a machine learning model.
All of this happens in `script.py`, which generates dummy data and trains a machine learning model.
With the argument `--no_mlflow_logging`, you can build a model without logging it to MLflow (`poetry run python ml_project/scripts.py --no_mlflow_logging`).
Make sure that you have built a model before starting the fastapi app.

## Data Analysis
There is a `jupyter notebook` for some EDA. You can have a quick look at the data after

## MLflow
MLflow is my favourite tool for model management. In this project I am using it only for performance tracking, not for model staging and serving.
You can start the server with `poetry run mlflow server` and open the UI on `http://127.0.0.1:5000`.


## Testing
I included some simple tests for the functions I wrote.
You can execute these tests by running `poetry run pytest`.


## Run fastapi app
To run the app without docker, simply execute `poetry run uvicorn ml_project.main:app --reload --host localhost --port 8000` in the root directory of the project.


## Docker
The docker image can be built by running
`docker build -t mlblueprintarch .`.
To start the container just run `docker run -d --name mlapp -p 8000:8000 mlblueprintarch`.


## API tests
You can test the API (either dockerized or not) with
`curl   --header "Content-Type: application/json"   --request POST   --data '{"x1": 1, "x2" : 1}'   http://localhost:8000/predict`
