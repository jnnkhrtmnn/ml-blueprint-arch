import logging

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def train_model(training_data: pd.DataFrame) -> LinearRegression:
    """Trains lin reg model.
    Saves model to folder.

    Parameters
    ----------
    training_data: pandas DataFrame with columns x1, x2 and y.

    Returns
    -------
    Trained Linear Regression model

    """

    feature_names = ["x1", "x2"]

    X = training_data[feature_names]
    y = training_data["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    logger.info("Training done!")

    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    logger.info(f"Test MSE: {mse} on {len(y_pred)} test samples")

    return regressor
