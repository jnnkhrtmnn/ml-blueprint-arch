import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from ml_project.src.train import train_model


@pytest.fixture
def random_data():
    return pd.DataFrame(np.random.rand(100, 3), columns=["x1", "x2", "y"])


def test_model_training(random_data):

    regression_model = train_model(training_data=random_data)

    assert isinstance(regression_model, LinearRegression)
    assert regression_model.coef_.shape == (2,)
    assert (regression_model.feature_names_in_ == ["x1", "x2"]).all()
