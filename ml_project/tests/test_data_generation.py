import pandas as pd

from ml_project.src.data_generation import generate_data


def test_data_generation():
    dummy_data = generate_data()

    assert isinstance(dummy_data, pd.DataFrame)
    assert dummy_data.shape == (1000, 3)
    assert "x1" in dummy_data.columns
    assert pd.Series(["x1", "x1", "y"]).isin(dummy_data.columns).all()
    assert pd.api.types.is_numeric_dtype(dummy_data["y"])
