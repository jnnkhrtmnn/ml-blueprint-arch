import pandas as pd
from scipy.stats import norm


def generate_data() -> pd.DataFrame:
    """Generates random data set with y, x1, x2 and epsilon.
    y is a linear combination of iid gaussian x1 and x2
    plus the gaussian error term epsilon

    Input
    ------
    path to store data

    Returns
    -------
    dummy_data : pandas dataframe of target variable and features.

    """

    OBSERVATIONS = 1000

    x1 = norm.rvs(10, 3, OBSERVATIONS)
    x2 = norm.rvs(30, 5, OBSERVATIONS)
    epsilon = norm.rvs(0, 1, OBSERVATIONS)

    y = x1 + x2 + epsilon

    dummy_data = pd.DataFrame(list(zip(y, x1, x2)), columns=["y", "x1", "x2"])

    return dummy_data
