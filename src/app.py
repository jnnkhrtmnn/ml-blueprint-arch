# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 23:24:13 2021

@author: janni
"""

import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import load


path = Path('C:/Users/janni/Desktop/blueprint/ml-blueprint-arch')

reg = load(path / 'models' / 'reg.joblib') 

