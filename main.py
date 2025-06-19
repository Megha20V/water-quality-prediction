# Import necessary libraries
import pandas as pd # data manipulation
import numpy as np # numerical python - linear algebra

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('afa2e701598d20110228.csv', sep=';')

df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')

df = df.sort_values(by=['id', 'date'])

df['year'] = df['date'].dt.year

df['month'] = df['date'].dt.month

pollutants = ['O2', 'NO3', 'NO2', 'SO4','PO4', 'CL']