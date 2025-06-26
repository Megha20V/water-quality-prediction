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

df = df.dropna(subset=pollutants)

X = df[['id', 'year']]
y = df[pollutants]

X_encoded = pd.get_dummies(X, columns=['id'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

for i, pollutant in enumerate(pollutants):
    print(f'{pollutant}:')
    print('   MSE:', mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    print('   R2:', r2_score(y_test.iloc[:, i], y_pred[:, i]))
    print()

