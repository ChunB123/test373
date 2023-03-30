import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.linear_model
import sklearn.model_selection

data = pd.read_csv("/Users/usfmichael/Library/CloudStorage/OneDrive-UniversityofSanFrancisco/USF/MATH373/auto-mpg.csv")

X = data[['cylinders', 'displacement',
          #'horsepower',
          'weight',
       'acceleration', 'model year', 'origin']]
"""
X['horsepower'] = pd.to_numeric(X['horsepower'], errors='coerce')
imput_dic = {'horsepower': np.mean(X['horsepower'])}
X = X.fillna(imput_dic)
"""
X = X.astype(float)

cylinders_dummy = pd.get_dummies(X['cylinders'], prefix="cy")

X = pd.concat([X,cylinders_dummy],axis=1)

y = data[['mpg']].astype(float)



X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2)

linearmodel = sk.linear_model.LinearRegression().fit(X_train, y_train)

y_pred = linearmodel.predict(X_test)
mse = sk.metrics.mean_squared_error(y_test, y_pred)