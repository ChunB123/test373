import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection
from sklearn import *

dataset = sklearn.datasets.load_diabetes()

X = dataset.data
y = dataset.target

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, train_size=.8)

model = sklearn.linear_model.LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

beta0 = model.intercept_

beta_vec = model.coef_

N_val = X_val.shape[0]

y_pred_check = []

for i in range(N_val):
    xi = X_val[i]
    yi_pred = beta0 + np.vdot(beta_vec, xi)
    y_pred_check.append(yi_pred)

