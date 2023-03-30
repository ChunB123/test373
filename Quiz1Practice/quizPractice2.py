import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.model_selection
import sklearn.linear_model

data = pd.read_csv("/Users/usfmichael/Library/CloudStorage/OneDrive-UniversityofSanFrancisco/USF/MATH373/HousePriceContest/house-prices-advanced-regression-techniques/train.csv")

X = data[['LotArea', 'LotFrontage']].astype(float)
y = data['SalePrice'].astype(float)

# fillna
imput_val = {'LotFrontage': np.mean(data['LotFrontage'])}
X = X.fillna(imput_val)

# hotcode OverallQual
oqDummyData = pd.get_dummies(data['OverallQual'], prefix="oq")
X = pd.concat([X, oqDummyData], axis=1)

X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2)

reg = sk.linear_model.LinearRegression().fit(X_train, y_train)

y_pred = reg.predict(X_test)

mse = sum((y_test-y_pred)**2)/len(y_test)

mse2 = sk.metrics.mean_squared_error(y_test, y_pred)


