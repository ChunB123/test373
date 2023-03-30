import numpy as np
import pandas as pd
import matplotlib
import sklearn as sk
import sklearn.linear_model

data_dir = "/Users/usfmichael/Library/CloudStorage/OneDrive-UniversityofSanFrancisco/USF/MATH373/HousePriceContest/house-prices-advanced-regression-techniques/"
train_data = pd.read_csv(data_dir + "train.csv")
test_data = pd.read_csv(data_dir + "test.csv")

train_data.isna().sum().sum()
train_data.info()
train_df = train_data[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt',
                       'YearRemodAdd', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd']]
train_labels = train_data['SalePrice']
"""
# One-hot encode the "Neighborhood" feature
train_neighborhood_dummies = pd.get_dummies(train_data['Neighborhood'], prefix='Neighborhood')
train_df = pd.concat([train_df, train_neighborhood_dummies], axis=1)
"""

############
reg = sk.linear_model.LinearRegression().fit(train_df, train_labels)

test_df = test_data[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt',
                     'YearRemodAdd', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd']]
"""
test_neighborhood_dummies = pd.get_dummies(test_data['Neighborhood'], prefix='Neighborhood')
test_df = pd.concat([test_df, test_neighborhood_dummies], axis=1)
"""
test_df.isna().sum()
# fill the missing data
impute_data = {'GarageCars': test_data['GarageCars'].mean(), 'TotalBsmtSF': test_data['TotalBsmtSF'].mean()}
test_df = test_df.fillna(impute_data)

y_pred = reg.predict(test_df)


d_sub = {'Id': test_data['Id'], 'SalePrice': y_pred}
df_sub = pd.DataFrame(d_sub)
df_sub.to_csv(data_dir + 'JiazhengSub.csv', index=False)
