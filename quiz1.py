# Quiz 1 consists of three coding problems.
# You can write your answers (your code) in the spaces provided below.

# Problem 1
# The code below loads the breast cancer dataset
# and splits the data into training and validation datasets.
# Notice that the target values (that is, the labels) are all 0 or 1.
# Use the training data to train a model to predict the labels.
# (This is a binary classification problem.)
# What is your classification accuracy when you use your model
# to predict the labels for the validation dataset?

import numpy as np
import sklearn as sk
import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model

dataset = sk.datasets.load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X, y, random_state=42)

# YOUR CODE GOES HERE.

bcModel = sk.linear_model.LogisticRegression(max_iter = 100000).fit(X_train, y_train)
y_pred = bcModel.predict(X_val)
bcModeAcc0 = 1 - sum(np.absolute(y_val - y_pred)) / len(y_val)
bcModeAcc1 = 1 - sk.metrics.mean_squared_error(y_val, y_pred)


# %%
# Problem 2
# Implement the sigmoid function, also called the logistic function.
# This function is useful in machine learning because it converts
# a real number to a probability (that is, a number between 0 and 1).

def sigmoid(u):
    # YOUR CODE GOES HERE.
    # e^u / (1 + e^u)
    return np.exp(u) / (1 + np.exp(u))  # This line should be modified


# %%
# Problem 3

# The following code loads the diabetes dataset
# and splits the data into training and validation datasets.
# Notice that the target values are numeric, not categorical
# (so this is a regression problem, not a classification problem).

dataset = sk.datasets.load_diabetes()
X = dataset.data
y = dataset.target

X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X, y, random_state=42)

# Suppose that a linear regression model has been trained to solve
# this regression problem, and that the beta coefficients for this linear
# regression model are
# beta0 = 151.89 (this is the "bias term" in the model),
# beta1 = -48.93, 
# beta2 = -219.74,  
# beta3 = 503.1 ,  
# beta4 = 332.1 , 
# beta5 = -568.53,  
# beta6 = 328.26,   
# beta7 = 33.43,
# beta8 = 87.66,
# beta9 = 636.94,
# beta10 = 153.37

# What is the mean squared error when using this model
# (with the given beta coefficients)
# to predict the target values for the validation dataset?

beta0 = 151.89
beta_coef = np.array([-48.93, -219.74, 503.1, 332.1, -568.53, 328.26, 33.43,
                      87.66, 636.94, 153.37])
"""
beta0Vector = np.array([beta0 for x in range(X_val.shape[0])])
y_pred = beta0Vector + np.sum(beta_coef * X_val, axis=1)
"""
y_pred = beta0 +  X_val @ beta_coef
acc = sk.metrics.mean_squared_error(y_val, y_pred)

# YOUR CODE GOES HERE.


##### This is the end of quiz 1. ########
