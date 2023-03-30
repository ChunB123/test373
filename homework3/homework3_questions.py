import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model

#################
### Problem 1 ###
#################

# The following code reads in the breast cancer dataset
# and splits it into training and validation datasets.

dataset = sk.datasets.load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X, y)

logisticModel = sk.linear_model.LogisticRegression().fit(X_train, y_train)

beta = np.concatenate((logisticModel.intercept_, logisticModel.coef_[0]))
ace = avg_cross_entropy(beta, X_train, y_train)


# In logistic regression for binary classification,
# we use average cross-entropy
# rather than mean squared error as our objective function.
# Write a function that computes the average cross-entropy
# for predictions made using a coefficient vector beta.
# Hint: It might help to first implement the sigmoid function
# and the binary cross-entropy loss function.

def avg_cross_entropy(beta, X, y):
    # X is an N by d numpy array. 
    # Each row of X contains a feature vector.
    # Corresponding target values are stored in y,
    # which is a numpy array of length N.
    # beta is a numpy array of length d+1.
    # beta contains coefficients for logistic regression.

    # Your code goes here.

    return binary_cross_entropy(y, sigmoid(beta[0] + X @ beta[1:])).mean()


def sigmoid(u):
    return 1 / (1 + np.exp(-u))


def binary_cross_entropy(p, q):
    return -p * np.log(q) - (1 - p) * np.log(1 - q)


#################
### Problem 2 ###
#################

# In multiclass classification, we typically use the softmax
# function as one of the ingredients of our prediction function.
# Recall that the softmax function is useful in machine learning
# because it converts a vector into a "probability vector".
# Implement the softmax function using numpy:

def softmax(u):
    # u is a numpy array of length K.

    # your code goes here

    return np.exp(u) / (np.exp(u).sum())  # You will change this line


#################
### Problem 3 ###
#################

# In multiclass classification, we typically use the 
# cross-entropy loss function.
# Implement the cross-entropy loss function in Python:

def cross_entropy(p, q):
    # p and q are probability vectors of length K.
    # We think of q as being a "predicted" probability vector
    # and of p as being a "ground truth" probability vector.
    # The components of q are assumed to satisfy 0 < q_k < 1.
    # This function will compute the cross-entropy to measure
    # how well the predicted probability vector q agrees
    # with the ground truth probability vector p.

    # Your code goes here.

    return - p @ np.log(q)  # You will change this line

#################
### Problem 4 ###
#################

# Participate in the Kaggle Digit Recognizer contest:
# https://www.kaggle.com/c/digit-recognizer
# What classification accuracy do you achieve?
# Along with your code, upload a screenshot showing your Kaggle score.

# Your code goes here.
data_dic = "/Users/usfmichael/Library/CloudStorage/OneDrive-UniversityofSanFrancisco/USF/MATH373/digit-recognizer/"

train_data = pd.read_csv(data_dic + "train.csv")
test_data = pd.read_csv(data_dic + "test.csv")

lables_train = train_data['label']

df_train = train_data.iloc[:, 1:]

##############
model = sk.linear_model.LogisticRegression()
model.fit(df_train, lables_train)
##################

labels_pred = model.predict(test_data)
ImageId = np.array([x+1 for x in range(len(labels_pred))])
d_sub = {'ImageId': ImageId, 'Label': labels_pred}
df_sub = pd.DataFrame(d_sub)
df_sub.to_csv(data_dic + 'JiazhengSub.csv', index=False)