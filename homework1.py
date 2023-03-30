import numpy as np
import sklearn as sk
from sklearn.datasets import load_diabetes
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from PIL import Image

# The code that you submit must run without errors.
# If you need help with these coding problems, feel free to ask!
# Sometimes people get a lot of help in office hours 
# if they are stuck or struggling.
# You are also allowed to discuss these problems with classmates.
# A major goal of this class is to practice programming in Python
# and to learn how to use libraries such as numpy and sklearn.

# Remember that, while it's ok to google to learn Python syntax
# and how to use various Python functions, 
# simply copying code from online sources is considered cheating.
# Any code snippets that are copied from online must be labeled
# explicitly with comments, and links to the source must be given.

# Problem 1
# Use numpy's np.random.randn function to
# randomly generate an 8 x 8 matrix M.
# Your code goes here.
M = np.random.randn(8, 8)

# Problem 2
# Use numpy to randomly generates a vector v of dimension 8.
# Your code goes here.
v = np.random.randn(8)

# Problem 3
# Write code that uses numpy to multiply the above matrix M by the above vector v.
# Your code goes here.
r1 = np.matmul(M, v)

# Problem 4
# Write code that computes the sum of the squares of the components of v.
# Can you do it in one line?
# Your code goes here.
r2 = np.matmul(v, v)

# Problem 5
# Write numpy code that randomly generates a vector w of dimension 8
# and then computes the dot product of v and w. 
# Use numpy's vdot function to compute the dot product.
# Your code goes here.
w = np.random.randn(8)
r3 = np.vdot(v, w)


# Problem 6:
# Complete the definition of the following function.
# How concise can you make your code?
# See if you can make good use of matrix and vector operations.
def mean_squared_error2(beta, X, y):
    # beta is a vector of dimension d+1 (stored as a numpy array)
    # X is a matrix (numpy array) with shape N by d.
    # y is a vector of dimension N (stored as a numpy array)
    # Each row of X is a feature vector.
    # Each entry of y is a corresponding target value.
    # beta is a vector of coefficients for linear regression.
    # This function computes the mean squared error L(beta),
    # as it was defined in class.

    # Your code goes here.
    pred_vals = np.matmul(X, beta[1:]) + beta[0]
    return sum((pred_vals - y) ** 2) / len(y)


# Problem 7
# Load the California housing dataset and split it into 
# training and validation datasets.
# What is the value of d for this dataset?
# (Recall that d is the dimension of the feature vectors.)
# Randomly generate a vector beta of linear regression coefficients.
# What should be the dimension of beta?
# Compute the mean squared error on the validation dataset
# using this vector beta to make predictions.
caDataset = fetch_california_housing()
X = caDataset.data
y = caDataset.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# d is 8
d = len(X[0,])
beta = np.random.randn(d + 1)
MSEofRandBeta = mean_squared_error2(beta, X_test, y_test)
XwithIntercepts = np.hstack((np.ones([X_test.shape[0], 1], X_test.dtype), X_test))
y_predRandBeta = np.matmul(XwithIntercepts, beta)

# Problem 8
# Randomly generate 1000 different beta vectors.
# Which vector gives you the least mean squared error on the training dataset?
# Print out the best vector beta that you found
# and the corresponding mean squared error on the training dataset.
beta_vecs = [np.random.randn(d + 1) for x in range(1000)]
r = [mean_squared_error2(x, X_train, y_train) for x in beta_vecs]
bestBeta = beta_vecs[r.index(min(r))]
MSEofBestBeta = mean_squared_error2(bestBeta, X_train, y_train)
print("bestBeta:", bestBeta, " MSE:", MSEofBestBeta)


# Problem 9
# Write Python code that reads in an image, and then creates a new image where the R, G, and B values
# at each pixel are weighted average of nearby R, G, and B values
# in the original image. Display the new image.
def generateImage(fileLocation):
    img = Image.open(fileLocation).crop((1000, 1000, 2000, 2000))
    numpydata = np.asarray(img)

    newNumpydata = np.array(
        [[weightedAverage(rIndex, cIndex, numpydata) for cIndex in range(numpydata[rIndex].shape[0])] for rIndex in
         range(numpydata.shape[0])])

    Image.fromarray(newNumpydata.astype(np.uint8)).show()


def weightedAverage(rIndex, cIndex, numpydata):
    nearbyPixels = []

    # not the upperost one
    if rIndex > 0:
        nearbyPixels.append(numpydata[rIndex - 1, cIndex])

    # not the lowermost one
    if rIndex < numpydata.shape[0] - 1:
        nearbyPixels.append(numpydata[rIndex + 1, cIndex])

    # not the leftmost one
    if cIndex > 0:
        nearbyPixels.append(numpydata[rIndex, cIndex - 1])

    # not the rightmost one
    if cIndex < numpydata.shape[1] - 1:
        nearbyPixels.append(numpydata[rIndex, cIndex + 1])

    nearbyPixelsMatrix = np.matrix(nearbyPixels)

    return np.asarray(nearbyPixelsMatrix.mean(0, dtype="int"))[0]


def testProblem6():
    # test for problem 6
    dataset = load_diabetes()
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
    reg = LinearRegression().fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    skMSE = sk.metrics.mean_squared_error(y_test, y_pred)

    beta = np.append(np.array([reg.intercept_]), reg.coef_)
    myMSE = mean_squared_error2(beta, X_test, y_test)
    print(np.around(skMSE, 3) == np.around(myMSE, 3))


if __name__ == '__main__':
    testProblem6()
    generateImage("/Users/usfmichael/Library/CloudStorage/OneDrive-UniversityofSanFrancisco/USF/MATH373/VisualDifferentialGeometry.jpg")
