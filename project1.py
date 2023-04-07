import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.datasets import load_breast_cancer
import sklearn.model_selection
import matplotlib.pyplot as plt
import random


def sigmoid(u):
    return 1 / (1 + np.exp(-u))


def binary_cross_entropy(p, q):
    return -p * np.log(q) - (1 - p) * np.log(1 - q)


def Xhat(X):
    if len(X.shape) == 1:
        return np.insert(X, 0, 1)
    return np.insert(X, 0, 1, axis=1)


def grad_L(X, y, beta):
    if len(X.shape) == 1:
        return (sigmoid(Xhat(X) @ beta) - y) * Xhat(X)
    return np.average(np.array([(sigmoid(x @ beta) - y[index]) * x for index, x in enumerate(Xhat(X))]), axis=0)


def eval_L(X, y, beta):
    return np.average([binary_cross_entropy(y[index], sigmoid(xi @ beta)) for index, xi in enumerate(Xhat(X))])


def train_model_using_grad_descent(X, y, alpha=0.1, max_iter=100):
    beta = np.zeros(X.shape[1] + 1)
    L_vals = []
    for _ in range(max_iter):
        beta = beta - alpha * grad_L(X, y, beta)
        L_vals.append(eval_L(X, y, beta))
    return beta, L_vals


def train_model_using_stochastic_grad_descent(X, y, alpha=0.1, epoch=500):
    beta = np.zeros(X.shape[1] + 1)
    L_vals = []
    for _ in range(epoch):
        indexList = random.sample(range(X.shape[0]), X.shape[0])
        for i in indexList:
            beta = beta - alpha * grad_L(X[i], y[i], beta)
        L_vals.append(eval_L(X, y, beta))
    return beta, L_vals


def draw(datas, iterations=500):
    # create a new figure
    fig, ax = plt.subplots()
    # Plot each curve on the axes object
    for index, data in enumerate(datas):
        ax.plot(data, label="Curve " + str(index))

    # Add a legend to the axes object
    ax.legend()

    # Set the x-axis and y-axis labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Set the title of the figure
    ax.set_title('Multiple curves in one figure')

    # Display the figure
    plt.show()


if __name__ == '__main__':
    # load cancer data
    cancer_data = load_breast_cancer()
    X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(cancer_data.data, cancer_data.target,
                                                                         test_size=0.2)
    mean = np.average(X_train, axis=0)
    s = np.std(X_train, axis=0)
    X_train = (X_train - mean) / s
    X_val = (X_val - mean) / s

    # 1.a
    # Try several different learning rates for gradient descent. Make a plot of the cost function value vs. iteration
    # for each learning rate. So, one single figure will contain several curves, comparing several different learning
    # rates. What did you find to be the best learning rate when using gradient descent? How many iterations are
    # required until the gradient descent method has converged?

    """
    The best learning rate should be 0.9 because it converges faster than other rates. 
    Approximately, it needs 150 iterations to converge based on the graph.
    """

    max_iter = 500
    beta1, L_vals_1 = train_model_using_grad_descent(X_train, y_train, 0.9, max_iter)
    beta2, L_vals_2 = train_model_using_grad_descent(X_train, y_train, 0.6, max_iter)
    beta3, L_vals_3 = train_model_using_grad_descent(X_train, y_train, 0.3, max_iter)
    beta4, L_vals_4 = train_model_using_grad_descent(X_train, y_train, 0.1, max_iter)
    beta5, L_vals_5 = train_model_using_grad_descent(X_train, y_train, 0.01, max_iter)
    draw([L_vals_1, L_vals_2, L_vals_3, L_vals_4, L_vals_5], max_iter)

    beta1, L_vals_1 = train_model_using_stochastic_grad_descent(X_train, y_train, 0.9, max_iter)
    beta2, L_vals_2 = train_model_using_stochastic_grad_descent(X_train, y_train, 0.6, max_iter)
    beta3, L_vals_3 = train_model_using_stochastic_grad_descent(X_train, y_train, 0.3, max_iter)
    beta4, L_vals_4 = train_model_using_stochastic_grad_descent(X_train, y_train, 0.1, max_iter)
    beta5, L_vals_5 = train_model_using_stochastic_grad_descent(X_train, y_train, 0.01, max_iter)
    draw([L_vals_1, L_vals_2, L_vals_3, L_vals_4, L_vals_5], max_iter)

    # 1.b Try several different learning rates for the stochastic gradient method. Make a plot of the cost function
    # value vs. epoch for each learning rate. So, one single figure will contain several curves, comparing several
    # different learning rates. What was the best learning rate when using the stochastic gradient method? How many
    # epochs are required until the stochastic gradient method has converged?

    beta = train_model_using_grad_descent(X_train, y_train, alpha=0.01, max_iter=50)
    y_pred = [1 if _ > 0.5 else 0 for _ in sigmoid(Xhat(X_val) @ beta)]
    acc = np.average(y_pred == y_val)
    print(acc)

# Make a plot that shows the cost function value vs. iteration for gradient descent, using the best learning rate
# that you found for gradient descent. In the same figure, plot the cost function value vs. epoch for the stochastic
# gradient method, using the best learning rate that you found for the stochastic gradient method. Which method
# converges faster? Do both methods eventually find a minimizer for the cost function?

# Report your classification accuracy on the validation dataset.
