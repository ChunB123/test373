import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.datasets import load_breast_cancer
import sklearn.model_selection
import matplotlib.pyplot as plt


def sigmoid(u):
    return 1 / (1 + np.exp(-u))


def binary_cross_entropy(p, q):
    return -p * np.log(q) - (1 - p) * np.log(1 - q)


def Xhat(X):
    return np.insert(X, 0, 1, axis=1)


def grad_L(X, y, beta):
    return np.average(np.array([(sigmoid(x @ beta) - y[index]) * x for index, x in enumerate(Xhat(X))]), axis=0)


def eval_L(X, y, beta):
    return np.average([binary_cross_entropy(y[index],sigmoid(xi @ beta)) for index, xi in enumerate(Xhat(X))])


def train_model_using_grad_descent(X, y, alpha=0.1, max_iter=100):
    beta = np.zeros(X.shape[1] + 1)
    L_vals = []
    for _ in range(max_iter):
        beta = beta - alpha * grad_L(X, y, beta)
        print(alpha * grad_L(X, y, beta))

        L_vals.append(eval_L(X, y, beta))
    # create a new figure
    fig, ax = plt.subplots()

    # plot the array against its index
    ax.plot(L_vals)

    # set the title and labels
    ax.set_title('Array plot')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    # show the plot
    plt.show()
    return beta


if __name__ == '__main__':
    # Load dataset
    cancer_data = load_breast_cancer()
    X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(cancer_data.data, cancer_data.target,
                                                                         test_size=0.2)

    mean = np.average(X_train, axis=0)
    s = np.std(X_train, axis=0)
    X_train = (X_train - mean) / s
    X_val = (X_val - mean) / s

    beta = train_model_using_grad_descent(X_train, y_train, alpha=0.01, max_iter=500)
    y_pred = [1 if _ > 0.5 else 0 for _ in sigmoid(Xhat(X_val) @ beta)]

    acc = np.average(y_pred == y_val)
    print(acc)

