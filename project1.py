from tqdm import tqdm
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.datasets import load_breast_cancer
import sklearn.model_selection
import matplotlib.pyplot as plt
import random


def sigmoid(u):
    # use identity to prevent overflow
    if isinstance(u, (int, float)):
        if u >= 0:
            return 1.0 / (1.0 + np.exp(-u))
        else:
            return np.exp(u) / (1.0 + np.exp(u))
    else:
        positive_mask = u >= 0
        output = np.array([0 for x in range(len(u))]).astype("float64")
        output[positive_mask] = 1 / (1 + np.exp(-u[positive_mask]))
        output[~positive_mask] = np.exp(u[~positive_mask]) / (1 + np.exp(u[~positive_mask]))
        return output


def softmax(u):
    # return np.exp(u) / np.sum(np.exp(u))
    # use log_softmax instead to prevent overflow
    return np.exp(u - np.max(u) - np.log(np.sum(np.exp(u - np.max(u)))))


def binary_cross_entropy(p, q, eps=1e-10):
    # prevent ln(0)
    q = np.clip(q, eps, 1 - eps)
    return -p * np.log(q) - (1 - p) * np.log(1 - q)


def cross_entropy(p, q, eps=1e-10):
    q = np.clip(q, eps, 1 - eps)
    return -p @ np.log(q)


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


def grad_L_m(X, y, beta):
    if len(X.shape) == 1:
        return np.outer(Xhat(X), (softmax(Xhat(X) @ beta) - y))


def eval_L_m(X, y, beta):
    return np.average([cross_entropy(y[index], sigmoid(xi @ beta)) for index, xi in enumerate(Xhat(X))])


def train_model_using_stochastic_grad_descent_multi(X, y, alpha=0.1, epoch=500):
    beta = np.zeros((X.shape[1] + 1, y.shape[1])).astype("float64")
    L_vals = []
    for _ in tqdm(range(epoch)):
        indexList = random.sample(range(X.shape[0]), X.shape[0])
        for i in indexList:
            beta = beta - alpha * grad_L_m(X[i], y[i], beta)
        # use cross-entropy
        L_vals.append(eval_L_m(X, y, beta))
    return beta, L_vals


def train_model_using_grad_descent(X, y, alpha=0.1, max_iter=100):
    beta = np.zeros(X.shape[1] + 1)
    L_vals = []
    for _ in tqdm(range(max_iter)):
        beta = beta - alpha * grad_L(X, y, beta)
        L_vals.append(eval_L(X, y, beta))
    return beta, L_vals


def train_model_using_stochastic_grad_descent(X, y, alpha=0.1, epoch=500):
    beta = np.zeros(X.shape[1] + 1)
    L_vals = []
    for _ in tqdm(range(epoch)):
        indexList = random.sample(range(X.shape[0]), X.shape[0])
        for i in indexList:
            beta = beta - alpha * grad_L(X[i], y[i], beta)
        L_vals.append(eval_L(X, y, beta))
    return beta, L_vals


def draw(datas, legends, xlabel, ylabel, title):
    # create a new figure
    fig, ax = plt.subplots()
    # Plot each curve on the axes object
    for index, data in enumerate(datas):
        ax.plot(data, label=str(legends[index]))

    # Add a legend to the axes object
    ax.legend()

    # Set the x-axis and y-axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set the title of the figure
    ax.set_title(title)

    # Display the figure
    plt.show()


def breast_cancer_classification():
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
    The best learning rates among [0.1, 0.05, 0.025, 0.0125, 0.00625] should be 0.1 
    because it converges faster than others and get the lowest value of cost function. 
    Approximately, it needs 300 iterations to converge based on my graph.
    """

    max_iter = 500
    learningRates_G = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    betas_G = []
    L_vals_G = []
    for lr in learningRates_G:
        beta, L_vals = train_model_using_grad_descent(X_train, y_train, lr, max_iter)
        betas_G.append(beta)
        L_vals_G.append(L_vals)
    draw(L_vals_G, learningRates_G, "iterations", "cost function value",
         "gradient descent")

    # 1.b
    # Try several different learning rates for the stochastic gradient method. Make a plot of the cost function
    # value vs. epoch for each learning rate. So, one single figure will contain several curves, comparing several
    # different learning rates. What was the best learning rate when using the stochastic gradient method? How many
    # epochs are required until the stochastic gradient method has converged?
    """
    The best learning rates among [0.1, 0.05, 0.025, 0.0125, 0.00625] should be 0.1 
    because it converges faster than others and get the lowest value of cost function. 
    Approximately, it needs 400 epochs to converge .
    """
    learningRates_SG = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    betas_SG = []
    L_vals_SG = []
    for lr in learningRates_SG:
        beta, L_vals = train_model_using_stochastic_grad_descent(X_train, y_train, lr, max_iter)
        betas_SG.append(beta)
        L_vals_SG.append(L_vals)
    draw(L_vals_SG, learningRates_SG, "epochs", "cost function value",
         "stochastic gradient method")

    # 1.c
    # Make a plot that shows the cost function value vs. iteration for gradient descent, using the best learning rate
    # that you found for gradient descent. In the same figure, plot the cost function value vs. epoch for the stochastic
    # gradient method, using the best learning rate that you found for the stochastic gradient method. Which method
    # converges faster? Do both methods eventually find a minimizer for the cost function?
    """
    Stochastic gradient method starts to converge after 100 epochs 
    and gradient method needs 300 iterations. Thus stochastic gradient method 
    converges faster than gradient method. 
    And both methods eventually find a minimizer beta for cost function 
    and the beta found by Stochastic gradient method leads to a lower value of cost function.
    """

    betaG, L_vals_1 = train_model_using_grad_descent(X_train, y_train, 0.1, max_iter)
    betaSG, L_vals_2 = train_model_using_stochastic_grad_descent(X_train, y_train, 0.1, max_iter)
    draw([L_vals_1, L_vals_2], ["gradient", "stochastic gradient"], "iterations", "cost function value",
         "gradient (lr=10) VS stochastic gradient (lr=0.1)")

    # 1.d
    # Report your classification accuracy on the validation dataset.
    """
    Gradient descent's accuracy:  0.9736842105263158  
    Stochastic gradient descent accuracy 0.9473684210526315
    """

    y_pred_G = [1 if _ > 0.5 else 0 for _ in sigmoid(Xhat(X_val) @ betaG)]
    y_pred_SG = [1 if _ > 0.5 else 0 for _ in sigmoid(Xhat(X_val) @ betaSG)]
    print("Gradient descent's accuracy: ", str(np.average(y_pred_G == y_val)),
          " Stochastic gradient descent accuracy", str(np.average(y_pred_SG == y_val)))


def digit_classification():
    mnist_data = sk.datasets.fetch_openml('mnist_784')
    X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(np.array(mnist_data.data).astype("float64"),
                                                                         np.array(mnist_data.target.astype("int64")),
                                                                         test_size=10000)
    X_train /= 255.0
    X_val /= 255.0

    # Hot encoding y_train
    y_train_he = np.zeros((len(y_train), 10))
    y_train_he[np.arange(len(y_train)), y_train] = 1

    # 2.a
    # Try several different learning rates for the stochastic gradient method. Make a plot of the cost function value
    # vs. epoch for each learning rate. So, one single figure will contain several curves, comparing several
    # different learning rates. What was the best learning rate when using the stochastic gradient method? How many
    # epochs are required until the stochastic gradient method has converged?
    """
    The best learning rate among [0,2, 0.05, 0.01] is 0.01. 
    It converges after 10 epochs, which is faster than others.
    """
    max_iter = 20
    learningRates = [0.2, 0.05, 0.01]
    beta0, L_vals_0 = train_model_using_stochastic_grad_descent_multi(X_train, y_train_he, learningRates[0], max_iter)
    beta1, L_vals_1 = train_model_using_stochastic_grad_descent_multi(X_train, y_train_he, learningRates[1], max_iter)
    beta2, L_vals_2 = train_model_using_stochastic_grad_descent_multi(X_train, y_train_he, learningRates[2], max_iter)
    draw([L_vals_0, L_vals_1, L_vals_2], learningRates, "epochs", "cost function value",
         "stochastic gradient method")

    # 2.b
    # Report your classification accuracy on the validation dataset.
    """
    Stochastic gradient descent's accuracy (learning rate: 0.01): 0.9225
    """
    y_pred = [np.argmax(yi) for yi in (Xhat(X_val) @ beta2)]
    print("Stochastic gradient descent's accuracy (learning rate: 0.01): ",
          str(np.average(y_pred == y_val)))

    # 2.c
    # In a single figure, display the 8 images which most confused your model. ("Confused" means that your model was
    # confident in its prediction, but also wrong.) Comment on why these images are confusing.
    """
    These images are confusing because they are very similar to others digits. Model makes wrong classification
    to the similar ones. For example, ID 3253 digit is nine but the little circle in "9" is relatively small,
    which makes it has a similar shape as "7" so my model classified it as "7" instead of "9". Another example
    ID 195 digit, which is "7" but has a weird horizontal line across the middle of digit. That line confused 
    my model to classify it as "2".
    """
    y_pred_prob = np.array([softmax(x) for x in (Xhat(X_val) @ beta2)])
    confusedImageIds = np.array([index for index, x in enumerate(y_pred == y_val) if x == 0])
    imageCounts = 8
    mostConfusedIds = confusedImageIds[
        np.argpartition(np.array([np.max(x) for x in y_pred_prob[confusedImageIds]]), -imageCounts)[-imageCounts:]]

    # plot eight images
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 20))
    for i, ax in zip(mostConfusedIds, axs.flatten()):
        # plot the image as a grayscale array
        ax.imshow(X_val[i].reshape(28, 28), cmap='gray')
        # label the image with its ID
        ax.text(0, -3, f"ID: {i} " + "pred: " + str(y_pred[i]) + " conf: " + f"{np.max(y_pred_prob[i]) * 100:.4f}%",
                fontsize=15, color='b')
    plt.show()


if __name__ == '__main__':
    breast_cancer_classification()
    digit_classification()
