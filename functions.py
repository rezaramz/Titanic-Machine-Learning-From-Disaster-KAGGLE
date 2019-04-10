import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost(theta, X, y):  # x
    m, n = X.shape
    y = np.reshape(y, (m, 1))
    # theta is (n+1, ) array
    theta = np.reshape(theta, (n+1, 1))
    bias = np.ones([m, 1])
    X = np.concatenate((bias, X), axis=1)

    t1 = np.log(sigmoid(np.matmul(X, theta)))
    t1 = np.multiply(y, t1)
    t2 = np.log(1 - sigmoid(np.matmul(X, theta)))
    t2 = np.multiply(1 - y, t2)
    t = (t1 + t2)
    j = (-1 / m) * sum(t)
    return j
