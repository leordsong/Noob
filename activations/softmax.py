import numpy as np


def exp(x):
    shiftx = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(shiftx)

def d_exp(_, y):
    return np.log(y)

def unstablesoftmax(x):
    exp = np.exp(x)
    exp_sum = np.sum(exp, axis=-1, keepdims=True)
    return exp / exp_sum

def softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shiftx)
    exp_sum = np.sum(exp, axis=-1, keepdims=True)
    return exp / exp_sum

def d_softmax(_, y):
    if len(y.shape) == 1:
        return _d_softmax(y)
    else:

        values = np.zeros(y.shape)
        for i in range(y.shape[0]):
            row = y[i]
            values.append(_d_softmax(row))
        return np.array(values)


def _d_softmax(y):
    y_matrix = np.outer(y, y)
    return np.diag(y) - y_matrix
