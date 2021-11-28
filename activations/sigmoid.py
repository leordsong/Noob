import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Calculate the output of a sigmoid function

    Args:
        x (np.ndarray): the input array

    Returns:
        np.ndarray: the output array
    """

    return 1 / (1 + np.exp(-x))

def d_sigmoid(_: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y * (1 - y)