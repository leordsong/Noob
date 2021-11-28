import numpy as np


def linear(x: np.ndarray) -> np.ndarray:
    """Calculate the linear output

    Args:
        x (np.ndarray): the input array

    Returns:
        np.ndarray: the output array
    """

    return x

def d_linear(_: np.ndarray, __: np.ndarray) -> np.ndarray:
    return 1