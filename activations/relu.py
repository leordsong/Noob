import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Calculate the relu output.
    Return the linear output if greater zero.
    Otherwise, return zero.

    Args:
        x (np.ndarray): the input array

    Returns:
        np.ndarray: the output array
    """
    
    return (x > 0) * x

def d_relu(x: np.ndarray, _: np.ndarray) -> np.ndarray:
    return (x > 0) * 1