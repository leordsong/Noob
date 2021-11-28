import numpy as np

from activations import sigmoid, d_sigmoid


class Dense:
  """
  A class that represents one fully connected neural network
  """

  def __init__(self, alpha: float, inputs: int, units: int,
               activation=(sigmoid, d_sigmoid)):
    """Initialize a new instance of this class

    Args:
        alpha (float): The learning rate
        inputs (int): The number of input units
        units (int): The number of hidden units
        activation (Tuple(any, any)): The activation function
    """
    self._lr = alpha
    self._inputs = inputs
    self._units = units
    
    self._weights = np.random.rand(self._inputs, self._units)
    self._bias = np.random.rand(1, self._units)
    self._activation = activation[0]
    self._d_activation = activation[1]

  def feedforward(self, x):
    values = np.dot(x, self._weights)
    values = np.add(values, self._bias)
    return self._activation(values)

  def backprop(self, x, delta):
    batch = x.shape[0]

    y = self.feedforward(x)
    d = delta * self._d_activation(None, y)
    x_delta = np.dot(d, self._weights.T)

    self._weights -= self._lr * np.dot(x.T, d) / batch
    self._bias    -= self._lr * np.mean(d, axis=0, keepdims=True) / batch
    return x_delta