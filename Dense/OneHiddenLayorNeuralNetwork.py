import numpy as np

class OneHiddenLayerNerualNetwork:
  """
  A class that represents one hidden layer neural network
  """

  def __init__(self, alpha, input, hidden):
    """
    Parameters
    ----------
    alpha : int
      The learning rate
    input : int
      The number of input units
    hidden : int
      The number of hidden units
    """
    self._input = input
    self._hidden = hidden
    
    self._weight_one = np.random.rand(self._input, self._hidden)
    self._bias_one   = np.random.rand(1, self._hidden)
    self._weight_two = np.random.rand(self._hidden, 1)
    self._bias_two   = np.random.rand()
    self._rate = alpha

  def predict(self, input):
    """Predict the expected value using this nerual network

    (numpy.ndarray) -> numpy.ndarray
    """

    hidden = self._feedforward(input , self._weight_one, self._bias_one)
    output = self._feedforward(hidden, self._weight_two, self._bias_two)
    return output

  def _feedforward(self, input, weight, bias):
    value = np.dot(input, weight)
    value = np.add(value, bias)
    return self._sigmoid(value)

  def _sigmoid(x):
    """Calculate the output of a sigmoid function

    (number) -> number
    """

    return 1 / (1 + np.exp(-x))
  
  def train(self, input, expected):
    """Train the nerual network using backpropagation

    (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """

    # feedforward
    hidden = self._feedforward(input , self._weight_one, self._bias_one)
    output = self._feedforward(hidden, self._weight_two, self._bias_two)

    # backward propagation
    # derivative of bias 2
    db_two = 2 * (output - expected) * output * (1 - output)
    # derivative of weight 2
    dw_two = np.dot(hidden.T, db_two)
    # derivative of bias 1
    db_one = np.dot(2 * (output - expected) * output * (1 - output), 
                    self._weight_two.T) * hidden * (1 - hidden)
    # derivative of weight 1
    dw_one = np.dot(input.T, db_one)

    self._weight_one -= self._rate * dw_one
    self._bias_one   -= self._rate * np.sum(db_one, axis=0, keepdims=True)
    self._weight_two -= self._rate * dw_two
    self._bias_two   -= self._rate * np.sum(db_two, axis=0, keepdims=True)
