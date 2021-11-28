import numpy as np

class Embedding:
  """An embedding net that maps a value to a multi-dimensional space (a vector)
  """

  def __init__(self, vocab_size = 1000, embedding_dim = 128, alpha=1):
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.lr = alpha
    self.weights = np.random.uniform(-1, 1, (self.vocab_size, self.embedding_dim))

  def feedforward(self, inputs):
    assert inputs is not None
    assert len(inputs.shape) == 1
    assert np.amax(inputs) < self.vocab_size
    assert np.amin(inputs) >= 0
    return np.take(self.weights, inputs, axis=0)

  def feedback(self, inputs, expected_outputs):
    outputs = np.take(self.weights, inputs, axis=0)
    outputs += self.lr * (expected_outputs - outputs)
    inputs = np.expand_dims(inputs, axis=1)
    np.put_along_axis(self.weights, inputs, outputs, axis=0)