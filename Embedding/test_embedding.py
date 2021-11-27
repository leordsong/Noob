import unittest

import numpy as np

from .Embedding import Embedding


class EmbeddingTests(unittest.TestCase):

    vocab = 5
    dim = 10

    def test_feedforward(self):
        emebdding = Embedding(self.vocab, self.dim)
        inputs = np.ones((3,), dtype=np.int64)
        outputs = emebdding.feedforward(inputs)
        assert outputs.shape == (3, 10)

    def test_feedback(self):
        emebdding = Embedding(self.vocab, self.dim)
        inputs = np.array([0, 1], dtype=np.int64)
        outputs = emebdding.feedforward(inputs)
        assert outputs.shape == (2, 10)

        expected_outputs = np.zeros((2, 10))
        emebdding.feedback(inputs, expected_outputs)

        outputs = emebdding.feedforward(inputs)
        assert np.all(outputs == expected_outputs)

    def test_feedback2(self):
        emebdding = Embedding(self.vocab, self.dim, alpha=0.5)
        inputs = np.zeros((2,), dtype=np.int64)
        outputs = emebdding.feedforward(inputs)
        assert outputs.shape == (2, 10)

        expected_outputs = np.zeros((2, 10))
        emebdding.feedback(inputs, expected_outputs)
        new_outputs = emebdding.feedforward(inputs)
        assert np.all(outputs / 2 == new_outputs)


if __name__ == '__main__':
    unittest.main()