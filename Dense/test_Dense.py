import unittest

import numpy as np

from Dense import Dense
from losses import MSE
from activations import linear, d_linear


class MSETests(unittest.TestCase):

    def test_feedforward(self):
        input_shape = 2
        output_shape = 3
        dense = Dense(0.5, input_shape, output_shape, activation=(linear, d_linear))

        self.assertEqual(dense._weights.shape, (input_shape, output_shape))
        self.assertEqual(dense._bias.shape, (1, output_shape))
        dense._weights = np.ones((input_shape, output_shape))
        dense._bias = np.zeros((1, output_shape))

        inputs = np.array([[0., 1.], [1., 0.], [1., 1.], [0., 0.]])
        y_pred = dense.feedforward(inputs)
        y_true = np.array([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.], [0., 0., 0.]])
        self.assertTrue(np.all(y_pred == y_true))

    def test_backprob(self):
        input_shape = 2
        batch = 2
        output_shape = 3
        dense = Dense(0.5, input_shape, output_shape, activation=(linear, d_linear))
        mse = MSE()
        dense._weights = np.ones((input_shape, output_shape))
        dense._bias = np.zeros((1, output_shape))

        self.assertEqual(dense._weights.shape, (input_shape, output_shape))
        self.assertEqual(dense._bias.shape, (1, output_shape))

        inputs = np.array([[0., 1.], [1., 0.]])
        y_pred = dense.feedforward(inputs)
        y_true = np.array([[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]])
        y_diff = mse.backprob(y_true, y_pred)
        self.assertTrue(np.all(np.full((input_shape, output_shape), -0.5) == y_diff))

        x_diff = dense.backprop(inputs, y_diff)
        self.assertTrue(np.all(np.full((batch, input_shape), -1.5) == x_diff))

        y = dense.feedforward(inputs)
        self.assertTrue(np.all(np.full((input_shape, output_shape), 1.25) == y))

