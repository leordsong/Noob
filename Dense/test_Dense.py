import unittest

import numpy as np

from Dense import Dense
from losses import MSE
from activations import sigmoid, d_sigmoid


class DenseTests(unittest.TestCase):

    def test_feedforward(self):
        input_shape = 2
        output_shape = 3
        dense = Dense(0.5, input_shape, output_shape)

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
        dense = Dense(0.5, input_shape, output_shape)
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

    def _run_model(self, raw_input, model):
        outputs = [raw_input]
        for layer in model:
            y_pred = layer.feedforward(outputs[-1])
            outputs.append(y_pred)
        return outputs

    def test_two_layers(self):
        model = [
            Dense(0.5, 3, 2),
            Dense(0.5, 2, 1, activation=[sigmoid, d_sigmoid])
        ]
        loss = MSE()
        output_shape = (4, 1)
        y_true = np.full(output_shape, 0.5)
        raw_input = np.random.uniform(-1, 1, (4, 3))

        outputs = self._run_model(raw_input, model)
        self.assertEqual(outputs[-1].shape, output_shape)
        loss1 = loss.feedforward(y_true, outputs[-1])

        diff = loss.backprob(y_true, outputs[-1])
        for i in range(len(model) - 1, -1, -1):
            layer = model[i]
            diff = layer.backprop(outputs[i], diff)
        outputs = self._run_model(raw_input, model)
        loss2 = loss.feedforward(y_true, outputs[-1])
        
        self.assertLess(loss2, loss1)

