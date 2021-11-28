import unittest

import numpy as np

from .MSE import MSE


class MSETests(unittest.TestCase):

    def test_output(self):
        mse = MSE()
        y_true = np.array([[0., 1.], [0., 0.]])
        y_pred = np.array([[1., 1.], [1., 0.]])
        loss = mse.feedforward(y_true, y_pred)
        self.assertEqual(loss, 0.5)

    def test_derivative(self):
        mse = MSE()
        y_true = np.array([[0., 1.], [0., 0.]])
        y_pred = np.array([[1., 1.], [1., 0.]])
        div = mse.backprob(y_true, y_pred)
        diff = (y_pred - y_true)
        self.assertTrue(np.all(div == diff))