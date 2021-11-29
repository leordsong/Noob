import unittest

import numpy as np

from .CNN import CNN
from losses import MSE
from activations import relu, d_relu, sigmoid, d_sigmoid, softmax, d_softmax


class CNNTests(unittest.TestCase):

    def test_padding(self):
        w, h = 4, 4
        pad = 1
        channel = 3
        batch = 5

        inputs = np.ones((w, h))
        inputs = np.pad(inputs, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
        self.assertEqual(inputs.shape, (w + 2 * pad, h + 2 * pad))
        
        inputs = np.ones((w, h, channel))
        inputs = np.pad(inputs, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=((0, 0),(0, 0), (0, 0)))
        self.assertEqual(inputs.shape, (w + 2 * pad, h + 2 * pad, channel))

        for i in range(w):
            for j in range(h):
                if i == 0 or i == (w + 1) or j == 0 or j == (h + 1):
                    self.assertTrue(np.all(inputs[i][j] == np.zeros((channel,))))
                else:
                    self.assertTrue(np.all(inputs[i][j] == np.ones((channel,))))

        inputs = np.ones((batch, w, h, channel))
        inputs = np.pad(inputs, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=((0, 0), (0, 0),(0, 0), (0, 0)))
        self.assertEqual(inputs.shape, (batch, w + 2 * pad, h + 2 * pad, channel))
        for b in range(batch):
            input_row = inputs[b]
            for i in range(w):
                for j in range(h):
                    if i == 0 or i == (w + 1) or j == 0 or j == (h + 1):
                        self.assertTrue(np.all(input_row[i][j] == np.zeros((channel,))))
                    else:
                        self.assertTrue(np.all(input_row[i][j] == np.ones((channel,))))


    def test_kernels(self):
        w, h = 4, 4
        channel = 3
        batch = 5
        filters = 6
        out_shape = (batch, 2, 2, filters)
        mse = MSE()

        inputs = np.ones((batch, w, h, channel))

        cnn = CNN(filters, activation=(relu, d_relu))
        outputs = cnn.feedforward(inputs)
        self.assertEqual(outputs.shape, out_shape)
        y_true = np.zeros(out_shape)
        loss1 = mse.feedforward(y_true, outputs)
        delta = mse.backprob(y_true, outputs)
        cnn.backprop(inputs, outputs, delta)
        outputs = cnn.feedforward(inputs)
        loss2 = mse.feedforward(y_true, outputs)
        self.assertLess(loss2, loss1)

        cnn = CNN(filters, activation=(sigmoid, d_sigmoid))
        outputs = cnn.feedforward(inputs)
        self.assertEqual(outputs.shape, out_shape)
        y_true = np.zeros(out_shape)
        loss1 = mse.feedforward(y_true, outputs)
        delta = mse.backprob(y_true, outputs)
        cnn.backprop(inputs, outputs, delta)
        outputs = cnn.feedforward(inputs)
        loss2 = mse.feedforward(y_true, outputs)
        self.assertLess(loss2, loss1)

        cnn = CNN(filters, activation=(softmax, d_softmax))
        outputs = cnn.feedforward(inputs)
        self.assertEqual(outputs.shape, out_shape)
        y_true = np.zeros(out_shape)
        loss1 = mse.feedforward(y_true, outputs)
        delta = mse.backprob(y_true, outputs)
        cnn.backprop(inputs, outputs, delta)
        outputs = cnn.feedforward(inputs)
        loss2 = mse.feedforward(y_true, outputs)
        self.assertLess(loss2, loss1)


