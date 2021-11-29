import unittest

import numpy as np

from .softmax import unstablesoftmax, softmax, d_softmax


class SoftmaxTests(unittest.TestCase):

    def test_softmax(self):
        inputs = np.random.randn(3)
        outputs = unstablesoftmax(inputs)
        self.assertAlmostEqual(np.sum(outputs), 1)

        inputs = np.random.randn(3, 4)
        outputs = unstablesoftmax(inputs)
        self.assertEqual(outputs.shape, (3, 4))
        outputs = np.sum(outputs, axis=1)
        self.assertAlmostEqual(outputs[0], 1)
        self.assertAlmostEqual(outputs[1], 1)
        self.assertAlmostEqual(outputs[2], 1)

    def test_softmax_safe(self):
        inputs = np.arange(3)
        outputs = softmax(inputs)
        self.assertAlmostEqual(np.sum(outputs), 1)

        inputs = np.ones((3, 2))
        outputs = softmax(inputs)
        self.assertEqual(outputs.shape, (3, 2))
        outputs = np.sum(outputs, axis=1)
        self.assertAlmostEqual(outputs[0], 1)
        self.assertAlmostEqual(outputs[1], 1)
        self.assertAlmostEqual(outputs[2], 1)
    
    def test_d_softmax(self):
        inputs = np.arange(3)
        outputs = softmax(inputs)
        d_outputs = d_softmax(None, outputs)
        self.assertEqual(d_outputs.shape, (3, 3))

        inputs = np.ones((3, 2))
        outputs = softmax(inputs)
        d_outputs = d_softmax(None, outputs)
        self.assertEqual(d_outputs.shape, (3, 2, 2))
        expected = np.array([[0.25, -0.25], [-0.25, 0.25]])
        expected = np.stack([expected, expected, expected])
        self.assertTrue(np.all(d_outputs == expected))