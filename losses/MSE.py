import numpy as np


class MSE:

    def feedforward(self, y_true, y_pred):
        diff = y_true - y_pred
        square_diff = diff * diff
        mse = np.mean(square_diff)
        return mse

    def backprob(self, y_true, y_pred):
        # remove constant scalar
        # return 2 * (y_true - y_pred) * (-1)
        return y_pred - y_true