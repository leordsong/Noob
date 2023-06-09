import numpy as np

class MaxPooling2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.mask = None

    def forward(self, x):
        # batch, channel, height, width
        n, c, h, w = x.shape
        h_out = (h - self.pool_size) // self.stride + 1
        w_out = (w - self.pool_size) // self.stride + 1

        out = np.zeros((n, c, h_out, w_out))
        self.mask = np.zeros_like(x)

        for i in range(h_out):
            for j in range(w_out):
                # construct the pools
                x_masked = x[
                    :, :,
                    i * self.stride : i * self.stride + self.pool_size,
                    j * self.stride : j * self.stride + self.pool_size
                ]
                out[:, :, i, j] = np.amax(x_masked, axis=(2, 3)) # max value
                # get the max value out of each pool
                mask = (x_masked == np.amax(x_masked, axis=(2, 3), keepdims=True))
                self.mask[
                    :, :,
                    i * self.stride : i * self.stride + self.pool_size,
                    j * self.stride : j * self.stride + self.pool_size
                ] += mask

        return out

    def backward(self, dout):
        dx = np.zeros_like(self.mask)
        dx[self.mask > 0] = dout[self.mask > 0]
        return dx


class MaxPooling1D:

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.mask = None

    def forward(self, x):
        # batch, channel, width
        n, c, w = x.shape
        w_out = (w - self.pool_size) // self.stride + 1

        out = np.zeros((n, c, w_out))
        self.mask = np.zeros_like(x)

        for i in range(w_out):
            x_masked = x[:, :, i * self.stride:i * self.stride + self.pool_size]
            out[:, :, i] = np.amax(x_masked, axis=(2))
            mask = (x_masked == np.amax(x_masked, axis=(2), keepdims=True))
            self.mask[:, :, i * self.stride:i * self.stride + self.pool_size] += mask

        return out

    def backward(self, dout):
        dx = np.zeros_like(self.mask)
        dx[self.mask > 0] = dout[self.mask > 0]
        return dx