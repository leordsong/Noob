import numpy as np

from activations import linear, d_linear


class CNN:

    def __init__(self, filters, alpha=0.5, kernel_shape=(3, 3), padding=0, stride=1, activation=(linear, d_linear)):
        self.filters = filters
        self.kernel_shape = kernel_shape
        self.padding = padding
        self.stride = stride
        self.channel = 3
        self._activation = activation[0]
        self._d_activation = activation[1]
        self.lr = alpha

        self.kernels = np.random.randn(self.filters, self.kernel_shape[0], self.kernel_shape[1])

    def iterate_regions(self, x, new_width, new_height):
        batch, _, _, _ = x.shape
        for b in range(batch):
            for i in range(0, new_width):
                for j in range(0, new_height):
                    wi = i * self.stride
                    hj = j * self.stride
                    to_wi = wi + self.kernel_shape[0]
                    to_hj = hj + self.kernel_shape[1]
                    local = x[b, wi : to_wi, hj : to_hj, :]
                    yield b, i, j, (wi, to_wi, hj, to_hj), local

    def _feedforward_local(self, local):
        result = []
        for i in range(self.filters):
            kernel = self.kernels[i]
            channel_result = 0
            for j in range(self.channel):
                channel = local[:, :, j]
                value = np.multiply(channel, kernel)
                value = np.sum(value)
                channel_result += value
            result.append(channel_result)
        return np.array(result)

    def feedforward(self, x):
        assert len(x.shape) == 4, f'Shape must with shape [batch, width, height, channel]. Got {x.shape} instead'
        batch, width, height, channels = x.shape
        assert channels == self.channel, 'Only support rbg channels'
        p = self.padding
        assert (width + 2 * p - self.kernel_shape[0]) % self.stride == 0, \
            f'Width {width} not compatiable with kernel size {self.kernel_shape[0]} stride {self.stride} pad {p}'
        assert (height + 2 * p - self.kernel_shape[1]) % self.stride == 0, \
            f'Height {height} not compatiable with kernel size {self.kernel_shape[1]} stride {self.stride} pad {p}'

        new_width = int((width + 2 * p - self.kernel_shape[0]) / self.stride) + 1
        new_height = int((height + 2 * p - self.kernel_shape[1]) / self.stride) + 1
        x = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant', constant_values=((0, 0), (0, 0),(0, 0), (0, 0)))

        output = np.zeros((batch, new_width, new_height, self.filters))
        for b, i, j, _, local in self.iterate_regions(x, new_width, new_height):
            local_output = self._feedforward_local(local)
            output[b, i, j] = local_output 
        return self._activation(output)

    
    def backprop(self, x, y, delta):
        df = np.zeros(self.kernels.shape)
        new_output = np.zeros(x.shape)
        output = np.zeros((x.shape[0], y.shape[1], y.shape[2], self.filters))
        
        for b, i, j, _, local in self.iterate_regions(x, y.shape[1], y.shape[2]):
            local_output = self._feedforward_local(local)
            output[b, i, j] = local_output 
        d = delta * self._d_activation(output, y)

        for b, i, j, loc, local in self.iterate_regions(x, y.shape[1], y.shape[2]):
            for f in range(self.filters):
                df[f] += d[b, i, j, f] * np.sum(local, axis=-1)
                value = d[b, i, j, f] * self.kernels[f]
                value = np.tile(np.expand_dims(value, -1), (1, 1, 3)) / 3
                new_output[b, loc[0]:loc[1], loc[2]:loc[3]] += value

        #update filters
        self.kernels -= self.lr * df / x.shape[0]

        return new_output
