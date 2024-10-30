from scipy.signal import convolve2d
from layer import Layer
import numpy as np

"""
Lent car calculs manuels
class Conv2D(Layer):
    def __init__(self, input_shape, num_filters, filter_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape  # (canaux, hauteur, largeur)

        # Initialisation des filtres et des biais
        self.filters = np.random.randn(num_filters, input_shape[0], filter_size, filter_size) * 0.1
        self.biases = np.random.randn(num_filters) * 0.1

    def forward_propagation(self, input_data):
        self.input = np.pad(input_data, [(0, 0), (self.padding, self.padding), (self.padding, self.padding)], mode='constant')
        self.output_shape = ((self.input_shape[1] - self.filter_size + 2 * self.padding) // self.stride + 1,
                             (self.input_shape[2] - self.filter_size + 2 * self.padding) // self.stride + 1)
        self.output = np.zeros((self.num_filters, *self.output_shape))

        for f in range(self.num_filters):
            for i in range(0, self.output_shape[0], self.stride):
                for j in range(0, self.output_shape[1], self.stride):
                    region = self.input[:, i:i+self.filter_size, j:j+self.filter_size]
                    self.output[f, i, j] = np.sum(region * self.filters[f]) + self.biases[f]
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        input_error = np.zeros_like(self.input)

        for f in range(self.num_filters):
            for i in range(0, self.output_shape[0], self.stride):
                for j in range(0, self.output_shape[1], self.stride):
                    region = self.input[:, i:i+self.filter_size, j:j+self.filter_size]
                    d_filters[f] += output_error[f, i, j] * region
                    d_biases[f] += output_error[f, i, j]
                    input_error[:, i:i+self.filter_size, j:j+self.filter_size] += output_error[f, i, j] * self.filters[f]

        # Mise à jour des poids
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases
        return input_error[:, self.padding:-self.padding, self.padding:-self.padding]  # Retirer le padding
"""

"""
Plus rapide car utilises numpy et scipy et les calculs optimisés
"""

class Conv2D(Layer):
    def __init__(self, input_shape, num_filters, filter_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape  # (channels, height, width)
        self.filters = np.random.randn(num_filters, input_shape[0], filter_size, filter_size) * 0.1
        self.biases = np.random.randn(num_filters, 1, 1) * 0.1

    def forward_propagation(self, input_data):
        self.input = input_data
        if input_data.ndim == 3:  # Without batch
            padded_input = np.pad(input_data, [(0, 0), (self.padding, self.padding), (self.padding, self.padding)], mode='constant')
            output_height = (padded_input.shape[1] - self.filter_size) // self.stride + 1
            output_width = (padded_input.shape[2] - self.filter_size) // self.stride + 1
            output = np.zeros((self.num_filters, output_height, output_width))

        elif input_data.ndim == 4:  # With batch
            padded_input = np.pad(input_data, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], mode='constant')
            output_height = (padded_input.shape[2] - self.filter_size) // self.stride + 1
            output_width = (padded_input.shape[3] - self.filter_size) // self.stride + 1
            output = np.zeros((input_data.shape[0], self.num_filters, output_height, output_width))

        else:
            raise ValueError(f"Unexpected input dimensions: {input_data.shape}")

        # Perform convolution
        for i in range(self.num_filters):
            current_filter = self.filters[i]
            if input_data.ndim == 3:  # No batch
                conv_sum = np.zeros((output_height, output_width))
                for j in range(self.input_shape[0]):
                    conv_sum += convolve2d(padded_input[j], current_filter[j], mode='valid')[::self.stride, ::self.stride]
                output[i] = conv_sum + self.biases[i]

            elif input_data.ndim == 4:  # With batch
                for b in range(input_data.shape[0]):
                    conv_sum = np.zeros((output_height, output_width))
                    for j in range(self.input_shape[0]):
                        conv_sum += convolve2d(padded_input[b, j], current_filter[j], mode='valid')[::self.stride, ::self.stride]
                    output[b, i] = conv_sum + self.biases[i]

        self.output = output
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # Initialize gradients
        d_filters = np.zeros_like(self.filters)
        d_biases = np.sum(output_error, axis=(0, 2, 3) if output_error.ndim == 4 else (1, 2))

        if output_error.ndim == 3:  # Without batch
            padded_input = np.pad(self.input, [(0, 0), (self.padding, self.padding), (self.padding, self.padding)], mode='constant')
            input_error = np.zeros_like(padded_input)

            for i in range(self.num_filters):
                for j in range(self.input_shape[0]):
                    d_filters[i, j] += convolve2d(padded_input[j], output_error[i], mode='valid')
                    input_error[j] += convolve2d(output_error[i], self.filters[i, j][::-1, ::-1], mode='full')

            input_error = input_error[:, self.padding:-self.padding, self.padding:-self.padding]

        elif output_error.ndim == 4:  # With batch
            padded_input = np.pad(self.input, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], mode='constant')
            input_error = np.zeros_like(padded_input)

            for b in range(output_error.shape[0]):  # For each sample in the batch
                for i in range(self.num_filters):
                    for j in range(self.input_shape[0]):
                        d_filters[i, j] += convolve2d(padded_input[b, j], output_error[b, i], mode='valid')
                        input_error[b, j] += convolve2d(output_error[b, i], self.filters[i, j][::-1, ::-1], mode='full')

            input_error = input_error[:, :, self.padding:-self.padding, self.padding:-self.padding]

        # Update filters and biases
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases.reshape(self.biases.shape)

        return input_error
