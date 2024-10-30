from layer import Layer
import numpy as np

"""
# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
"""
class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        # Initialisation des poids et biais
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.random.randn(1, output_size) * 0.1

    def forward_propagation(self, input_data):
        self.input = input_data
        
        # Vérifie si l'entrée contient un batch (2D) ou non (1D)
        if input_data.ndim == 1:
            # Cas sans batch
            self.output = np.dot(input_data, self.weights) + self.biases.flatten()
        else:
            # Cas avec batch
            self.output = np.dot(input_data, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # Calcul des gradients
        if self.input.ndim == 1:
            # Cas sans batch
            d_weights = np.outer(self.input, output_error)
            d_biases = output_error
            input_error = np.dot(output_error, self.weights.T)
        else:
            # Cas avec batch
            d_weights = np.dot(self.input.T, output_error) / self.input.shape[0]
            d_biases = np.mean(output_error, axis=0)
            input_error = np.dot(output_error, self.weights.T)

        # Mise à jour des poids et biais
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases.reshape(self.biases.shape)
        
        return input_error
