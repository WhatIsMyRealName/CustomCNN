import numpy as np
from layer import Layer

# Dropout Layer
class Dropout(Layer):
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward_propagation(self, input_data):
        if self.dropout_rate > 0:
            # Générer le masque pour le dropout
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=input_data.shape)
            return input_data * self.mask / (1 - self.dropout_rate)
        return input_data  # Pas de dropout si le taux est 0

    def backward_propagation(self, output_error, learning_rate):
        if self.mask is not None:
            return output_error * self.mask
        return output_error  # Pas de masque si pas de dropout


# Batch Normalization Layer
class BatchNormalization(Layer):
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        self.input = None
        self.mean = None
        self.variance = None

    def forward_propagation(self, input_data):
        if len(input_data.shape) == 2:  # Si on est dans un mode batch
            self.input = input_data
            self.mean = np.mean(input_data, axis=0)
            self.variance = np.var(input_data, axis=0)
            self.normalized_input = (input_data - self.mean) / np.sqrt(self.variance + self.epsilon)

            if self.gamma is None or self.beta is None:
                self.gamma = np.ones_like(self.mean)
                self.beta = np.zeros_like(self.mean)

            self.output = self.gamma * self.normalized_input + self.beta
            return self.output
        else:  # Si on est en mode non-batch
            return input_data  # Pas de normalisation

    def backward_propagation(self, output_error, learning_rate):
        if self.input is not None:
            input_error = (1. / self.input.shape[0]) * self.gamma / np.sqrt(self.variance + self.epsilon) * \
                          (self.input.shape[0] * output_error - np.sum(output_error, axis=0) -
                           self.normalized_input * np.sum(output_error * self.normalized_input, axis=0))

            # Mise à jour de gamma et beta
            self.gamma -= learning_rate * np.sum(output_error * self.normalized_input, axis=0)
            self.beta -= learning_rate * np.sum(output_error, axis=0)
            return input_error
        return output_error  # Pas d'erreur si pas de normalisation


# Softmax Activation Layer
class Softmax(Layer):
    def forward_propagation(self, input_data):
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = self.output * (output_error - np.sum(output_error * self.output, axis=1, keepdims=True))
        return input_error

# Embedding Layer (for integer input indices)
class Embedding(Layer):
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embeddings = np.random.randn(vocab_size, embed_dim)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.embeddings[input_data]
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        for i, index in enumerate(self.input):
            self.embeddings[index] -= learning_rate * output_error[i]
        return None  # No input error needed for embedding layer

# Recurrent Layers - Simplified LSTM
class LSTM(Layer):
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Weights for input, forget, output and candidate gates
        self.W = np.random.randn(4, input_dim + hidden_dim, hidden_dim)
        self.b = np.random.randn(4, hidden_dim)
        
    def forward_propagation(self, input_data):
        self.h, self.c = np.zeros(self.hidden_dim), np.zeros(self.hidden_dim)
        self.outputs = []
        for t in range(input_data.shape[1]):  # Sequence length
            x_t = input_data[:, t, :]
            combined = np.concatenate((x_t, self.h), axis=1)
            i_t = self._sigmoid(np.dot(combined, self.W[0]) + self.b[0])  # Input gate
            f_t = self._sigmoid(np.dot(combined, self.W[1]) + self.b[1])  # Forget gate
            o_t = self._sigmoid(np.dot(combined, self.W[2]) + self.b[2])  # Output gate
            g_t = np.tanh(np.dot(combined, self.W[3]) + self.b[3])        # Candidate
            self.c = f_t * self.c + i_t * g_t
            self.h = o_t * np.tanh(self.c)
            self.outputs.append(self.h)
        return np.stack(self.outputs, axis=1)

    def backward_propagation(self, output_error, learning_rate):
        # LSTM backward pass can be quite complex due to time dependencies
        # We'll leave this unimplemented here for simplicity
        pass
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))