from layer import Layer
import numpy as np

class MaxPooling(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
    """
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output_shape = (input_data.shape[0],
                             input_data.shape[1] // self.pool_size,
                             input_data.shape[2] // self.pool_size)
        self.output = np.zeros(self.output_shape)

        for c in range(self.input.shape[0]):
            for i in range(0, self.output_shape[1], self.stride):
                for j in range(0, self.output_shape[2], self.stride):
                    region = self.input[c, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]
                    self.output[c, i, j] = np.max(region)
        return self.output
        """

    # prend en charge les batch
    def forward_propagation(self, input_data):
        # Vérifie si l'entrée a trois dimensions, auquel cas on ajoute une dimension batch de taille 1
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)  # Ajoute une dimension batch

        batch_size, channels, height, width = input_data.shape
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        output = np.zeros((batch_size, channels, output_height, output_width))

        # Boucle sur chaque canal et chaque échantillon
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size
                        output[b, c, i, j] = np.max(input_data[b, c, start_i:end_i, start_j:end_j])

        # Si l'entrée était sans batch, on retourne une sortie sans batch pour conserver la cohérence
        return output[0] if batch_size == 1 else output
class MaxPooling:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward_propagation(self, input_data):
        if input_data.ndim == 3:  # Sans batch
            input_data = np.expand_dims(input_data, axis=0)  # Ajout d'une dimension batch

        self.input = input_data  # On stocke l'entrée pour l'utiliser dans la rétropropagation
        batch_size, channels, height, width = input_data.shape
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        self.output = np.zeros((batch_size, channels, output_height, output_width))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size
                        self.output[b, c, i, j] = np.max(input_data[b, c, start_i:end_i, start_j:end_j])

        return self.output[0] if batch_size == 1 else self.output

    def backward_propagation(self, output_error, learning_rate):
        if output_error.ndim == 3:  # Sans batch
            output_error = np.expand_dims(output_error, axis=0)  # Ajout d'une dimension batch

        input_error = np.zeros_like(self.input)  # Initialiser l'erreur d'entrée avec la même forme que l'entrée

        batch_size, channels, output_height, output_width = output_error.shape
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size

                        # Identifie l'index du maximum dans la région correspondante
                        region = self.input[b, c, start_i:end_i, start_j:end_j]
                        max_pos = np.unravel_index(np.argmax(region), region.shape)

                        # Propager l'erreur à l'emplacement du maximum
                        input_error[b, c, start_i:end_i, start_j:end_j][max_pos] = output_error[b, c, i, j]

        return input_error[0] if batch_size == 1 else input_error
