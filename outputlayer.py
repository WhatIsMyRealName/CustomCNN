import numpy as np
from layer import Layer

class OutputLayer(Layer):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.weights = np.random.rand(input_size, num_classes) - 0.5  # Initialisation des poids
        self.bias = np.random.rand(1, num_classes) - 0.5  # Initialisation des biais

    # Forward propagation
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias  # Calcul des logits
        return self.softmax(self.output)  # Applique softmax pour obtenir les probabilités

    # Fonction softmax
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))  # Pour stabilité numérique
        return e_x / e_x.sum(axis=1, keepdims=True)  # Normalisation pour obtenir des probabilités

    # Backward propagation
    def backward_propagation(self, output_error, learning_rate):
        # On n'a pas besoin de mettre à jour les poids ici, car on utilise la sortie softmax
        input_error = np.dot(output_error, self.weights.T)  # Calcul de l'erreur d'entrée
        weights_error = np.dot(self.input.T, output_error)  # Calcul de l'erreur de poids
        
        # Mise à jour des poids et biais
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
