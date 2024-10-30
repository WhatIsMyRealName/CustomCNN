import pickle
import sys

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                sys.stdout.write(f"\rforward propagation... {i}/{samples}")
                sys.stdout.flush()
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                sys.stdout.write(f"\rforward propagation... {i}/{samples}   ")
                sys.stdout.flush()
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            sys.stdout.write('\r')
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

    def fit2(self, x_train, y_train, epochs, learning_rate, batch_size=32):
        import numpy as np
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(0, samples, batch_size):
                # Sélection du mini-lot
                x_batch = x_train[j:j+batch_size]
                y_batch = y_train[j:j+batch_size]

                sys.stdout.write(f"\rforward propagation... {j/batch_size}/{int(samples/batch_size) +1}   ")
                sys.stdout.flush()

                # Propagation avant
                output = x_batch
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # Calcul de la perte pour le mini-lot
                batch_loss = np.mean([self.loss(y_batch[k], output[k]) for k in range(len(y_batch))])
                err += batch_loss

                # Rétropropagation pour le mini-lot
                sys.stdout.write(f"\rbackward propagation... {j/batch_size}/{int(samples/batch_size) +1}")
                sys.stdout.flush()
                batch_error = np.array([self.loss_prime(y_batch[k], output[k]) for k in range(len(y_batch))])
                for layer in reversed(self.layers):
                    batch_error = layer.backward_propagation(batch_error, learning_rate)

            # Calcul de la perte moyenne pour cette époque
            err /= (samples // batch_size)
            sys.stdout.write('\r')
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

    def count_parameters(self):
        total_params = 0
        for layer in self.layers:
            # Pour les couches convolutives
            if hasattr(layer, 'filters'):
                # Calcul des paramètres : (taille du filtre * canaux d'entrée + 1 pour le biais) * nombre de filtres
                filter_size = layer.filter_size ** 2
                in_channels = layer.input_shape[0]  # Nombre de canaux d'entrée (sans tenir compte du batch)
                total_params += (filter_size * in_channels + 1) * layer.num_filters
            # Pour les couches entièrement connectées
            elif hasattr(layer, 'weights'):
                total_params += layer.weights.size
            # Ajout des biais si présents
            if hasattr(layer, 'bias'):
                total_params += layer.bias.size
        return total_params

    def summary(self):
        print("Model Summary:")
        for i, layer in enumerate(self.layers):
            layer_type = type(layer).__name__

            # Récupération de la forme de sortie si disponible
            output_shape = getattr(layer, 'output', None)
            output_shape = output_shape.shape if output_shape is not None else "N/A"

            # Compte des paramètres pour la couche actuelle
            params_count = 0
            if hasattr(layer, 'filters'):
                # Calcul des paramètres pour une couche conv
                filter_size = layer.filter_size ** 2
                in_channels = layer.input_shape[0]
                params_count += (filter_size * in_channels + 1) * layer.num_filters
            elif hasattr(layer, 'weights'):
                params_count += layer.weights.size
            if hasattr(layer, 'bias'):
                params_count += layer.bias.size

            print(f"Layer {i + 1}: {layer_type} | Output Shape: {output_shape} | Parameters: {params_count}")
        print(f"Total Parameters: {self.count_parameters()}")
    
    # Save the model to a file
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {filename}")

    # Load the model from a file
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {filename}")
        return model