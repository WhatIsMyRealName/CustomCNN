from layer import Layer
import numpy as np

'''
class FlattenLayer(Layer):
    """
    FlattenLayer is a layer that flattens the input data into a 1-dimensional array.

    This layer is typically used in Convolutional Neural Networks (CNNs) 
    to transform the multi-dimensional output of convolutional or pooling layers 
    into a flat vector, which can then be passed to fully connected layers.

    Parameters
    ----------
    Layer : Layer
        The base class for all layers, providing a structure for 
        forward and backward propagation methods.
    
    Attributes
    ----------
    output_shape : tuple
        Desired shape of the output after flattening. Default is (1, -1), 
        meaning the output will have one row and an automatically 
        determined number of columns.
    output : np.ndarray
        The output of the layer, initialized as a zero array of shape output_shape.

    Methods
    -------
    forward_propagation(input_data)
        Flattens the input data and reshapes it to the specified output_shape.
        
    backward_propagation(output_error, learning_rate)
        Reshapes the output error back to the shape of the input data 
        for backpropagation.
    """

    def __init__(self, output_shape=(1, -1)):
        """
        Initializes the FlattenLayer with the specified output shape.

        Parameters
        ----------
        output_shape : tuple, optional
            Desired shape of the output after flattening, by default (1, -1).
        """
        self.output_shape = output_shape  # Définit la forme de sortie souhaitée
        try:
            self.output = np.zeros(output_shape)
        except ValueError:
            self.output = np.array([])

    def forward_propagation(self, input_data):
        """
        Flattens the input data into a one-dimensional array and reshapes it.

        Parameters
        ----------
        input_data : np.ndarray
            The input data to be flattened.

        Returns
        -------
        np.ndarray
            The flattened and reshaped output data.

        Raises
        ------
        ValueError
            If the specified output_shape is incompatible with the number of elements 
            in the input data after flattening.
        """
        self.input_shape = input_data.shape  # Enregistre la forme d'entrée pour le backprop
        flattened_output = input_data.flatten()
        
        # Tente de reformer l'output en output_shape, lève une erreur si incompatible
        try:
            return flattened_output.reshape(self.output_shape)
        except ValueError:
            raise ValueError("La forme de sortie spécifiée n'est pas compatible avec le nombre d'éléments.")

    def backward_propagation(self, output_error, learning_rate):
        """
        Reshapes the output error back to the original input shape for backpropagation.

        Parameters
        ----------
        output_error : np.ndarray
            The error signal coming from the next layer, which needs to be reshaped.
        learning_rate : float
            The learning rate is not used in this layer since it does not have learnable parameters.

        Returns
        -------
        np.ndarray
            The reshaped error to match the input shape for backpropagation.
        """
        # Reforme l'erreur à la forme d'origine pour le backprop
        return output_error.reshape(self.input_shape)

    def __getstate__(self):
        # Copiez l'état de l'objet
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Restaurez l'état de l'objet
        self.__dict__.update(state)
'''
class FlattenLayer(Layer):
    def __init__(self, output_shape=(1, -1)):
        """
        Initialize the FlattenLayer with the specified output shape.
        
        Parameters
        ----------
        output_shape : tuple, optional
            Desired shape of the output after flattening, by default (1, -1).
        """
        self.output_shape = output_shape
        self.output = np.zeros(output_shape) if output_shape != (1, -1) else None

    def forward_propagation(self, input_data):
        # Enregistre la forme d'entrée pour backward propagation
        self.input_shape = input_data.shape  
        
        # Ajuste la sortie en fonction des dimensions d'entrée
        if input_data.ndim == 3:
            # Cas sans batch
            self.output = input_data.flatten()
        elif input_data.ndim == 4:
            # Cas avec batch
            batch_size = input_data.shape[0]
            self.output = input_data.reshape(batch_size, -1)
        else:
            raise ValueError("Unexpected input dimensions in FlattenLayer.")
            
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # Remodèle l'erreur de sortie pour correspondre à la forme d'entrée d'origine
        return output_error.reshape(self.input_shape)

