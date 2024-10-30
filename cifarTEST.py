from network import Network
from fclayer import FCLayer
from convlayer import Conv2D
from poolinglayer import MaxPooling
from flattenlayer import FlattenLayer
from activationlayer import ActivationLayer
from otherlayers import Dropout, BatchNormalization
from activations import relu, relu_prime, tanh, tanh_prime, softmax, softmax_prime
from lossfunction import mse, mse_prime, cross_entropy_loss, cross_entropy_loss_prime

from scipy.ndimage import zoom
from keras.datasets import cifar100
from keras.utils import to_categorical 
import numpy as np
from random import randrange

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

net = Network()
net.add(Conv2D(input_shape=(3, 32, 32), num_filters=32, filter_size=3, stride=1, padding=1))  # Augmenter le nombre de filtres
net.add(BatchNormalization())  # Normalisation après convolution
net.add(ActivationLayer(relu, relu_prime))
net.add(MaxPooling(pool_size=2, stride=2))
net.add(Conv2D(input_shape=(32, 16, 16), num_filters=64, filter_size=3, stride=1, padding=1))  # Augmenter le nombre de filtres
net.add(BatchNormalization())  # Normalisation après convolution
net.add(ActivationLayer(relu, relu_prime))
net.add(MaxPooling(pool_size=2, stride=2))
net.add(Conv2D(input_shape=(64, 8, 8), num_filters=128, filter_size=3, stride=1, padding=1))  # Augmenter le nombre de filtres
net.add(BatchNormalization())  # Normalisation après convolution
net.add(ActivationLayer(relu, relu_prime))
net.add(MaxPooling(pool_size=2, stride=2))
net.add(FlattenLayer())  # Transforme les données en vecteur
net.add(FCLayer(2048, 256))  # Réduire le nombre de neurones
net.add(BatchNormalization())  # Normalisation après convolution
net.add(ActivationLayer(relu, relu_prime))
net.add(Dropout(dropout_rate=0.5))  # Ajout d'un Dropout
net.add(FCLayer(256, 100))  # Couche de sortie pour 100 classes
net.add(ActivationLayer(softmax, softmax_prime))
net.use(mse, mse_prime)

from random import randrange
t = randrange(0, 5000)
# Redimensionner les images
#x_train = resize_images(x_train[t:t+1000])
#x_test = resize_images(x_test[t:t+1000])

# Display the first 4 images
import matplotlib.pyplot as plt

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(x_train[i])
    plt.axis('off')

plt.show()

x_train = x_train.transpose(0, 3, 1, 2)
x_test = x_test.transpose(0, 3, 1, 2)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train[t:t+500], num_classes=100)
y_test = to_categorical(y_test[t:t+500], num_classes=100)

print("starting")

import time
timing = time.time()
net.fit2(x_train[t:t+500], y_train, epochs=10, learning_rate=0.01, batch_size=16)
# net.fit(x_train[t:t+500], y_train, epochs=5, learning_rate=0.01)
print("model trained in ", time.time() - timing, "seconds")

# Test sur 3 exemples
out = net.predict(x_test[0:3])
print("\nPrédictions :")
print(out)

array = []
for prediction in out:
    # Vérifie si la prédiction est un array 1D (cas probable) et applique np.argmax pour obtenir l'indice de la valeur maximale
    if isinstance(prediction, np.ndarray) and prediction.ndim == 1:
        normalized_pred = np.argmax(prediction)  # Utilise np.argmax pour obtenir l'indice de la valeur maximale
        array.append(normalized_pred)
    else:
        print("Format inattendu pour la prédiction:", prediction)
print("Valeurs normalisées :", array)
print("\nValeurs réelles :")
print([np.argmax(y) for y in y_test[0:3]])

# Affichez un résumé du modèle
net.summary()
# Prédire sur une entrée et obtenir le temps d'inférence
timing = time.time()
output = net.predict(x_test[10:15])[0]
inference_time = time.time() - timing
print(f"Output: {output}, Inference Time: {inference_time:.6f} seconds")
print("solution : ", y_test[10:15])

net.save("monPremierModele.pkl")