from network import Network
from fclayer import FCLayer
from convlayer import Conv2D
from poolinglayer import MaxPooling
from flattenlayer import FlattenLayer
from activationlayer import ActivationLayer
from activations import relu, relu_prime, tanh, tanh_prime
from lossfunction import mse, mse_prime

from scipy.ndimage import zoom
from keras.datasets import mnist
from keras.utils import to_categorical 
import numpy as np
from random import randrange

# Charger MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Créer le réseau avec des couches Conv2D, MaxPooling, et Fully Connected
net = Network()
net.add(Conv2D(input_shape=(1, 56, 56), num_filters=8, filter_size=3, stride=1, padding=1))  # Sortie : (8, 28, 28)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(MaxPooling(pool_size=2, stride=2))  # Sortie : (8, 14, 14)
net.add(FlattenLayer())  # Transforme (8, 14, 14) en (1, 1568) si batch_size=1
net.add(FCLayer(8 * 28 * 28, 100))  # Sortie : (1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))           # Sortie : (1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))            # Sortie : (1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

# Fonction de redimensionnement avec interpolation
def resize_images(images, target_size=(56, 56)):
    resized_images = []
    for img in images:
        # Zoomer pour passer de 28x28 à 56x56
        zoom_factor = (target_size[0] / img.shape[0], target_size[1] / img.shape[1])
        resized_img = zoom(img, zoom=zoom_factor, order=1)  # order=1 pour interpolation bilinéaire
        resized_images.append(resized_img)
    return np.array(resized_images)

# Entraînement sur un sous-échantillon des données pour accélérer
net.use(mse, mse_prime)

from random import randrange
t = randrange(0, 5000)
# Redimensionner les images
x_train = resize_images(x_train[t:t+1000])
x_test = resize_images(x_test[t:t+1000])

# Mise en forme des données pour le modèle (nombre d'exemples, canaux, hauteur, largeur)
# # Mise en forme des données aplaties pour la couche FCLayer (nombre d'exemples, 3136)
x_train = x_train.reshape(x_train.shape[0], 1, 56, 56)
x_test = x_test.reshape(x_test.shape[0], 1, 56, 56)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train[t:t+1000])
y_test = to_categorical(y_test[t:t+1000])
print("starting")

import time
timing = time.time()
# net.fit2(x_train, y_train, epochs=10, learning_rate=0.1, batch_size=10)
net.fit(x_train, y_train, epochs=10, learning_rate=0.1)
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