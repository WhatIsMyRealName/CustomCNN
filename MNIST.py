from network import Network
from fclayer import FCLayer
from activationlayer import ActivationLayer
from activations import tanh, tanh_prime
from lossfunction import mse, mse_prime

from keras.datasets import mnist
from keras.utils import to_categorical 
import numpy as np
from scipy.ndimage import zoom

# Charger MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Fonction de redimensionnement avec interpolation
def resize_images(images, target_size=(56, 56)):
    resized_images = []
    for img in images:
        # Zoomer pour passer de 28x28 à 56x56
        zoom_factor = (target_size[0] / img.shape[0], target_size[1] / img.shape[1])
        resized_img = zoom(img, zoom=zoom_factor, order=1)  # order=1 pour interpolation bilinéaire
        resized_images.append(resized_img)
    return np.array(resized_images)

# Network
net = Network()
net.add(FCLayer(56*56, 100))                # input_shape=(1, 56*56)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.use(mse, mse_prime)
from random import randrange
t = randrange(0, 5000)
# Redimensionner les images
x_train = resize_images(x_train[t:t+1000])
x_test = resize_images(x_test[t:t+1000])

# Mise en forme des données pour le modèle (nombre d'exemples, canaux, hauteur, largeur)
# # Mise en forme des données aplaties pour la couche FCLayer (nombre d'exemples, 3136)
x_train = x_train.reshape(x_train.shape[0], 56 * 56)
x_test = x_test.reshape(x_test.shape[0], 56 * 56)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train[t:t+1000])
y_test = to_categorical(y_test[t:t+1000])
print("starting")

import time
timing = time.time()
net.fit(x_train, y_train, epochs=10, learning_rate=0.1)
print("model trained in ", time.time() - timing, "seconds")

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("predicted values normalised : ")
array = [[i for i in j[0]] for j in out]
print([i.index(max(i)) for i in array])
print("true values : ")
print(y_test[0:3])

timing = time.time()
output = net.predict(x_test[10:15])
inference_time = time.time() - timing
print(f"Output: {output}, Inference Time: {inference_time:.6f} seconds")
print("solution : ", y_test[10:15])