from network import Network
from fclayer import FCLayer
from activationlayer import ActivationLayer
from activations import tanh, tanh_prime
from lossfunction import mse, mse_prime

from keras.datasets import cifar100
from keras.utils import to_categorical
import numpy as np
from scipy.ndimage import zoom

# Charger CIFAR-100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Network
net = Network()
net.add(FCLayer(32 * 32 * 3, 512))          # input_shape=(1, 3072)    ;   output_shape=(1, 512)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(512, 256))                  # input_shape=(1, 512)     ;   output_shape=(1, 256)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(256, 100))                  # input_shape=(1, 256)     ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.use(mse, mse_prime)
# Préparation des données
from random import randrange
t = randrange(0, 4000)

# Mise en forme des données pour le modèle
x_train = x_train.reshape(x_train.shape[0], 32 * 32 * 3).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 32 * 32 * 3).astype('float32') / 255
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)

# Limitez les données pour des tests rapides
x_train, y_train = x_train[t:t+1000], y_train[t:t+1000]
x_test, y_test = x_test[t:t+1000], y_test[t:t+1000]

print("starting")

# Entraînement du modèle
import time
timing = time.time()
net.fit(x_train, y_train, epochs=5, learning_rate=0.1)
print("model trained in ", time.time() - timing, "seconds")

# Test sur quelques échantillons
out = net.predict(x_test[0:3])
#print("\nPredicted values:")
#print(out)

# Normalisation et affichage des prédictions
predictions = [np.argmax(pred) for pred in out]
print("Predicted class indices:", predictions)
print("True class indices:", [np.argmax(y) for y in y_test[0:3]])

# Mesure du temps d'inférence
timing = time.time()
output = net.predict(x_test[10:15])
inference_time = time.time() - timing
print(f"Output: {output}, Inference Time: {inference_time:.6f} seconds")
print("Solution:", y_test[10:15])