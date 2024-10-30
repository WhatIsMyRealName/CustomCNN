import numpy as np

from network import Network
from fclayer import FCLayer
from activationlayer import ActivationLayer
from activations import tanh, tanh_prime, relu, relu_prime # see below
from lossfunction import mse, mse_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(relu, relu_prime)) # choose here
net.add(FCLayer(3, 1))
net.add(ActivationLayer(relu, relu_prime)) # and here

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)