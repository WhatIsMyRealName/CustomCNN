import numpy as np

# tanh activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

# ReLU activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exps = np.exp(x - np.max(x))  # pour éviter les débordements numériques
    return exps / np.sum(exps, axis=0, keepdims=True)

def softmax_prime(x):
    s = softmax(x)
    return s * (1 - s)  # Dérivée simplifiée
