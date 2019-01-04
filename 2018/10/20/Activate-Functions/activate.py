import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 100)

def sigmoid(x):
    return 1 / (1 + np.e**(-x))
def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    x = np.exp(2*x)
    return (x-1) / (x+1)
def dtanh(x):
    return 1 - tanh(x)**2

def BNLL(x):
    return np.log(1 + np.exp(x))
def dBNLL(x):
    x = np.exp(x)
    return x / (x + 1)

def power(x, alpha, beta, gama):
    return (alpha * x + beta)**gama
def dpower(x, alpha, beta, gama):
    return alpha * gama * ((alpha * x + beta) ** (gama - 1))

def ReLU(x):
    x = np.c_[np.zeros(shape=(x.shape[0])), x]
    return np.max(x, axis=1)
def dReLU(x):
    x = np.sign(x)
    x[x<0] = 0
    return x

def ELU(x, alpha):
    for i in range(x.shape[0]):
        if x[i] < 0: x[i] = alpha * (np.exp(x[i]) - 1)
    return x
def dELU(x, alpha):
    y = np.ones(shape=x.shape)
    for i in range(x.shape[0]):
        if x[i] < 0:
            y[i] = ELU(x[i], alpha) + alpha

def PReLU(x, alpha):
    for i in range(x.shape[0]):
        if x[i] < 0: x[i] = alpha * x[i]
    return x
def dPReLU(x, alpha):
    y = np.ones(shape=x.shape)
    for i in range(x.shape[0]):
        if x[i] < 0:
            y[i] = alpha

def exp(x, alpha, beta, gama):
    return gama ** (alpha * x + beta)
def dexp(x, alpha, beta, gama):
    return exp(x, alpha, beta, gama) * np.log(gama) * alpha

def log(x, alpha, beta, gama):
    return np.log(alpha * x + beta) / np.log(gama)
def dlog(x, alpha, beta, gama):
    return alpha * (np.log(gama) * (alpha * x + beta))


