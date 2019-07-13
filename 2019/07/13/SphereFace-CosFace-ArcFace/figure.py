import numpy as np
import math
import matplotlib.pyplot as plt

def monocos(x):
    n = x // np.pi
    y = np.cos(x - np.pi*n) - 2*n
    return y

def taylorArccos(x, n):

    general_term = lambda x, n: (math.factorial(2 * n - 1) /\
                                math.factorial(2 * n)) *\
                                (x**(2 * n + 1) / (2 * n + 1))
    y = np.pi / 2 - x
    for i in range(1, n):
        y -= general_term(x, i)
    return y

sigmoid = lambda x, s: 1. / (1 + np.e ** (-s * x))

if __name__ == '__main__':

    x = np.linspace(0, np.pi, 100)
    plt.figure('cos')
    plt.title("cos")
    plt.plot(x, np.cos(x))

    x = np.linspace(-1, 1, 100)
    plt.figure('arccos')
    plt.title("arccos")
    plt.plot(x, np.arccos(x))

    plt.figure('taylor arccos, n = 5')
    plt.title("taylor arccos, n = 5")
    plt.plot(x, taylorArccos(x, 5))

    x = np.linspace(0, 3*np.pi, 200)
    plt.figure('cos & monocos', figsize=(10, 4))
    plt.subplot(121)
    plt.title("not monotonical")
    y = np.cos(x)
    plt.vlines(np.pi, np.min(y), np.cos(np.pi), color='r', linestyles='dashdot')
    plt.vlines(np.pi*2, np.min(y), np.cos(np.pi*2), color='r', linestyles='dashdot')
    plt.plot(x, y)
    plt.subplot(122)
    plt.title("monotonical")
    y = monocos(x)
    plt.vlines(np.pi, np.min(y), monocos(np.pi), color='r', linestyles='dashdot')
    plt.vlines(np.pi*2, np.min(y), monocos(np.pi*2), color='r', linestyles='dashdot')
    plt.plot(x, y)

    x = np.linspace(-5, 5, 100)
    plt.figure('s')
    plt.subplot(221)
    plt.plot(x, sigmoid(x, 1))
    plt.title("s=1")
    plt.subplot(222)
    plt.plot(x, sigmoid(x, 4))
    plt.title("s=4")
    plt.subplot(223)
    plt.plot(x, sigmoid(x, 8))
    plt.title("s=8")
    plt.subplot(224)
    plt.plot(x, sigmoid(x, 16))
    plt.title("s=16")

    plt.show()

