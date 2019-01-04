import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import integrate


theta = np.arange(-0.1, 1.1, 0.01)

def MLE(theta):
    return theta**7 * (1-theta)**3
def gaussian(x, mu, sigma):
    x = (x - mu) / sigma
    return np.e**(- (x / np.sqrt(2))**2) / (np.sqrt(2*np.pi) * sigma)

def MAP(theta, theta0, sigma0):
    return MLE(theta) * gaussian(theta, theta0, sigma0)

def showMAP(theta, theta0, sigma0):
    fig, ax = plt.subplots(figsize=(9,2),ncols=3,nrows=1)

    
    L_MLE = MLE(theta)
    ax[0].plot(theta, L_MLE, label = "MLE")
    ax[0].legend()
    P = gaussian(theta, theta0, sigma0)
    ax[1].plot(theta, P, label = "P(θ)")
    ax[1].legend()
    L_MAP = MAP(theta, theta0, sigma0)
    ax[2].plot(theta, L_MAP, label = "MAP")
    ax[2].legend()

    idx = np.argmax(np.array(L_MAP))
    print(theta[idx])
    
    name = "MAP_theta0_{:} sigma0_{:}".format(theta0, sigma0)
    plt.savefig(name + '.png')
    plt.show()


theta0 = 0.3; sigma0 = 0.1
showMAP(theta, theta0, sigma0)

theta0 = 0.5; sigma0 = 0.1
showMAP(theta, theta0, sigma0)

theta0 = 0.7; sigma0 = 0.1
showMAP(theta, theta0, sigma0)

theta0 = 0.5; sigma0 = 0.01
showMAP(theta, theta0, sigma0)

theta0 = 0.5; sigma0 = 1.0
showMAP(theta, theta0, sigma0)


def BE(theta, theta0, sigma0):
    be = lambda theta: MLE(theta) * gaussian(theta, theta0, sigma0)
    area, error = integrate.quad(be, 0, 1)
    return MLE(theta) * gaussian(theta, theta0, sigma0) / area

def showBE(theta, theta0, sigma0):
    fig, ax = plt.subplots(figsize=(9,2),ncols=3,nrows=1)

    L_MLE = MLE(theta)
    ax[0].plot(theta, L_MLE, label = "MLE")
    ax[0].legend()
    P = gaussian(theta, theta0, sigma0)
    ax[1].plot(theta, P, label = "P(θ)")
    ax[1].legend()
    
    P_BE = BE(theta, theta0, sigma0)
    ax[2].plot(theta, P_BE, label = "BE")
    ax[2].legend()
    
    
    name = "BE_theta0_{:} sigma0_{:}".format(theta0, sigma0)
    plt.savefig(name + '.png')
    plt.show()
    
theta0 = 0.3; sigma0 = 0.1
showBE(theta, theta0, sigma0)

theta0 = 0.5; sigma0 = 0.1
showBE(theta, theta0, sigma0)

theta0 = 0.7; sigma0 = 0.1
showBE(theta, theta0, sigma0)

theta0 = 0.5; sigma0 = 0.01
showBE(theta, theta0, sigma0)

theta0 = 0.5; sigma0 = 1.0
showBE(theta, theta0, sigma0)

