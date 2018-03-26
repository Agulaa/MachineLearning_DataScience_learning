import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def plot_(z):
    phi_z = sigmoid(z)
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
    plt.axhline(y=0.5, ls='dotted', color='k')
    plt.yticks([0.0, 0.5, 1.0])
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('phi(z)')
    plt.show()


if __name__ == '__main__':
    z = np.arange(-7, 7, 0.1)
    plot_(z)
