# Denoising utils

import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage

def imshow(img):
    return plt.imshow(np.clip(a = img, a_min=0, a_max=255), cmap="gray")

def add_0MWGN(X, std_dev=40):
    '''Zero-Mean White Gaussian Noise'''
    return X + np.random.normal(loc=0, scale=std_dev, size=X.shape).reshape(X.shape)

def get_gaussian_kernel(sigma=1):
    number_of_coeffs = 3
    number_of_zeros = 0
    while number_of_zeros < 2 :
        delta = np.zeros(number_of_coeffs)
        delta[delta.size//2] = 1
        coeffs = scipy.ndimage.gaussian_filter1d(delta, sigma=sigma)
        number_of_zeros = coeffs.size - np.count_nonzero(coeffs)
        number_of_coeffs += 1
    return coeffs[1:-1]

def generate_PN(X, gamma=0.1):
    '''Poisson Noise'''
    return np.random.poisson(X * gamma) / gamma

def generate_MPGN(X, std_dev=10.0, gamma=0.1, poisson_ratio=0.5):
    '''Mixel Poisson Gaussian Noise'''
    N_poisson = np.random.poisson(X * gamma)/gamma
    N_gaussian = np.random.normal(loc=0, scale=std_dev, size=X.size)
    N_gaussian = np.reshape(N_gaussian, X.shape)
    Y = (1 - poisson_ratio) * (X + N_gaussian) + poisson_ratio * N_poisson
    #Y = np.clip(Y, 0, 255)
    #Y = N_gaussian + N_poisson
    #Y = N_gaussian + gamma*N_poisson
    #Y = N_poisson
    #Y = N_gaussian + X
    return Y

