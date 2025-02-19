import numpy as np
import scipy.ndimage

def get_gaussian_kernel(sigma=1):
    """Generates a 1D Gaussian kernel using scipy.ndimage.gaussian_filter1d()."""
    number_of_coeffs = 3
    number_of_zeros = 0
    while number_of_zeros < 2 :
        delta = np.zeros(number_of_coeffs)
        delta[delta.size//2] = 1 # Impulse at center
        coeffs = scipy.ndimage.gaussian_filter1d(delta, sigma=sigma)
        number_of_zeros = coeffs.size - np.count_nonzero(coeffs)
        number_of_coeffs += 1
    return coeffs[1:-1]  # Remove the first and last zero elements

def gaussian_noise(shape, mean=0, sigma=20):
    noise = np.random.normal(mean, sigma, shape).reshape(shape)
    return noise
