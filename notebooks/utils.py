# Denoising utils

import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage
from PIL import Image
import cv2

def normalize(X):
    X = cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return X

def RGB_normalize(X):
    X = cv2.cvtColor(X, cv2.COLOR_RGB2YCrCb)
    Y, Cb, Cr = cv2.split(X)
    Y = cv2.normalize(Y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    X = cv2.merge([Y, Cb, Cr])
    X = cv2.cvtColor(X, cv2.COLOR_YCrCb2RGB)
    return X

def imshow(img):
    return plt.imshow(np.clip(a = img, a_min=0, a_max=255), cmap="gray")

def add_0MWGN(X, std_dev=40):
    '''Zero-Mean White Gaussian Noise'''
    print("Better use generate_0MWGN()")
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

def generate_0MWGN(rows=512, cols=512, mean=128.0, stddev=5.0):
    """
    Generates a 2D NumPy array with zero-mean white Gaussian noise.

    Args:
        rows: Number of rows in the array.
        cols: Number of columns in the array.
        mean: Mean (average) of the Gaussian distribution (default: 0.0).
        stddev: Standard deviation of the Gaussian distribution (default: 1.0).

    Returns:
        A 2D NumPy array with Gaussian random noise.
    """
    noise = np.random.normal(loc=mean, scale=stddev, size=(rows, cols))
    print(np.min(noise), np.max(noise))
    return noise

    #'''Poisson Noise'''
    #return np.random.poisson(X * gamma) / gamma

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

def clip(X):
    return np.clip(X, 0, 255)

def normalize_image_to_255(image_float):
    """
    Normalizes a single-channel float image to the range [0, 255] and converts it to uint8.

    Args:
        image_float (np.ndarray): A single-channel image (2D NumPy array)
                                  with float pixel values.

    Returns:
        np.ndarray: The normalized image as a 2D NumPy array with dtype uint8,
                    scaled to the range [0, 255].
    """
    if not isinstance(image_float, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")
    if image_float.ndim != 2:
        raise ValueError("Input image must be a single-channel (2D) array.")

    min_val = image_float.min()
    max_val = image_float.max()

    if min_val == max_val:
        # Handle the case where all pixel values are the same
        # If all values are 0, output all 0s. Otherwise, output all 127s (mid-gray).
        if min_val == 0:
            normalized_image = np.zeros_like(image_float, dtype=np.uint8)
        else:
            normalized_image = np.full_like(image_float, 127, dtype=np.uint8)
    else:
        # Perform min-max scaling to [0, 1]
        normalized_image = (image_float - min_val) / (max_val - min_val)
        # Scale to [0, 255] and convert to uint8
        normalized_image = (normalized_image * 255).astype(np.uint8)

    return normalized_image

def equalize_grayscale_image(input_image_array):
    """
    Performs histogram equalization on a grayscale image provided as a NumPy array.

    Args:
        input_image_array (numpy.ndarray): A 2D NumPy array representing the grayscale image.
                                          Expected to be of dtype uint8 (0-255).

    Returns:
        PIL.Image.Image: The equalized grayscale image as a Pillow Image object,
                         or None if the input array is invalid (e.g., not 2D).
    """
    # 1. Validate the input array
    if not isinstance(input_image_array, np.ndarray):
        print("Error: Input must be a NumPy array.")
        return None
    
    # Ensure it's a 2D grayscale array (height x width)
    if input_image_array.ndim != 2:
        print(f"Error: Input array must be 2-dimensional (grayscale). Found {input_image_array.ndim} dimensions.")
        return None
    
    # Ensure image data type is uint8 (0-255) for consistent processing
    # If it's not uint8, scale it to 0-255 and convert.
    if input_image_array.dtype != np.uint8:
        print("Warning: Input array is not of dtype uint8. Scaling to 0-255 and converting.")
        # Ensure values are within 0-255 range after scaling
        img_array = (input_image_array / input_image_array.max() * 255).astype(np.uint8)
    else:
        img_array = input_image_array # Use the array directly if it's already uint8

    # 2. Calculate the histogram
    # np.histogram returns (counts, bin_edges). We only need the counts.
    # .flatten() converts the 2D image array into a 1D array of pixel values.
    hist, bins = np.histogram(img_array.flatten(), bins=256, range=[0, 256])

    # 3. Calculate the Cumulative Distribution Function (CDF)
    cdf = hist.cumsum()

    # 4. Normalize the CDF to create the lookup table (LUT)
    # Find the minimum non-zero CDF value. This is crucial for proper mapping,
    # ensuring the lowest effective pixel intensity maps to 0 in the equalized image.
    cdf_min = cdf[cdf > 0].min()
    
    # Total number of pixels in the image
    total_pixels = img_array.size
    
    # Create the lookup table (LUT)
    # The standard formula for histogram equalization transformation:
    # T(v) = round(((cdf(v) - cdf_min) / (M*N - cdf_min)) * (L-1))
    # Where:
    #   v is the original pixel intensity
    #   cdf(v) is the cumulative distribution function value for intensity v
    #   cdf_min is the minimum non-zero value of the CDF
    #   M*N is the total number of pixels (img_array.size)
    #   L-1 is the maximum intensity value (255 for 8-bit images)

    if total_pixels == cdf_min:
        # This edge case implies all pixels in the image have the same value.
        # In such a scenario, equalization doesn't change anything meaningfully.
        # Return an image with the same uniform intensity.
        equalized_img_array = np.full_like(img_array, img_array[0,0], dtype=np.uint8)
    else:
        # Calculate the scaling factor for normalization.
        # This maps the range [cdf_min, total_pixels] to [0, 255].
        scale_factor = 255.0 / (total_pixels - cdf_min)
        lut = np.round((cdf - cdf_min) * scale_factor).astype(np.uint8)

        # 5. Apply the transformation using the lookup table
        # We use the original image's pixel values as indices into the lookup table.
        # This efficiently maps each old pixel value to its new equalized value.
        equalized_img_array = lut[img_array]

    # Convert the NumPy array back to a Pillow Image object
    equ_img_pil = Image.fromarray(equalized_img_array)

    return equ_img_pil
