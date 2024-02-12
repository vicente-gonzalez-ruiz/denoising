import numpy as np
import cv2
import scipy
import math
#from . import kernels
#pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms import YCoCg as YUV
#import image_denoising

#image_denoising.logger.info(f"Logging level: {image_denoising.logger.getEffectiveLevel()}")

import logging
logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger = logging.getLogger(__name__)
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

import scipy.ndimage
import information_theory
from matplotlib import pyplot as plt

def vertical_gaussian_filtering(noisy_image, kernel, mean):
    KL = kernel.size
    KL2 = KL//2
    extended_noisy_image = np.full(fill_value=mean, shape=(noisy_image.shape[0] + KL, noisy_image.shape[1]))
    extended_noisy_image[KL2:noisy_image.shape[0] + KL2, :] = noisy_image[:, :]
    filtered_noisy_image = []
    #filtered_noisy_image = np.empty_like(noisy_image, dtype=np.float32)
    N_rows = noisy_image.shape[0]
    N_cols = noisy_image.shape[1]
    #horizontal_line = np.empty(N_cols, dtype=np.float32)
    #print(horizontal_line.shape)
    for y in range(N_rows):
        #horizontal_line.fill(0)
        horizontal_line = np.zeros(N_cols, dtype=np.float32)
        for i in range(KL):
            horizontal_line += extended_noisy_image[y + i, :] * kernel[i]
        filtered_noisy_image.append(horizontal_line)
        #filtered_noisy_image[y, :] = horizontal_line[:]
    filtered_noisy_image = np.stack(filtered_noisy_image, axis=0)
    return filtered_noisy_image

def horizontal_gaussian_filtering(noisy_image, kernel, mean):
    KL = kernel.size
    KL2 = KL//2
    extended_noisy_image = np.full(fill_value=mean, shape=(noisy_image.shape[0], noisy_image.shape[1] + KL))
    extended_noisy_image[:, KL2:noisy_image.shape[1] + KL2] = noisy_image[:, :]
    #filtered_noisy_image = []
    filtered_noisy_image = np.empty_like(noisy_image, dtype=np.float32)
    N_rows = noisy_image.shape[0]
    N_cols = noisy_image.shape[1]
    vertical_line = np.empty(N_rows, dtype=np.float32)
    for x in range(N_cols):
        #vertical_line = np.zeros(N_rows, dtype=np.float32)
        vertical_line.fill(0)
        for i in range(KL):
            vertical_line += extended_noisy_image[:, x + i] * kernel[i]
        #filtered_noisy_image.append(vertical_line)
        filtered_noisy_image[:, x] = vertical_line[:]
    #filtered_noisy_image = np.stack(filtered_noisy_image, axis=1)
    return filtered_noisy_image

def gray_gaussian_filtering(noisy_image, kernel):
    mean = np.average(noisy_image)
    #t0 = time.perf_counter()
    filtered_noisy_image_Y = vertical_gaussian_filtering(noisy_image, kernel, mean)
    #t1 = time.perf_counter()
    #print(t1 - t0)
    filtered_noisy_image_YX = horizontal_gaussian_filtering(filtered_noisy_image_Y, kernel, mean)
    #t2 = time.perf_counter()
    #print(t2 - t1)
    return filtered_noisy_image_YX

def normalize(img):
    min_img = np.min(img)
    max_img = np.max(img)
    return 255*((img - min_img)/(max_img - min_img))

def RGB_gaussian_filtering(noisy_image, kernel):
    filtered_noisy_image_R = gray_gaussian_filtering(noisy_image[..., 0], kernel)
    filtered_noisy_image_G = gray_gaussian_filtering(noisy_image[..., 1], kernel)
    filtered_noisy_image_B = gray_gaussian_filtering(noisy_image[..., 2], kernel)
    return np.stack([filtered_noisy_image_R, filtered_noisy_image_G, filtered_noisy_image_B], axis=2)

class Monochrome_Image_Gaussian_Denoising:

    def __init__(self, sigma=1.5, verbosity=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)
        print(f"logging level = {self.logger.level}")
        self.sigma = sigma
        self.logger.info(f"sigma={self.sigma}")
        self.gaussian_filtering = gray_gaussian_filtering

    def get_gaussian_kernel(self):
        number_of_coeffs = 3
        number_of_zeros = 0
        while number_of_zeros < 2 :
            delta = np.zeros(number_of_coeffs)
            delta[delta.size//2] = 1
            coeffs = scipy.ndimage.gaussian_filter1d(delta, sigma=self.sigma)
            number_of_zeros = coeffs.size - np.count_nonzero(coeffs)
            number_of_coeffs += 1
        return coeffs[1:-1]

    def filter(self, noisy_image, GT=None, N_iters=1):
        if self.logger.getEffectiveLevel() < logging.INFO:
            PSNR_vs_iteration = []
        kernel = self.get_gaussian_kernel()
        denoised_image = noisy_image.copy()
        for i in range(N_iters):
            if self.logger.getEffectiveLevel() < logging.INFO:
                prev = denoised_image
            denoised_image = self.gaussian_filtering(denoised_image, kernel)
            if self.logger.getEffectiveLevel() < logging.INFO:
                if isinstance(GT, np.ndarray):
                    _PSNR = information_theory.distortion.avg_PSNR(denoised_image, GT)
                else:
                    _PSNR = 0.0
                PSNR_vs_iteration.append(_PSNR)
                fig, axs = plt.subplots(1, 2, figsize=(10, 20))
                axs[0].imshow(normalize(denoised_image), cmap="gray")
                axs[0].set_title(f"iter {i} " + f"({_PSNR:4.2f}dB)")
                axs[1].imshow(normalize(denoised_image - prev + 128), cmap="gray")
                axs[1].set_title(f"diff")
                plt.show()
        print()
        if self.logger.getEffectiveLevel() < logging.INFO:
            return denoised_image, PSNR_vs_iteration
        else:
            return denoised_image, None

class Color_Image_Gaussian_Denoising(Monochrome_Image_Gaussian_Denoising):

    def __init__(self, sigma=1.5, verbosity=logging.INFO):
        super().__init__(sigma=sigma, verbosity=verbosity)
        self.gaussian_filtering = RGB_gaussian_filtering

    def filter(self, noisy_image, GT=None, N_iters=1):
        if self.logger.getEffectiveLevel() < logging.INFO:
            PSNR_vs_iteration = []
        kernel = self.get_gaussian_kernel()
        denoised_image = noisy_image.copy()
        for i in range(N_iters):
            if self.logger.getEffectiveLevel() < logging.INFO:
                prev = denoised_image
            denoised_image = self.gaussian_filtering(denoised_image, kernel)
            if self.logger.getEffectiveLevel() < logging.INFO:
                if isinstance(GT, np.ndarray):
                    _PSNR = information_theory.distortion.avg_PSNR(denoised_image, GT)
                else:
                    _PSNR = 0.0
                PSNR_vs_iteration.append(_PSNR)
                fig, axs = plt.subplots(1, 2, figsize=(10, 20))
                axs[0].imshow(normalize(denoised_image).astype(np.uint8))
                axs[0].set_title(f"iter {i} " + f"({_PSNR:4.2f}dB)")
                axs[1].imshow(normalize(denoised_image - prev + 128).astype(np.uint8))
                axs[1].set_title(f"diff")
                plt.show()
        print()
        if self.logger.getEffectiveLevel() < logging.INFO:
            return denoised_image, PSNR_vs_iteration
        else:
            return denoised_image, None


