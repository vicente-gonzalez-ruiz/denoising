'''Gaussian image denoising.'''

import numpy as np
import cv2
import scipy
import math
#from . import kernels
#pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms import YCoCg as YUV
#import img_denoising
#from . import OF_gaussian

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

def normalize(img):
    min_img = np.min(img)
    max_img = np.max(img)
    return 255*((img - min_img)/(max_img - min_img))

class Monochrome_Denoising:

    def __init__(self, verbosity=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)

    def get_kernel(self, sigma=1.5):
        self.logger.info(f"sigma={sigma}")
        number_of_coeffs = 3
        number_of_zeros = 0
        while number_of_zeros < 2 :
            delta = np.zeros(number_of_coeffs)
            delta[delta.size//2] = 1
            coeffs = scipy.ndimage.gaussian_filter1d(delta, sigma=sigma)
            number_of_zeros = coeffs.size - np.count_nonzero(coeffs)
            number_of_coeffs += 1
        return coeffs[1:-1]
    
    def project_A_to_B(self, A, B):
        return A
    
    def filter_vertical(self, noisy_img, kernel, mean):
        KL = kernel.size
        KL2 = KL//2
        extended_noisy_img = np.full(fill_value=mean, shape=(noisy_img.shape[0] + KL, noisy_img.shape[1]))
        extended_noisy_img[KL2:noisy_img.shape[0] + KL2, :] = noisy_img[:, :]
        filtered_noisy_img = []
        #filtered_noisy_img = np.empty_like(noisy_img, dtype=np.float32)
        N_rows = noisy_img.shape[0]
        N_cols = noisy_img.shape[1]
        #horizontal_line = np.empty(N_cols, dtype=np.float32)
        #print(horizontal_line.shape)
        for y in range(N_rows):
            print(y, end=' ')
            #horizontal_line.fill(0)
            horizontal_line = np.zeros(N_cols, dtype=np.float32)
            for i in range(KL):
                line = self.project_A_to_B(A=extended_noisy_img[y + i, :], B=extended_noisy_img[y, :])
                horizontal_line += line * kernel[i]
            filtered_noisy_img.append(horizontal_line)
            #filtered_noisy_img[y, :] = horizontal_line[:]
        filtered_noisy_img = np.stack(filtered_noisy_img, axis=0)
        return filtered_noisy_img
    
    def filter_horizontal(self, noisy_img, kernel, mean):
        KL = kernel.size
        KL2 = KL//2
        extended_noisy_img = np.full(fill_value=mean, shape=(noisy_img.shape[0], noisy_img.shape[1] + KL))
        extended_noisy_img[:, KL2:noisy_img.shape[1] + KL2] = noisy_img[:, :]
        #filtered_noisy_img = []
        filtered_noisy_img = np.empty_like(noisy_img, dtype=np.float32)
        N_rows = noisy_img.shape[0]
        N_cols = noisy_img.shape[1]
        vertical_line = np.empty(N_rows, dtype=np.float32)
        for x in range(N_cols):
            print(x, end=' ')
            #vertical_line = np.zeros(N_rows, dtype=np.float32)
            vertical_line.fill(0)
            for i in range(KL):
                line = self.project_A_to_B(A=extended_noisy_img[:, x + i], B=extended_noisy_img[:, x])
                vertical_line += line * kernel[i]
            #filtered_noisy_img.append(vertical_line)
            filtered_noisy_img[:, x] = vertical_line[:]
        #filtered_noisy_img = np.stack(filtered_noisy_img, axis=1)
        return filtered_noisy_img
    
    def _filter(self, noisy_img, kernel):
        mean = np.average(noisy_img)
        #t0 = time.perf_counter()
        filtered_in_vertical = self.filter_vertical(noisy_img, kernel, mean)
        print(filtered_in_vertical.dtype)
        #t1 = time.perf_counter()
        #print(t1 - t0)
        filtered_in_horizontal = self.filter_horizontal(noisy_img, kernel, mean)
        #t2 = time.perf_counter()
        #print(t2 - t1)
        filtered_noisy_img = (filtered_in_vertical + filtered_in_horizontal)/2
        return filtered_noisy_img

    def filter(self, noisy_img, kernel):
        mean = np.average(noisy_img)
        #t0 = time.perf_counter()
        #filtered_noisy_img_Y = self.filter_vertical(noisy_img, kernel, mean)
        #print(filtered_noisy_img_Y.dtype)
        #t1 = time.perf_counter()
        #print(t1 - t0)
        #mean = np.average(filtered_noisy_img_Y)
        #filtered_noisy_img_YX = self.filter_horizontal(filtered_noisy_img_Y, kernel, mean)
        filtered_noisy_img_YX = self.filter_horizontal(noisy_img, kernel, mean)
        #t2 = time.perf_counter()
        #print(t2 - t1)
        return filtered_noisy_img_YX

    def filter_iterate(self, noisy_img, sigma=1.5, GT=None, N_iters=1):
        self.logger.info(f"sigma={sigma}")
        if self.logger.getEffectiveLevel() < logging.INFO:
            PSNR_vs_iteration = []
        kernel = self.get_kernel(sigma)
        denoised_img = noisy_img.copy()
        for i in range(N_iters):
            if self.logger.getEffectiveLevel() < logging.INFO:
                prev = denoised_img
            denoised_img = self.filter(denoised_img, kernel)
            denoised_img = np.clip(denoised_img, a_min=0, a_max=255)
            if self.logger.getEffectiveLevel() < logging.INFO:
                if isinstance(GT, np.ndarray):
                    _PSNR = information_theory.distortion.avg_PSNR(denoised_img, GT)
                else:
                    _PSNR = 0.0
                PSNR_vs_iteration.append(_PSNR)
                fig, axs = plt.subplots(1, 2, figsize=(10, 20))
                axs[0].imshow(denoised_img, cmap="gray")
                axs[0].set_title(f"iter {i} " + f"({_PSNR:4.2f}dB)")
                axs[1].imshow(denoised_img - GT + 128, cmap="gray")
                axs[1].set_title(f"diff")
                plt.show()
        print()
        if self.logger.getEffectiveLevel() < logging.INFO:
            return denoised_img, PSNR_vs_iteration
        else:
            return denoised_img, None

class Color_Denoising(Monochrome_Denoising):

    def __init__(self, verbosity=logging.INFO):
        super().__init__(verbosity=verbosity)

    def filter(self, noisy_img, kernel):
        filtered_noisy_img_R = super().filter(noisy_img[..., 0], kernel)
        filtered_noisy_img_G = super().filter(noisy_img[..., 1], kernel)
        filtered_noisy_img_B = super().filter(noisy_img[..., 2], kernel)
        return np.stack([filtered_noisy_img_R, filtered_noisy_img_G, filtered_noisy_img_B], axis=2)

    def filter_iterate(self, noisy_img, sigma=1.5, GT=None, N_iters=1):
        if self.logger.getEffectiveLevel() < logging.INFO:
            PSNR_vs_iteration = []
        kernel = self.get_kernel()
        denoised_img = noisy_img.copy()
        for i in range(N_iters):
            if self.logger.getEffectiveLevel() < logging.INFO:
                prev = denoised_img
            denoised_img = self.filter(denoised_img, kernel)
            denoised_img = np.clip(denoised_img, a_min=0, a_max=255)
            if self.logger.getEffectiveLevel() < logging.INFO:
                if isinstance(GT, np.ndarray):
                    _PSNR = information_theory.distortion.avg_PSNR(denoised_img, GT)
                else:
                    _PSNR = 0.0
                PSNR_vs_iteration.append(_PSNR)
                fig, axs = plt.subplots(1, 2, figsize=(10, 20))
                axs[0].imshow(denoised_img)
                axs[0].set_title(f"iter {i} " + f"({_PSNR:4.2f}dB)")
                axs[1].imshow(denoised_img - GT + 128)
                axs[1].set_title(f"diff")
                plt.show()
        print()
        if self.logger.getEffectiveLevel() < logging.INFO:
            return denoised_img, PSNR_vs_iteration
        else:
            return denoised_img, None


