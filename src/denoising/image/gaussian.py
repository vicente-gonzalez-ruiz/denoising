import time
import numpy as np
import cv2
import scipy
import math
from . import kernels
#pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms import YCoCg as YUV
#import image_denoising

#image_denoising.logger.info(f"Logging level: {image_denoising.logger.getEffectiveLevel()}")

import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

if logger.getEffectiveLevel() < logging.WARNING:
    from matplotlib import pyplot as plt
    
    def normalize(img):
        min_img = np.min(img)
        max_img = np.max(img)
        return 255*((img - min_img)/(max_img - min_img))

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

def RGB_gaussian_filtering(noisy_image, kernel):
    filtered_noisy_image_R = gray_gaussian_filtering(noisy_image[..., 0], kernel)
    filtered_noisy_image_G = gray_gaussian_filtering(noisy_image[..., 1], kernel)
    filtered_noisy_image_B = gray_gaussian_filtering(noisy_image[..., 2], kernel)
    return np.stack([filtered_noisy_image_R, filtered_noisy_image_G, filtered_noisy_image_B], axis=2)

def filter_gray_image(noisy_image, sigma=2.5, N_iters=1.0, GT=None):

    logger.info(f"N_iters={N_iters} sigma={sigma}")
    if logger.getEffectiveLevel() < logging.WARNING:
        PSNR_vs_iteration = []

    kernel = kernels.get_gaussian_kernel(sigma)
    denoised_image = noisy_image.copy()
    for i in range(N_iters):
        if logger.getEffectiveLevel() < logging.WARNING:
            prev = denoised_image
        denoised_image = gray_gaussian_filtering(denoised_image, kernel)
        if logger.getEffectiveLevel() < logging.WARNING:
            if GT != None:
                _PSNR = information_theory.distortion.avg_PSNR(denoised_image_image, GT)
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

    if logger.getEffectiveLevel() < logging.WARNING:
        return denoised_image, PSNR_vs_iteration
    else:
        return denoised_image, None

def filter_RGB_image(noisy_image, sigma=2.5, N_iters=1.0, GT=None):

    logger.info(f"N_iters={N_iters} sigma={sigma}")
    if logger.getEffectiveLevel() < logging.WARNING:
        PSNR_vs_iteration = []

    kernel = kernels.get_gaussian_kernel(sigma)
    denoised_image = noisy_image.copy()
    for i in range(N_iters):
        if logger.getEffectiveLevel() < logging.WARNING:
            prev = denoised_image
        denoised_image = RGB_gaussian_filtering(denoised_image, kernel)
        if logger.getEffectiveLevel() < logging.WARNING:
            if GT != None:
                _PSNR = information_theory.distortion.avg_PSNR(denoised_image_image, GT)
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

    if logger.getEffectiveLevel() < logging.WARNING:
        return denoised_image, PSNR_vs_iteration
    else:
        return denoised_image, None
