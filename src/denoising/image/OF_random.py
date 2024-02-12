'''Random image denoising using the optical flow.'''

import numpy as np
import cv2
from . import flow_estimation
from color_transforms import YCoCg as YUV #pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from information_theory.distortion import PSNR #pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"
import information_theory
#import image_denoising
from matplotlib import pyplot as plt
import logging
#logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)
    
class Filter_Monochrome_Image(flow_estimation.Farneback_Flow_Estimator):

    def __init__(
            self,
            levels=3,       # Pyramid slope. Multiply by 2^levels the searching area if the OFE
            window_side=15, # Applicability window side
            iters=3,        # Number of iterations at each pyramid level
            poly_n=5,       # Size of the pixel neighborhood used to the find polynomial expansion in each pixel
            poly_sigma=1.0, # Standard deviation of the Gaussian basis used in the polynomial expansion
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
            verbosity=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)
        print(f"logging level = {self.logger.level}")
        self.logger.debug(f"levels={levels}, window_side={window_side}, iters={iters}, poly_n={poly_n}, poly_sigma={poly_sigma}")
        super().__init__(levels, window_side, iters, poly_n, poly_sigma, flags)
        self.logger.debug(f"levels={levels}, window_side={window_side}, iters={iters}, poly_n={poly_n}, poly_sigma={poly_sigma}")

    def project_A_to_B(self, A, B):
        flow = self.get_flow_to_project_A_to_B(A, B)
        self.logger.info(f"np.average(np.abs(flow))={np.average(np.abs(flow))}")
        return flow_estimation.project(A, flow)

    def normalize(self, img):
        min_img = np.min(img)
        max_img = np.max(img)
        return 255*((img - min_img + 1)/(max_img - min_img + 1))

    def randomize(self, image, mean=0, std_dev=1.0):
        height, width = image.shape[:2]
        self.logger.debug(f"image.shape={image.shape}")
        x_coords, y_coords = np.meshgrid(range(width), range(height)) # Create a grid of coordinates
        flattened_x_coords = x_coords.flatten()
        flattened_y_coords = y_coords.flatten()
        displacements_x = np.random.normal(mean, std_dev, flattened_x_coords.shape)
        displacements_y = np.random.normal(mean, std_dev, flattened_y_coords.shape)
        #displacements_x *= max_distance_x # Scale the displacements by the maximum distance
        #displacements_y *= max_distance_y
        displacements_x = displacements_x.astype(np.int32)
        displacements_y = displacements_y.astype(np.int32)

        self.logger.info(f"np.max(displacements_x)={np.max(displacements_x)} np.max(displacements_y)={np.max(displacements_y)}")
        randomized_x_coords = flattened_x_coords + displacements_x
        randomized_y_coords = flattened_y_coords + displacements_y
        #randomized_x_coords = np.clip(randomized_x_coords, 0, width - 1) # Clip the randomized coordinates to stay within image bounds
        #randomized_y_coords = np.clip(randomized_y_coords, 0, height - 1)
        randomized_x_coords = np.mod(randomized_x_coords, width) # Apply periodic extension to handle border pixels
        randomized_y_coords = np.mod(randomized_y_coords, height)
        randomized_image = np.zeros_like(image)
        randomized_image[randomized_y_coords, randomized_x_coords] = image[flattened_y_coords, flattened_x_coords]
        return randomized_image

    def filter(self,
               noisy_image,
               RD_iters=50,
               RD_sigma=1.0, # Standard deviation of the maximum random (gaussian-distributed) displacements of the pixels
               RD_mean=0.0, # Mean of the randomized distances
               #RD_sigma=1.0,
               #levels=3,
               #window_side=2,
               #poly_n=5,
               #poly_sigma=0.3,
               GT=None):

        #logger.info(f"RD_iters={RD_iters} RD_mean={RD_mean} RD_sigma={sigma} levels={levles} window_side={window_side} poly_n={poly_n} poly_sigma={poly_sigma}")
        self.logger.info(f"RD_iters={RD_iters} RD_mean={RD_mean} RD_sigma={RD_sigma}")
        if self.logger.level <= logging.INFO:
            PSNR_vs_iteration = []

        acc_image = np.zeros_like(noisy_image, dtype=np.float32)
        acc_image[...] = noisy_image
        if self.logger.level <= logging.DEBUG:
            denoised_image = noisy_image
        for i in range(RD_iters):
            self.logger.info(f"Iteration {i}/{RD_iters}")
            if self.logger.level <= logging.DEBUG:
                fig, axs = plt.subplots(1, 2)
                prev = denoised_image
            denoised_image = acc_image/(i+1)
            if self.logger.level <= logging.INFO:
                try:
                    _PSNR = information_theory.distortion.PSNR(denoised_image, GT)
                except:
                    _PSNR = 0.0
                PSNR_vs_iteration.append(_PSNR)
            if self.logger.level <= logging.DEBUG:
                axs[0].imshow(denoised_image.astype(np.uint8))
                axs[0].set_title(f"iter {i} " + f"({_PSNR:4.2f}dB)")
                axs[1].imshow(self.normalize(prev - denoised_image + 128).astype(np.uint8), cmap="gray")
                axs[1].set_title(f"diff")
                plt.show()
            randomized_noisy_image = self.randomize(
                noisy_image,
                RD_mean,
                RD_sigma).astype(np.float32)
            #randomized_noisy_image = randomize(noisy_image)
            randomized_and_compensated_noisy_image = self.project_A_to_B(
                A=denoised_image,
                B=randomized_noisy_image)
            acc_image += randomized_and_compensated_noisy_image
        denoised_image = acc_image/(RD_iters + 1)
        #print(flush=True)

        if self.logger.level <= logging.INFO:
            return denoised_image, PSNR_vs_iteration
        else:
            return denoised_image, None

class Filter_Color_Image(Filter_Monochrome_Image):

    def __init__(
            self,
            levels=3, # Pyramid slope. Multiply by 2^levels the searching area if the OFE
            window_side=15, # Applicability window side
            iters=3, # Number of iterations at each pyramid level
            poly_n=5, # Size of the pixel neighborhood used to find polynomial expansion in each pixel
            poly_sigma=1.0, # Standard deviation of the Gaussian basis used in the polynomial expansion
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
            verbosity=logging.INFO):
        super().__init__(levels, window_side, iters, poly_n, poly_sigma, flags, verbosity)

    def project_A_to_B(self, A, B):
        self.logger.debug(f"A.shape={A.shape} B.shape={B.shape}")
        A_luma = YUV.from_RGB(A.astype(np.int16))[..., 0]
        B_luma = YUV.from_RGB(B.astype(np.int16))[..., 0]
        #A_luma = np.log(YUV.from_RGB(A.astype(np.int16))[..., 0] + 1)
        #B_luma = np.log(YUV.from_RGB(B.astype(np.int16))[..., 0] + 1)
        flow = self.get_flow_to_project_A_to_B(A_luma, B_luma)
        self.logger.info(f"np.average(np.abs(flow))={np.average(np.abs(flow))}")
        return flow_estimation.project(A, flow)
        #return super().warp_B_to_A(A_luma,
        #                           B_luma)

'''
def _randomize(image, max_distance_x=10, max_distance_y=10):
    height, width, _ = image.shape
    x_coords, y_coords = np.meshgrid(range(width), range(height)) # Create a grid of coordinates
    flattened_x_coords = x_coords.flatten()
    flattened_y_coords = y_coords.flatten()
    displacements_x = np.random.randint(-max_distance_x, max_distance_x + 1, flattened_x_coords.shape)
    displacements_y = np.random.randint(-max_distance_y, max_distance_y + 1, flattened_y_coords.shape)
    randomized_x_coords = flattened_x_coords + displacements_x
    randomized_y_coords = flattened_y_coords + displacements_y
    randomized_x_coords = np.mod(randomized_x_coords, width) # Use periodic extension to handle border pixels
    randomized_y_coords = np.mod(randomized_y_coords, height)
    randomized_image = np.empty_like(image)
    randomized_image[...] = image
    randomized_image[randomized_y_coords,
                     randomized_x_coords, :] = image[flattened_y_coords,
                                                     flattened_x_coords, :]
    return randomized_image

def RGB_warp_B_to_A(A, B, l=3, w=15, prev_flow=None, sigma=1.5):
    A_luma = YUV.from_RGB(A.astype(np.int16))[..., 0]
    B_luma = YUV.from_RGB(B.astype(np.int16))[..., 0]
    #A_luma = np.log(YUV.from_RGB(A.astype(np.int16))[..., 0] + 1)
    #B_luma = np.log(YUV.from_RGB(B.astype(np.int16))[..., 0] + 1)
    flow = flow_estimation.get_flow_to_project_A_to_B(A_luma, B_luma, l, w, prev_flow, sigma)
    return flow_estimation.project(B, flow)

def warp_B_to_A(A, B, l=3, w=15, prev_flow=None, sigma=1.5):
    flow = flow_estimation.get_flow_to_project_A_to_B(A, B, l, w, prev_flow, sigma)
    return flow_estimation.project(B, flow)
                                   
def filter_image(
        warp_B_to_A,
        noisy_image,
        N_iters=50,
        mean_RD=0.0,
        sigma_RD=1.0,
        l=3,
        w=2,
        sigma_OF=0.3,
        GT=None):

    logger.info(f"N_iters={N_iters} mean_RD={mean_RD} sigma_RD={sigma_RD} l={l} w={w} sigma_OF={sigma_OF}")
    if logger.level <= logging.INFO:
        PSNR_vs_iteration = []

    acc_image = np.zeros_like(noisy_image, dtype=np.float32)
    acc_image[...] = noisy_image
    denoised_image = noisy_image
    for i in range(N_iters):
        print(f"{i}/{N_iters}", end=' ')
        if logger.level <= logging.DEBUG:
            fig, axs = plt.subplots(1, 2)
            prev = denoised_image
        denoised_image = acc_image/(i+1)
        if logger.level <= logging.DEBUG:
            if GT != None:
                _PSNR = information_theory.distortion.PSNR(denoised_image, GT)
            else:
                _PSNR = 0.0
            PSNR_vs_iteration.append(_PSNR)
            axs[0].imshow(denoised_image.astype(np.uint8))
            axs[0].set_title(f"iter {i} " + f"({_PSNR:4.2f}dB)")
            axs[1].imshow(normalize(prev - denoised_image + 128).astype(np.uint8), cmap="gray")
            axs[1].set_title(f"diff")
            plt.show()
        randomized_noisy_image = randomize(noisy_image, mean_RD, sigma_RD).astype(np.float32)
        randomized_and_compensated_noisy_image = warp_B_to_A(
            A=randomized_noisy_image,
            B=denoised_image,
            l=l,
            w=w,
            sigma=sigma_OF)
        acc_image += randomized_and_compensated_noisy_image
    denoised_image = acc_image/(N_iters + 1)
    print()

    if logger.level <= logging.INFO:
        return denoised_image, PSNR_vs_iteration
    else:
        return denoised_image, None

def _denoise(warp_B_to_A, noisy_image, N_iters=50, mean_RD=0.0, sigma_RD=1.0, l=3, w=2, sigma_OF=0.3):
    logger.info(f"N_iters={N_iters} mean_RD={mean_RD} sigma_RD={sigma_RD} l={l} w={w} sigma_OF={sigma_OF}")
    acc_image = np.zeros_like(noisy_image, dtype=np.float32)
    acc_image[...] = noisy_image
    for i in range(N_iters):
        print(f"iter={i}", end=' ')
        denoised_image = acc_image/(i+1)
        randomized_noisy_image = randomize(noisy_image, mean_RD, sigma_RD).astype(np.float32)
        randomized_and_compensated_noisy_image = warp_B_to_A(A=randomized_noisy_image, B=denoised_image, l=l, w=w, sigma=sigma_OF)
        acc_image += randomized_and_compensated_noisy_image
    denoised_image = acc_image/(N_iters + 1)
    print()
    return denoised_image
'''
