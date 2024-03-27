'''Gaussian image denoising using optical flow (OpenCV version).'''

#import time
import numpy as np
import cv2
#from . import kernels
#from . import flow_estimation
#pip install "motion_estimation @ git+https://github.com/vicente-gonzalez-ruiz/motion_estimation"
from motion_estimation._2D.farneback_OpenCV import Estimator_in_CPU as Estimator
from motion_estimation._2D.project import project
#from color_transforms import YCoCg as YUV #pip install "pip install color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
#import image_denoising
from . import gaussian
import logging # borrame

N_POLY = 7
PYRAMID_LEVELS = 2
WINDOW_SIDE = 7
NUM_ITERATIONS = 3
FLAGS = 0

class Monochrome_Denoising(gaussian.Monochrome_Denoising):

    def __init__(self, logger,
                 pyramid_levels=PYRAMID_LEVELS,
                 window_side=WINDOW_SIDE,
                 N_poly=N_POLY,
                 num_iterations=NUM_ITERATIONS,
                 flags=FLAGS):
        super().__init__(logger)
        self.estimator = Estimator(logger)
        self.pyramid_levels = pyramid_levels
        self.window_side = window_side
        self.N_poly = N_poly
        self.num_iterations = num_iterations
        self.flags = flags
        for attr, value in vars(self).items():
            self.logger.info(f"{attr}: {value}")

    def warp_slice(self, slice, flow):
        warped_slice = project(
            self.logger,
            image=slice,
            flow=flow,
            interpolation_mode=cv2.INTER_LINEAR,
            extension_mode=cv2.BORDER_REPLICATE)
        return warped_slice

    def pyramid_get_flow(
        self,
        reference,
        target,
        flow=None,
        pyramid_levels=PYRAMID_LEVELS, # Number of pyramid layers
        window_side=WINDOW_SIDE, # Applicability window side
        num_iterations=NUM_ITERATIONS, # Number of iterations at each pyramid level
        N_poly=N_POLY, # Standard deviation of the Gaussian basis used in the polynomial expansion
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW | cv2.OPTFLOW_FARNEBACK_GAUSSIAN):
        #flow = np.zeros(shape=(reference.size, 1), dtype=np.float32)
        #return flow
        flow = self.estimator.pyramid_get_flow(
            target, reference,
            flow=flow,
            pyramid_levels=pyramid_levels,
            window_side=window_side,
            num_iterations=num_iterations,
            N_poly=N_poly,
            flags=flags)
        return flow

    def filter_Y_line(self, img, padded_img, kernel, y):
        KL = kernel.size
        KL2 = KL//2
        print(y, end=' ')
        horizontal_line = np.zeros(shape=(img.shape[1] + self.window_side), dtype=np.float32)
        target_slice = padded_img[y + KL2:y + KL2 + self.window_side]
        for i in range(KL):
            reference_slice = padded_img[y + i:y + i + self.window_side]
            flow = self.pyramid_get_flow(
                reference=reference_slice,
                target=target_slice,
                flow=None,
                flags=0)
            OF_compensated_slice = self.warp_slice(reference_slice, flow)
            OF_compensated_line = OF_compensated_slice[(self.window_side + 1) >> 1, :]
            OF_compensated_line = np.roll(OF_compensated_line, -self.window_side//2) # Creo que si intercambiamos reference_slice por target_slice, esta línea sobra
            print(horizontal_line.shape, OF_compensated_line.shape)
            horizontal_line += OF_compensated_line * kernel[i]
        return horizontal_line

    def filter(self, img, kernel):
        mean = img.mean()
        filtered_img_Y = self.filter_Y(img, kernel[0], mean)
        transposed_filtered_img_Y = np.transpose(filtered_img_Y, (1, 0))
        transposed_filtered_img_YX = self.filter_Y(transposed_filtered_noisy_img_Y, kernel, mean)
        filtered_img_YX = np.transpose(transposed_filtered_img_YX, (1, 0))
        return filtered_img_YX



