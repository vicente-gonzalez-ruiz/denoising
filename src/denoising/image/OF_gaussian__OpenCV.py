'''Gaussian image denoising using optical flow (OpenCV version).'''

#import time
import numpy as np
import cv2
#from . import kernels
#from . import flow_estimation
#pip install "motion_estimation @ git+https://github.com/vicente-gonzalez-ruiz/motion_estimation"
from motion_estimation._2D.farneback_OpenCV import OF_Estimation
from motion_estimation._2D.project import Projection
#from color_transforms import YCoCg as YUV #pip install "pip install color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
#import image_denoising
from . import gaussian
import logging # borrame

N_POLY = 7
PYRAMID_LEVELS = 2
WINDOW_SIDE = 7
NUM_ITERS = 3
FLAGS = 0

class Monochrome_Denoising(gaussian.Monochrome_Denoising):

    def __init__(self, logger,
                 pyramid_levels=PYRAMID_LEVELS,
                 window_side=WINDOW_SIDE,
                 N_poly=N_POLY,
                 num_iters=NUM_ITERS,
                 flags=FLAGS):
        super().__init__(logger)
        self.estimator = OF_Estimation(logger)
        self.projector = Projection(logger)
        self.pyramid_levels = pyramid_levels
        self.window_side = window_side
        self.N_poly = N_poly
        self.sigma_poly = (N_poly - 1)/4
        self.num_iters = num_iters
        self.flags = flags
        
        for attr, value in vars(self).items():
            self.logger.info(f"{attr}: {value}")

    def warp_slice(self, image, flow):
        #warped_slice = Projection(
        #    self.logger,
        #    image=slice,
        #    flow=flow,
        #    interpolation_mode=cv2.INTER_LINEAR,
        #    extension_mode=cv2.BORDER_REPLICATE)
        warped_slice = self.projector.remap(
            image=image,
            flow=flow,
            interpolation_mode=cv2.INTER_LINEAR,
            extension_mode=cv2.BORDER_REPLICATE
        )
        return warped_slice

    def get_flow(
        self,
        reference,
        target,
        flow=None):
        flow = cv2.calcOpticalFlowFarneback(
            prev=target,
            next=reference,
            flow=flow,
            pyr_scale=0.5,
            levels=self.pyramid_levels,
            winsize=self.window_side,
            iterations=self.num_iters,
            poly_n=7,
            poly_sigma=self.sigma_poly,
            flags=self.flags)
        #flow[...] = 0.0
        if self.logger.getEffectiveLevel() <= logging.DEBUG:
            print(np.max(np.abs(flow)), end=' ')
        return flow

    def __get_flow(
        self,
        reference,
        target,
        flow=None,
        pyramid_levels=PYRAMID_LEVELS, # Number of pyramid layers
        window_side=WINDOW_SIDE, # Applicability window side
        num_iters=NUM_ITERS, # Number of iterations at each pyramid level
        N_poly=N_POLY, # Standard deviation of the Gaussian basis used in the polynomial expansion
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW | cv2.OPTFLOW_FARNEBACK_GAUSSIAN):
        #flow = np.zeros(shape=(reference.size, 1), dtype=np.float32)
        #return flow
        flow = self.estimator.pyramid_get_flow(
            target, reference,
            flow=flow,
            pyramid_levels=pyramid_levels,
            window_side=window_side,
            num_iters=num_iters,
            N_poly=N_poly,
            flags=flags)
        return flow

    def gray_vertical_OF_gaussian_filtering(self, noisy_image, kernel):
        print("v3")
        KL = kernel.size
        KL2 = KL//2
        w2 = self.window_side//2
        N_rows = noisy_image.shape[0]
        N_cols = noisy_image.shape[1]
    
        # Opción 0: Los márgenes son 128
        #extended_noisy_image = np.full(shape=(noisy_image.shape[0] + KL + w, noisy_image.shape[1] + w, noisy_image.shape[2]), fill_value=128, dtype=np.uint8)
    
        # Opción 1: Usando padding (no terminó de funcionar)
        #extended_noisy_image = np.empty(shape=(noisy_image.shape[0] + KL + w, noisy_image.shape[1] + w, noisy_image.shape[2]), dtype=np.uint8)
        #extended_noisy_image[..., 0] = np.pad(array=noisy_image[..., 0],
        #                              pad_width=(((KL + w)//2, (KL + w)//2), ((w + 1)//2, (w + 1)//2)),
        #                              mode="constant")
        #extended_noisy_image[..., 1] = np.pad(array=noisy_image[..., 1], pad_width=(KL2 + w2, w2), mode="constant")
        #extended_noisy_image[..., 2] = np.pad(array=noisy_image[..., 2], pad_width=(KL2 + w2, w2), mode="constant")
    
        # Opción 2: Los márgenes son la propia imagen, ampliada
        extended_noisy_image = cv2.resize(
            src = noisy_image,
            dsize = (noisy_image.shape[1] + self.window_side,
                     noisy_image.shape[0] + KL + self.window_side))
        #print(extended_noisy_image.shape)
        #extended_noisy_image[KL2 + w2:noisy_image.shape[0] + KL2 + w2, w2:noisy_image.shape[1] + w2] = noisy_image[...]
        #extended_noisy_image[(KL + w)//2 - 1:noisy_image.shape[0] + (KL + w)//2 - 1, w2 - 1:noisy_image.shape[1] + w2 - 1] = noisy_image[...]
        extended_noisy_image[(KL + self.window_side)//2:noisy_image.shape[0] + (KL + self.window_side)//2, w2:noisy_image.shape[1] + w2] = noisy_image[...]
        extended_noisy_image = extended_noisy_image.astype(np.float32)
        extended_Y = extended_noisy_image
        filtered_noisy_image = []
        N_rows = noisy_image.shape[0]
        N_cols = noisy_image.shape[1]
        for y in range(N_rows):
            self.logger.info(f"row={y}")
            #print(y, end=' ')
            horizontal_line = np.zeros(shape=(N_cols + self.window_side), dtype=np.float32)
            target_slice_Y = extended_Y[y + KL2:y + KL2 + self.window_side]
            #print("<", target_slice_Y.shape, ">")
            target_slice = extended_noisy_image[y + KL2:y + KL2 + self.window_side]
            for i in range(KL):
                reference_slice_Y = extended_Y[y + i:y + i + self.window_side]
                reference_slice = extended_noisy_image[y + i:y + i + self.window_side]
                flow = self.get_flow(
                    reference=reference_slice_Y,
                    target=target_slice_Y,
                    flow=None)
                OF_compensated_slice = self.warp_slice(reference_slice, flow)
                OF_compensated_line = OF_compensated_slice[(self.window_side + 1) >> 1, :]
                OF_compensated_line = np.roll(OF_compensated_line, -w2) # Creo que si intercambiamos reference_slice por target_slice, esta línea sobra
                horizontal_line += OF_compensated_line * kernel[i]
            filtered_noisy_image.append(horizontal_line)
        filtered_noisy_image = np.stack(filtered_noisy_image, axis=0)[0:noisy_image.shape[0], 0:noisy_image.shape[1]]
        return filtered_noisy_image

    def __filter_Y_line(self, img, padded_img, kernel, y):
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

    def __filter(self, img, kernel):
        mean = img.mean()
        filtered_img_Y = self.filter_Y(img, kernel[0], mean)
        transposed_filtered_img_Y = np.transpose(filtered_img_Y, (1, 0))
        transposed_filtered_img_YX = self.filter_Y(transposed_filtered_noisy_img_Y, kernel, mean)
        filtered_img_YX = np.transpose(transposed_filtered_img_YX, (1, 0))
        return filtered_img_YX

    def filter(self, noisy_img, kernel, l=3, w=5, w_poly=5, sigma_poly=0.5, flags=0, iterations=3):
        filtered_noisy_img_Y = self.gray_vertical_OF_gaussian_filtering(noisy_img, kernel[0])
        transposed_filtered_noisy_img_Y = np.transpose(filtered_noisy_img_Y, (1, 0))
        transposed_filtered_noisy_img_YX = self.gray_vertical_OF_gaussian_filtering(transposed_filtered_noisy_img_Y, kernel[1])
        filtered_noisy_img_YX = np.transpose(transposed_filtered_noisy_img_YX, (1, 0))
        return filtered_noisy_img_YX



