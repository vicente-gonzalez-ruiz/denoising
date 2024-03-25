'''Gaussian volume denoising using optical flow (OpenCV version).'''

import numpy as np
import cv2
from . import gaussian

SIGMA_POLY = 1.0
N_POLY = 5
PYRAMID_LEVELS = 1
WINDOW_SIDE = 5
NUM_ITERS = 3

class Monochrome_Denoising(gaussian.Monochrome_Denoising):

    def __init__(self, logger, pyramid_levels=PYRAMID_LEVELS, window_side=WINDOW_SIDE, sigma_poly=SIGMA_POLY, N_poly=N_POLY, num_iters=NUM_ITERS):
        super().__init__(logger)
        self.pyramid_levels = pyramid_levels
        self.logger.info(f"l={self.pyramid_levels}")
        self.window_side = window_side
        self.logger.info(f"lw{self.window_side}")
        self.sigma_poly = sigma_poly
        self.logger.info(f"sigma_poly={self.sigma_poly}")
        self.num_iters = num_iters
        self.logger.info(f"num_iters={self.num_iters}")
        self.N_poly = N_poly
        self.logger.info(f"N_poly={self.N_poly}")

    def warp_slice(self, slice, flow):
        height, width = flow.shape[:2]
        map_x = np.tile(np.arange(width), (height, 1))
        map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
        map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
        warped_slice = cv2.remap(slice, map_xy, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped_slice

    def get_flow(self, reference, target, prev_flow, pyramid_levels, window_side):
        flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=prev_flow, pyr_scale=0.5, levels=self.pyramid_levels, winsize=self.window_side, iterations=self.num_iters, poly_n=self.N_poly, poly_sigma=self.sigma_poly, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        return flow

