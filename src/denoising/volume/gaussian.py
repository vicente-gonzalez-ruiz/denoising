'''Gaussian volume denoising.'''

import threading
import time
import numpy as np

class Monochrome_Denoising:

    def __init__(
            self,
            logger,
            quality_estimator=None
    ):
        self.logger = logger
        self.logger = logger
        self.Q_estimator = quality_estimator
        self.show_event = threading.Event()
        self.logger_daemon = threading.Thread(target=self.show_log)
        self.logger_daemon.daemon = True
        self.time_0 = time.perf_counter()
        self.logger_daemon.start()

    def show_log(self):
        while self.show_event.wait():
            time_1 = time.perf_counter()
            running_time = time_1 - self.time_0
            print(f"{self.iter:>5d}/{self.dim}", end='')
            print(f"{running_time:>16.3f}", end='')
            if self.Q_estimator != None:
                print(f"{self.quality_index:>16.4f}", end='')
            print()
            self.show_event.clear()
            self.time_0 = time.perf_counter()

    def warp_slice(self, slice, flow):
        return slice

    def get_flow(self, reference, target, prev_flow=None, l=0, w=0):
        return prev_flow

    def filter_Z(self, vol, kernel, mean, l=0, w=0):
        assert kernel.size % 2 != 0 # kernel.size must be odd
        filtered_vol = np.zeros_like(vol).astype(np.float32)
        shape_of_vol = np.shape(vol)
        padded_vol = np.full(shape=(shape_of_vol[0] + kernel.size, shape_of_vol[1], shape_of_vol[2]), fill_value=mean)
        padded_vol[kernel.size//2:shape_of_vol[0] + kernel.size//2, :, :] = vol
        Z_dim = vol.shape[0]
        self.dim = Z_dim
        for z in range(Z_dim):
            self.iter = z
            self.show_event.set()
            tmp_slice = np.zeros_like(vol[z]).astype(np.float32)
            prev_flow = np.zeros(shape=(shape_of_vol[1], shape_of_vol[2], 2), dtype=np.float32)
            for i in range((kernel.size//2) - 1, -1, -1):
                flow = self.get_flow(padded_vol[z + i, :, :], vol[z, :, :], prev_flow, l, w)
                self.logger.debug(f"{np.average(np.abs(flow))}")
                prev_flow = flow
                warped_slice = self.warp_slice(padded_vol[z + i, :, :], flow)
                tmp_slice += warped_slice * kernel[i]
            tmp_slice += vol[z, :, :] * kernel[kernel.size//2]
            prev_flow = np.zeros(shape=(shape_of_vol[1], shape_of_vol[2], 2), dtype=np.float32)
            for i in range(kernel.size//2+1, kernel.size):
                flow = self.get_flow(padded_vol[z + i, :, :], vol[z, :, :], prev_flow, l, w)
                self.logger.debug(f"{np.average(np.abs(flow))}")
                prev_flow = flow
                warped_slice = self.warp_slice(padded_vol[z + i, :, :], flow)
                tmp_slice += warped_slice * kernel[i]
            filtered_vol[z, :, :] = tmp_slice
        return filtered_vol

    def filter_Y(self, vol, kernel, mean, l=0, w=0):
        assert kernel.size % 2 != 0 # kernel.size must be odd
        filtered_vol = np.zeros_like(vol).astype(np.float32)
        shape_of_vol = np.shape(vol)
        padded_vol = np.full(shape=(shape_of_vol[0], shape_of_vol[1] + kernel.size, shape_of_vol[2]), fill_value=mean)
        padded_vol[:, kernel.size//2:shape_of_vol[1] + kernel.size//2, :] = vol
        Y_dim = vol.shape[1]
        self.dim = Y_dim
        for y in range(Y_dim):
            self.iter = y
            self.show_event.set()
            tmp_slice = np.zeros_like(vol[:, y, :]).astype(np.float32)
            prev_flow = np.zeros(shape=(shape_of_vol[0], shape_of_vol[2], 2), dtype=np.float32)
            for i in range((kernel.size//2) - 1, -1, -1):
                flow = self.get_flow(padded_vol[:, y + i, :], vol[:, y, :], prev_flow, l, w)
                self.logger.debug(f"{np.average(np.abs(flow))}")
                prev_flow = flow                     
                warped_slice = self.warp_slice(padded_vol[:, y + i, :], flow)
                tmp_slice += warped_slice * kernel[i]
            tmp_slice += vol[:, y, :] * kernel[kernel.size//2]
            prev_flow = np.zeros(shape=(shape_of_vol[0], shape_of_vol[2], 2), dtype=np.float32)
            for i in range(kernel.size//2+1, kernel.size):
                flow = self.get_flow(padded_vol[:, y + i, :], vol[:, y, :], prev_flow, l, w)
                self.logger.debug(f"{np.average(np.abs(flow))}")
                prev_flow = flow                      
                warped_slice = self.warp_slice(padded_vol[:, y + i, :], flow)
                tmp_slice += warped_slice * kernel[i]
            filtered_vol[:, y, :] = tmp_slice
        return filtered_vol

    def filter_X(self, vol, kernel, mean, l=0, w=0):
        assert kernel.size % 2 != 0 # kernel.size must be odd
        filtered_vol = np.zeros_like(vol).astype(np.float32)
        shape_of_vol = np.shape(vol)
        padded_vol = np.full(shape=(shape_of_vol[0], shape_of_vol[1], shape_of_vol[2] + kernel.size), fill_value=mean)
        padded_vol[:, :, kernel.size//2:shape_of_vol[2] + kernel.size//2] = vol
        X_dim = vol.shape[2]
        self.dim = X_dim
        for x in range(X_dim):
            self.iter = x
            self.show_event.set()
            tmp_slice = np.zeros_like(vol[:, :, x]).astype(np.float32)
            prev_flow = np.zeros(shape=(shape_of_vol[0], shape_of_vol[1], 2), dtype=np.float32)
            for i in range((kernel.size//2) - 1, -1, -1):
                flow = self.get_flow(padded_vol[:, :, x + i], vol[:, :, x], prev_flow, l, w)
                self.logger.debug(f"{np.average(np.abs(flow))}")
                prev_flow = flow
                warped_slice = self.warp_slice(padded_vol[:, :, x + i], flow)
                tmp_slice += warped_slice * kernel[i]
            tmp_slice += vol[:, :, x] * kernel[kernel.size//2]
            prev_flow = np.zeros(shape=(shape_of_vol[0], shape_of_vol[1], 2), dtype=np.float32)
            for i in range(kernel.size//2+1, kernel.size):
                flow = self.get_flow(padded_vol[:, :, x + i], vol[:, :, x], prev_flow, l, w)
                self.logger.debug(f"{np.average(np.abs(flow))}")
                prev_flow = flow
                warped_slice = self.warp_slice(padded_vol[:, :, x + i], flow)
                tmp_slice += warped_slice * kernel[i]
            filtered_vol[:, :, x] = tmp_slice
        return filtered_vol

    def filter(self, vol, kernel, l=0, w=0):
        mean = vol.mean()
        self.logger.info(f"mean={mean}")
        filtered_vol_Z = self.filter_Z(vol, kernel[0], mean, l, w)
        self.logger.info(f"filtered along Z")
        filtered_vol_ZY = self.filter_Y(filtered_vol_Z, kernel[1], mean, l, w)
        self.logger.info(f"filtered along Y")
        filtered_vol_ZYX = self.filter_X(filtered_vol_ZY, kernel[2], mean, l, w)
        self.logger.info(f"filtered along X")
        return filtered_vol_ZYX

