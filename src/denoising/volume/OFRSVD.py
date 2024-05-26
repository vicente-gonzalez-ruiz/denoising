'''Optical Flow-based Random Shaking Iterative Volume Denoising.'''

import logging
import threading
import time
import numpy as np
# pip install "motion_estimation @ git+https://github.com/vicente-gonzalez-ruiz/motion_estimation"
from motion_estimation._3D.farneback_opticalflow3d import Farneback_Estimator as OF_Estimation 
from motion_estimation._3D.project_opticalflow3d import Volume_Projection

PYRAMID_LEVELS = 3
WINDOW_SIDE = 5
ITERATIONS = 5
N_POLY = 11

class Random_Shaking_Denoising(OF_Estimation, Volume_Projection):
    def __init__(
        self,
        logging_level=logging.INFO
    ):
        OF_Estimation.__init__(self, logging_level)
        Volume_Projection.__init__(self, logging_level)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

        if self.logger.getEffectiveLevel() <= logging.INFO:
            self.max = 0
            self.min = 0
        print(f"{'iter':>5s}", end='')
        print(f"{'min_shaking':>15s}", end='')
        print(f"{'max_shaking':>15s}", end='')
        print(f"{'min_flow':>15s}", end='')
        print(f"{'avg_abs_flow':>15s}", end='')
        print(f"{'max_flow':>15s}", end='')
        print(f"{'time':>15s}", end='')
        print()

        self.stop_event = threading.Event()
        self.logger_daemon = threading.Thread(target=self.show_log)
        self.logger_daemon.daemon = True
        self.time_0 = time.perf_counter()
        self.logger_daemon.start()

    def show_log(self):
        #while not self.stop_event.is_set():
        while self.stop_event.wait():
            time_1 = time.perf_counter()
            running_time = time_1 - self.time_0
            print(f"{self.iter:>5d}", end='')
            print(f"{np.min(self.displacements):>15.2f}", end='')
            print(f"{np.max(self.displacements):>15.2f}", end='')
            print(f"{np.min(self.flow):>15.2f}", end='')
            print(f"{np.average(np.abs(self.flow)):>15.2f}", end='')
            print(f"{np.max(self.flow):>15.2f}", end='')
            print(f"{running_time:>15.2f}", end='')
            print()
            self.stop_event.clear()
            self.time_0 = time.perf_counter()

    def shake_vector(self, x, mean=0.0, std_dev=1.0):
        y = np.arange(len(x))
        self.displacements = np.random.normal(mean, std_dev, len(x))
        return np.stack((y + self.displacements, x), axis=1)

    def shake_volume(self, volume, mean=0.0, std_dev=1.0):
        shaked_volume = np.empty_like(volume)

        # Shaking in Z
        values = np.arange(volume.shape[0]).astype(np.int16)
        for y in range(volume.shape[1]):
            for x in range(volume.shape[2]):
                pairs = self.shake_vector(x=values, mean=mean, std_dev=std_dev).astype(np.int16)
                pairs = pairs[pairs[:, 0].argsort()]
                shaked_volume[values, y, x] = volume[pairs[:, 1], y , x]
        volume = shaked_volume
    
        # Shaking in Y
        values = np.arange(volume.shape[1]).astype(np.int16)
        for z in range(volume.shape[0]):
            for x in range(volume.shape[2]):
                pairs = self.shake_vector(values, mean=mean, std_dev=std_dev).astype(np.int16)
                pairs = pairs[pairs[:, 0].argsort()]
                shaked_volume[z, values, x] = volume[z, pairs[:, 1], x]
        volume = shaked_volume

        # Shaking in X
        values = np.arange(volume.shape[2]).astype(np.int16)
        for z in range(volume.shape[0]):
            for y in range(volume.shape[1]):
                pairs = self.shake_vector(values, mean=mean, std_dev=std_dev).astype(np.int16)
                pairs = pairs[pairs[:, 0].argsort()]
                shaked_volume[z, y, values] = volume[z, y, pairs[:, 1]]
                
        return shaked_volume

    def project_volume_reference_to_target(self, reference, target, pyramid_levels, window_side, iterations, N_poly, block_size, overlap, threads_per_block):
        self.flow = self.pyramid_get_flow(
            target=target,
            reference=reference,
            flow=None,
            pyramid_levels=pyramid_levels,
            window_side=window_side,
            iterations=iterations,
            N_poly=N_poly,
            block_size=block_size,
            overlap=overlap,
            threads_per_block=threads_per_block)
        projection = self.remap(reference, self.flow)
        return projection

    def filter_volume(
        self,
        noisy_volume,
        N_iters=25,
        mean=0.0,
        std_dev=1.0,
        pyramid_levels=PYRAMID_LEVELS,
        window_side=WINDOW_SIDE,
        iterations=ITERATIONS,
        N_poly=N_POLY,
        block_size=(256, 256, 256),
        overlap=(64, 64, 64),
        threads_per_block=(8, 8, 8)
    ):
        acc_volume = np.zeros_like(noisy_volume, dtype=np.float32)
        acc_volume[...] = noisy_volume
        for i in range(N_iters):
            self.iter = i
            denoised_volume = acc_volume/(i+1)
            shaked_noisy_volume = self.shake_volume(noisy_volume, mean=mean, std_dev=std_dev)
            shaked_and_compensated_noisy_volume = self.project_volume_reference_to_target(
                reference=denoised_volume,
                target=shaked_noisy_volume,
                pyramid_levels=pyramid_levels,
                window_side=window_side,
                iterations=iterations,
                N_poly=N_poly,
                block_size=block_size,
                overlap=overlap,
                threads_per_block=threads_per_block)
            acc_volume += shaked_and_compensated_noisy_volume
            self.stop_event.set()
        denoised_volume = acc_volume/(N_iters + 1)

        return denoised_volume

