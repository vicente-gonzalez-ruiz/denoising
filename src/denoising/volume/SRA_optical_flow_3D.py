'''Volume denoising using Suffle, Register, and Average (SRaA)'''
#'''Optical Flow-based Random Shaking Iterative Volume Denoising.'''
#'''Optical Flow-Compensated Random-Shaking Iterative Volume Denoising (RandomDenoising).'''

import threading
import time
import numpy as np
#from motion_estimation._3D.farneback_opticalflow3d import OF_Estimation # pip install "motion_estimation @ git+https://github.com/vicente-gonzalez-ruiz/motion_estimation"
#from motion_estimation._3D.project_opticalflow3d import Projection
#import information_theory
#from matplotlib import pyplot as plt
import logging
import inspect

N_ITERS = 25
STD_DEV = 1.5

GAUSS_SIZE = 9    # Side of the Gaussian applicability window used
                    # during the polynomial expansion. Applicability (that is, the relative importance of the points in the neighborhood) size should match the scale of the structures we wnat to estimate orientation for (page 77). However, small applicabilities are more sensitive to noise.
SIGMA_K = 0.15      # Scaling factor used to calculate the standard
                    # deviation of the Gaussian applicability. The
                    # formula to calculate the standard deviation is
                    # sigma = sigma_k*(gauss_size - 1).

# OF estimation
FILTER_TYPE = "box" # Shape of the filer used to average the flow. It
                    # can be "box" or "gaussian".
FILTER_SIZE = 21    # Size of the filter used to average the G and
                    # matrices (see Eqs. 4.7 and 4.27 of the thesis).
PYRAMID_LEVELS = 3  # Number of pyramid layers
ITERATIONS = 5      # Number of iterations at each pyramid level
PYRAMID_SCALE = 0.5

class Shuffle_Register_and_Average:
    def __init__(
        self,
        OF_estimator,
        projector,
        logger,
        quality_estimator,
        show_image=False
    ):

        self.estimator = OF_estimator
        self.projector = projector
        self.logger = logger
        self.Q_estimator = quality_estimator
        self.show_image = show_image
        self.quality_index = 0.0

        if self.logger.level <= logging.INFO:
            self.max = 0
            self.min = 0

        print(f"{'iter':>5s}", end='')
        print(f"{'min':>16s}", end='')
        print(f"{'avg_abs':>16s}", end='')
        print(f"{'max':>16s}", end='')
        print(f"{'min_flow':>16s}", end='')
        print(f"{'avg_abs_flow':>16s}", end='')
        print(f"{'max_flow':>16s}", end='')
        print(f"{'time':>16s}", end='')
        if self.Q_estimator != None:
            print(f"{'quality_index':>16s}", end='')
        print()

        self.stop_event = threading.Event()
        self.logger_daemon = threading.Thread(target=self.show_log)
        self.logger_daemon.daemon = True
        self.time_0 = time.perf_counter()
        self.logger_daemon.start()

    def show_log(self):
        while self.stop_event.wait():
            time_1 = time.perf_counter()
            running_time = time_1 - self.time_0
            print(f"{self.iter:>5d}", end='')
            print(f"{np.min(self.displacements):>16.2f}", end='')
            print(f"{np.average(np.abs(self.displacements)):>16.2f}", end='')
            print(f"{np.max(self.displacements):>16.2f}", end='')
            print(f"{np.min(self.flow):>16.2f}", end='')
            print(f"{np.average(np.abs(self.flow)):>16.2f}", end='')
            print(f"{np.max(self.flow):>16.2f}", end='')
            print(f"{running_time:>16.2f}", end='')
            if self.Q_estimator != None:
                print(f"{self.quality_index:>16.4f}", end='')
            print()
            self.stop_event.clear()
            self.time_0 = time.perf_counter()

    def shuffle_vector(self, x, mean=0.0, std_dev=1.0):
        y = np.arange(len(x))
        self.displacements = np.random.normal(mean, std_dev, len(x))
        return np.stack((y + self.displacements, x), axis=1)

    def shuffle_volume(self, volume, mean=0.0, std_dev=1.0):
        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
        if self.logger.level < logging.INFO:
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        shuffled_volume = np.empty_like(volume)

        # Shuffling in Z
        values = np.arange(volume.shape[0]).astype(np.int16)
        for y in range(volume.shape[1]):
            for x in range(volume.shape[2]):
                pairs = self.shuffle_vector(x=values, mean=mean, std_dev=std_dev).astype(np.int16)
                pairs = pairs[pairs[:, 0].argsort()]
                shuffled_volume[values, y, x] = volume[pairs[:, 1], y , x]
        volume = shuffled_volume
    
        # Shuffling in Y
        values = np.arange(volume.shape[1]).astype(np.int16)
        for z in range(volume.shape[0]):
            for x in range(volume.shape[2]):
                pairs = self.shuffle_vector(x=values, mean=mean, std_dev=std_dev).astype(np.int16)
                pairs = pairs[pairs[:, 0].argsort()]
                shuffled_volume[z, values, x] = volume[z, pairs[:, 1], x]
        volume = shuffled_volume

        # Shuffing in X
        values = np.arange(volume.shape[2]).astype(np.int16)
        for z in range(volume.shape[0]):
            for y in range(volume.shape[1]):
                pairs = self.shuffle_vector(x=values, mean=mean, std_dev=std_dev).astype(np.int16)
                pairs = pairs[pairs[:, 0].argsort()]
                shuffled_volume[z, y, values] = volume[z, y, pairs[:, 1]]
                
        return shuffled_volume

    def project_volume_reference_to_target(
        self,
        reference,
        target,
        pyramid_levels,
        gauss_size,
        iterations,
        sigma_k,
        filter_type,
        filter_size,
        presmoothing
    ):
        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
        if self.logger.level < logging.INFO:
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        self.flow = self.estimator.pyramid_get_flow(
            target=target,
            reference=reference,
            pyramid_levels=pyramid_levels,
            gauss_size=gauss_size,
            iterations=iterations,
            sigma_k=sigma_k,
            filter_type=filter_type,
            filter_size=filter_size,
            presmoothing=presmoothing)
        projection = self.projector.remap(volume=reference, flow=self.flow)
        return projection

    def filter(
        self,
        noisy_volume,
        N_iters=25,
        mean=0.0,
        std_dev=1.0,
        pyramid_levels=PYRAMID_LEVELS,
        gauss_size=GAUSS_SIZE,
        iterations=ITERATIONS,
        sigma_k=SIGMA_K,
        filter_type=FILTER_TYPE,
        filter_size=FILTER_SIZE,
        presmoothing=None
    ):
        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
        if self.logger.level < logging.INFO:
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        acc_volume = np.zeros_like(noisy_volume, dtype=np.float32)
        acc_volume[...] = noisy_volume
        for i in range(N_iters):
            self.iter = i
            denoised_volume = acc_volume/(i+1)
            shuffled_noisy_volume = self.shuffle_volume(noisy_volume, mean=mean, std_dev=std_dev)
            shuffled_and_compensated_noisy_volume = self.project_volume_reference_to_target(
                reference=denoised_volume,
                target=shuffled_noisy_volume,
                pyramid_levels=pyramid_levels,
                gauss_size=gauss_size,
                iterations=iterations,
                sigma_k=sigma_k,
                filter_type=filter_type,
                filter_size=filter_size,
                presmoothing=presmoothing)
            acc_volume += shuffled_and_compensated_noisy_volume

            if self.Q_estimator != None:
                denoised = acc_volume/(i + 2)
                self.quality_index = self.Q_estimator(noisy_volume, denoised)
                title = f"iter={i+1} DQI={self.quality_index:6.5f} min={np.min(denoised):5.2f} max={np.max(denoised):5.2f} avg={np.average(denoised):5.2f}"
            else:
                title = ''

            if self.show_image:
                self.show_image(denoised, title)

            self.stop_event.set()
        denoised_volume = acc_volume/(N_iters + 1)

        return denoised_volume

from motion_estimation._3D.farneback_optical_flow_3D import OF_Estimation # pip install "motion_estimation @ git+https://github.com/vicente-gonzalez-ruiz/motion_estimation"
from motion_estimation._3D.project_optical_flow_3D import Project
import information_theory
from matplotlib import pyplot as plt

class Registered_Shuffling_Means(OF_Estimation, Project):
#class Random_Shaking_Denoising:
    def __init__(
        self,
        logging_level=logging.INFO,
        block_size=(256, 256, 256),
        overlap=(8, 8, 8),
        threads_per_block=(8, 8, 8),
        use_gpu=True,
        device_id=0,
        show_image=False
        #estimator="opticalflow3d"
    ):
        #self.estimator = estimator
        #OF_Estimation.__init__(self, logging_level)
        #self.logger = logging.getLogger(__name__)
        #self.logger.setLevel(logging_level)
        self.logger.level = logging_level

        #if self.logging_level <= logging.INFO:
        #    print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
        #    '''
        #    args, _, _, values = inspect.getargvalues(inspect.currentframe())
        #    for arg in args:
        #        if isinstance(values[arg], np.ndarray):
        #            print(f"{arg}.shape: {values[arg].shape}", end=' ')
        #            print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
        #        else:
        #            print(f"{arg}: {values[arg]}")
        #    '''

        OF_Estimation.__init__(
            self,
            logging_level = logging_level,
            block_size = block_size,
            overlap = overlap,
            threads_per_block = threads_per_block,
            use_gpu = use_gpu,
            device_id = device_id)

        Project.__init__(self, logging_level)

        self.show_image = show_image
        self.get_quality = get_quality
        self.quality_index = 0.0

        if self.logger.level <= logging.INFO:
            self.max = 0
            self.min = 0
        print(f"{'iter':>5s}", end='')
        print(f"{'min_shuffling':>16s}", end='')
        print(f"{'avg_abs_shuffling':>16s}", end='')
        print(f"{'max_shuffling':>16s}", end='')
        print(f"{'min_flow':>16s}", end='')
        print(f"{'avg_abs_flow':>16s}", end='')
        print(f"{'max_flow':>16s}", end='')
        print(f"{'time':>16s}", end='')
        if get_quality!=None:
            print(f"{'quality_index':>16s}", end='')
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
            print(f"{np.min(self.displacements):>16.2f}", end='')
            print(f"{np.average(np.abs(self.displacements)):>16.2f}", end='')
            print(f"{np.max(self.displacements):>16.2f}", end='')
            print(f"{np.min(self.flow):>16.2f}", end='')
            print(f"{np.average(np.abs(self.flow)):>16.2f}", end='')
            print(f"{np.max(self.flow):>16.2f}", end='')
            print(f"{running_time:>16.2f}", end='')
            if self.get_quality!=None:
                print(f"{self.quality_index:>16.4f}", end='')
            print()
            self.stop_event.clear()
            self.time_0 = time.perf_counter()

    def shuffling_vector(self, x, mean=0.0, std_dev=1.0):
        y = np.arange(len(x))
        self.displacements = np.random.normal(mean, std_dev, len(x))
        return np.stack((y + self.displacements, x), axis=1)

    def shuffle_volume(self, volume, mean=0.0, std_dev=1.0):

        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
        if self.logger.level < logging.INFO:
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        shuffled_volume = np.empty_like(volume)

        # Shuffling in Z
        values = np.arange(volume.shape[0]).astype(np.int16)
        for y in range(volume.shape[1]):
            for x in range(volume.shape[2]):
                pairs = self.shuffle_vector(x=values, mean=mean, std_dev=std_dev).astype(np.int16)
                pairs = pairs[pairs[:, 0].argsort()]
                shuffled_volume[values, y, x] = volume[pairs[:, 1], y , x]
        volume = shuffled_volume
    
        # Shuffling in Y
        values = np.arange(volume.shape[1]).astype(np.int16)
        for z in range(volume.shape[0]):
            for x in range(volume.shape[2]):
                pairs = self.shuffle_vector(x=values, mean=mean, std_dev=std_dev).astype(np.int16)
                pairs = pairs[pairs[:, 0].argsort()]
                shuffled_volume[z, values, x] = volume[z, pairs[:, 1], x]
        volume = shuffled_volume

        # Shuffing in X
        values = np.arange(volume.shape[2]).astype(np.int16)
        for z in range(volume.shape[0]):
            for y in range(volume.shape[1]):
                pairs = self.shuffle_vector(x=values, mean=mean, std_dev=std_dev).astype(np.int16)
                pairs = pairs[pairs[:, 0].argsort()]
                shuffled_volume[z, y, values] = volume[z, y, pairs[:, 1]]
                
        return shuffled_volume

    def project_volume_reference_to_target(self, reference, target, pyramid_levels, gauss_size, iterations, sigma_k, filter_type, filter_size, presmoothing):

        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
        if self.logger._level < logging.INFO:
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        self.flow = self.pyramid_get_flow(
            target=target,
            reference=reference,
            pyramid_levels=pyramid_levels,
            gauss_size=gauss_size,
            iterations=iterations,
            sigma_k=sigma_k,
            filter_type=filter_type,
            filter_size=filter_size,
            presmoothing=presmoothing)
        projection = self.remap(volume=reference, flow=self.flow, use_gpu=self.use_gpu)
        return projection

    def project_volume_reference_to_target_old(
        self,
        reference,
        target,
        OF_estimator,
        projector
    ):

        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
        if self.logger.level < logging.INFO:
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        self.flow = OF_estimator.pyramid_get_flow(
            target=target,
            reference=reference,
            OF_estimator = OF_estimator)
        projection = projector.remap(volume=reference, flow=self.flow, use_gpu=use_gpu)
        return projection

    def filter_volume_old(
        self,
        noisy_volume,
        OF_estimator,
        projector,
        N_iters=N_ITERS,
        mean=0.0,
        std_dev=STD_DEV
    ):
        
        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
        if self.logger.level < logging.INFO:
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}") 

        #total_vol=(noisy_volume.shape[0], noisy_volume.shape[1], noisy_volume.shape[2]),

        acc_volume = np.zeros_like(noisy_volume, dtype=np.float32)
        acc_volume[...] = noisy_volume
        for i in range(N_iters):
            self.iter = i
            denoised_volume = acc_volume/(i+1)
            shuffled_noisy_volume = self.shuffle_volume(noisy_volume, mean=mean, std_dev=std_dev)
            shaffled_and_compensated_noisy_volume = self.project_volume_reference_to_target(
                reference=denoised_volume,
                target=shuffled_noisy_volume,
                OF_estimator=OF_estimator,
                projector=projector)
            acc_volume += shuffled_and_compensated_noisy_volume

            if self.quality_index != None:
                denoised = acc_volume/(i + 2)
                self.quality_index = self.get_quality(noisy_volume, denoised)
                title = f"iter={i+1} DQI={self.quality_index:6.5f} min={np.min(denoised):5.2f} max={np.max(denoised):5.2f} avg={np.average(denoised):5.2f}"
            else:
                title = ''
            if self.show_image:
                self.show_image(denoised, title)

            self.stop_event.set()
        denoised_volume = acc_volume/(N_iters + 1)

        return denoised_volume

    def filter_volume(
        self,
        noisy_volume,
        N_iters=25,
        mean=0.0,
        std_dev=1.0,
        pyramid_levels=PYRAMID_LEVELS,
        gauss_size=GAUSS_SIZE,
        iterations=ITERATIONS,
        sigma_k=SIGMA_K,
        filter_type=FILTER_TYPE,
        filter_size=FILTER_SIZE,
        presmoothing=None
    ):

        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
        if self.logger.level < logging.INFO:
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        acc_volume = np.zeros_like(noisy_volume, dtype=np.float32)
        acc_volume[...] = noisy_volume
        for i in range(N_iters):
            self.iter = i
            denoised_volume = acc_volume/(i+1)
            shuffled_noisy_volume = self.shuffle_volume(noisy_volume, mean=mean, std_dev=std_dev)
            shuffled_and_compensated_noisy_volume = self.project_volume_reference_to_target(
                reference=denoised_volume,
                target=shuffled_noisy_volume,
                pyramid_levels=pyramid_levels,
                gauss_size=gauss_size,
                iterations=iterations,
                sigma_k=sigma_k,
                filter_type=filter_type,
                filter_size=filter_size,
                presmoothing=presmoothing)
            acc_volume += shuffled_and_compensated_noisy_volume

            if self.quality_index != None:
                denoised = acc_volume/(i + 2)
                self.quality_index = self.get_quality(noisy_volume, denoised)
                title = f"iter={i+1} DQI={self.quality_index:6.5f} min={np.min(denoised):5.2f} max={np.max(denoised):5.2f} avg={np.average(denoised):5.2f}"
            else:
                title = ''
            if self.show_image:
                self.show_image(denoised, title)

            self.stop_event.set()
        denoised_volume = acc_volume/(N_iters + 1)

        return denoised_volume

from motion_estimation._2D.farneback_OpenCV import OF_Estimation as _2D_OF_Estimation 
from motion_estimation._2D.project import Projection as Slice_Projection
import cv2

class Random_Shuffling_Means_by_Slices(Shuffle_Register_and_Average, _2D_OF_Estimation, Slice_Projection):
    def __init__(
        self,
        logging_level=logging.INFO
    ):
        Random_Shuffing_Denoising.__init__(self, logging_level)
        _2D_OF_Estimation.__init__(self, logging_level)
        Slice_Projection.__init__(self, logging_level)

    def shuffle_slice(self, slc, mean=0.0, std_dev=1.0):
        shuffled_slice = np.empty_like(slc)
    
        # Shuffling in Y
        values = np.arange(slc.shape[0]).astype(np.int16)
        for x in range(slc.shape[1]):
            pairs = self.shuffle_vector(x=values, mean=mean, std_dev=std_dev).astype(np.int16)
            pairs = pairs[pairs[:, 0].argsort()]
            shuffled_slice[values, x] = slc[pairs[:, 1], x]
        slc = shuffled_slice
    
        # Shuffling in X
        values = np.arange(slc.shape[1]).astype(np.int16)
        for y in range(slc.shape[0]):
            pairs = self.shuffle_vector(x=values, mean=mean, std_dev=std_dev).astype(np.int16)
            pairs = pairs[pairs[:, 0].argsort()]
            shuffled_slice[y, values] = slc[y, pairs[:, 1]]
    
        return shuffled_slice

    def project_slice_reference_to_target(self, reference, target, pyramid_levels, window_side, iterations, N_poly, interpolation_mode, extension_mode):
        self.flow = _2D_OF_Estimation.pyramid_get_flow(self,
            target=target,
            reference=reference,
            flow=None,
            pyramid_levels=pyramid_levels,
            window_side=window_side,
            iterations=iterations,
            N_poly=N_poly)
        projection = Slice_Projection.remap(self, reference, self.flow, interpolation_mode, extension_mode)
        return projection

    def filter_slice(self, noisy_slice, denoised_slice, mean, std_dev, pyramid_levels, window_side, iterations, N_poly, interpolation_mode, extension_mode):
        shuffled_noisy_slice = self.shuffle_slice(slc=noisy_slice, mean=mean, std_dev=std_dev)
        
        shuffled_and_compensated_noisy_slice = self.project_slice_reference_to_target(
            reference=denoised_slice,
            target=shuffled_noisy_slice,
            pyramid_levels=pyramid_levels,
            window_side=window_side,
            iterations=iterations,
            N_poly=N_poly,
            interpolation_mode=interpolation_mode,
            extension_mode=extension_mode)
        return shuffled_and_compensated_noisy_slice

    def filter_volume(
        self,
        noisy_vol,
        N_iters=25,
        mean=0.0,
        std_dev=1.0,
        pyramid_levels=PYRAMID_LEVELS,
        window_side=5,
        iterations=ITERATIONS,
        N_poly=1.2,
        interpolation_mode=cv2.INTER_LINEAR,
        extension_mode=cv2.BORDER_REPLICATE
    ):
        acc_vol = np.zeros_like(noisy_vol, dtype=np.float32)
        acc_vol[...] = noisy_vol
        for i in range(N_iters):
            self.iter = i
            denoised_vol = acc_vol/(i+1)

            for z in range(noisy_vol.shape[0]):
                acc_vol[z, :, :] += self.filter_slice(
                    noisy_slice=noisy_vol[z, :, :],
                    denoised_slice=denoised_vol[z, :, :],
                    mean=mean,
                    std_dev=std_dev,
                    pyramid_levels=pyramid_levels,
                    window_side=window_side,
                    iterations=iterations,
                    N_poly=N_poly,
                    interpolation_mode=interpolation_mode,
                    extension_mode=extension_mode)
            self.stop_event.set()

            for y in range(noisy_vol.shape[1]):
                acc_vol[:, y, :] += self.filter_slice(
                    noisy_slice=noisy_vol[:, y, :],
                    denoised_slice=denoised_vol[:, y, :],
                    mean=mean,
                    std_dev=std_dev,
                    pyramid_levels=pyramid_levels,
                    window_side=window_side,
                    iterations=iterations,
                    N_poly=N_poly,
                    interpolation_mode=interpolation_mode,
                    extension_mode=extension_mode)
            self.stop_event.set()

            for x in range(noisy_vol.shape[2]):
                acc_vol[:, :, x] += self.filter_slice(
                    noisy_slice=noisy_vol[:, :, x],
                    denoised_slice=denoised_vol[:, :, x],
                    mean=mean,
                    std_dev=std_dev,
                    pyramid_levels=pyramid_levels,
                    window_side=window_side,
                    iterations=iterations,
                    N_poly=N_poly,
                    interpolation_mode=interpolation_mode,
                    extension_mode=extension_mode)
            self.stop_event.set()

        denoised_vol = acc_vol/(N_iters + 1)
        return denoised_vol
