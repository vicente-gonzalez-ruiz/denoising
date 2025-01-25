'''Gaussian image denoising using optical flow (Python version).'''

#import time
import numpy as np
import cv2
#from . import kernels
#from . import flow_estimation
#pip install "motion_estimation @ git+https://github.com/vicente-gonzalez-ruiz/motion_estimation"
from motion_estimation._1D.farneback_python import OF_Estimation
from motion_estimation._1D.project import Projection
#from color_transforms import YCoCg as YUV #pip install "pip install color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
#import image_denoising
from . import gaussian
import logging # borrame

from numpy.linalg import LinAlgError

N_POLY = 17
PYRAMID_LEVELS = 2
WINDOW_LENGTH = 17
NUM_ITERS = 3
MODEL="constant"
MU=0

class Monochrome_Denoising(gaussian.Monochrome_Denoising):

    def __init__(self, logger,
                 pyramid_levels=PYRAMID_LEVELS,
                 window_length=WINDOW_LENGTH,
                 N_poly=N_POLY,
                 num_iters=NUM_ITERS,
                 model=MODEL,
                 mu=MU):
        super().__init__(logger)
        self.estimator = OF_Estimation(logger)
        self.singular_matrices_found = 0
        self.pyramid_levels = pyramid_levels
        self.window_length = window_length
        self.N_poly = N_poly
        self.num_iters = num_iters
        self.model = model
        self.mu = mu
        for attr, value in vars(self).items():
            self.logger.info(f"{attr}: {value}")

    def warp_line(self, line, flow):
        length = flow.shape[0]
        warped_line = Projection.remap(self.logger, signal=line, flow=np.squeeze(flow))
        return warped_line

    def get_flow(self, reference, target, flow):
        #flow = np.zeros(shape=(reference.size, 1), dtype=np.float32)
        #return flow
        try: 
            flow = self.estimator.pyramid_get_flow(
                target, reference,
                N_poly=self.N_poly,
                window_length=self.window_length,
                iterations=self.num_iters,
                pyramid_levels=self.pyramid_levels,
                flow=flow,
                model=self.model,
                mu=self.mu)
        except LinAlgError as e:
            flow = np.zeros(shape=(reference.size, 1), dtype=np.float32)
            print(f"Caught exception: {e} counter={self.singular_matrices_found} ")
            self.singular_matrices_found += 1
        return flow

    def filter_Y_line(self, img, padded_img, kernel, y):
        tmp_line = np.zeros_like(img[y, :]).astype(np.float32)
        prev_flow = np.zeros(shape=(img.shape[1], 1), dtype=np.float32)
        for i in range((kernel.size//2) - 1, -1, -1):
            flow = self.get_flow(
                target=img[y, :],
                reference=padded_img[y + i, :],
                flow=prev_flow)
            self.logger.debug(f"{np.average(np.abs(flow))}")
            prev_flow = flow                     
            warped_line = self.warp_line(padded_img[y + i, :], flow)
            tmp_line += warped_line * kernel[i]
        tmp_line += img[y, :] * kernel[kernel.size//2]
        prev_flow = np.zeros(shape=(img.shape[1], 1), dtype=np.float32)
        for i in range(kernel.size//2+1, kernel.size):
            flow = self.get_flow(
                target=img[y, :],
                reference=padded_img[y + i, :],
                flow=prev_flow)
            self.logger.debug(f"{np.average(np.abs(flow))}")
            prev_flow = flow                      
            warped_line = self.warp_line(padded_img[y + i, :], flow)
            tmp_line += warped_line * kernel[i]
        return tmp_line

    def filter_X_line(self, img, padded_img, kernel, x):
        tmp_line = np.zeros_like(img[:, x]).astype(np.float32)
        prev_flow = np.zeros(shape=(img.shape[0], 1), dtype=np.float32)
        for i in range((kernel.size//2) - 1, -1, -1):
            flow = self.get_flow(
                target=img[:, x],
                reference=padded_img[:, x + i],
                flow=prev_flow)
            self.logger.debug(f"{np.average(np.abs(flow))}")
            prev_flow = flow
            warped_line = self.warp_line(padded_img[:, x + i], flow)
            tmp_line += warped_line * kernel[i]
        tmp_line += img[:, x] * kernel[kernel.size//2]
        prev_flow = np.zeros(shape=(img.shape[0], 1), dtype=np.float32)
        for i in range(kernel.size//2+1, kernel.size):
            flow = self.get_flow(
                target=img[:, x],
                reference=padded_img[:, x + i],
                flow=prev_flow)
            self.logger.debug(f"{np.average(np.abs(flow))}")
            prev_flow = flow
            warped_line = self.warp_line(padded_img[:, x + i], flow)
            tmp_line += warped_line * kernel[i]
        return tmp_line

    def _filter(self, img, kernel):
        print("1")
        mean = img.mean()
        self.logger.info(f"mean={mean}")
        filtered_img_Y = self.filter_Y(img, kernel[0], mean)
        self.logger.info(f"filtered along Y")
        filtered_img_YX = self.filter_X(filtered_img_Y, kernel[1], mean)
        self.logger.info(f"filtered along X")
        return filtered_img_YX

    def filter(self, img, kernel):
        print("2")
        mean = np.average(img)
        #t0 = time.perf_counter()
        filtered_img_Y = self.filter_Y(img, kernel[0], mean)
        #t1 = time.perf_counter()
        #print(t1 - t0)
        filtered_img_X = self.filter_X(img, kernel[1], mean)
        #t2 = time.perf_counter()
        #print(t2 - t1)
        filtered_img_YX = (filtered_img_Y + filtered_img_X)/2
        return filtered_img_YX
        
    def __iterate_filter(self, noisy_img, sigma_kernel=1.5, GT=None, N_iters=1):
        self.logger.info(f"sigma_kernel={sigma_kernel}")
        _ = super().iterate_filter(noisy_img, sigma_kernel, GT, N_iters)
        self.logger.warning(f"Number of singular matrices = {self.counter}")
        return _

    def __project_A_to_B(self, A, B):
        try:
            flow = self.estimator.pyramid_get_flow(target=B, reference=A, flow=None)
        except LinAlgError as e:
            print(f"Caught exception: {e}")
            self.counter += 1
            return A
        self.logger.debug(f"{np.average(np.abs(flow))}")
        #MVs = np.zeros_like(A)
        projection = Projectiont(self.logger, signal=A, flow=np.squeeze(flow))
        #print("A", A.shape, "projection", projection.shape)
        #print("A", np.sum(A)/len(A), "p", np.sum(projection)/len(projection))
        return projection

###################################33

def get_flow(reference, target, l=3, w=5, prev_flow=None, sigma=0.5):
    flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=prev_flow,
                                            pyr_scale=0.5, levels=l, winsize=w,
                                            iterations=3, poly_n=5, poly_sigma=sigma,
                                            flags=0)
    #flow[...] = 0.0
    print(np.max(np.abs(flow)), end=' ')
    return flow

def __warp_slice(reference, flow):
    height, width = flow.shape[:2]
    map_x = np.tile(np.arange(width), (height, 1))
    map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
    map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
    warped_slice = cv2.remap(reference, map_xy, None,
                             #interpolation=cv2.INTER_LANCZOS4, #INTER_LINEAR,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return warped_slice

def __project(img, flow):
    return motion_estimation.helpers.project(img, flow, interpolation_mode=cv2.INTER_LINEAR, extension_mode=cv2.BORDER_REPLICATE)

def filter_vertical(noisy_img, kernel, l=3, w=5, sigma=0.5):
    KL = kernel.size
    KL2 = KL//2
    w2 = w//2
    N_rows = noisy_img.shape[0]
    N_cols = noisy_img.shape[1]

    # Opción 0: Los márgenes son 128
    #extended_noisy_img = np.full(shape=(noisy_img.shape[0] + KL + w, noisy_img.shape[1] + w, noisy_img.shape[2]), fill_value=128, dtype=np.uint8)

    # Opción 1: Usando padding (no terminó de funcionar)
    #extended_noisy_img = np.empty(shape=(noisy_img.shape[0] + KL + w, noisy_img.shape[1] + w, noisy_img.shape[2]), dtype=np.uint8)
    #extended_noisy_img[..., 0] = np.pad(array=noisy_img[..., 0],
    #                              pad_width=(((KL + w)//2, (KL + w)//2), ((w + 1)//2, (w + 1)//2)),
    #                              mode="constant")
    #extended_noisy_img[..., 1] = np.pad(array=noisy_img[..., 1], pad_width=(KL2 + w2, w2), mode="constant")
    #extended_noisy_img[..., 2] = np.pad(array=noisy_img[..., 2], pad_width=(KL2 + w2, w2), mode="constant")

    # Opción 2: Los márgenes son la propia imagen, ampliada
    extended_noisy_img = cv2.resize(src = noisy_img, dsize = (noisy_img.shape[1] + w, noisy_img.shape[0] + KL + w))
    #print(extended_noisy_img.shape)
    #extended_noisy_img[KL2 + w2:noisy_img.shape[0] + KL2 + w2, w2:noisy_img.shape[1] + w2] = noisy_img[...]
    #extended_noisy_img[(KL + w)//2 - 1:noisy_img.shape[0] + (KL + w)//2 - 1, w2 - 1:noisy_img.shape[1] + w2 - 1] = noisy_img[...]
    extended_noisy_img[(KL + w)//2:noisy_img.shape[0] + (KL + w)//2, w2:noisy_img.shape[1] + w2] = noisy_img[...]
    extended_noisy_img = extended_noisy_img.astype(np.float32)
    extended_Y = extended_noisy_img
    filtered_noisy_img = []
    N_rows = noisy_img.shape[0]
    N_cols = noisy_img.shape[1]
    for y in range(N_rows):
        print(y, end=' ')
        horizontal_line = np.zeros(shape=(N_cols + w), dtype=np.float32)
        target_slice_Y = extended_Y[y + KL2:y + KL2 + w]
        #print("<", target_slice_Y.shape, ">")
        target_slice = extended_noisy_img[y + KL2:y + KL2 + w]
        for i in range(KL):
            reference_slice_Y = extended_Y[y + i:y + i + w]
            reference_slice = extended_noisy_img[y + i:y + i + w]
            flow = get_flow(
                reference=reference_slice_Y,
                target=target_slice_Y,
                l=l,
                w=w,
                prev_flow=None,
                sigma=sigma)
            OF_compensated_slice = Projection.remap(reference_slice, flow)
            OF_compensated_line = OF_compensated_slice[(w + 1) >> 1, :]
            OF_compensated_line = np.roll(OF_compensated_line, -w2)
            horizontal_line += OF_compensated_line * kernel[i]
        filtered_noisy_img.append(horizontal_line)
    filtered_noisy_img = np.stack(filtered_noisy_img, axis=0)[0:noisy_img.shape[0], 0:noisy_img.shape[1]]
    return filtered_noisy_img

def __filter(noisy_img, kernel, l=3, w=5, sigma=0.5):
    filtered_noisy_img_in_vertical = gray_vertical_OF_gaussian_filtering(noisy_img, kernel, l, w, sigma)
    transposed_noisy_img = np.transpose(noisy_img, (1, 0))
    transposed_and_filtered_noisy_img_in_horizontal = gray_vertical_OF_gaussian_filtering(transposed_noisy_img, kernel, l, w, sigma)
    filtered_noisy_img_in_horizontal = np.transpose(transposed_and_filtered_noisy_img_in_horizontal, (1, 0))
    filtered_noisy_img = (filtered_noisy_img_in_vertical + filtered_noisy_img_in_horizontal)/2
    return filtered_noisy_img

def filter(noisy_img, kernel, l=3, w=5, sigma=0.5):
    filtered_noisy_img_Y = gray_vertical_OF_gaussian_filtering(noisy_img, kernel, l, w, sigma)
    transposed_filtered_noisy_img_Y = np.transpose(filtered_noisy_img_Y, (1, 0))
    transposed_filtered_noisy_img_YX = gray_vertical_OF_gaussian_filtering(transposed_filtered_noisy_img_Y, kernel, l, w, sigma)
    filtered_noisy_img_YX = np.transpose(transposed_filtered_noisy_img_YX, (1, 0))
    return filtered_noisy_img_YX

def filter_vertical_RGB(noisy_img, kernel, l=3, w=5, sigma=0.5):
    #print("v1")
    KL = kernel.size
    KL2 = KL//2
    w2 = w//2
    N_rows = noisy_img.shape[0]
    N_cols = noisy_img.shape[1]
    #print(f"KL={KL} l={l} w={w}")

    # Opción 0: Los márgenes son 128
    #extended_noisy_img = np.full(shape=(noisy_img.shape[0] + KL + w, noisy_img.shape[1] + w, noisy_img.shape[2]), fill_value=128, dtype=np.uint8)

    # Opción 1: Usando padding (no terminó de funcionar)
    #extended_noisy_img = np.empty(shape=(noisy_img.shape[0] + KL + w, noisy_img.shape[1] + w, noisy_img.shape[2]), dtype=np.uint8)
    #extended_noisy_img[..., 0] = np.pad(array=noisy_img[..., 0],
    #                              pad_width=(((KL + w)//2, (KL + w)//2), ((w + 1)//2, (w + 1)//2)),
    #                              mode="constant")
    #extended_noisy_img[..., 1] = np.pad(array=noisy_img[..., 1], pad_width=(KL2 + w2, w2), mode="constant")
    #extended_noisy_img[..., 2] = np.pad(array=noisy_img[..., 2], pad_width=(KL2 + w2, w2), mode="constant")

    # Opción 2: Los márgenes son la propia imgn, ampliada
    extended_noisy_img = cv2.resize(src = noisy_img, dsize = (noisy_img.shape[1] + w, noisy_img.shape[0] + KL + w))
    #print(extended_noisy_img.shape)
    extended_noisy_img[(KL + w)//2:noisy_img.shape[0] + (KL + w)//2, w2:noisy_img.shape[1] + w2] = noisy_img[...]
    extended_Y = YUV.from_RGB(extended_noisy_img.astype(np.int16))[..., 0]
    extended_Y = extended_Y.astype(np.float32)
    extended_noisy_img = extended_noisy_img.astype(np.float32)
    #print(np.max(extended_Y), np.min(extended_Y))
    filtered_noisy_img = []
    N_rows = noisy_img.shape[0]
    N_cols = noisy_img.shape[1]
    for y in range(N_rows):
        print(y, end=' ')
        horizontal_line = np.zeros(shape=(N_cols + w, noisy_img.shape[2]), dtype=np.float32)
        target_slice_Y = extended_Y[y + KL2:y + KL2 + w]
        #print("<", target_slice_Y.shape, w, ">")
        target_slice = extended_noisy_img[y + KL2:y + KL2 + w, :]
        for i in range(KL):
            reference_slice_Y = extended_Y[y + i:y + i + w, :]
            reference_slice = extended_noisy_img[y + i:y + i + w, :]
            flow = get_flow(
                reference=reference_slice_Y,
                target=target_slice_Y,
                l=l,
                w=w,
                prev_flow=None,
                sigma=sigma)
            OF_compensated_slice = Projection.remap(reference_slice, flow)
            OF_compensated_line = OF_compensated_slice[(w + 1) >> 1, :, :]
            #OF_compensated_line = OF_compensated_slice[(w + 0) >> 1, :, :]
            OF_compensated_line = np.roll(a=OF_compensated_line, shift=-w2, axis=0)
            horizontal_line += OF_compensated_line * kernel[i]
        filtered_noisy_img.append(horizontal_line)
    filtered_noisy_img = np.stack(filtered_noisy_img, axis=0)[0:noisy_img.shape[0], 0:noisy_img.shape[1], :]
    return filtered_noisy_img

def filter_RGB(noisy_img, kernel, l=3, w=5, sigma=0.5):
    filtered_noisy_img_Y = RGB_vertical_OF_gaussian_filtering(noisy_img, kernel, l, w, sigma)
    filtered_noisy_img_YX = RGB_vertical_OF_gaussian_filtering(np.transpose(filtered_noisy_img_Y, (1, 0, 2)), kernel, l, w, sigma)
    OF_filtered_noisy_img = np.transpose(filtered_noisy_img_YX, (1, 0, 2))
    return OF_filtered_noisy_img

class _Monochrome_Denoising(gaussian.Monochrome_Denoising):

    def __init__(self, sigma_gaussian=1.5, l=3, w=9, sigma_OF=2.5, verbosity=logging.INFO):
        super().__init__(sigma=sigma_gaussian, verbosity=verbosity)
        self.l = l
        self.w = 9
        self.sigma_OF = sigma_OF
        self.logger.info(f"l={self.l}")
        self.logger.info(f"w={self.w}")
        self.logger.info(f"sigma_OF={self.sigma_OF}")
        self.logger.info(f"sigma_gaussian={self.sigma_gaussian}")
        self.gaussian_filtering = filter

class Color_Denoising(gaussian.Color_Denoising):

    def __init__(self, sigma_gaussian=1.5, l=3, w=9, sigma_OF=2.5, verbosity=logging.INFO):
        super().__init__(sigma=sigma_gaussian, verbosity=verbosity)
        self.l = l
        self.w = w
        self.sigma_OF = sigma_OF
        self.logger.info(f"l={self.l}")
        self.logger.info(f"w={self.w}")
        self.logger.info(f"sigma_OF={self.sigma_OF}")
        self.logger.info(f"sigma_gaussian={self.sigma_gaussian}")
        self.gaussian_filtering = filter_RGB
    
def filter_RGB_img(noisy_img, sigma_gaussian=2.5, N_iters=1, l=3, w=9, sigma_OF=2.5, GT=None):

    logger.info(f"sigma_kernel={sigma_kernel} N_iters={N_iters} l={l} w={w} sigma_OF={sigma_OF}")
    if logger.getEffectiveLevel() < logging.WARNING:
        PSNR_vs_iteration = []
    
    kernel = kernels.get_gaussian_kernel(sigma_gaussian)
    denoised_img = noisy_img.copy()
    for i in range(N_iters):
        if logger.getEffectiveLevel() < logging.WARNING:
            prev = denoised_img
        denoised_img = RGB_OF_gaussian_filtering(denoised_img, kernel, l, w, sigma_OF)
        if logger.getEffectiveLevel() < logging.WARNING:
            if GT != None:
                _PSNR = information_theory.distortion.avg_PSNR(denoised_img_img, GT)
            else:
                _PSNR = 0.0
            PSNR_vs_iteration.append(_PSNR)
            fig, axs = plt.subplots(1, 2, figsize=(10, 20))
            axs[0].imshow(normalize(denoised_img).astype(np.uint8))
            axs[0].set_title(f"iter {i} " + f"({_PSNR:4.2f}dB)")
            axs[1].imshow(normalize(denoised_img - prev + 128).astype(np.uint8))
            axs[1].set_title(f"diff")
            plt.show()
    print()

    if logger.getEffectiveLevel() < logging.WARNING:
        return denoised_img, PSNR_vs_iteration
    else:
        return denoised_img, None


