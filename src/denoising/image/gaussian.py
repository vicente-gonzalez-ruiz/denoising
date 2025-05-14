'''Gaussian image denoising.'''

import numpy as np
import cv2

#pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
#from color_transforms import YCoCg as YUV

class Monochrome_Denoising:

    def __init__(self, logger):
        self.logger = logger
        self.window_size = 0 # Ojo, no sé si se va a usar

    def filter(self, img, kernel):
        #mean = img.mean()
        #self.logger.info(f"mean={mean}")
        #padded_img = cv2.resize(src=img, dsize=(img.shape[1] + kernel.size, img.shape[0] + kernel.size))
        #padded_img[kernel.size//2:shape_of_img[0] + (kernel.size//2), kernel.size//2:shape_of_img[1] + (kernel.size//2)] = img
        self.logger.info(f"filtering along Y")
        filtered_img_Y = self.filter_Y(img, kernel[0])
        self.logger.info(f"done")
        self.logger.info(f"filtering along X")
        filtered_img_YX = self.filter_X(filtered_img_Y, kernel[1])
        self.logger.info(f"done")
        return filtered_img_YX

    def filter_Y(self, img, kernel):
        assert kernel.size % 2 != 0 # kernel.size must be odd
        filtered_img = np.zeros_like(img).astype(np.float32)
        shape_of_img = np.shape(img)
        #padded_img = np.full(shape=(shape_of_img[0] + kernel.size, shape_of_img[1]), fill_value=mean)
        # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize
        padded_img = cv2.resize(src=img, dsize=(shape_of_img[1], shape_of_img[0] + kernel.size))
        padded_img[kernel.size//2:shape_of_img[0] + (kernel.size//2), :] = img
        Y_dim = img.shape[0]
        for y in range(Y_dim):
            filtered_img[y, :] = self.filter_Y_line(img, padded_img, kernel, y)
        return filtered_img

    def filter_X(self, img, kernel):
        assert kernel.size % 2 != 0 # kernel.size must be odd
        filtered_img = np.zeros_like(img).astype(np.float32)
        shape_of_img = np.shape(img)
        #padded_img = np.full(shape=(shape_of_img[0], shape_of_img[1] + kernel.size), fill_value=mean)
        # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize
        padded_img = cv2.resize(img, (shape_of_img[1] + kernel.size, shape_of_img[0]))
        padded_img[:, kernel.size//2:shape_of_img[1] + (kernel.size//2)] = img
        X_dim = img.shape[1]
        for x in range(X_dim):
            filtered_img[:, x] = self.filter_X_line(img, padded_img, kernel, x)
        return filtered_img

    def filter_Y_line(self, img, padded_img, kernel, y):
        tmp_line = np.zeros_like(img[y, :]).astype(np.float32)
        #for i in range((kernel.size//2) - 1, 0, -1):
        for i in range(kernel.size):
            tmp_line += padded_img[y + i, :] * kernel[i]
        '''
        for i in range(0, kernel.size//2):
            tmp_line += padded_img[y + i, :] * kernel[i]
        tmp_line += img[y, :] * kernel[kernel.size//2]
        for i in range(kernel.size//2+1, kernel.size):
            tmp_line += padded_img[y + i, :] * kernel[i]
        '''
        return tmp_line

    def filter_X_line(self, img, padded_img, kernel, x):
        tmp_line = np.zeros_like(img[:, x]).astype(np.float32)
        for i in range(kernel.size):
            tmp_line += padded_img[:, x + i] * kernel[i]
        '''
        for i in range((kernel.size//2) - 1, -1, -1):
            tmp_line += padded_img[:, x + i] * kernel[i]
        tmp_line += img[:, x] * kernel[kernel.size//2]
        for i in range(kernel.size//2+1, kernel.size):
            tmp_line += padded_img[:, x + i] * kernel[i]
        '''
        return tmp_line

class old:
    
    def unused__warp_line(self, line, flow):
        return line

    def unused__get_flow(self, reference, target, N_poly=0, window_length=0, num_iters=0, pyramid_levels=0, flow=None, model="", mu=0):
        return flow

    def unused__filter_Y(self, img, kernel, mean, l=0, w=0):
        assert kernel.size % 2 != 0 # kernel.size must be odd
        filtered_img = np.zeros_like(img).astype(np.float32)
        shape_of_img = np.shape(img)
        padded_img = np.full(shape=(shape_of_img[0] + kernel.size, shape_of_img[1]), fill_value=mean)
        padded_img[kernel.size//2:shape_of_img[0] + kernel.size//2, :] = img
        Y_dim = img.shape[0]
        for y in range(Y_dim):
            tmp_line = np.zeros_like(img[y, :]).astype(np.float32)
            prev_flow = np.zeros(shape=(shape_of_img[1], 1), dtype=np.float32)
            for i in range((kernel.size//2) - 1, -1, -1):
                flow = self.get_flow(padded_img[y + i, :], img[y, :], prev_flow, l, w)
                self.logger.debug(f"{np.average(np.abs(flow))}")
                prev_flow = flow                     
                warped_line = self.warp_line(padded_img[y + i, :], flow)
                tmp_line += warped_line * kernel[i]
            tmp_line += img[y, :] * kernel[kernel.size//2]
            prev_flow = np.zeros(shape=(shape_of_img[1], 1), dtype=np.float32)
            for i in range(kernel.size//2+1, kernel.size):
                flow = self.get_flow(padded_img[y + i, :], img[y, :], prev_flow, l, w)
                self.logger.debug(f"{np.average(np.abs(flow))}")
                prev_flow = flow                      
                warped_line = self.warp_line(padded_img[y + i, :], flow)
                tmp_line += warped_line * kernel[i]
            filtered_img[y, :] = tmp_line
        return filtered_img

    def unused_filter_X(self, img, kernel, mean, l=0, w=0):
        assert kernel.size % 2 != 0 # kernel.size must be odd
        filtered_img = np.zeros_like(img).astype(np.float32)
        shape_of_img = np.shape(img)
        padded_img = np.full(shape=(shape_of_img[0], shape_of_img[1] + kernel.size), fill_value=mean)
        padded_img[:, kernel.size//2:shape_of_img[1] + kernel.size//2] = img
        X_dim = img.shape[1]
        for x in range(X_dim):
            tmp_line = np.zeros_like(img[:, x]).astype(np.float32)
            prev_flow = np.zeros(shape=(shape_of_img[0], 1), dtype=np.float32)
            for i in range((kernel.size//2) - 1, -1, -1):
                flow = self.get_flow(padded_img[:, x + i], img[:, x], prev_flow, l, w)
                self.logger.debug(f"{np.average(np.abs(flow))}")
                prev_flow = flow
                warped_line = self.warp_line(padded_img[:, x + i], flow)
                tmp_line += warped_line * kernel[i]
            tmp_line += img[:, x] * kernel[kernel.size//2]
            prev_flow = np.zeros(shape=(shape_of_img[0], 1), dtype=np.float32)
            for i in range(kernel.size//2+1, kernel.size):
                flow = self.get_flow(padded_img[:, x + i], img[:, x], prev_flow, l, w)
                self.logger.debug(f"{np.average(np.abs(flow))}")
                prev_flow = flow
                warped_line = self.warp_line(padded_img[:, x + i], flow)
                tmp_line += warped_line * kernel[i]
            filtered_img[:, x] = tmp_line
        return filtered_img

    def unused_filter(self, noisy_img, kernel):
        #print("2")
        mean = np.average(noisy_img)
        #t0 = time.perf_counter()
        filtered_in_vertical = self.filter_Y(noisy_img, kernel[0], mean)
        #t1 = time.perf_counter()
        #print(t1 - t0)
        filtered_in_horizontal = self.filter_X(noisy_img, kernel[1], mean)
        #t2 = time.perf_counter()
        #print(t2 - t1)
        filtered_noisy_img = (filtered_in_vertical + filtered_in_horizontal)/2
        return filtered_noisy_img

class old:
        
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
        #print(filtered_in_vertical.dtype)
        #t1 = time.perf_counter()
        #print(t1 - t0)
        filtered_in_horizontal = self.filter_horizontal(noisy_img, kernel, mean)
        #t2 = time.perf_counter()
        #print(t2 - t1)
        filtered_noisy_img = (filtered_in_vertical + filtered_in_horizontal)/2
        return filtered_noisy_img

    def _filter(self, noisy_img, kernel):
        mean = np.average(noisy_img)
        #t0 = time.perf_counter()
        filtered_noisy_img_Y = self.filter_vertical(noisy_img, kernel, mean)
        #print(filtered_noisy_img_Y.dtype)
        #t1 = time.perf_counter()
        #print(t1 - t0)
        mean = np.average(filtered_noisy_img_Y)
        filtered_noisy_img_YX = self.filter_horizontal(filtered_noisy_img_Y, kernel, mean)
        #filtered_noisy_img_YX = self.filter_horizontal(noisy_img, kernel, mean)
        #t2 = time.perf_counter()
        #print(t2 - t1)
        return filtered_noisy_img_YX

    def iterate_filter(self, noisy_img, sigma_kernel=1.5, GT=None, N_iters=1):
        self.logger.info(f"sigma_kernel={sigma_kernel}")
        if self.logger.getEffectiveLevel() < logging.INFO:
            PSNR_vs_iteration = []
        kernel = self.get_kernel(sigma_kernel)
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

    def __init__(self, logger):
        super().__init__(logger)

    def filter(self, noisy_img, kernel):
        filtered_noisy_img_R = super().filter(noisy_img[..., 0], kernel)
        filtered_noisy_img_G = super().filter(noisy_img[..., 1], kernel)
        filtered_noisy_img_B = super().filter(noisy_img[..., 2], kernel)
        return np.stack([filtered_noisy_img_R, filtered_noisy_img_G, filtered_noisy_img_B], axis=2)

    def iterate_filter(self, noisy_img, sigma_kernel=1.5, GT=None, N_iters=1):
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


