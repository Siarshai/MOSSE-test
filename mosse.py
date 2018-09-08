import cv2
import numpy as np
from scipy import signal as sg

from utils import normalize, show_image, get_random_transform_mat, get_hanning_window2d, show_freq_image, \
    preprocess_image


class MOSSETracker(object):
    def __init__(self, sigma_amplitude=0.2, transforms_number=1, learning_rate=0.2, epsilon=0.0001,
                 psr_threshold=5.7, debug_mode=False):
        self._is_initialized = False
        self._sigma_amplitude = sigma_amplitude
        self._transforms_number = transforms_number
        self._debug_mode = debug_mode
        self._learning_rate = learning_rate
        self._epsilon = epsilon
        self._psr_threshold = psr_threshold
        self._prev_roi = None
        self._frames_counter = 0
        self.g_desired_response = None
        self.G_desired_response_freq = None

    def is_inited(self):
        return self._is_initialized

    def initialize(self, image, roi):
        image = preprocess_image(image)
        self.g_desired_response = self._construct_desired_response(roi)
        self.G_desired_response_freq = np.fft.fft2(self.g_desired_response)
        self._prev_roi = [x for x in roi]
        self._train_on_image(image, roi, 1.0)
        self._is_initialized = True
        self._frames_counter = 1

    def track(self, image):
        # Scale adjustment not implemented for simplicity
        print("Frame {}".format(self._frames_counter))
        image = preprocess_image(image)
        prev_object_neighbourhood = image[self._prev_roi[1]:self._prev_roi[1] + self._prev_roi[3],
                self._prev_roi[0]:self._prev_roi[0] + self._prev_roi[2]]

        hanning_window = get_hanning_window2d(prev_object_neighbourhood.shape[1], prev_object_neighbourhood.shape[0])
        prev_object_neighbourhood = hanning_window*prev_object_neighbourhood + (1 - hanning_window)*np.mean(prev_object_neighbourhood)

        G_response_freq = self._W_filter_weights_freq * np.fft.fft2(prev_object_neighbourhood)
        g_response = np.abs(np.fft.ifft2(G_response_freq))
        if self._debug_mode:
            show_image(normalize(prev_object_neighbourhood), "f", wait=False,
                    size=(4*prev_object_neighbourhood.shape[1], 4*prev_object_neighbourhood.shape[0]))
            show_image(normalize(g_response), "g_response", wait=False,
                       size=(4*g_response.shape[1], 4*g_response.shape[0]))
            w_filter_weights = np.fft.ifft2(self._W_filter_weights_freq).real
            real_convolution_image = sg.convolve2d(prev_object_neighbourhood, w_filter_weights, mode='same', boundary='wrap')
            show_image(normalize(real_convolution_image), "real_convolution_image",
                       size=(4*real_convolution_image.shape[1], 4*real_convolution_image.shape[0]))

        max_response = g_response.max()
        max_position = np.where(g_response == max_response)
        x_max_position = int(np.mean(max_position[1]))
        y_max_position = int(np.mean(max_position[0]))
        dx = x_max_position - g_response.shape[1] // 2  # sic
        dy = y_max_position - g_response.shape[0] // 2

        if self._debug_mode:
            print("ROI ", self._prev_roi[0], self._prev_roi[1])
        print("Delta ", dx, dy)

        # MOSSE paper tells to exclude 11x11 window around the peak, but that's dubious
        # when g sigma is variable, so we are doing this:
        g_sidelobes = g_response - self.g_desired_response
        g_sidelobes[g_sidelobes < 0] = 0.0
        g_sidelobes[y_max_position, x_max_position] = 0.0
        mean_sidelobes = np.mean(g_sidelobes)
        std_sidelobes = np.std(g_sidelobes)

        psr = (max_response - mean_sidelobes)/(std_sidelobes + self._epsilon)

        if psr >= self._psr_threshold:
            ok = True
            self._prev_roi[0] += int(dx)
            self._prev_roi[1] += int(dy)

            if self._prev_roi[0] < 0:
                self._prev_roi[0] = 0
            elif self._prev_roi[0] >= image.shape[1] - self._prev_roi[2]:
                self._prev_roi[0] = image.shape[1] - self._prev_roi[2] - 1

            if self._prev_roi[1] < 0:
                self._prev_roi[1] = 0
            elif self._prev_roi[1] >= image.shape[0] - self._prev_roi[3]:
                self._prev_roi[1] = image.shape[0] - self._prev_roi[3] - 1

            self._train_on_image(image, self._prev_roi, self._learning_rate)
        else:
            ok = False

        if self._debug_mode:
            print("PSR: {} {}".format(psr, self._psr_threshold))

        self._frames_counter += 1
        return self._prev_roi, ok

    def _train_on_image(self, image, roi, learning_rate):
        if self._debug_mode:
            print("Training on original image")
        A, B = self._train_on_single_transform(image, roi)

        for i in range(1, self._transforms_number):
            if self._debug_mode:
                print("Training on transform {}".format(i))
            affine_transform_matrix = get_random_transform_mat(roi)
            transformed_image = cv2.warpAffine(image, affine_transform_matrix, (image.shape[1], image.shape[0]))
            Ai, Bi = self._train_on_single_transform(transformed_image, roi)
            A += Ai
            B += Bi

        if learning_rate >= 1.0:
            self._A = A
            self._B = B
        else:
            self._A = learning_rate*A + (1 - learning_rate)*self._A
            self._B = learning_rate*B + (1 - learning_rate)*self._B

        self._W_filter_weights_freq = np.divide(self._A, self._B + self._epsilon)

        if self._debug_mode:
            w_filter_weights = np.fft.ifft2(self._W_filter_weights_freq).real
            show_image(normalize(w_filter_weights), "w", wait=False,
                    size=(4*w_filter_weights.shape[1], 4*w_filter_weights.shape[0]))
            show_freq_image(self._W_filter_weights_freq, "W", wait=False,
                    size=(4*self._W_filter_weights_freq.shape[1], 4*self._W_filter_weights_freq.shape[0]))

    def _train_on_single_transform(self, image, roi):
        f_object_image = image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        hanning_window = get_hanning_window2d(f_object_image.shape[1], f_object_image.shape[0])
        f_object_image = hanning_window*f_object_image + (1 - hanning_window)*np.mean(f_object_image)

        F_object_image_freq = np.fft.fft2(f_object_image)

        A = self.G_desired_response_freq * np.conjugate(F_object_image_freq)
        B = F_object_image_freq * np.conjugate(F_object_image_freq)

        if self._debug_mode:
            show_image(normalize(self.g_desired_response), "g_desired", wait=False,
                    size=(4*self.g_desired_response.shape[1], 4*self.g_desired_response.shape[0]))
            show_image(normalize(f_object_image), "f_initial", wait=False,
                    size=(4*f_object_image.shape[1], 4*f_object_image.shape[0]))
            show_freq_image(F_object_image_freq, "F_initial", wait=False,
                    size=(4*F_object_image_freq.shape[1], 4*F_object_image_freq.shape[0]))

            if self._transforms_number > 1:
                W = np.divide(A, B + self._epsilon)
                w = np.fft.ifft2(W).real
                real_convolution_image = sg.convolve2d(f_object_image, w, mode='same', boundary='wrap')

                show_image(normalize(w), "w_partial", wait=False,
                        size=(4*w.shape[1], 4*w.shape[0]))
                show_image(normalize(real_convolution_image), "real_convolution_image_partial", wait=False,
                        size=(4*real_convolution_image.shape[1], 4*real_convolution_image.shape[0]))

        return A, B

    def _construct_desired_response(self, roi):
        # Notice that this can be optimized if target_shift_amplitude is not used
        xx, yy = np.meshgrid(np.arange(roi[2]), np.arange(roi[3]))
        center_x, center_y = roi[2] / 2, roi[3] / 2
        sigma = self._sigma_amplitude * np.sqrt(roi[2]*roi[3])  # adjusting sigma to ROI size
        distance_map = (np.square(xx - center_x) + np.square(yy - center_y)) / (sigma*sigma)
        response = np.exp(-distance_map)
        response = normalize(response)
        return response
