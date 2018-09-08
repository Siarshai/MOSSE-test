import cv2
import numpy as np


def normalize(array):
    maximum = array.max()
    minimum = array.min()
    return (array - minimum) / (maximum - minimum)


def preprocess_image(image):
    # Preprocessing suggested by ASEF paper
    image = image.astype(np.float32)
    # image = np.log(image + 1) # TODO: huh, seems too stong... try sqrt?
    image = np.sign(image - 127.5) * np.sqrt(np.abs(image - 127.5))
    image -= np.mean(image)
    image /= np.linalg.norm(image)
    return image


def get_hanning_window2d(width, height):
    xx = np.hanning(width)
    yy = np.hanning(height)
    xx_window, yy_window = np.meshgrid(xx, yy)
    window = xx_window * yy_window
    return window


def get_random_transform_mat(roi):
    amplitude = 0.025
    min_angle, max_angle = -3, 3
    angle = np.random.uniform(min_angle, max_angle)
    origin_pts = np.zeros(shape=(3, 2), dtype=np.float32)

    origin_pts[0, 0] = roi[0]
    origin_pts[0, 1] = roi[1]
    origin_pts[1, 0] = roi[0] + roi[2]
    origin_pts[1, 1] = roi[1]
    origin_pts[2, 0] = roi[0]
    origin_pts[2, 1] = roi[1] + roi[3]

    center_x = roi[0] + roi[2] / 2
    center_y = roi[1] + roi[3] / 2
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    extended_rotation_matrix = np.concatenate((rotation_matrix, np.asarray([[0.0, 0.0, 1.0]])))

    target_pts = np.copy(origin_pts)
    affine_noise = np.asarray(np.random.uniform(-amplitude, amplitude, size=(3, 2)), dtype=np.float32)
    origin_pts[0, 0] += roi[2] * affine_noise[0, 0]
    origin_pts[0, 1] += roi[3] * affine_noise[0, 1]
    origin_pts[1, 0] += roi[2] * affine_noise[1, 0]
    origin_pts[1, 1] += roi[3] * affine_noise[1, 1]
    origin_pts[2, 0] += roi[2] * affine_noise[2, 0]
    origin_pts[2, 1] += roi[3] * affine_noise[2, 1]
    affine_matrix = cv2.getAffineTransform(origin_pts, target_pts)
    extended_affine_matrix = np.concatenate((affine_matrix, np.asarray([[0.0, 0.0, 1.0]])))
    full_transform = np.matmul(extended_rotation_matrix, extended_affine_matrix)

    return full_transform[0:2]


def show_freq_image(fourier_image, caption, wait=True, size=None):
    F = np.abs(np.fft.fftshift(fourier_image))
    F_log = np.log(F + 0.0000000001)
    show_image(normalize(F), caption, False, size)
    show_image(normalize(F_log), caption + "_log", wait, size)


def show_image(image, caption, wait=True, size=None):
    if caption not in show_image.windows:
        show_image.windows.add(caption)
        cv2.namedWindow(caption, cv2.WINDOW_NORMAL)
        if size is not None:
            cv2.resizeWindow(caption, size[0], size[1])
    cv2.imshow(caption, image)
    if wait:
        cv2.waitKey(0)


show_image.windows = set()


class VideoBatchReader(object):
    def __init__(self, filename, normalized, target_fps=None, debug_mode=False):
        self.cap = cv2.VideoCapture(filename)
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self._fps is None or self._fps <= 0:
            self._fps = 25
        if target_fps is None:
            target_fps = self._fps
        self.drop_divisor = self._fps // target_fps
        self._frame_read_total = 0
        self._normalized = normalized
        self._debug_mode = debug_mode

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()
        cv2.destroyAllWindows()

    def frames(self):
        while True:
            really_read_frame = False
            frame = None
            while not really_read_frame:
                ok, frame = self.cap.read()
                if not ok:
                    raise StopIteration

                really_read_frame = not (self._frame_read_total % self.drop_divisor)
                self._frame_read_total += 1

            assert(frame is not None)

            if self._debug_mode:
                cv2.imshow('frame', frame)
                k = cv2.waitKey(1) & 0xff
                if k == 32:
                    k = cv2.waitKey() & 0xff
                if k == 27:
                    break

            if self._normalized:
                frame = 2.0 * (frame.astype(np.float32) / 255.5) - 1.0
            yield frame