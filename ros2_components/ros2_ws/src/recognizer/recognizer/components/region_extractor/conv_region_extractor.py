# coding: utf-8

import cv2
import numpy as np


class ConvolutionRegionExtractor:
    def __init__(self, input_size: int, kernel_size: int):

        if kernel_size > input_size:
            raise ValueError(f"kernel size should be smaller than input image size")

        self.kernel = self.__create_uniform_kernel(kernel_size)

    def get_diff_conv(self, ref: np.ndarray, target: np.ndarray) -> np.ndarray:
        diff = ref - target

        diff = self.__conv(diff)

        return diff

    def get_conv_diff(self, ref: np.ndarray, target: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            ref (np.ndarray): Reference image
            target (np.ndarray): Traget image to extract something
        """
        ref = self.__conv(ref)
        target = self.__conv(target)
        diff = ref - target

        return diff

    def __conv(self, input: np.ndarray) -> np.ndarray:
        """Convolution filter

        Args:
            input (np.ndarray): Input image

        Returns:
            np.ndarray: Filetered image
        """
        #        input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        return cv2.filter2D(input, -1, self.kernel)

    def __create_uniform_kernel(self, kernel_size: int) -> np.ndarray:
        """Returns an uniform kernel with input kernel_size

        Args:
            kernel_size (int): kernel size

        Returns:
            np.ndarray: kernel
        """
        return np.ones((kernel_size, kernel_size))
