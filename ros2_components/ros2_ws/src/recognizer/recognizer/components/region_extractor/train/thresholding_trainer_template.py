# coding: utf-8

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import cv2
import numpy as np


class ThresholdingTrainerTemplate(metaclass=ABCMeta):
    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def target(self, x: List[int]) -> float:
        pass

    def loss(self, img: np.ndarray, contours: List[Tuple[int]]) -> float:
        """Contours are not contours, but bounding boxes

        Args:
            img (np.ndarray): _description_
            contours (List[Tuple[int]]): _description_

        Returns:
            float: _description_
        """

        loss_value = 0
        n_contour = 0
        if len(contours) == 0:
            return 1.0e+06
        for i_contour, contour in enumerate(contours):
            #if len(contour) != 1:
            #    return 1.0e+06
            print(contour)
            # x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = contour
            print(f"w: {w}, h: {h}")
            loss_value += (
                self.w_aspect_ratio
                * abs(self.aspect_ratio - w / h)
                * (abs(self.w - w) + abs(self.h - h))
            )
            n_contour = i_contour
        loss_value *= 1.0 / (abs(n_contour - 1) + 1)

        return loss_value
