# coding: utf-8

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import cv2
import numpy as np


class DetectorTemplate(metaclass=ABCMeta):
    """An interface of rule baseddetectors

    Args:
        metaclass (_type_, optional): Inhering abstract class. Defaults to ABCMeta.
    """

    @abstractmethod
    def detect(self, input: np.ndarray) -> List[np.ndarray]:
        """Detect something and return a list of contours

        Args:
            input (np.ndarray): Input image of OpenCV

        Returns:
            List[np.ndarray]: A list of OpenCV contours of detected things.
        """
        pass

    def _create_object_coordinate_list(
        self, input: np.ndarray, contours: List[Tuple[int]]
    ) -> List[Tuple[int]]:
        """Create a list of bounding box coordinate list

        Args:
            input (np.ndarray): Input image
            contours (List[Tuple[int]]): Contours of objects (plums)

        Returns:
            Lisst[Tuple[int]]: A list of bbox coordinate
        """
        coordinate_list = []
        for contour in contours:
            print("contour before bounding rect: ", contour)
            # if -1 in contour:
            #    print("no contour")
            #    continue
            # if len(contour) == 0:
            #    continue
            ### bounding rect retuns:

            x, y, w, h = cv2.boundingRect(contour)
            margin = min(w, h)
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, input.shape[1])
            y2 = min(y + h + margin, input.shape[0])

            coordinate_list.append((x1, y1, x2 - x1, y2 - y1))
        print("bboxes: ", coordinate_list)
        return coordinate_list
