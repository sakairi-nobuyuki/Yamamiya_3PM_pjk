# coding: utf-8

from typing import Any, List, Tuple, Union

import cv2
import numpy as np

from . import DetectorTemplate

class ThresholdingDetectorSaturate(DetectorTemplate):
    def __init__(
        self,
        threshold_value: int = None,
    ):
        """Plum size might be 100px X 100px

        Args:
            type (str, optional): _description_. Defaults to "hsv".
            threshold_value (int, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        self.threshold_value = threshold_value
        self.__detector = self.detect_saturated

    @property
    def threshold_value(self):
        return self.__threshold_value

    @threshold_value.setter
    def threshold_value(self, threshold_value: Union[int, None]):
        if threshold_value is not None:
            self.__threshold_value = threshold_value
        else:
            self.__threshold_value = 20

    def detect(self, input: np.ndarray) -> List[Tuple[int]]:
        """Detect items, and returns its list of bboxes

        Args:
            input (np.ndarray): input image

        Returns:
            List[Tuple[int]]: bbox coordinate list
        """

        contours = self.__detector(input)

        bboxes_list = self._create_object_coordinate_list(input, contours)

        return bboxes_list

    def detect_saturated(self, input: np.ndarray) -> List[Tuple[int]]:

        input = cv2.medianBlur(input, 5)
        hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV_FULL)
        target = cv2.split(hsv)[1]
        self.gray = target
        img_bin = cv2.threshold(
            target, self.threshold_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
#        contours = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
#        contours = cv2.findContours(target, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        contours = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

        bbox_list = self._create_object_coordinate_list(input, contours)


        return bbox_list

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
        coordinate_list = super()._create_object_coordinate_list(input, contours)

        return coordinate_list

