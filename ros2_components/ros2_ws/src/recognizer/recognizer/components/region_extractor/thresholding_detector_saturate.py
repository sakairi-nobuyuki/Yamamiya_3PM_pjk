# coding: utf-8

from typing import Any, List, Tuple

import cv2
import numpy as np

from ...io import S3ConfigIO
from . import DetectorTemplate

class ThresholdingDetectorSaturate(DetectorTemplate):
    def __init__(
        self,
        config_s3: S3ConfigIO = None,
        type: str = "hsv",
        threshold_value: int = None,
        object_filter_threshold: float = 0.01,
    ):
        """Plum size might be 100px X 100px

        Args:
            type (str, optional): _description_. Defaults to "hsv".
            threshold_value (int, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        self.object_filter_threshold = object_filter_threshold

        if threshold_value is None:
            self.threshold_value = 22
            # self.threshold_value = 5
        else:
            self.threshold_value = threshold_value

        self.__detector = self.detect_saturated

        if config_s3 is not None:
            self.__filter = self.__filter_small_bboxes
            self.__load_object_filter_cofig(config_s3)
        else:
            self.__filter = self.__identity_function_1d
        self.get_contours = self.__detector

    def detect(self, input: np.ndarray) -> List[np.ndarray]:
        """Detect items, and returns its list of bboxes

        Args:
            input (np.ndarray): input image

        Returns:
            List[List[int]]: bbox coordinate list
        """

        contours = self.__detector(input)

        bboxes_list = self.__create_object_coordinate_list(input, contours)

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

        return contours

    @classmethod
    def __identity_function_1d(cls, x: Any) -> Any:
        return x

    def __create_object_coordinate_list(
        self, input: np.ndarray, contours: List[Tuple[int]]
    ) -> List[Tuple[int]]:
        """Create a list of cropped image from the contours and input image

        Args:
            input (np.ndarray): Input image
            contours (List[Tuple[int]]): Contours of objects (plums)

        Returns:
            Lisst[np.ndarray]: A list of cropped images
        """
        coordinate_list = []
        for contour in contours:
            if -1 in contour:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if self.__filter((x, y, w, h)):
                # if w > 10 or h > 10:
                # img_list.append(cv2.rectangle(input, (x, y), (x+w, y+h), (0, 0, 255), 2))
                margin = min(w, h)
                x1 = max(x - margin, 0)
                y1 = max(y - margin, 0)
                x2 = min(x + w + margin, input.shape[1])
                y2 = min(y + h + margin, input.shape[0])
                coordinate_list.append((x1, y1, x2 - x1, y2 - y1))

        return coordinate_list

