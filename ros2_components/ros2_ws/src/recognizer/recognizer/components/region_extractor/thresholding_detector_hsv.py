# coding: utf-8

from typing import List, Tuple, Union
import cv2
import numpy as np

from . import DetectorTemplate

class ThresholdingDetectorHsv(DetectorTemplate):
    def __init__(
        self,
        threshold_value: int = None,
        lower_green: np.ndarray = None, 
        upper_green: np.ndarray = None
    ):
        """Plum size might be 100px X 100px

        Args:
            type (str, optional): _description_. Defaults to "hsv".
            threshold_value (int, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        self.threshold_value = threshold_value
        self.lower_green = lower_green
        self.upper_green = upper_green
        self.__detector = self.detect_green_lsv

    @property
    def lower_green(self) -> np.ndarray:
        return self.__lower_green

    @lower_green.setter
    def lower_green(self, lower_green: Union[np.ndarray, None]) -> None:
        if lower_green is not None:
            self.__lower_green = lower_green
        else:
            self.__lower_green = np.array([59, 75, 25])
        
    @property
    def uppser_green(self) -> np.ndarray:
        return self.__upper_green

    @lower_green.setter
    def upper_green(self, upper_green: Union[np.ndarray, None]) -> None:
        if upper_green is not None:
            self.__upper_green = upper_green
        else:
            self.__upper_green = np.array([100, 254, 255])

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
            List[np.ndarray]: bbox coordinate list
        """

        contours = self.__detector(input)

        bboxes_list = self._create_object_coordinate_list(input, contours)

        return bboxes_list

    def detect_green_lsv(self, input: np.ndarray) -> List[Tuple[int]]:
        """Detect plums from an image in LSV space, where plums are supposed to be green

        Args:
            input (np.ndarray): Image

        Returns:
            List[Tuple[int]]: A list of contours
        """
        hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV_FULL)
        self.gray = hsv
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        return contours

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

