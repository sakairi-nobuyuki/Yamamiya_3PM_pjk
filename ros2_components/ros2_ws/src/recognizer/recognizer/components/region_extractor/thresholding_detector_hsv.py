# coding: utf-8

from typing import Any, List, Tuple

import cv2
import numpy as np

from ...io import S3ConfigIO


class ThresholdingDetector:
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

        if type == "bgr":
            self.__detector = self.detect_green_bgr
        elif type == "saturation":
            self.__detector = self.detect_saturated
        elif type is None or type == "hsv":
            self.__detector = self.detect_green_lsv
            self.lower_green = np.array([59, 75, 25])
            self.upper_green = np.array([100, 254, 255])
            # self.lower_green = np.array([75, 50, 50])
            # self.upper_green = np.array([110, 255, 255])
        else:
            raise NotImplementedError(f"detector type: {type} is not implemented")

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
            List[np.ndarray]: bbox coordinate list
        """

        contours = self.__detector(input)

        img_list = self.__create_cropped_list(input, contours)
        # print("contour: ", type(contours))

        return img_list

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

    def detect_green_bgr(self, input: np.ndarray) -> List[Tuple[int]]:
        """Detecting plums from an image in BGR space, where plums are supposed to be green.

        Args:
            input (np.ndarray): Input image

        Returns:
            List[Tuple[int]]: A list of contours
        """
        # img = np.abs(4 * cv2.split(input)[1] - cv2.split(input)[0] - cv2.split(input)[2])
        img = cv2.split(input)[1]
        # img = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        self.gray = img
        img = cv2.threshold(img, self.threshold_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        return contours

    def __create_cropped_list(
        self, input: np.ndarray, contours: List[Tuple[int]]
    ) -> List[np.ndarray]:
        """Create a list of cropped image from the contours and input image

        Args:
            input (np.ndarray): Input image
            contours (List[Tuple[int]]): Contours of objects (plums)

        Returns:
            Lisst[np.ndarray]: A list of cropped images
        """
        img_list = []
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
                img_list.append(input[y1:y2, x1:x2])

        return img_list

    def __get_probability(self, x: float) -> float:
        return (
            1
            / (self.sigma * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)
        )

    def __filter_small_bboxes(self, bbox_feature: Tuple[int]) -> bool:
        print("object filter")
        print("  filter input: ", bbox_feature[2], bbox_feature[3])
        prob = self.__get_probability(max(bbox_feature[2], bbox_feature[3]))
        print("  probability: ", prob)

        if prob < self.object_filter_threshold:
            return False
        else:
            return True

    def __load_object_filter_cofig(self, s3_config: S3ConfigIO) -> None:
        config_dict = s3_config.load("detector/object_filter_config.yaml")

        self.sigma = config_dict["sigma"]
        self.mu = config_dict["mu"]

    @classmethod
    def __identity_function_1d(cls, x: Any) -> Any:
        return x
