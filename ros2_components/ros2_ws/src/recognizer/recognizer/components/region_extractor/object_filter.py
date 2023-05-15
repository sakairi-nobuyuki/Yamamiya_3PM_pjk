# coding: utf-8


from typing import Any, List, Tuple, Dict, Union

import numpy as np


class ThresholdingDetector:
    def __init__(
        self,
        config_dict: Dict[str, Union[int, float]] = None
    ):
        """Plum size might be 100px X 100px

        Args:
            type (str, optional): _description_. Defaults to "hsv".
            threshold_value (int, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """

        if config_dict is None:
            self.sigma = config_dict["sigma"]
            self.mu = config_dict["mu"]
            self.object_filter_threshold = config_dict["threshold"]
        else:
            self.sigma = 1.0
            self.mu = 50
            self.object_filter_threshold = 0.1

    def run(self, bboxes_list: List[List[int]]) -> List[List[int]]:
        """Detect items, and returns its list of bboxes

        Args:
            input (np.ndarray): input image

        Returns:
            List[np.ndarray]: bbox coordinate list
        """

        filtered_bboxes_list = [bbox for bbox in bboxes_list if self.__filter_small_bboxes(bbox)]

        return filtered_bboxes_list


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


