# coding: utf-8


from typing import Any, Dict, List, Tuple, Union

import numpy as np


class ObjectFilter:
    def __init__(
        self, filtering_flag: bool = True, config_dict: Dict[str, Union[int, float]] = None
    ):
        """Plum size might be 100px X 100px

        Args:
            type (str, optional): _description_. Defaults to "hsv".
            threshold_value (int, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """

        if config_dict is not None:
            self.sigma = config_dict["sigma"]
            self.mu = config_dict["mu"]
            self.object_filter_threshold = config_dict["threshold"]
        else:
            self.sigma = 1.0
            self.mu = 50
            self.object_filter_threshold = 0.1

        if filtering_flag is True:
            self.__filter_small_bboxes = lambda x_list: [
                self.__filter_small_bbox(x) for x in x_list
            ]
        else:
            self.__filter_small_bboxes = self.__identity_function_1d

    def run(self, bboxes_list: List[Tuple[int]]) -> List[Tuple[int]]:
        """Detect items, and returns its list of bboxes

        Args:
            input (np.ndarray): input image

        Returns:
            List[Tuple[int]]: bbox coordinate list
        """

        return self.__filter_small_bboxes(bboxes_list)

    def __get_probability(self, x: float) -> float:
        return (
            1
            / (self.sigma * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)
        )

    def __filter_small_bbox(self, bbox_feature: Tuple[int]) -> bool:
        print("object filter")
        print("  filter input: ", bbox_feature[2], bbox_feature[3])
        prob = self.__get_probability(max(bbox_feature[2], bbox_feature[3]))
        print("  probability: ", prob)

        if prob < self.object_filter_threshold:
            return False
        else:
            return True

    @classmethod
    def __identity_function_1d(cls, x: Any) -> Any:
        return x
