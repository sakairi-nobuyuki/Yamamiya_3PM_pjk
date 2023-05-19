# coding: utf-8

from typing import Dict

import cv2
import numpy as np
from recognizer.components.region_extractor import ThresholdingDetectorHsv as ThresholdingDetector
from recognizer.io import S3ImageIO


class ObjectFilterTrainer:
    def __init__(self, image_s3: S3ImageIO = None):
        """Constructor.

        Args:
            image_s3 (S3ImageIO, optional): S3 resources for downloading dataset for train. Defaults to None.
            config_s3 (S3ConfigIO, optional): S3 resources for saving trained parameters. Defaults to None.
        """
        if image_s3 is None:
            self.image_s3 = S3ImageIO(
                endpoint_url="http://192.168.1.194:9000",
                access_key="sigma-chan",
                secret_key="sigma-chan-dayo",
                bucket_name="dataset",
            )
        else:
            self.image_s3 = image_s3
        self.file_name_list = [
            item for item in self.image_s3.blob if "region_extractor/thresholding_train" in item
        ]
        self.detector = ThresholdingDetector(type="saturation", threshold_value=100)

    def run(self) -> Dict[str, str]:
        """Run optimization.
        The trained parameters for making a probability distribution of cropping image size should be saved to a
        filtering config file.
        """
        object_bbox_array = self.__get_object_size()

        std_deviation = np.std(object_bbox_array)
        mean = np.mean(object_bbox_array)
        res_dict = dict(sigma=float(std_deviation), mu=float(mean))
        print("trained results: ", res_dict)

        return res_dict

    def __get_object_size(self) -> np.ndarray:

        object_list = []
        for file_name in self.file_name_list:
            img = self.image_s3.load(file_name)
            contours = self.detector.get_contours(img)
            contour_list = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                contour_list.append([h, w])
            object_list.append(max(list(map(lambda x: x[np.argmax(x[0] * x[1])], contour_list))))
            # print("  object_list: ", object_list)
            # object_list.append(max(list(map(lambda x: x[0] * x[1], contour_list))))
        size_array = np.array(object_list)
        print("object size: ", size_array)
        return size_array
