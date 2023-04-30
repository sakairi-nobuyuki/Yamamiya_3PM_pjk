# coding: utf-8

import cv2
import numpy as np

from recognizer.io import S3ConfigIO, S3ImageIO
from recognizer.components.region_extractor import ThresholdingDetector


class ObjectFilterTrainer:
    def __init__(self):
        self.image_s3 = S3ImageIO(
            endpoint_url="http://192.168.1.194:9000", 
            access_key="sigma-chan", 
            secret_key="sigma-chan-dayo", 
            bucket_name="dataset")
        self.config_s3 = S3ConfigIO(
            endpoint_url="http://192.168.1.194:9000", 
            access_key="sigma-chan", 
            secret_key="sigma-chan-dayo", 
            bucket_name="config")
        self.file_name_list = [item for item in self.image_s3.blob if "region_extractor/thresholding_train" in item]
        self.detector = ThresholdingDetector(type="saturation", threshold_value=100)

    def run(self) -> None:
        object_bbox_array = self.__get_object_size()

        std_deviation = np.std(object_bbox_array)
        mean = np.mean(object_bbox_array)
        res_dict = dict(sigma=float(std_deviation), mu=float(mean))
        print("trained results: ", res_dict)
        self.config_s3.save(res_dict, "detector/object_filter_config.yaml")


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
                


