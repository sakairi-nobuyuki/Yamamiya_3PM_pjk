# coding: utf-8

from typing import Any, Dict
from uuid import uuid4

import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real

from ...components.region_extractor import DetectorFactory
from ...components.region_extractor.train import (
    ObjectFilterTrainer,
    ThresholdingHsvTrainer,
    ThresholdingSaturationTrainer,
)
from ...io import S3ConfigIO, S3ImageIO


class ThresholdingTrainer:
    # def __init__(self, train_config: Dict[str, str], thresholding_trainer: ThresholdingTrainer, object_filter_trainer: ObjectFilterTrainer) -> None:
    def __init__(
        self, train_config: Dict[str, str], dataset_s3: S3ImageIO, config_s3: S3ConfigIO
    ) -> None:
        """Initialize various training modules"""
        if not isinstance(dataset_s3, S3ImageIO):
            raise TypeError(f"dataset_s3 should be an instance of S3ImageIO: {type(dataset_s3)}")
        if not isinstance(config_s3, S3ConfigIO):
            raise TypeError(f"config_s3 should be an instance of S3ConfigIO: {type(config_s3)}")

        self.train_config = train_config
        print("Thresholoding trainer: \n  train config: ", train_config)
        print("  initializing hsv trainer:")
        self.thresholding_hsv_trainer = ThresholdingHsvTrainer(**train_config, s3=dataset_s3)
        print("  initializing saturation trainer:")
        self.thresholding_saturation_trainer = ThresholdingSaturationTrainer(
            **train_config, s3=dataset_s3
        )
        self.s3_config = config_s3
        self.s3_dataset = dataset_s3
        print("  initialized storage")
        self.trained_thresholding_parameter_file = "detector/thresholding.yaml"

    def run(self) -> Dict[str, Any]:
        ### train thresholding
        res_threshold = self.train_thresholding()
        self.s3_config.save(res_threshold, self.trained_thresholding_parameter_file)

        ### create dataset for object_filter training
        self.__create_object_filter_dataset(res_threshold)
        print("res thresholding: ", res_threshold, type(res_threshold))

    def __create_object_filter_dataset(self, config_dict: Dict[str, Any]) -> None:
        detector_factory = DetectorFactory(config_dict)
        detector = detector_factory.create()
        file_name_list = [
            item for item in self.s3_dataset.blob if "region_extractor/thresholding_train" in item
        ]

        for file_name in file_name_list:
            img = self.s3_dataset.load(file_name)
            bboxes = detector.detect(img)
            for bbox in bboxes:
                x, y, w, h = bbox
                img_cropped = img[y : y + h, x : x + w]
                file_name_cropped = (
                    f"region_extractor/object_filter_train/{str(uuid4()).replace('-', '')}.png"
                )
                assert isinstance(img_cropped, np.ndarray)
                self.s3_dataset.save(img_cropped, file_name_cropped)

    def train_thresholding(self) -> Dict[str, Any]:
        ## train thresholding things
        print("training thresholoing")
        res_thresholding_hsv = self.thresholding_hsv_trainer.run()
        res_thresholding_saturation = self.thresholding_saturation_trainer.run()

        min_loss_thresholding_hsv = np.min(res_thresholding_hsv.func_vals)
        min_loss_thresholding_saturation = np.min(res_thresholding_saturation.func_vals)

        if min_loss_thresholding_hsv > min_loss_thresholding_saturation:
            res_list = res_thresholding_hsv.x_iters[np.argmin(res_thresholding_hsv.func_vals)]
            res_dict = dict(
                type="hsv",
                threshold_value=int(res_list[0]),
                lower_green=[int(res_list[1]), int(res_list[2]), int(res_list[3])],
                upper_green=[int(res_list[4]), int(res_list[5]), int(res_list[6])],
            )
        else:
            res_list = res_thresholding_saturation.x_iters[
                np.argmin(res_thresholding_saturation.func_vals)
            ]
            res_dict = dict(type="saturation", threshold_value=int(res_list[0]))

        print("train threshold res: ", res_dict)

        return res_dict
