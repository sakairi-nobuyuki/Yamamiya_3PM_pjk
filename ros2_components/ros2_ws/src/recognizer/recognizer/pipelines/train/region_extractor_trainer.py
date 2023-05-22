# coding: utf-8

from typing import Dict, Any

import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real

from ...components.region_extractor.train import ObjectFilterTrainer, ThresholdingHsvTrainer, ThresholdingSaturationTrainer
from ...components.region_extractor import DetectorFactory
from ...io import S3ConfigIO, S3ImageIO


class RegionExtractorTrainPipeline:
    # def __init__(self, train_config: Dict[str, str], thresholding_trainer: ThresholdingTrainer, object_filter_trainer: ObjectFilterTrainer) -> None:
    def __init__(
        self, phase: str, config_path: str, train_config: Dict[str, str], dataset_s3: S3ImageIO, config_s3: S3ConfigIO
    ) -> None:
        """Initialize various training modules"""
        if not isinstance(dataset_s3, S3ImageIO):
            raise TypeError(f"dataset_s3 should be an instance of S3ImageIO: {type(dataset_s3)}")
        if not isinstance(config_s3, S3ConfigIO):
            raise TypeError(f"config_s3 should be an instance of S3ConfigIO: {type(config_s3)}")
        self.train_config = train_config
        
        self.thresholding_hsv_trainer = ThresholdingHsvTrainer()
        self.thresholding_saturation_trainer = ThresholdingSaturationTrainer()
        self.object_filter_trainer = ObjectFilterTrainer(image_s3=dataset_s3)
        self.s3_config = config_s3
        self.trained_thresholding_parameter_file = "detector/thresholding_config.yaml"

    def run(self) -> Dict[str, Any]:
        ### train thresholding
        res_threshold = self.train_thresholding()

        ## train object filter
        ### create dataset for object_filter training
        detector = DetectorFactory(**res_threshold)
        

        ## create a config dict and save
        pass

    def __train_object_filter(self):
        # load threshold config
        
        # create dataset
        
        # train
        pass

    def train_thresholding(self)  -> Dict[str, Any]:
        ## train thresholding things
        print("training thresholoing")
        res_thresholding_hsv = self.thresholding_hsv_trainer.run()
        res_thresholding_saturation = self.thresholding_saturation_trainer.run()
        
        min_loss_thresholding_hsv = np.min(res_thresholding_hsv.func_value)
        min_loss_thresholding_saturation = np.min(res_thresholding_saturation.func_value)
        
        if min_loss_thresholding_hsv > min_loss_thresholding_saturation:
            res_list = res_thresholding_hsv.x_iters[np.argmin(res_thresholding_hsv.func_vals)]
            res_dict = dict(type="hsv", thresholding_value=res_list[0], lower_green=[res_list[1], res_list[2], res_list[3]], upper_green=[res_list[4], res_list[5], res_list[6]])
        else:
            res_list = res_thresholding_saturation.x_iters[np.argmin(res_thresholding_saturation.func_vals)]
            res_dict = dict(type="saturation", thresholding_value=res_list[0])

        print("train threshold res: ", res_dict)

        return res_dict