# coding: utf-8

import socket

from recognizer.io import S3ConfigIO, S3ImageIO
from recognizer.pipelines.train import RegionExtractorTrainPipeline


class TestRegionExtractorTrainPipeline:
    def test_init(self, mock_s3_dataset: S3ImageIO, mock_s3_config: S3ConfigIO):
        train_config = {
            "threshold": {
                "w": 110,
                "h": 100,
                "aspect_ratio": 1.0,
            }
        }
        trainer = RegionExtractorTrainPipeline(
            train_config, dataset_s3=mock_s3_dataset, config_s3=mock_s3_config
        )

        assert isinstance(trainer, RegionExtractorTrainPipeline)
        assert trainer.thresholding_trainer.w == train_config["threshold"]["w"]
        assert trainer.thresholding_trainer.h == train_config["threshold"]["h"]
        assert (
            trainer.thresholding_trainer.aspect_ratio == train_config["threshold"]["aspect_ratio"]
        )
        assert trainer.thresholding_trainer.w_aspect_ratio is not None
