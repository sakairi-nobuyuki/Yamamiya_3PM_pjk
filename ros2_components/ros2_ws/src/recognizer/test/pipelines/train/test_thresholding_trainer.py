# coding: utf-8

from typing import Any, Dict

import numpy as np
import pytest
from recognizer.io import S3ConfigIO, S3ImageIO
from recognizer.pipelines.train import ThresholdingTrainer


@pytest.fixture
def mock_train_config() -> Dict[str, Any]:
    return dict(w=100, h=100, aspect_ratio=100, w_aspect_ratio=1.0, n_calls=10)


class TestThresholdingTrainer:
    def test_run(
        self,
        mock_train_config: Dict[str, Any],
        mock_s3_dataset: S3ImageIO,
        mock_s3_config: S3ConfigIO,
    ):
        trainer = ThresholdingTrainer(
            mock_train_config, dataset_s3=mock_s3_dataset, config_s3=mock_s3_config
        )

        assert isinstance(trainer, ThresholdingTrainer)

    def test_thresholding(
        self,
        mock_train_config: Dict[str, Any],
        mock_s3_dataset: S3ImageIO,
        mock_s3_config: S3ConfigIO,
    ):
        trainer = ThresholdingTrainer(
            mock_train_config, dataset_s3=mock_s3_dataset, config_s3=mock_s3_config
        )
        res_dict = trainer.train_thresholding()

        assert isinstance(res_dict, dict)
        assert "type" in res_dict.keys()
        assert "threshold_value" in res_dict.keys()
        assert res_dict["type"] == "hsv" or res_dict["type"] == "saturation"
        assert res_dict["threshold_value"] > 0 and res_dict["threshold_value"] < 256
        if res_dict["type"] == "hsv":
            assert "upper_green" in res_dict.keys()
            assert len(res_dict["upper_green"]) == 3
            assert "lower_green" in res_dict.keys()
            assert len(res_dict["lower_green"]) == 3

    def test_run(
        self,
        mock_train_config: Dict[str, Any],
        mock_s3_dataset: S3ImageIO,
        mock_s3_config: S3ConfigIO,
    ):
        trainer = ThresholdingTrainer(
            mock_train_config, dataset_s3=mock_s3_dataset, config_s3=mock_s3_config
        )
        trainer.run()

        cropped_image_list = [
            item for item in mock_s3_dataset.blob if "region_extractor/object_filter_train" in item
        ]
        assert len(cropped_image_list) > 0
        for file_name in cropped_image_list:
            img = mock_s3_dataset.load(file_name)

            assert isinstance(img, np.ndarray)
            assert img.size > 1
            assert img.shape[2] == 3
