# coding: utf-8

import json
from typing import Dict

import numpy as np
import pytest

from ml_components.components.factory import IoModuleFactory
from ml_components.data_structures import PredictionParameters
from ml_components.pipelines import InferencePipeline


@pytest.mark.vgg_pipeline
@pytest.mark.skip(reason="no longer needed")
class TestInferencePipeline:
    def test_init(self, mock_io_module_config_dict: Dict[str, str]):
        model_path = "recognizer/dog_cat/checkpoint_20230823_18epoch_train_loss0.04602494465797839_val_loss0.059299631915891014.pth"
        io_factory = IoModuleFactory(**mock_io_module_config_dict)
        config_s3 = io_factory.create(**dict(type="config", bucket_name="models"))
        print("if model is in the bucket: ", model_path in config_s3.blob)
        assert model_path in config_s3.blob
        parameters = PredictionParameters(
            **dict(
                model_path=model_path, base_model="VGG19", category="dnn", type="binary"
            )
        )

        assert isinstance(parameters, PredictionParameters)
        parameters_str = json.dumps(parameters.model_dump())
        assert isinstance(parameters_str, str)

        inference_pipeline = InferencePipeline(parameters_str)
        assert isinstance(inference_pipeline, InferencePipeline)

        ### TODO: input an image to test inference_pipeline.predict
        img = np.zeros([1000, 1000, 3], dtype=np.uint8)
        img.fill(255)
        res = inference_pipeline.run(img)
        print(res)


@pytest.mark.vgg_umap_pipeline
class TestInferenceVggUmapPipeline:
    def test_init(self, mock_io_module_config_dict: Dict[str, str]):
        # model_path = "recognizer/dog_cat/checkpoint_20230823_18epoch_train_loss0.04602494465797839_val_loss0.059299631915891014.pth"
        # model_path = "classifier/vgg_umap/20240715014130/feature_extractor.pth"
        model_path = "classifier/vgg_umap/20240715014130"
        io_factory = IoModuleFactory(**mock_io_module_config_dict)
        config_s3 = io_factory.create(**dict(type="config", bucket_name="models"))
        print("if model is in the bucket: ", model_path in config_s3.blob)
        assert f"{model_path}/feature_extractor.pth" in config_s3.blob
        parameters = PredictionParameters(
            **dict(
                model_path=model_path,
                base_model="VGG19",
                category="vgg-umap",
                type="binary",
            )
        )

        assert isinstance(parameters, PredictionParameters)
        parameters_str = json.dumps(parameters.model_dump())
        assert isinstance(parameters_str, str)

        inference_pipeline = InferencePipeline(parameters_str)
        assert isinstance(inference_pipeline, InferencePipeline)

        ### TODO: input an image to test inference_pipeline.predict
        img = np.zeros([1000, 1000, 3], dtype=np.uint8)
        img.fill(255)
        res = inference_pipeline.run(img)
        print(res)
