# coding: utf-8

import json

import pytest

from ml_components.pipelines import AccuracyMeasPipeline


@pytest.fixture
def accuracy_meas_parameters_str() -> str:
    parameters_dict = {
        "type": "classification",
        "n_class": 2,
        "base_model": "VGG19",
        "models_directory_path": "dog_cat/test",
        "dataset": {
            "type": "kaggle",
            "train_data_rate": 0.7,
            "val_data_rate": 0.2,
            "dataset_name": "karakaggle/kaggle-cat-vs-dog-dataset",
            "local_dir": "classifier/kaggle/dog_cat",
            "s3_dir": "dog_cat",
        },
    }
    return json.dumps(parameters_dict)


class TestAccuracyMeasPipeline:
    def test_init(self, accuracy_meas_parameters_str: str) -> None:
        pipeline = AccuracyMeasPipeline(accuracy_meas_parameters_str)
        assert isinstance(pipeline, AccuracyMeasPipeline)

        print(pipeline.model_list)
        assert len(pipeline.model_list) > 0
        assert isinstance(pipeline.file_list_dict, dict)
        assert len(pipeline.file_list_dict) > 0
        assert len(pipeline.file_list_dict.keys()) > 0

        pipeline.run()