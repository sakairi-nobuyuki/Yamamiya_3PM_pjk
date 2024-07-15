# coding: utf-8

from typing import Any, Dict

from ml_components.components.factory import IoModuleFactory
from ml_components.components.operators import load_train_parameters
from ml_components.data_structures import DatasetLoaderParameters, TrainParameters
from ml_components.io import S3ConfigIO


class TestLoadTrainParameters:
    def test_init(
        self,
        mock_io_module_config_dict: Dict[str, str],
        mock_train_parameters_dict: Dict[str, Any],
    ) -> None:
        print(mock_train_parameters_dict)

        io_factory = IoModuleFactory(**mock_io_module_config_dict)
        config_io = io_factory.create(**dict(type="config", bucket_name="config"))

        ## save mock config dict to the bucket
        mock_parameters_path = "train/mock_train_parameters.yaml"
        config_io.save(mock_train_parameters_dict, mock_parameters_path)
        config_io.blob = config_io.get_blob()

        print("blob: ", config_io.blob)

        print("mock train dict: ", mock_train_parameters_dict)

        assert isinstance(io_factory, IoModuleFactory)
        assert isinstance(config_io, S3ConfigIO)
        parameters = load_train_parameters(mock_parameters_path, config_io)
        assert isinstance(parameters, TrainParameters)
        assert isinstance(parameters.dataset, DatasetLoaderParameters)
        assert isinstance(parameters.dataset.type, str)
        assert parameters.dataset.type == mock_train_parameters_dict["dataset"]["type"]
        assert isinstance(parameters.dataset.kaggle, KaggleDatasetLoaderParameters)
