# coding: utf-8

from typing import Dict

import pytest

from ml_components.data_structures import DatasetLoaderParameters


class TestDatasetLoaderParameters:
    def test_dataset_loader_parametres(
        self, mock_dataset_loader_parameters_dict: Dict[str, str]
    ):
        print(mock_dataset_loader_parameters_dict)
        dataset_loader_parameters = DatasetLoaderParameters(
            **mock_dataset_loader_parameters_dict
        )
        assert isinstance(dataset_loader_parameters, DatasetLoaderParameters)
        assert isinstance(dataset_loader_parameters.type, str)
        # print(dataset_loader_parameters)
