# coding: utf-8

from typing import Dict

from ml_components.components.dataset_loader import KaggleDatasetLoader
from ml_components.data_structures import DatasetLoaderParameters
from ml_components.io import S3ImageIO


class TestKaggleDatasetLoader:
    def test_is_dataset_in_the_storage(
        self,
        mock_dataset_loader_parameters_dict: Dict[str, str],
        mock_s3_dataset: S3ImageIO,
    ) -> None:
        parameters_dict = mock_dataset_loader_parameters_dict
        parameters_dict["s3_dir"] = "hoge"
        kaggle_parameters = DatasetLoaderParameters(**mock_dataset_loader_parameters_dict)

        assert isinstance(kaggle_parameters, DatasetLoaderParameters)

        dataset_loader = KaggleDatasetLoader(kaggle_parameters, mock_s3_dataset)
        assert isinstance(dataset_loader, KaggleDatasetLoader)

        assert dataset_loader.is_dataset_in_the_storage() is False

        parameters_dict["s3_dir"] = "classifier/train"
        kaggle_parameters = DatasetLoaderParameters(**mock_dataset_loader_parameters_dict)
        dataset_loader = KaggleDatasetLoader(kaggle_parameters, mock_s3_dataset)
        assert dataset_loader.is_dataset_in_the_storage() is True

        assert len(dataset_loader.get_label_list()) == 2
