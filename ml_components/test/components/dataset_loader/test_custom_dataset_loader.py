# coding: utf-8

import pytest

from ml_components.components.dataset_loader import CustomDatasetLoader
from ml_components.data_structures import DatasetLoaderParameters
from ml_components.io import S3ImageIO


@pytest.fixture(scope="function")
def mock_datasetloader_parameters() -> DatasetLoaderParameters:
    return DatasetLoaderParameters(
        **dict(
            type="custom",
            train_data_rate=0.7,
            val_data_rate=0.2,
            dataset_name="yamamiya_pm",
            local_dir="classifier/kaggle",
            s3_dir="classifier/train",
        )
    )


class TestCustomDatasetLoader:
    def test_init(
        self,
        mock_datasetloader_parameters: DatasetLoaderParameters,
        mock_s3_dataset: S3ImageIO,
    ):
        dataset_loader = CustomDatasetLoader(
            mock_datasetloader_parameters, mock_s3_dataset
        )
        assert isinstance(dataset_loader, CustomDatasetLoader)

        data_dict_list = dataset_loader.load()

        assert len(data_dict_list) > 2
        print(data_dict_list)
        assert len(list(set(map(lambda x: list(x.values())[0], data_dict_list)))) == 2
