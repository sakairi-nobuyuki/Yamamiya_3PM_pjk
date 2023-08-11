# coding: utf-8

import pytest
import torch
from typing import Dict

from ml_components.io import S3ImageIO, OnnxS3

minio_endpoint_url="http://192.168.1.194:9000"

@pytest.fixture
def mock_dataloader() -> torch.utils.data.DataLoader:
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return torch.tensor(self.data[idx])

    class CustomDataLoader(torch.utils.data.DataLoader):
        def __init__(self):
            dataset = CustomDataset()
            super().__init__(dataset=dataset, batch_size=2)

    data_loader = CustomDataLoader()

    return data_loader


@pytest.fixture
def mock_s3_dataset():
    return S3ImageIO(
        endpoint_url=minio_endpoint_url,
        access_key="sigma-chan",
        secret_key="sigma-chan-dayo",
        bucket_name="dataset",
    )

@pytest.fixture
def mock_s3_onnx():
    return OnnxS3(
        endpoint_url=minio_endpoint_url,
        access_key="sigma-chan",
        secret_key="sigma-chan-dayo",
        bucket_name="models")

@pytest.fixture
def mock_io_module_config_dict() -> Dict[str, str]:
    return dict(endpoint_url=minio_endpoint_url, access_key="sigma-chan",
        secret_key="sigma-chan-dayo")