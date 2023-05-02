# coding: utf-8

import torch

from ml_components.components.dataloader import BinaryClassifierDataloaderFactory
from ml_components.data_structures import ClassifierDataloaderDataclass
from ml_components.io import S3ImageIO


class TestBinaryClassifierDataloaderFactory:
    def test_init(self, mock_s3_dataset: S3ImageIO) -> None:
        dataloader_factory = BinaryClassifierDataloaderFactory(mock_s3_dataset)

        assert isinstance(dataloader_factory, BinaryClassifierDataloaderFactory)

    def test_create(self, mock_s3_dataset: S3ImageIO) -> None:
        dataloader_factory = BinaryClassifierDataloaderFactory(mock_s3_dataset)
        dataloader = dataloader_factory.create("classifier/train")

        assert isinstance(dataloader, ClassifierDataloaderDataclass)
        assert isinstance(dataloader.train_loader, torch.utils.data.DataLoader)
        assert isinstance(dataloader.validation_loader, torch.utils.data.DataLoader)

        for inputs, labels in dataloader.train_loader:
            print("labels", labels, len(labels))
