# coding: utf-8

from ml_components.components.dataloader import BinaryClassifierDataloaderFactory
from ml_components.io import S3ImageIO

class TestBinaryClassifierDataloaderFactory:
    def test_init(self, mock_s3_dataset: S3ImageIO) -> None:

        dataloader_factory = BinaryClassifierDataloaderFactory(S3ImageIO)

        assert isinstance(dataloader_factory, BinaryClassifierDataloaderFactory)

    def test_create(self, mock_s3_dataset: S3ImageIO) -> None:

        dataloader_factory = BinaryClassifierDataloaderFactory(S3ImageIO)
        dataloader_factory.create(" classifier/train/")
        