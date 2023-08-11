# coding: utf-8

from typing import Dict

from ml_components.components.dataloader import BinaryClassifierDataloaderFactory
from ml_components.io import S3ImageIO, OnnxS3, DataTransferS3
from ml_components.models.factory import VggLikeClassifierFactory 
from ml_components.train import VggLikeClassifierTrainer
from ml_components.components.factory import IoModuleFactory


class TrainPipeline:
    def __init__(self, io_config: Dict[str, str]) -> None:
        io_factory = IoModuleFactory(**io_config)
        self.image_s3 = io_factory.create(**dict(type="image", bucket_name="dataset"))

        self.trainer = VggLikeClassifierTrainer(
            "classifier/train",
            VggLikeClassifierFactory(),
            BinaryClassifierDataloaderFactory(self.image_s3),
            io_factory.create(**dict(type="transfer", bucket_name="models")),
            n_epoch=100,
        )

    def run(self):
        self.trainer.train()