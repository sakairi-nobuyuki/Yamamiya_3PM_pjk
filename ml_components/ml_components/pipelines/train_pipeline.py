# coding: utf-8

from typing import Dict

from ml_components.components.dataloader import BinaryClassifierDataloaderFactory
from ml_components.components.dataset_loader import KaggleDatasetLoader
from ml_components.components.factory import IoModuleFactory
from ml_components.components.operators import load_train_parameters
from ml_components.io import DataTransferS3, OnnxS3, S3ImageIO
from ml_components.models.factory import VggLikeClassifierFactory
from ml_components.train import VggLikeClassifierTrainer


class TrainPipeline:
    def __init__(self, io_config: Dict[str, str], yaml_path: str) -> None:
        """Initialize train pipeline.
        Tasks:
        - Loading parameters
        - Download dataset
        - Configure trainer

        Args:
            io_config (Dict[str, str]): IO configuration
            yaml_path (str): Parameters yaml path
        """
        io_factory = IoModuleFactory(**io_config)
        self.image_s3 = io_factory.create(**dict(type="image", bucket_name="dataset"))

        ### parameter loading
        print(f">> loading parameters: {yaml_path}")
        self.parameters = load_train_parameters(
            yaml_path, io_factory.create(**dict(type="config", bucket_name="config"))
        )
        print(f">> parameters: {self.parameters}")

        ### configure dataset loader
        print(">> loading dataset")
        # TODO: in future dataset loader should be created with a factory
        self.dataset_loader = KaggleDatasetLoader(self.parameters.dataset, self.image_s3)

        self.dataset_loader.load()

        ### configure trainer
        self.trainer = VggLikeClassifierTrainer(
            self.parameters.dataset.s3_dir,
            # "classifier/train",
            VggLikeClassifierFactory(),
            BinaryClassifierDataloaderFactory(self.image_s3),
            io_factory.create(**dict(type="transfer", bucket_name="models")),
            n_epoch=100,
        )

    def run(self):
        ### download dataset
        self.dataset_loader.load()

        ### train
        self.trainer.train()
