# coding: utf-8

import json
import os
from typing import Dict

from ml_components.components.dataloader import BinaryClassifierDataloaderFactory
from ml_components.components.dataset_loader import (
    CustomDatasetLoader,
)
from ml_components.components.factory import IoModuleFactory
from ml_components.components.operators import load_train_parameters
from ml_components.models.factory import VggLikeClassifierFactory
from ml_components.train import VggLikeClassifierTrainer, VggLikeUmapClassifierTrainer

from ..data_structures import TrainParameters


class TrainPipeline:
    """Train model
    Tasks:
    - Train model and create a model and a label list
    """

    def __init__(self, parameters_str: str) -> None:
        """Initialize train pipeline.
        Tasks:
        - Loading parameters
        - Download dataset
        - Configure trainer

        Args:
            io_config (Dict[str, str]): IO configuration
            yaml_path (str): Parameters yaml path
        """
        io_config = dict(
            endpoint_url=f"http://{os.environ['ENDPOINT_URL']}:9000",
            access_key=os.environ["ACCESS_KEY"],
            secret_key=os.environ["SECRET_KEY"],
        )

        io_factory = IoModuleFactory(**io_config)
        self.image_s3 = io_factory.create(**dict(type="image", bucket_name="dataset"))

        parameters_dict = json.loads(parameters_str)
        self.parameters = TrainParameters(**parameters_dict)
        print(f">> parameters: {self.parameters}")

        ### VGG classifier
        if self.parameters.type == "vgg_classification":
            ### configure dataset loader
            print("VGG like classifier: ")
            print(">> loading dataset")

            ### configure trainer
            self.trainer = VggLikeClassifierTrainer(
                self.parameters.dataset.s3_dir,
                VggLikeClassifierFactory(),
                BinaryClassifierDataloaderFactory(self.image_s3),
                io_factory.create(**dict(type="transfer", bucket_name="models")),
                n_epoch=100,
            )

            ### download dataset
            self.dataset_loader.load()

        elif self.parameters.type == "umap_vgg_classification":
            print("VGG like UMAP classifier: ")
            print(">> loading dataset")
            dataset_loader = CustomDatasetLoader(self.parameters.dataset, self.image_s3)
            label_file_path_dict_list = dataset_loader.load()
            print("label_file_path_dict_list: ", label_file_path_dict_list)

            self.trainer = VggLikeUmapClassifierTrainer(
                label_file_path_dict_list,
                VggLikeClassifierFactory(),
                io_factory,
                n_layer=-3,
            )

    def run(self):
        ### train
        return self.trainer.train()
