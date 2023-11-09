# coding: utf-8

import glob
import os
from typing import Dict, List
import cv2
import json
import numpy as np
import pytest
import torch
import torchvision

from ml_components.components.dataset_loader import CustomDatasetLoader
from ml_components.components.factory import IoModuleFactory
from ml_components.models.factory import VggLikeClassifierFactory
from ml_components.train import VggLikeUmapClassifierTrainer
from ml_components.data_structures import DatasetLoaderParameters

@pytest.fixture
def mock_dataset() -> List[Dict[str, str]]:
    dataset_parameters_dict = dict(
        type="custom",
        train_data_rate=0.7,
        val_data_rate=0.2,
        dataset_name="binary_classifier_test",
        s3_dir="classifier/trainer_test",
    )
    
    parameters = DatasetLoaderParameters(**dataset_parameters_dict)
    io_factory = IoModuleFactory(
        **dict(
            endpoint_url=f"http://{os.environ['ENDPOINT_URL']}:9000",
            access_key=os.environ["ACCESS_KEY"],
            secret_key=os.environ["SECRET_KEY"],
        )
    )
    image_s3 = io_factory.create(**dict(type="image", bucket_name="dataset"))

    dataset_loader = CustomDatasetLoader(parameters, image_s3)
    label_file_path_dict_list = dataset_loader.load()

    return label_file_path_dict_list

class TestVggLikeUmapClassifierTrainer:
    ### create mock model cofig
    model_name = "hoge.pth"
    factory = VggLikeClassifierFactory()
    tmp_model = factory.create_model()
    optimizer = torch.optim.SGD(tmp_model.parameters(), lr=0.001, momentum=0.9)
    checkpoint = {
        "epoch": 0,
        "model_state_dict": tmp_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": 0.9,
    }

    io_factory = IoModuleFactory(
        **dict(
            endpoint_url=f"http://{os.environ['ENDPOINT_URL']}:9000",
            access_key=os.environ["ACCESS_KEY"],
            secret_key=os.environ["SECRET_KEY"],
        )
    )
    image_s3 = io_factory.create(**dict(type="image", bucket_name="dataset"))

    def test_init(self, mock_dataset: List[Dict[str, str]]) -> None:
        ### create mock model
        torch.save(self.checkpoint, self.model_name)

        trainer = VggLikeUmapClassifierTrainer(
            mock_dataset,
            VggLikeClassifierFactory(),
            self.io_factory.create(**dict(type="image", bucket_name="dataset")),
            n_layer=-3,
        )

        assert isinstance(trainer, VggLikeUmapClassifierTrainer)
        # assert isinstance(trainer.vgg.model, torchvision.models.vgg.VGG)
        # dataset_dict = trainer.configure_dataset("classifier", "train")
        # print(dataset_dict)

        os.remove(self.model_name)

    #    @pytest.mark.skip("not now")
    @pytest.mark.parametrize("n_layer", [-3, -1])
    def test_various_layers(self, n_layer: int, mock_dataset: List[Dict[str, str]]) -> None:
        # torch.save(self.checkpoint, self.model_name)
        

        trainer = VggLikeUmapClassifierTrainer(
            mock_dataset,
            VggLikeClassifierFactory(),
            self.image_s3,
            n_layer=n_layer,
        )

        assert isinstance(trainer, VggLikeUmapClassifierTrainer)
        assert isinstance(trainer.vgg.model, torchvision.models.vgg.VGG)


        reduced_feat = trainer.train()

        assert isinstance(reduced_feat, np.ndarray)
        np.testing.assert_array_equal(reduced_feat.shape, (8, 2))
        print("reduced feat: ", reduced_feat)
        print("reduced feat shape: ", reduced_feat.shape)
        print("reduced feat type: ", type(reduced_feat))
