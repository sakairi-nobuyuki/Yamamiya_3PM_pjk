# coding: utf-8

import glob
import os

import cv2
import numpy as np
import pytest
import torch
import torchvision

from ml_components.components.factory import IoModuleFactory
from ml_components.components.inference import (
    VggLikeFeatureExtractor,
    VggLikeUmapPredictor,
)
from ml_components.models.factory import VggLikeClassifierFactory
from ml_components.train import VggLikeUmapClassifierTrainer


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
    config_s3 = io_factory.create(**dict(type="config", bucket_name="models"))

    def test_init(self) -> None:
        ### create mock model
        torch.save(self.checkpoint, self.model_name)

        trainer = VggLikeUmapClassifierTrainer(
            [
                {"hoge/hoge/hoge.png": "ok"},
                {"piyo/piyo/piuo": "ng"},
                {"fuga/guga/futa/fuga": "ok"},
            ],
            VggLikeClassifierFactory(),
            #            self.image_s3,
            #            self.config_s3,
            self.io_factory,
            n_layer=-3,
        )

        assert isinstance(trainer, VggLikeUmapClassifierTrainer)
        assert isinstance(trainer.vgg.model, torchvision.models.vgg.VGG)
        # dataset_dict = trainer.configure_dataset("classifier", "train")
        # print(dataset_dict)

        os.remove(self.model_name)

    @pytest.mark.parametrize("n_layer", [-3, -1])
    def test_various_layers(self, n_layer: int) -> None:
        torch.save(self.checkpoint, self.model_name)

        trainer = VggLikeUmapClassifierTrainer(
            [
                {"classifier_test/ng/fuga_13_0.png": "ng"},
                {"classifier_test/ng/fuga_13_0.png": "ng"},
                {"classifier_test/ok/fuga_32_0.png": "ok"},
                {"classifier_test/ok/fuga_32_0.png": "ok"},
                {"classifier_test/ok/fuga_32_0.png": "ok"},
            ],
            VggLikeClassifierFactory(),
            #            self.io_factory.create(**dict(type="image", bucket_name="dataset")),
            #            self.io_factory.create(**dict(type="config", bucket_name="models")),
            self.io_factory,
            n_layer=n_layer,
        )

        assert isinstance(trainer, VggLikeUmapClassifierTrainer)
        assert isinstance(trainer.vgg.model, torchvision.models.vgg.VGG)
        assert len(trainer.aggregate_label_map_dict(trainer.data_path_dict_list)) == 2

        os.remove(self.model_name)
