# coding: utf-8

import glob
import os

import cv2
import numpy as np
import pytest
import torch
import torchvision

from ml_components.components.inference import VggLikeFeatureExtractor
from ml_components.models.factory import VggLikeClassifierFactory


class TestVggLikeFeatureExtractor:
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

    def test_init(self):
        """Run a prediction with mock model and mock data."""

        torch.save(self.checkpoint, self.model_name)

        extractor = VggLikeFeatureExtractor(self.factory, model_path=self.model_name)

        assert isinstance(extractor, VggLikeFeatureExtractor)
        assert isinstance(extractor.model, torchvision.models.vgg.VGG)

        os.remove(self.model_name)

    @pytest.mark.parametrize(
        # "n_layer", [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]
        "n_layer",
        [-1, -2],
    )
    def test_various_layers(self, n_layer: int) -> None:
        torch.save(self.checkpoint, self.model_name)

        extractor = VggLikeFeatureExtractor(
            self.factory, n_layer=n_layer, model_path=self.model_name
        )

        assert isinstance(extractor, VggLikeFeatureExtractor)
        assert isinstance(extractor.model, torchvision.models.vgg.VGG)

        os.remove(self.model_name)

        this_file_path = os.path.dirname(os.path.abspath(__file__))
        file_list = glob.glob(f"{this_file_path}/*png")
        for file_path in file_list:
            image = cv2.imread(file_path)
            assert isinstance(image, np.ndarray)
            # print(file_path)
            res = extractor.predict(image)
            print(type(res), res.shape)
            # assert res in [0, 1]
            assert isinstance(res, np.ndarray)
            assert res.size > 0
            # assert [n for n in list(res.size())][0] > 0
            # assert len(list(res["feature"].size())) > 0
            # assert len(res.keys()) > 0
