# coding: utf-8

import os
import glob
import numpy as np
import pytest
import cv2
import torch
import torchvision

from ml_components.components.inference import (
    VggLikeUmapPredictor,
    VggLikeFeatureExtractor,
)
from ml_components.models.factory import VggLikeClassifierFactory


class TestVggLikeUmapPredictor:
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

    def test_init(self) -> None:
        torch.save(self.checkpoint, self.model_name)

        extractor = VggLikeUmapPredictor(self.model_name, self.factory, n_layer=-1)

        assert isinstance(extractor, VggLikeUmapPredictor)
        assert isinstance(extractor.vgg.model, torchvision.models.vgg.VGG)

        os.remove(self.model_name)

    @pytest.mark.parametrize("n_layer", [-1])
    def test_various_layers(self, n_layer: int) -> None:
        torch.save(self.checkpoint, self.model_name)

        extractor = VggLikeUmapPredictor(self.model_name, self.factory, n_layer=n_layer)

        assert isinstance(extractor, VggLikeUmapPredictor)
        assert isinstance(extractor.vgg.model, torchvision.models.vgg.VGG)

        this_file_path = os.path.dirname(os.path.abspath(__file__))
        file_list = glob.glob(f"{this_file_path}/*png")
        for file_path in file_list:
            print("file path: ", file_path)
            image = cv2.imread(file_path)
            print(image.mean())
            assert isinstance(image, np.ndarray)
            # print(file_path)
            res = extractor.predict(image)
            # print(res.mean())

            #            assert res.ndim == 2
            #            assert res.shape[0] == 1
            # assert res in [0, 1]

            #            assert isinstance(res, np.ndarray)
            #            assert res.size > 0
            # print("reduced res, mean: ", res, res.mean(), res.max(), res.min())
            #            print("reduced res size: ", res.size)
            print("reduced res: ", res, type(res))

        os.remove(self.model_name)
