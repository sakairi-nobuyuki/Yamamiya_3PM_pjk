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


class TestVggLikeUmapPredictor:
    model_path: str = "classifier/vgg_umap"
    factory = VggLikeClassifierFactory()
    io_factory = IoModuleFactory(
        **dict(
            endpoint_url=f"http://{os.environ['ENDPOINT_URL']}:9000",
            access_key=os.environ["ACCESS_KEY"],
            secret_key=os.environ["SECRET_KEY"],
        )
    )

    def test_init(self) -> None:
        transfer_s3 = self.io_factory.create(
            **dict(type="transfer", bucket_name="models")
        )
        print("blob: ", transfer_s3.blob)
        assert (
            len(
                [
                    file_path
                    for file_path in transfer_s3.blob
                    if "test_data" in file_path
                ]
            )
            > 0
        )
        assert (
            len(
                [
                    file_path
                    for file_path in transfer_s3.blob
                    if "test_data/labels.yaml" in file_path
                ]
            )
            > 0
        )
        assert (
            len(
                [
                    file_path
                    for file_path in transfer_s3.blob
                    if "test_data/umap_model.pickle" in file_path
                ]
            )
            > 0
        )
        assert (
            len(
                [
                    file_path
                    for file_path in transfer_s3.blob
                    if "test_data/feature_extractor.pth" in file_path
                ]
            )
            > 0
        )

        model_dir_path_name = "classifier/vgg_umap/test_data"
        predictor = VggLikeUmapPredictor(
            f"{model_dir_path_name}/feature_extractor.pth",
            f"{model_dir_path_name}/umap_model.pickle",
            f"{model_dir_path_name}/labels.yaml",
            VggLikeClassifierFactory(),
            self.io_factory,
        )
        assert isinstance(predictor, VggLikeUmapPredictor)


@pytest.mark.skip(reason="no longer effective")
class TestVggLikeUmapPredictorCollapsed:
    model_name = "hoge.pth"
    factory = VggLikeClassifierFactory()
    tmp_model = factory.create_model()
    optimizer = torch.optim.SGD(tmp_model.parameters(), lr=0.001, momentum=0.9)
    io_factory = IoModuleFactory(
        **dict(
            endpoint_url=f"http://{os.environ['ENDPOINT_URL']}:9000",
            access_key=os.environ["ACCESS_KEY"],
            secret_key=os.environ["SECRET_KEY"],
        )
    )
    label_dict = {0: "ok", 1: "ng"}
    s3 = io_factory.create(**dict(type="pickle", bucket_name="models"))
    checkpoint = {
        "epoch": 0,
        "model_state_dict": tmp_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": 0.9,
    }

    def test_init(self) -> None:
        torch.save(self.checkpoint, self.model_name)

        extractor = VggLikeUmapPredictor(
            self.model_name, self.label_dict, self.factory, self.io_factory, n_layer=-1
        )

        assert isinstance(extractor, VggLikeUmapPredictor)
        assert isinstance(extractor.vgg.model, torchvision.models.vgg.VGG)

        os.remove(self.model_name)

    @pytest.mark.parametrize("n_layer", [-1])
    def test_various_layers(self, n_layer: int) -> None:
        torch.save(self.checkpoint, self.model_name)

        extractor = VggLikeUmapPredictor(
            self.model_name,
            self.label_dict,
            self.factory,
            self.io_factory,
            n_layer=n_layer,
        )

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
