# coding: utf-8

import os
import glob
import numpy as np
import pytest
import cv2
import torch
import torchvision

from ml_components.train import VggLikeUmapClassifierTrainer
from ml_components.components.dataloader import BinaryClassifierDataloaderFactory
from ml_components.components.factory import IoModuleFactory
from ml_components.components.inference import (
    VggLikeUmapPredictor,
    VggLikeFeatureExtractor,
)
from ml_components.models.factory import VggLikeClassifierFactory


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

    io_factory = IoModuleFactory(**dict(
        endpoint_url=f"http://{os.environ['ENDPOINT_URL']}:9000",
        access_key=os.environ["ACCESS_KEY"],
        secret_key=os.environ["SECRET_KEY"],
    ))
    image_s3 = io_factory.create(**dict(type="image", bucket_name="dataset"))

    def test_init(self) -> None:
        ### create mock model
        torch.save(self.checkpoint, self.model_name)

        trainer = VggLikeUmapClassifierTrainer(
            "classifier/train",
            VggLikeClassifierFactory(),
            self.io_factory.create(**dict(type="image", bucket_name="dataset")),
            n_layer=-3
        )

        assert isinstance(trainer, VggLikeUmapClassifierTrainer)
        #assert isinstance(trainer.vgg.model, torchvision.models.vgg.VGG)
        dataset_dict = trainer.configure_dataset("classifier", "train")
        print(dataset_dict)

        os.remove(self.model_name)

    @pytest.mark.skip("not now")
    @pytest.mark.parametrize("n_layer", [-3, -1])
    def test_various_layers(self, n_layer: int) -> None:
        torch.save(self.checkpoint, self.model_name)

        extractor = VggLikeUmapClassifierTrainer(
            self.model_name, self.factory, n_layer=n_layer
        )

        assert isinstance(extractor, VggLikeUmapClassifierTrainer)
        assert isinstance(extractor.vgg.model, torchvision.models.vgg.VGG)

        this_file_path = os.path.dirname(os.path.abspath(__file__))
        file_list = glob.glob(f"{this_file_path}/*png")
        file_list = file_list + file_list + file_list + file_list + file_list + file_list
        img_list = []
        for file_path in file_list:
            img = cv2.imread(file_path)
            img_list.append(img)
            assert isinstance(img, np.ndarray)
        assert isinstance(img_list, list)
        reduced_feat = extractor.fit(img_list)

        assert isinstance(reduced_feat, np.ndarray)
        np.testing.assert_array_equal(reduced_feat.shape, (12, 2))
        print("reduced feat: ", reduced_feat)
        os.remove(self.model_name)

    @pytest.mark.skip("not now")
    def test_calculate_d0_cluster_simplices(self)-> None:

        input_1 = np.array([[0.0, 0.0], [1.0, -2.0], [2.0, 0.0], [1.0, -1.0]])
        input_2 = np.array([[0.0, -1.0], [1.0, 2.0], [2.0, -1.0], [1.0, 1.0]])

        torch.save(self.checkpoint, self.model_name)
        extractor = VggLikeUmapClassifierTrainer(
            self.model_name, self.factory, n_layer=-3
        )
        ch1 = extractor.get_convex_hull(input_1)
        assert ch1.simplices.size == 3 * 2
        for simplex in ch1.simplices:
            print("simplex: ", simplex)
            print(input_1[simplex, 0], input_1[simplex, 1])

        print("convex hull 1 simplices:", ch1.simplices)
        #extractor.calculate_d0_cluster_simplices(input_1, input_2)
        #print(extractor.calculate_cluster_haussdorf_distance(input_1, input_2))
        os.remove(self.model_name)