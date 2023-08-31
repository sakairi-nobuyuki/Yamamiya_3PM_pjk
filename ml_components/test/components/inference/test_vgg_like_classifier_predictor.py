# coding: utf-8

import glob
import os
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pytest
import torch
import torchvision

from ml_components.components.factory import IoModuleFactory
from ml_components.components.inference import VggLikeClassifierPredictor
from ml_components.io import S3ImageIO
from ml_components.models.factory import VggLikeClassifierFactory


class TestVggLikeClassifierPredictor:
    def test_init(self) -> None:
        """Run a prediction with mock model and mock data."""
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
        torch.save(checkpoint, model_name)

        predictor = VggLikeClassifierPredictor(model_name, factory)

        assert isinstance(predictor, VggLikeClassifierPredictor)
        assert isinstance(predictor.model, torchvision.models.vgg.VGG)

        os.remove(model_name)

        this_file_path = os.path.dirname(os.path.abspath(__file__))
        file_list = glob.glob(f"{this_file_path}/*png")
        for file_path in file_list:
            image = cv2.imread(file_path)
            assert isinstance(image, np.ndarray)
            # print(file_path)
            res = predictor.predict(image)
            assert res in [0, 1]

    # @pytest.mark.skip("dattemurinandamon")
    def test_f_value(self, mock_io_module_config_dict: Dict[str, str]) -> None:
        """Calculate F-value"""
        # find out latest model
        model_dir = os.path.join(
            str(Path(os.path.abspath(__file__)).parent.parent.parent.parent), "models"
        )
        pths = glob.glob(f"{model_dir}/*.pth", recursive=True)
        i_max = np.argmax(
            np.array([pth_name.split("/")[-1].split("_")[1] for pth_name in pths])
        )
        model_path = pths[i_max]

        assert os.path.exists(model_path)

        model_factory = VggLikeClassifierFactory()
        predictor = VggLikeClassifierPredictor(model_path, model_factory)

        # load image data path
        io_factory = IoModuleFactory(**mock_io_module_config_dict)
        img_io = io_factory.create(**dict(type="image", bucket_name="dataset"))
        assert isinstance(img_io, S3ImageIO)

        # run prediction
        val_data_ok_list = [
            img_path for img_path in img_io.blob if "validation" in img_path and "ok"
        ]
        ok_data_predict_res_list = [
            predictor.predict(img_io.load(img_path)) for img_path in val_data_ok_list
        ]

        val_data_ng_list = [
            img_path for img_path in img_io.blob if "validation" in img_path and "ng"
        ]
        ng_data_predict_res_list = [
            predictor.predict(img_io.load(img_path)) for img_path in val_data_ng_list
        ]

        print(ok_data_predict_res_list, ng_data_predict_res_list)

        tp = len([item for item in ok_data_predict_res_list if item == 1])
        fn = len([item for item in ok_data_predict_res_list if item == 0])
        fp = len([item for item in ng_data_predict_res_list if item == 1])
        tn = len([item for item in ng_data_predict_res_list if item == 0])

        print("tp: ", tp)
        print("f-value: ", 2 * tp / (2 * tp + fp + fn))
        print("accuracy: ", (tp + tn) / (tp + fp + fn + tn))
        print("precision: ", tp / (tp + fp))
        print("recall: ", tp / (tp + fn))
