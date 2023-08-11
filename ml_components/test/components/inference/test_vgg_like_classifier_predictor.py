# coding: utf-8

import os
import glob
import numpy as np
import cv2
import torch
import torchvision

from ml_components.components.inference import VggLikeClassifierPredictor
from ml_components.models.factory import VggLikeClassifierFactory

class TestVggLikeClassifierPredictor:
    def test_init(self) -> None:
        model_name = "hoge.pth"

        factory = VggLikeClassifierFactory()
        tmp_model = factory.create_model()
        optimizer = torch.optim.SGD(tmp_model.parameters(), lr=0.001, momentum=0.9)
        checkpoint = {
            'epoch': 0,
            'model_state_dict': tmp_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': 0.9,
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
            predictor.predict(image)