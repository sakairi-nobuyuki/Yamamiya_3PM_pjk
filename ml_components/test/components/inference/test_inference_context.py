# coding: utf-8

import os
import torch

from ml_components.components.inference import InferenceContext, TemplatePredictor, VggLikeClassifierPredictor
from ml_components.models.factory import VggLikeClassifierFactory

class TestInferenceContext:
    def test_init_vgg_like(self):
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

        inference_context = InferenceContext(VggLikeClassifierPredictor(model_name, factory)) 
        assert isinstance(inference_context, InferenceContext)

        os.remove(model_name)

        