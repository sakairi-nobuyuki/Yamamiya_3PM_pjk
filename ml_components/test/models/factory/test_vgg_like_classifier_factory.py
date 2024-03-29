# coding: utf-8


import torch
import torchvision.models as models

from ml_components.models.factory import ModelFactoryTemplate, VggLikeClassifierFactory


class TestVggLikeClassifierFactory:
    factory = VggLikeClassifierFactory()

    def test_init(self):
        assert isinstance(self.factory, VggLikeClassifierFactory)
        assert isinstance(self.factory, ModelFactoryTemplate)

    def test_create_model(self):
        model = self.factory.create_model()
        print(model.classifier[-1].in_features, model.classifier[-1].out_features)
        print("model.fetures: ", model.features)
        print("type(model): ", type(model))
        print("model parameters: ", model.parameters)
        print("model named parameters: ", model.named_parameters)
        print("model feat[0]: ", model.classifier[0].in_features)
        print("model feat[0]: ", model.features[0])
        model.eval()
        assert isinstance(model, models.vgg.VGG)
        assert model.classifier[-1].out_features == 2

    def test_create_forward(self):
        model = self.factory.create_model()

        input_size = (3, 224, 224)
        input_tensor = torch.randn(1, *input_size)

        forward = self.factory.create_forward()

        print("forward: ", forward(input_tensor))
