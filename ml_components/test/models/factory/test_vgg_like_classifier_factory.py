# coding: utf-8


from ml_components.models.factory import ModelFactoryTemplate, VggLikeClassifierFactory
import torchvision.models as models
import torch

class TestVggLikeClassifierFactory:
    factory = VggLikeClassifierFactory()

    def test_init(self):
        assert isinstance(self.factory, VggLikeClassifierFactory)
        assert isinstance(self.factory, ModelFactoryTemplate)

    def test_create_model(self):
        model = self.factory.create_model()
        print(model.classifier[-1].in_features, model.classifier[-1].out_features)
        print(model.features)
        assert isinstance(model, models.vgg.VGG)
        assert model.classifier[-1].out_features == 2

    def test_create_forward(self):
        model = self.factory.create_model()

        input_size = (3, 224, 224)
        input_tensor = torch.randn(1, *input_size)

        forward = self.factory.create_forward()

        print(forward(input_tensor))

