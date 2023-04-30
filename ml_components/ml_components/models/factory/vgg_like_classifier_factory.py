# coding:

from typing import Callable
import torch.nn as nn
import torchvision.models as models

from ml_components.models.factory import ModelFactoryTemplate


class VggLikeClassifierFactory(ModelFactoryTemplate):
    def __init__(self, n_classes: int = 2):
        """Create a VGG like classifier.

        Args:
            n_classes (int, optional): _description_. Defaults to 2.
        """
        self.n_classes = n_classes
        

    def create_model(self) -> models.vgg.VGG:
        self.model = models.vgg19(pretrained=True)
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_ftrs, self.n_classes)
        return self.model

    def create_forward(self) -> Callable:
        return self.model