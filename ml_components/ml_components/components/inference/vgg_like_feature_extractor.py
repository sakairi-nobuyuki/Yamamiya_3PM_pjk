# coding: utf-8

import os

import numpy as np
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

from ...models.factory import ModelFactoryTemplate
from .template_predictor import TemplatePredictor


class VggLikeFeatureExtractor(TemplatePredictor):
    def __init__(
        self,
        model_factory: ModelFactoryTemplate,
        n_layer: int = -1,
        model_path: str = None,
    ) -> None:
        """Initialize predictor.
        - load model

        Args:
            model_path (str): Model path
            model_factory (ModelFactoryTemplate): Model factory instance
            n_layer (int): The position of a layer to extract feature vector

        Raises:
            TypeError: _description_
            FileNotFoundError: _description_
        """
        print("Configuring a feature extractor")
        ### download and load the model, finally delete it.
        if not isinstance(model_factory, ModelFactoryTemplate):
            raise TypeError(f"{model_factory} model is not that of ModelFactoryTemplate")

        # Create a model instance with factory
        self.model = model_factory.create_model()

        if model_path is not None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"{model_path} is not found.")

            # Extract the model parameters
            checkpoint = torch.load(model_path)
            model_params = checkpoint["model_state_dict"]
            # Load the parameters to the model instance
            self.model.load_state_dict(model_params, strict=False)

        # Set the model to evaluation mode
        self.model.eval()

        self.feature_extractor = self.model
        for i_layer in range(n_layer, 0):
            print(f">> configuring {i_layer}th layer to be an identity mapping.")
            self.feature_extractor.classifier[i_layer] = torch.nn.Identity()
            # self.feature_extractor.classifier[-i_layer] = torch.nn.Identity()
        self.feature_extractor.classifier[n_layer] = torch.nn.Identity()
        print(">> feat extractor configuration: ", self.feature_extractor)

        self.prediction_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image: np.ndarray) -> bool:
        """Returns a predicted result of binary classification.

        Args:
            input (np.ndarray): Input RGB image in OpenCV style.

        Returns:
            bool: Classified result.
        """
        ### Transform input image to a tensor with some preprocess
        if not isinstance(image, np.ndarray):
            print(f"image is not np.array: {type(image), image}")
            return -1
        input = self.prediction_transform(image)

        # Add a batch dimension
        input = input.unsqueeze(0)

        # Make a prediction with the model
        output = self.feature_extractor(input).detach().numpy()

        return output
        # return predicted.item()
