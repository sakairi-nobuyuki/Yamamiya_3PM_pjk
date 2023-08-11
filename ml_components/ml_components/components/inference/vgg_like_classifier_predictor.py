# coding: utf-8

import os
import torch
import torchvision
import torchvision.models as models
import numpy as np
import cv2

from .template_predictor import TemplatePredictor
from ...models.factory import ModelFactoryTemplate

class VggLikeClassifierPredictor(TemplatePredictor):

    def __init__(self, model_path: str, model_factory: ModelFactoryTemplate)->None:
        """Initialize predictor.
        - load model

        Args:
            model_path (str): _description_
            model_factory (ModelFactoryTemplate): _description_

        Raises:
            TypeError: _description_
            FileNotFoundError: _description_
        """
        if not isinstance(model_factory, ModelFactoryTemplate):
            raise TypeError(f"{model_factory} model is not that of ModelFactoryTemplate")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} is not found.")

        checkpoint = torch.load(model_path)

        # Extract the model parameters
        model_params = checkpoint['model_state_dict']

        # Create a model instance with factory
        self.model = model_factory.create_model()

        # Load the parameters to the model instance
        self.model.load_state_dict(model_params)

        # Set the model to evaluation mode
        self.model.eval()

        self.prediction_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
        input = self.prediction_transform(image)

        # Add a batch dimension
        input = input.unsqueeze(0)

        # Make a prediction with the model
        output = self.model(input)

        # Convert the output to a class label
        # For example, use the maximum score as the predicted class
        _, predicted = torch.max(output.data, 1)

        print('Predicted class:', predicted.item())

        return predicted.item()
        