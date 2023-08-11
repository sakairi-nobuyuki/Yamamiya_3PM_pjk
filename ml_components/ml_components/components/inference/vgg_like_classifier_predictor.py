# coding: utf-8

import os
import torch
import torchvision
import torchvision.models as models
import numpy as np

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



    def predict(self, image: np.ndarray) -> bool:
        """Returns a predicted result of binary classification.

        Args:
            input (np.ndarray): Input RGB image in OpenCV style.

        Returns:
            bool: Classified result.
        """
        # Prepare your input data
        # For example, load an image and transform it to a tensor
        input = torch.from_numpy(image)

        # Resize and crop the image to 224 x 224
        input = torchvision.transforms.functional.resize(input, 224)
        input = torchvision.transforms.functional.center_crop(input, 224)

        # Permute the channels to (C, H, W)
        input = input.permute(2, 0, 1)

        # Normalize the image with the same mean and std as ImageNet
        input = torchvision.transforms.functional.normalize(
            input,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Add a batch dimension
        input = input.unsqueeze(0)

        # Make a prediction with the model
        output = self.model(input)

        # Convert the output to a class label
        # For example, use the maximum score as the predicted class
        _, predicted = torch.max(output.data, 1)

        print('Predicted class:', predicted.item())

        return predicted.item()
        