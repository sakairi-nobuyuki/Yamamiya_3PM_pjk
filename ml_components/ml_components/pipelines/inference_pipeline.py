# coding: utf-8

import os
import json
from typing import Any, Dict

import numpy as np

from ..components.factory import IoModuleFactory
from ..components.inference import InferenceContext, VggLikeClassifierPredictor
from ..data_structures import PredictionParameters
from ..models.factory import ModelFactoryTemplate, VggLikeClassifierFactory


class InferencePipeline:
    """Inference pipeline supposed to be used from a client code or production code."""

    def __init__(self, parameters_str: str) -> None:
        """Initialize the pipeline

        Args:
            parameters_str (str): Parameters in string

        Raises:
            TypeError: Parameters dataclass validation.
            NotImplementedError: Parameters are validated that such function is not implemented.
        """

        ### load parameters
        ### parse str into PredictParameters
        parameters_dict = json.loads(parameters_str)
        parameters = PredictionParameters(**parameters_dict)
        if isinstance(parameters, PredictionParameters):
            self.parameers = parameters
        else:
            raise TypeError(
                f"{type(parameters)} is not an instance of PredictionParameters"
            )

        ### load model factory
        print("Configuring DNN model")
        io_factory = IoModuleFactory()

        ### TODO: this model download part should be implemented in the parant class
        s3_model_io = io_factory.create(**{"type": "transfer", "bucket_name": "models"})
        model_path = s3_model_io.load(self.parameers.model_path)
        self.model_factory = VggLikeClassifierFactory()

        ### select predictor in accordance with parameters
        print("Configuring the predictor")
        if self.parameers.type == "binary":
            print(">> prediction type: binary classification")
            print(">> model path: ", self.parameers.model_path)
            self.predictor = InferenceContext(
                #VggLikeClassifierPredictor(self.parameers.model_path, self.model_factory)
                VggLikeClassifierPredictor(model_path["file_name"], self.model_factory)
            )
        else:
            raise NotImplementedError(f"{self.parameers.type} is not implemented")
        os.remove(model_path["file_name"])

    def predict(self, input: np.ndarray) -> Any:
        return self.predictor.run(input)
