# coding: utf-8

import json
import os
from typing import Any, Dict

import numpy as np

from ..components.factory import IoModuleFactory
from ..components.inference import (
    InferenceContext,
    VggLikeClassifierPredictor,
    VggLikeUmapPredictor,
)
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
            self.parameters = parameters
        else:
            raise TypeError(
                f"{type(parameters)} is not an instance of PredictionParameters"
            )

        ### load model factory
        print("Configuring DNN model")
        io_factory = IoModuleFactory()

        ### TODO: this model download part should be implemented in the parant class
        s3_transfer_io = io_factory.create(
            **{"type": "transfer", "bucket_name": "models"}
        )
        self.model_factory = VggLikeClassifierFactory()

        ### select predictor in accordance with parameters
        print("Configuring the predictor")
        if self.parameters.type == "binary":
            print(">> prediction type: binary classification")
            if self.parameters.category == "dnn":
                print(">> classifier category: vgg")
                print(">> model path: ", self.parameters.model_path)
                model_path = s3_transfer_io.load(self.parameters.model_path)
                self.predictor = InferenceContext(
                    VggLikeClassifierPredictor(
                        model_path["file_name"], self.model_factory
                    )
                )
                os.remove(model_path)
            elif self.parameters.category == "vgg-umap":
                print(">> classifier category: vgg umap")
                print(">> model dir path: ", self.parameters.model_path)

                ### download VGG model from cloud storage
                vgg_model_path = os.path.join(
                    self.parameters.model_path, "feature_extractor.pth"
                )
                model_path = s3_transfer_io.load(vgg_model_path)
                print(f">> model downloaded: {model_path}")

                self.predictor = InferenceContext(
                    VggLikeUmapPredictor(
                        f"{self.parameters.model_path}/feature_extractor.pth",
                        f"{self.parameters.model_path}/umap_model.pickle",
                        f"{self.parameters.model_path}/labels.yaml",
                        # vgg_model_path["file_name"],
                        self.model_factory,
                        io_factory,
                    )
                )
                # os.remove(f"data/{self.parameters.model_path}")
        else:
            raise NotImplementedError(f"{self.parameers.type} is not implemented")
        # os.remove(model_path["file_name"])

    def run(self, input: np.ndarray) -> Any:
        return self.predictor.run(input)
