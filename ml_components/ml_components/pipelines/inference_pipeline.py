# coding: utf-8

from typing import Any, Dict
import numpy as np
import json

from ..components.factory import IoModuleFactory
from ..data_structures import PredictionParameters
from ..models.factory import ModelFactoryTemplate
from ..components.inference import InferenceContext, VggLikeClassifierPredictor

class InferencePipeline:
    #def __init__(self, io_config: Dict[str, str], yaml_path: str, location: str) -> None:
    def __init__(self, parameters_str: str) -> None:
        
        
        ### load parameters
        ### parse str into PredictParameters
        parameters_dict = json.loads(parameters_str)
        parameters = PredictionParameters(**parameters_dict)
        if isinstance(parameters, PredictionParameters):
            self.parameers = parameters
        else:
            raise TypeError(f"{type(parameters)} is not an instance of PredictionParameters")

        ### load model factory
        ### TODO: initialize model factory in accordance with parameters
        print("Configuring DNN model")
        self.model_factory=VggLikeClassifierPredictor()

        ### select predictor in accordance with parameters
        print("Configuring the predictor")
        if self.parameers.type == "binary":
            print(">> prediction type: binary classification")
            print(">> model path: ", self.parameers.model_path)
            self.predictor = InferenceContext(VggLikeClassifierPredictor(self.parameers.model_path, self.model_factory))
        else:
            raise NotImplementedError(f"{self.parameers.type} is not implemented")

    def predict(self, input: np.ndarray) -> Any:
        
        return self.predictor.run(input)