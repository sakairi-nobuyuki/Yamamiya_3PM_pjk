# coding: utf-8

from typing import Any

from . import TemplatePredictor


class InferenceContext:
    def __init__(self, predictor: TemplatePredictor):
        if not isinstance(predictor, TemplatePredictor):
            raise TypeError(
                f"{type(predictor)} is not a concrete class of TemplatePredictor"
            )
        self.predictor_ = predictor

    @property
    def inference_strategy(self) -> TemplatePredictor:
        return self.predictor_

    @inference_strategy.setter
    def inference_strategy(self, predictor: TemplatePredictor) -> None:
        self.predictor_ = predictor

    def run(self, input: Any) -> Any:
        return self.predictor_.predict(input)
