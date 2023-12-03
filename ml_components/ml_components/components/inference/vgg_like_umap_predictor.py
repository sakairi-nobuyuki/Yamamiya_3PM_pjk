# coding: utf-8

from typing import Dict

import numpy as np

from ...models.factory import ModelFactoryTemplate
from . import TemplatePredictor, UmapReducingPredictor, VggLikeFeatureExtractor


class VggLikeUmapPredictor(TemplatePredictor):
    def __init__(
        self,
        model_path: str,
        label_dict: Dict[int, str],
        model_factory: ModelFactoryTemplate,
        n_layer: int = -1,
    ) -> None:
        print("VggLikeUmapPredictor")
        print(">> model path: ", model_path)
        print(">> n layer: ", n_layer)
        self.vgg = VggLikeFeatureExtractor(model_path, model_factory, n_layer)
        self.reducer = UmapReducingPredictor()
        self.label_dict = label_dict

    def predict(self, image: np.ndarray) -> str:
        """Get a feature vector of input image

        Args:
            image (np.ndarray): input image

        Returns:
            str: Classified result
        """
        feat = self.vgg.predict(image)
        predicted = self.reducer.predict(feat)
        res = self.label_dict[predicted]

        return res
