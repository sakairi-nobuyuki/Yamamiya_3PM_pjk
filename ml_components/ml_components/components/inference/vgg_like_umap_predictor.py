# coding: utf-8

import numpy as np

from ...models.factory import ModelFactoryTemplate
from . import TemplatePredictor, UmapReducingPredictor, VggLikeFeatureExtractor


class VggLikeUmapPredictor(TemplatePredictor):
    def __init__(
        self, model_path: str, model_factory: ModelFactoryTemplate, n_layer: int = -1
    ) -> None:
        print("VggLikeUmapPredictor")
        print(">> model path: ", model_path)
        print(">> n layer: ", n_layer)
        self.vgg = VggLikeFeatureExtractor(model_path, model_factory, n_layer)
        self.reducer = UmapReducingPredictor()

    def predict(self, image: np.ndarray) -> np.ndarray:
        feat = self.vgg.predict(image)
        print("feat: ", feat)
        # TODO: n_samples and n_features should be class attributes
        # n_samples = feat.shape[0]
        # n_features = np.prod(feat.shape[1:])

        # feat_reshaped = feat.reshape(n_samples, n_features)

        # print("feat reshaped shape: ", feat_reshaped.shape)
        # print(
        #    "feat reshaped, reshaped feat mean: ",
        #    feat_reshaped,
        #    feat_reshaped.mean(),
        #    feat_reshaped.max(),
        #    feat_reshaped.min(),
        # )

        # reduced_feat = self.reducer.predict(feat_reshaped)
        reduced_feat = self.reducer.predict(feat)

        return reduced_feat
