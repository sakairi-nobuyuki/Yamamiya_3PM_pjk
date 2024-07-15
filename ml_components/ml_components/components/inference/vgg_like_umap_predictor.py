# coding: utf-8

import os
from typing import Dict

import numpy as np

from ...components.factory import IoModuleFactory
from ...models.factory import ModelFactoryTemplate
from . import TemplatePredictor, UmapReducingPredictor, VggLikeFeatureExtractor


class VggLikeUmapPredictor(TemplatePredictor):
    def __init__(
        self,
        ### TODO: This model path should be in the storage, not local
        feature_extractor_path: str,
        umap_model_path: str,
        label_path: str,
        model_factory: ModelFactoryTemplate,
        io_factory: IoModuleFactory,
        n_layer: int = -1,
    ) -> None:
        print("VggLikeUmapPredictor")
        print(">> feature extractor model path: ", feature_extractor_path)
        print(">> UMAP model path: ", umap_model_path)
        print(">> label path: ", label_path)
        print(">> n layer: ", n_layer)

        ### Load feature extractor model
        trans_s3 = io_factory.create(**dict(type="transfer", bucket_name="models"))
        feature_extractor_path = trans_s3.load(feature_extractor_path)["file_name"]
        print(" >> downloaded feature extractor: ", feature_extractor_path)
        self.vgg = VggLikeFeatureExtractor(
            model_factory, n_layer=n_layer, model_path=feature_extractor_path
        )
        os.remove(feature_extractor_path)

        ### load UMAP model
        self.reducer = UmapReducingPredictor(
            "inference",
            model_path=umap_model_path,
            s3=io_factory.create(**dict(type="pickle", bucket_name="models")),
        )

        ### loading label
        label_s3 = io_factory.create(**dict(type="config", bucket_name="models"))
        self.label_dict = label_s3.load(label_path)

    def predict(self, image: np.ndarray) -> str:
        """Get a feature vector of input image

        Args:
            image (np.ndarray): input image

        Returns:
            str: Classified result
        """
        feat = self.vgg.predict(image)
        predicted = self.reducer.predict(feat)
        print("predicted: ", predicted, self.label_dict)
        res = self.label_dict[predicted]

        return res
