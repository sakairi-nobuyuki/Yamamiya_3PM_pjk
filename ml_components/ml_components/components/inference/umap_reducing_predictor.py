# coding: utf-8

import os
from typing import Dict, List, Tuple, Union

import numpy as np
import umap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ...io import IOTemplate, PickleIO
from .template_predictor import TemplatePredictor


class UmapReducingPredictor(TemplatePredictor):
    def __init__(self, mode: str, model_path: str = None, s3: IOTemplate = None):
        if mode == "train":
            self.reducer = umap.UMAP()
            self.regression = LogisticRegression()
            if not isinstance(s3, IOTemplate):
                raise TypeError(f"{type(s3)} is not that of IOTemplate")
            else:
                self.s3 = s3
        elif mode == "inference":
            if model_path is None or IOTemplate is None:
                raise Exception(f"model_path: {model_path}, IOTemplate: {IOTemplate}")
            if not isinstance(s3, IOTemplate):
                raise TypeError(f"{type(s3)} is not that of IOTemplate")
            else:
                self.s3 = s3
            self.reducer, self.regression = self.load_models(model_path)

    def predict(self, input: np.ndarray) -> np.ndarray:
        """Predict a classifier task

        Args:
            input (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        print(
            f"input: max: {np.max(input)}, size: {input.size}, dim: {input.ndim}, shape: {input.shape}"
        )
        scaled_input = StandardScaler().fit_transform(input)
        print(
            f"scaled input: max: {np.max(scaled_input)}, size: {scaled_input.size}, dim: {scaled_input.ndim}, shape: {scaled_input.shape}"
        )
        reduced = self.reducer.fit_transform(scaled_input)
        predicted = self.regression.predict(reduced)
        return predicted[0]

    def fit_transform(
        self, feat_array: np.ndarray, y: np.ndarray = None
    ) -> Union[Tuple[np.ndarray], np.ndarray]:
        """Train models of UMPA and UMAP based classifier.

        Args:
            feat_array (np.ndarray): Input data
            y (np.ndarray, optional): Label list. If this argument is given, the model will be a classifier. Defaults to None.

        Returns:
            Union[Tuple[np.ndarray], np.ndarray]: UMAP model or UMAP and lagarithmic regression model
        """
        if y is not None:
            X_embded = self.reducer.fit_transform(feat_array, y=y)
            self.regression.fit(X_embded, y)
            return self.reducer, self.regression
        else:
            # return self.reducer.fit_transform(feat_array)
            return self.reducer.fit_transform(feat_array)

    def load_models(
        self, model_path: str
    ) -> List[Union[umap.UMAP, LogisticRegression]]:
        print("downloading UMAP model from ", model_path)
        object_list = self.s3.load(model_path)

        if not isinstance(object_list[0], umap.UMAP):
            raise TypeError(
                f"The first element of object_list must be umap.UMAP: {type(object_list[0])}"
            )
        if not isinstance(object_list[1], LogisticRegression):
            raise TypeError(
                f"The second element of object_list must be LogisticRegresson: {type(object_list[1])}"
            )
        print(f">> downloaded {len(object_list)} objs for UMAP ")
        return object_list

    def save_models(
        self, object_list: List[Union[umap.UMAP, LogisticRegression]], model_path: str
    ) -> Dict[str, str]:
        if not isinstance(object_list, List):
            raise TypeError(f"input must be List: {type(object_list)}")

        if not isinstance(object_list[0], umap.UMAP):
            raise TypeError(
                f"The first element of object_list must be umap.UMAP: {type(object_list[0])}"
            )
        if not isinstance(object_list[1], LogisticRegression):
            raise TypeError(
                f"The second element of object_list must be LogisticRegresson: {type(object_list[1])}"
            )

        return self.s3.save(object_list, model_path)
