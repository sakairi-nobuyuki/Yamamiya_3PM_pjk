# coding: utf-8

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap

from .template_predictor import TemplatePredictor


class UmapReducingPredictor(TemplatePredictor):
    def __init__(self):
        self.reducer = umap.UMAP()

    def predict(self, input: np.ndarray):
        # return self.reducer.fit(input)
        print(
            f"input: max: {np.max(input)}, size: {input.size}, dim: {input.ndim}, shape: {input.shape}"
        )
        scaled_input = StandardScaler().fit_transform(input)
        print(
            f"scaled input: max: {np.max(scaled_input)}, size: {scaled_input.size}, dim: {scaled_input.ndim}, shape: {scaled_input.shape}"
        )
        return self.reducer.fit_transform(scaled_input)
