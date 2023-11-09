# coding: utf-8

import os
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from ..components.dataloader import BinaryClassifierDataloaderFactory
from ..components.dataset_loader import CustomDatasetLoader
from ..components.inference import UmapReducingPredictor, VggLikeFeatureExtractor
from ..data_structures import TrainParameters
from ..io import IOTemplate, S3ImageIO
from ..models.factory import ModelFactoryTemplate
from . import TemplateTrainer


class VggLikeUmapClassifierTrainer(TemplateTrainer):
    def __init__(
        self,
        data_path_list_dict: Dict[str, List[str]],
        factory: ModelFactoryTemplate,
        image_io: S3ImageIO,
        model_path: str = None,
        n_layer: int = -1,
    ) -> None:
        if not isinstance(factory, ModelFactoryTemplate):
            raise TypeError(f"{type(factory) is not {ModelFactoryTemplate}}")

        print("VggLikeUmapPredictorTrainer")
        self.data_path_dict_list = data_path_list_dict
        self.image_io = image_io
        self.factory = factory

        # Train the model on your dataset using binary cross-entropy loss and SGD optimizer
        print(">> model path: ", model_path)
        print(">> n layer: ", n_layer)
        if model_path is None:
            self.vgg = VggLikeFeatureExtractor(self.factory, n_layer)
        else:
            self.vgg = VggLikeFeatureExtractor(
                self.factory, n_layer, model_path=model_path
            )
        self.reducer = UmapReducingPredictor()


    def train(self) -> np.ndarray:
        """Train the UMAP parameter so that it will minimize the distance between clusters.
        - Create a list of combinations of two classes.
        - Calculate distances of each combinations.
        - Pertub the parameters and optimize so that it will minimize the total distances.

        Args:
            image_list_dict (Dict[List[np.ndarray]]): Input images list of classes

        Returns:
            Dict[str, Any]: UMAP parameters
        """
        data_path_dict_list = self.data_path_dict_list
        
        # create a set of feature vectors and label list
        feat_list = []
        label_list = []
        
        print(">> Extracting feat vectors with VGG ")
        with tqdm(
            data_path_dict_list, total=len(data_path_dict_list)
        ) as train_progress:
            for item in train_progress:
                input = self.image_io.load(list(item.keys())[0])
                feat = self.vgg.predict(input)
                feat_list.append(feat)
                label_list.append(list(item.values())[0])
        feat_array = np.concatenate(feat_list)

        label_dict = {label: i_label for i_label, label in enumerate(list(set(label_list)))}
        label_list_num = list(map(label_dict.get, label_list))

        label_array = np.array(label_list_num)
        print("feat list: ", feat_list, feat_array)
        print("label list: ", label_list, label_array)

        print(">> Obtaining reduced feat vectors with UMAP supervised")
        umap_feat = self.reducer.reducer.fit_transform(feat_array, y=label_array)

        return umap_feat

