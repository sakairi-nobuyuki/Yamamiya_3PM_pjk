# coding: utf-8

import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
import torchvision.models as models
import umap
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from ..components.factory import IoModuleFactory
from ..components.inference import UmapReducingPredictor, VggLikeFeatureExtractor
from ..models.factory import ModelFactoryTemplate
from . import TemplateTrainer


class VggLikeUmapClassifierTrainer(TemplateTrainer):
    def __init__(
        self,
        data_path_list_dict: Dict[str, List[str]],
        factory: ModelFactoryTemplate,
        io_factory: IoModuleFactory,
        model_path: str = None,
        save_model_path_base: str = None,
        n_layer: int = -1,
    ) -> None:
        """Initialize trainer

        Args:
            data_path_list_dict (Dict[str, List[str]]): a list of dict of which key is data file path in the storage and value is its label.
            factory (ModelFactoryTemplate): NN model factory. The model is used as a feature extractor.
            image_io (DataTransferS3): IO component to load image faile
            config_io (S3ConfigIO): IO component for save label
            model_path (str, optional): model name. Defaults to None.
            save_model_path_base (str, optional): model path to save. Defaults to None.
            n_layer (int, optional): The number of NN layer to reduce when using as a feature extractor. Defaults to -1, -3 could be promissing.

        Raises:
            TypeError: factory, io things
        """
        if not isinstance(factory, ModelFactoryTemplate):
            raise TypeError(f"{type(factory) is not {ModelFactoryTemplate}}")
        if not isinstance(io_factory, IoModuleFactory):
            raise TypeError(f"{io_factory} is not {IoModuleFactory}")

        print("VggLikeUmapPredictor")

        self.data_path_dict_list = data_path_list_dict
        self.label_map_dict = self.aggregate_label_map_dict(self.data_path_dict_list)
        self.image_io = io_factory.create(**dict(type="image", bucket_name="dataset"))
        self.config_io = io_factory.create(**dict(type="config", bucket_name="models"))
        self.transfer_io = io_factory.create(
            **dict(type="transfer", bucket_name="models")
        )
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
        self.reducer = UmapReducingPredictor(
            "train", s3=io_factory.create(**dict(type="pickle", bucket_name="models"))
        )

        # set model related things storage
        if save_model_path_base is None:
            self.save_model_path_base = f"classifier/vgg_umap/{self.__get_current_time()}"
        else:
            self.save_model_path_base = save_model_path_base

        print(">> part of data_path_dict_list: ", self.data_path_dict_list[:5])
        print(">> label map dict: ", self.label_map_dict)
        label_map_dict_inverse = {v: k for k, v in self.label_map_dict.items()}

        self.data_path_dict_list_int = [
            {k: label_map_dict_inverse[v] for k, v in data_path_dict.items()}
            for data_path_dict in self.data_path_dict_list
            if len(data_path_dict) == 1
        ]

    def train(self) -> str:
        """Train the UMAP parameter so that it will minimize the distance between clusters.
        - Create a list of combinations of two classes.
        - Calculate distances of each combinations.
        - Pertub the parameters and optimize so that it will minimize the total distances.

        Args:
            image_list_dict (Dict[List[np.ndarray]]): Input images list of classes

        Returns:
            Dict[str, Any]: UMAP parameters
        """
        # create a set of feature vectors and label list
        feat_list = []
        label_list = []

        print(">> train start: ")
        print(">> len self.data_path_dict_list_int: ", len(self.data_path_dict_list_int))
        print(">> feature extraction ")

        for train_progress in tqdm(self.data_path_dict_list_int):
            for file_path, label in train_progress.items():
                input = self.image_io.load(file_path)
                feat = self.vgg.predict(input)
                feat_list.append(feat)
                label_list.append(label)
        feat_array = np.concatenate(feat_list)
        label_array = np.array(label_list)
        print(type(feat_array), type(label_array))
        print(">> UMAP fit ")
        umap_model = self.reducer.fit_transform(feat_array, label_array)
        print(">> UMAP feat: ", umap_model, type(umap_model[0]), type(umap_model[1]))

        # save model and labels
        self.save_ml_model(umap_model[0], umap_model[1])
        self.save_nn_model(self.vgg.model)
        self.save_labels(self.get_label_map_dict())

        return self.save_model_path_base

    def get_label_map_dict(self) -> Dict[int, str]:
        """Get label map as a form of dict.
        Let's say there are a set of data and its label in a dataset like,
            file_1: label_1
            file_2: label_2
            file_3: label_1
            ...
        However the label itself is treated as integer, hence we should prepare a mapping from label in integer to
        label in string like this.

        {
            1: "label_1", 2: "label_2", ...
        }

        Returns:
            Dict[int, str]: A dict to map integer to string
        """
        return self.label_map_dict

    def aggregate_label_map_dict(self, data_path_dict_list) -> Dict[int, str]:
        """Get label map as a form of dict.
        Let's say there are a set of data and its label in a dataset like,
            file_1: label_1
            file_2: label_2
            file_3: label_1
            ...
        However the label itself is treated as integer, hence we should prepare a mapping from label in integer to
        label in string like this.

        {
            1: "label_1", 2: "label_2", ...
        }

        Returns:
            Dict[int, str]: A dict to map integer to string
        """
        label_map_dict = {
            i_label: label
            for i_label, label in enumerate(
                {
                    str(list(data_path_dict.values())[0])
                    for data_path_dict in data_path_dict_list
                }
            )
        }

        return label_map_dict

    def save_ml_model(
        self,
        umap_model: umap.UMAP,
        regression_model: LogisticRegression,
        model_name: str = None,
    ) -> str:
        if model_name is None:
            file_name = f"umap_model.pickle"
        else:
            file_name = model_name if "pickle" in model_name else f"{model_name}.pickle"

        model_path = os.path.join(self.save_model_path_base, file_name)
        self.reducer.save_models([umap_model, regression_model], model_path)

        return model_path

    def save_nn_model(self, model: models, model_name: str = None) -> str:
        if model_name is None:
            file_name = f"feature_extractor.pth"
        else:
            file_name = model_name if "pth" in model_name else f"{model_name}.pth"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            # "optimizer_state_dict": self.optimizer.state_dict(),
        }
        model_path = os.path.join(self.save_model_path_base, file_name)

        if model_name is None:
            file_name = f"feature_extractor.pth"
        else:
            file_name = model_name
        torch.save(checkpoint, file_name)

        self.transfer_io.save(file_name, model_path)

        os.remove(file_name)

        return model_path

    def save_labels(self, label_dict: Dict[str, str], file_name: str = None):
        if file_name is None:
            file_name = f"labels.yaml"
        else:
            file_name = file_name if "yaml" in file_name else f"{file_name}.yaml"

        label_path = os.path.join(self.save_model_path_base, file_name)
        self.config_io.save(label_dict, label_path)

        return label_path

    def __get_current_time(self) -> str:
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")

        return current_time
