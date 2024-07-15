# coding: utf-8

import os
from typing import Dict

import numpy as np
import yaml

from ..components.dataset_loader import KaggleDatasetLoader
from ..components.factory import IoModuleFactory
from ..components.inference import InferenceContext, VggLikeClassifierPredictor
from ..data_structures import AccuracyMeasurementParameters
from ..models.factory import ModelFactoryTemplate, VggLikeClassifierFactory
from . import TemplatePipeline


class AccuracyMeasPipeline(TemplatePipeline):
    """Accuracy measurement code.
    Specification:
        To choose a nice model from a models after training,
        - Load all the model in a directory of the model bucket.
        - Load all the test dataset
        - Run inference to measure confusion matrix or IoU

    Args:
        TemplatePipeline (_type_): Template class
    """

    def __init__(self, parameters_str: str) -> None:
        """Initialize the model

        Tasks:
            - Load a list of models
            - Load dataset file path list
            - Set a metrics
            - Load inference context
            - Load model factory
        """

        ### load parameters
        self.parameters = AccuracyMeasurementParameters(
            **yaml.safe_load(parameters_str)
        )

        io_factory = IoModuleFactory()
        self.config_io = io_factory.create(**dict(type="config", bucket_name="config"))

        ### configure dataset loader
        print(">> loading dataset")
        # TODO: in future dataset loader should be created with a factory
        self.img_io = io_factory.create(**dict(type="image", bucket_name="dataset"))
        self.dataset_loader = KaggleDatasetLoader(self.parameters.dataset, self.img_io)

        label_list = self.dataset_loader.load()

        # test_dataset_dir = os.path.dirname(self.parameters.dataset_directory_path)
        # test_dataset_dir = self.parameters.dataset_directory_path

        self.model_trans_io = io_factory.create(
            **dict(type="transfer", bucket_name="models")
        )

        ### create model lists to evaluate
        self.model_list = self.model_trans_io.blob

        ### create correct data list in a list of dicts [{"file_path": str, "correct_data": str}, ...]
        if self.parameters.type == "classification":
            label_list = self.dataset_loader.get_label_list()
            file_path_list = self.img_io.get_blob()
            self.label_dict = {
                i_label: label for i_label, label in enumerate(label_list)
            }
            print(f">> creating data list dict for {label_list}")
            self.file_list_dict = {
                # label: [label for label in label_list if label.split("/") is label]
                label: [
                    file_path
                    for file_path in file_path_list
                    if label in file_path and "test" in file_path
                ]
                for label in label_list
            }
            print(f">> file list dict: {self.file_list_dict}")

        else:
            raise NotImplementedError(f"{self.parameters.type} is not implemented.")

        self.model_factory = VggLikeClassifierFactory()

    def construct_predictor(self, model_path: str, factory: ModelFactoryTemplate):
        if not isinstance(factory, ModelFactoryTemplate):
            raise NotImplementedError(f"{factory} is not implemented")

        ### download model
        local_model_path = self.model_trans_io.load(model_path)["file_name"]

        ### create predictor
        predictor = InferenceContext(
            VggLikeClassifierPredictor(local_model_path, factory)
        )

        os.remove(local_model_path)

        return predictor

    def run(self):
        print("Run the accuracy measurement")
        res_list = []
        correct_list = []
        n_classes = len(self.label_dict)

        for model_path in self.model_list:
            print(f">> Meas: {model_path}")
            predictor = self.construct_predictor(model_path, self.model_factory)

            for label, file_path_list in self.file_list_dict.items():
                i_correct_label = [
                    i_label_ref
                    for i_label_ref, label_ref in self.label_dict.items()
                    if label == label_ref
                ][0]
                print(label, i_correct_label)
                for file_path in file_path_list:
                    img = self.img_io.load(file_path)
                    res = predictor.run(img)
                    # print(f"{file_path}: {res}")
                    correct_list.append(i_correct_label)
                    res_list.append(res)

            confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
            classes = np.unique(correct_list)
            for i in range(n_classes):
                for j in range(n_classes):
                    confusion_matrix[i, j] = np.sum(
                        (correct_list == classes[i]) & (res_list == classes[j])
                    )
            print(">> confusion matrix: ", confusion_matrix)

            tp = confusion_matrix[0][0]
            tn = confusion_matrix[1][1]
            fp = confusion_matrix[1][0]
            fn = confusion_matrix[0][1]

            accuracy = (tp + tn) / float(tp + tn + fp + fn)
            precision = tp / float(tp + fp)
            recall = tp / float(tp + fn)
            f1_score = 2 * (precision * recall) / (precision + recall)

            print(
                f">> precision: {precision}, accuracy: {accuracy}, f1: {f1_score}, recall: {recall}"
            )
