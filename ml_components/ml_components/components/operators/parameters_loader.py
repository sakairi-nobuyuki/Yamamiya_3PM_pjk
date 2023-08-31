# utf-8: coding

import os

import yaml

from ...data_structures import TrainParameters
from ...io import S3ConfigIO


def load_train_parameters(yaml_path: str, config_s3: S3ConfigIO) -> TrainParameters:
    """Loading a yaml file for parameters for training and load its contents.
    Theseparameters are validated and summerized into an instance of TrainParameters.

    Args:
        yaml_path (str): Parameters yaml file path.

    Returns:
        TrainParameters: Parameters instance.
    """

    if yaml_path not in config_s3.blob:
        raise FileNotFoundError(f"{yaml_path} is not found for loading train parameters.")

    parameters_dict = config_s3.load(yaml_path)

    return TrainParameters(**parameters_dict)
